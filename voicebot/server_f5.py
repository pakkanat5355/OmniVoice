#!/usr/bin/env python3
"""
OmniVoice Voicebot — F5-TTS Thai Edition (server_f5.py)

  TTS : F5-TTS Thai fine-tuned  (f5-tts library)
  ASR : typhoon-ai/typhoon-whisper-turbo  (transformers pipeline)

Install (on RunPod):
    pip install f5-tts transformers accelerate

Requires reference audio for voice cloning:
    voicebot/ref_voice.wav   (or .mp3 .flac .ogg)
    voicebot/ref_voice_text.txt  — transcript of that audio (Thai text)
                                   if missing, a default line is used

Usage:
    cd /app
    uv run python voicebot/server_f5.py
"""

import asyncio
import io
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import pipeline as hf_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = _best_device()

# ---------------------------------------------------------------------------
# Reference voice config
# ---------------------------------------------------------------------------

_VOICE_DIR = os.path.dirname(__file__)

_REF_VOICE_PATH: str | None = next(
    (os.path.join(_VOICE_DIR, f"ref_voice.{ext}")
     for ext in ("wav", "mp3", "flac", "ogg")
     if os.path.exists(os.path.join(_VOICE_DIR, f"ref_voice.{ext}"))),
    None,
)

_REF_TEXT_PATH = os.path.join(_VOICE_DIR, "ref_voice_text.txt")
if os.path.exists(_REF_TEXT_PATH):
    with open(_REF_TEXT_PATH, encoding="utf-8") as _f:
        _REF_VOICE_TEXT = _f.read().strip()
else:
    # Fallback: short Thai sentence — replace with the actual transcript of your ref audio
    _REF_VOICE_TEXT = "สวัสดีค่ะ ยินดีให้บริการค่ะ"

# Thai fine-tuned F5-TTS checkpoint on HuggingFace
# Override with env vars if you have a local checkpoint:
#   F5TTS_REPO  = HuggingFace repo id  (default: VIZINTZOR/F5-TTS-Thai)
#   F5TTS_CKPT  = filename of .pt file (default: auto-detect)
#   F5TTS_VOCAB = filename of vocab    (default: vocab.txt)
_F5TTS_HF_REPO    = os.environ.get("F5TTS_REPO",  "VIZINTZOR/F5-TTS-Thai")
_F5TTS_CKPT_NAME  = os.environ.get("F5TTS_CKPT",  "")   # empty = auto-detect
_F5TTS_VOCAB_NAME = os.environ.get("F5TTS_VOCAB", "vocab.txt")

# ---------------------------------------------------------------------------
# Load ASR — Typhoon Whisper Turbo (via transformers, no OmniVoice needed)
# ---------------------------------------------------------------------------

logger.info("Loading Typhoon Whisper Turbo ASR ...")
_asr_pipe = hf_pipeline(
    "automatic-speech-recognition",
    model="typhoon-ai/typhoon-whisper-turbo",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device=device,
)
logger.info("Typhoon Whisper ASR loaded.")

# ---------------------------------------------------------------------------
# Load TTS — F5-TTS Thai fine-tuned
# ---------------------------------------------------------------------------

from f5_tts.api import F5TTS  # noqa: E402 (import after torch setup)
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402


def _resolve_thai_checkpoint() -> tuple[str, str]:
    """Download Thai checkpoint + vocab from HuggingFace, return (ckpt_path, vocab_path)."""
    repo = _F5TTS_HF_REPO
    logger.info(f"Resolving F5-TTS Thai checkpoint from {repo} ...")

    # Find checkpoint filename if not specified
    ckpt_name = _F5TTS_CKPT_NAME
    if not ckpt_name:
        files = list(list_repo_files(repo))
        pt_files = sorted(f for f in files if f.endswith(".pt") or f.endswith(".safetensors"))
        if not pt_files:
            raise RuntimeError(f"No .pt checkpoint found in {repo}")
        ckpt_name = pt_files[-1]  # highest step (last alphabetically)
        logger.info(f"Auto-detected checkpoint: {ckpt_name}")

    ckpt_path  = hf_hub_download(repo_id=repo, filename=ckpt_name)
    vocab_path = hf_hub_download(repo_id=repo, filename=_F5TTS_VOCAB_NAME)
    return ckpt_path, vocab_path


logger.info(f"Loading F5-TTS Thai ({_F5TTS_HF_REPO}) ...")
_ckpt_path, _vocab_path = _resolve_thai_checkpoint()
_f5tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=_ckpt_path, vocab_file=_vocab_path)
logger.info("F5-TTS loaded.")

if _REF_VOICE_PATH:
    logger.info(f"Reference voice : {_REF_VOICE_PATH}")
    logger.info(f"Reference text  : {_REF_VOICE_TEXT[:60]}...")
else:
    logger.warning("No ref_voice file found — F5-TTS requires a reference audio!")
    logger.warning("Place ref_voice.wav (+ ref_voice_text.txt) in voicebot/ and restart.")

# Serialise GPU calls
_gpu_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

def _transcribe_sync(audio_array: np.ndarray, sample_rate: int) -> str:
    audio_input = {"array": audio_array, "sampling_rate": sample_rate}
    result = _asr_pipe(
        audio_input,
        generate_kwargs={"language": "th", "task": "transcribe"},
    )
    return result["text"].strip()


async def transcribe(audio_array: np.ndarray, sample_rate: int) -> str:
    async with _gpu_lock:
        return await asyncio.get_event_loop().run_in_executor(
            None, _transcribe_sync, audio_array, sample_rate
        )


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

_TTS_SPEED = 0.85  # F5-TTS speed multiplier (< 1 = slower)


def _tts_sync(text: str) -> tuple[np.ndarray, int]:
    """Returns (float32 waveform, sample_rate)."""
    wav, sr, _ = _f5tts.infer(
        ref_file=_REF_VOICE_PATH,
        ref_text=_REF_VOICE_TEXT,
        gen_text=text,
        speed=_TTS_SPEED,
        seed=-1,
        remove_silence=True,
    )
    return np.asarray(wav, dtype=np.float32), int(sr)


async def tts(text: str) -> tuple[np.ndarray, int]:
    async with _gpu_lock:
        return await asyncio.get_event_loop().run_in_executor(None, _tts_sync, text)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def pcm8k_to_float32(audio_bytes: bytes) -> np.ndarray:
    pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0


def float32_to_pcm8k_bytes(wav: np.ndarray, src_sr: int) -> bytes:
    """Resample from src_sr → 8kHz 16-bit PCM bytes for Asterisk."""
    tensor = torch.from_numpy(wav).unsqueeze(0)
    tensor_8k = torchaudio.functional.resample(tensor, src_sr, 8000)
    pcm16 = (tensor_8k.squeeze(0).numpy() * 32768).clip(-32768, 32767).astype(np.int16)
    return pcm16.tobytes()


async def _tts_pcm8k(text: str) -> bytes:
    wav, sr = await tts(text)
    return float32_to_pcm8k_bytes(wav, sr)


# ---------------------------------------------------------------------------
# IVR — ABC Call Center  (identical to server_gpu.py)
# ---------------------------------------------------------------------------

IVR_GREETING = (
    "สวัสดีค่ะ ติดต่อบริษัท ABC ยินดีให้บริการค่ะ "
    "กรุณาแจ้งเรื่องที่ต้องการได้เลยค่ะ "
    "เช่น สอบถามคะแนนสะสม สอบถามโปรโมชั่น แจ้งปัญหา "
    "ตรวจสอบออเดอร์ สมัครสมาชิก หรือติดต่อเจ้าหน้าที่ค่ะ"
)

IVR_INTENTS = {
    "check_points": {
        "name": "สอบถามคะแนนสะสม",
        "keywords": ["คะแนน", "แต้ม", "แนน", "point", "reward", "สะสม"],
        "response": (
            "ขณะนี้คะแนนสะสมของคุณลูกค้ามีทั้งหมด หนึ่งพันสองร้อยห้าสิบ คะแนนค่ะ "
            "สามารถนำคะแนนไปแลกรับสิทธิ์พิเศษได้ที่เว็บไซต์ abc.com ค่ะ"
        ),
    },
    "promotions": {
        "name": "สอบถามโปรโมชั่น",
        "keywords": ["โปรโมชั่น", "ส่วนลด", "ดีล", "โปร", "promotion", "discount", "offer"],
        "response": (
            "ขณะนี้บริษัท ABC มีโปรโมชั่นพิเศษ ลด ยี่สิบ เปอร์เซ็นต์ "
            "สำหรับสินค้าทุกรายการถึงสิ้นเดือนนี้ค่ะ "
            "สามารถดูรายละเอียดเพิ่มเติมได้ที่เว็บไซต์ abc.com ค่ะ"
        ),
    },
    "complaint": {
        "name": "แจ้งปัญหาหรือร้องเรียน",
        "keywords": ["ปัญหา", "ร้องเรียน", "เสีย", "ไม่ได้", "แจ้ง", "complaint", "บกพร่อง", "ผิดพลาด"],
        "response": (
            "รับทราบค่ะ ดิฉันจะบันทึกเรื่องร้องเรียนของคุณลูกค้าไว้ "
            "และทีมงานจะติดต่อกลับภายใน ยี่สิบสี่ ชั่วโมงค่ะ ขอบคุณที่แจ้งให้ทราบนะคะ"
        ),
    },
    "order_status": {
        "name": "ตรวจสอบสถานะออเดอร์",
        "keywords": ["ออเดอร์", "คำสั่งซื้อ", "สถานะ", "จัดส่ง", "order", "delivery", "tracking", "พัสดุ", "ของ"],
        "response": (
            "กรุณาแจ้งหมายเลขคำสั่งซื้อของคุณลูกค้าได้เลยค่ะ "
            "หรือทีมงานจะส่ง SMS แจ้งสถานะให้ที่หมายเลขที่ลงทะเบียนไว้ค่ะ"
        ),
    },
    "membership": {
        "name": "สมัครสมาชิก",
        "keywords": ["สมัคร", "สมาชิก", "member", "register", "ลงทะเบียน", "เปิดบัญชี"],
        "response": (
            "สามารถสมัครสมาชิกได้ง่ายๆ ผ่านเว็บไซต์ abc.com "
            "หรือดาวน์โหลดแอปพลิเคชัน ABC ได้เลยค่ะ การสมัครใช้เวลาไม่ถึง ห้า นาทีค่ะ"
        ),
    },
    "transfer": {
        "name": "ติดต่อเจ้าหน้าที่",
        "keywords": ["เจ้าหน้าที่", "คุยกับคน", "agent", "operator", "โอนสาย", "transfer", "พนักงาน", "คน"],
        "response": "กรุณาถือสายรอสักครู่นะคะ กำลังโอนสายให้เจ้าหน้าที่ค่ะ",
    },
}

IVR_REPROMPT      = "คุณลูกค้าต้องการสอบถามเรื่องอะไรนะคะ รบกวนพูดอีกทีค่ะ"
IVR_NOT_CONFIRMED = "รับทราบค่ะ กรุณาแจ้งเรื่องที่ต้องการใหม่อีกครั้งได้เลยค่ะ"
IVR_ASK_MORE      = "คุณลูกค้ามีคำถามอื่นเพิ่มเติมไหมคะ"
IVR_GOODBYE       = (
    "ขอบคุณที่ใช้บริการบริษัท ABC ค่ะ "
    "ช่วงนี้อากาศเปลี่ยนแปลงบ่อย ดูแลสุขภาพด้วยนะคะ ขอบคุณค่ะ"
)


class IVRState(Enum):
    AWAITING         = "awaiting"
    CONFIRMING       = "confirming"
    ASKING_PHONE     = "asking_phone"
    CONFIRMING_PHONE = "confirming_phone"
    ASKING_MORE      = "asking_more"


_TH_DIGIT = {"0":"ศูนย์","1":"หนึ่ง","2":"สอง","3":"สาม","4":"สี่",
              "5":"ห้า","6":"หก","7":"เจ็ด","8":"แปด","9":"เก้า"}

_TH_WORD_TO_DIGIT = {
    "ศูนย์":"0","หนึ่ง":"1","สอง":"2","สาม":"3","สี่":"4",
    "ห้า":"5","หก":"6","เจ็ด":"7","แปด":"8","เก้า":"9","เอ็ด":"1",
}


def extract_phone(text: str) -> str:
    t = text
    for word, digit in _TH_WORD_TO_DIGIT.items():
        t = t.replace(word, digit)
    return re.sub(r"[^\d]", "", t)


def phone_to_thai_speech(digits: str) -> str:
    return " ".join(_TH_DIGIT.get(d, d) for d in digits)


def detect_intent(text: str) -> str | None:
    t = text.lower()
    for key, intent in IVR_INTENTS.items():
        if any(kw in t for kw in intent["keywords"]):
            return key
    return None


def detect_yes_no(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["ไม่", "ผิด", "no", "เปลี่ยน", "cancel", "ยกเลิก"]):
        return "no"
    if any(k in t for k in ["ใช่", "ถูก", "ครับ", "ค่ะ", "คะ", "ได้", "ok", "yes", "right", "ยืนยัน"]):
        return "yes"
    return "unknown"


# ---------------------------------------------------------------------------
# FastAPI — pre-generate greeting
# ---------------------------------------------------------------------------

_greeting_pcm8k: bytes | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _greeting_pcm8k
    if _REF_VOICE_PATH:
        logger.info("Pre-generating greeting audio (F5-TTS)...")
        _greeting_pcm8k = await _tts_pcm8k(IVR_GREETING)
        logger.info(f"Greeting ready: {len(_greeting_pcm8k)} bytes")
    else:
        logger.warning("Skipping greeting pre-gen — no reference audio found.")
    yield


app = FastAPI(title="OmniVoice Voicebot — F5-TTS Thai", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---- TTS test endpoint ----

class ChatRequest(BaseModel):
    text: str


@app.post("/api/tts")
async def tts_endpoint(req: ChatRequest):
    wav, sr = await tts(req.text)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Asterisk WebSocket (/asterisk_ws)
# ---------------------------------------------------------------------------

_VAD_ENERGY_THRESHOLD = 50
_VAD_SILENCE_CHUNKS   = 20
_MAX_TURN_BYTES       = 16000 * 10  # 10s fallback
_BOT_COOLDOWN_SECS    = 0.8


async def _send_audio(ws: WebSocket, pcm8k: bytes) -> None:
    FRAME = 320
    for i in range(0, len(pcm8k), FRAME):
        await ws.send_bytes(pcm8k[i : i + FRAME])
        await asyncio.sleep(0.018)


@app.websocket("/asterisk_ws")
async def asterisk_ws(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[IVR {session_id}] Connected")

    recv_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def _receiver():
        try:
            while True:
                msg = await ws.receive()
                if "bytes" in msg:
                    recv_queue.put_nowait(msg["bytes"])
                elif msg.get("type") == "websocket.disconnect":
                    recv_queue.put_nowait(None)
                    break
        except WebSocketDisconnect:
            recv_queue.put_nowait(None)
        except Exception:
            recv_queue.put_nowait(None)

    recv_task = asyncio.create_task(_receiver())

    ivr_state      = IVRState.AWAITING
    pending_intent = None

    audio_buf      = bytearray()
    silence_chunks = 0
    is_speaking    = False
    current_task   = None

    hangup_flag        = [False]
    pending_phone      = [None]
    bot_speaking_until = [0.0]

    async def speak(text: str) -> None:
        pcm = await _tts_pcm8k(text)
        await _send_audio(ws, pcm)
        bot_speaking_until[0] = time.time() + _BOT_COOLDOWN_SECS

    async def process_turn(audio_bytes: bytes) -> None:
        nonlocal ivr_state, pending_intent
        try:
            f32 = pcm8k_to_float32(audio_bytes)
            t0  = time.time()
            user_text = await transcribe(f32, 8000)
            logger.info(f"[IVR {session_id}] ASR {(time.time()-t0)*1000:.0f}ms → '{user_text}'")

            if not user_text.strip():
                return

            words = user_text.split()
            if len(words) >= 5 and len(set(words)) <= 2:
                logger.info(f"[IVR {session_id}] Hallucination detected, discarding: '{user_text[:50]}...'")
                return

            if ivr_state == IVRState.AWAITING:
                intent_key = detect_intent(user_text)
                if intent_key:
                    pending_intent = intent_key
                    ivr_state      = IVRState.CONFIRMING
                    await speak(f"ต้องการ{IVR_INTENTS[intent_key]['name']} ถูกต้องใช่ไหมคะ")
                else:
                    await speak(IVR_REPROMPT)

            elif ivr_state == IVRState.CONFIRMING:
                answer = detect_yes_no(user_text)
                if answer == "yes":
                    if pending_intent == "check_points":
                        ivr_state = IVRState.ASKING_PHONE
                        await speak("กรุณาแจ้งเบอร์มือถือที่ลงทะเบียนไว้ได้เลยค่ะ")
                    else:
                        response       = IVR_INTENTS[pending_intent]["response"]
                        ivr_state      = IVRState.ASKING_MORE
                        pending_intent = None
                        await speak(response)
                        await speak(IVR_ASK_MORE)
                elif answer == "no":
                    ivr_state      = IVRState.AWAITING
                    pending_intent = None
                    await speak(IVR_NOT_CONFIRMED)
                else:
                    await speak(f"ต้องการ{IVR_INTENTS[pending_intent]['name']} ถูกต้องใช่ไหมคะ")

            elif ivr_state == IVRState.ASKING_PHONE:
                digits = extract_phone(user_text)
                if len(digits) >= 9:
                    pending_phone[0] = digits
                    ivr_state = IVRState.CONFIRMING_PHONE
                    await speak(f"เบอร์โทรของคุณลูกค้าคือ {phone_to_thai_speech(digits)} ถูกต้องไหมคะ")
                else:
                    await speak("ขอโทษค่ะ ไม่ได้ยินเบอร์ครบ รบกวนพูดเบอร์มือถือ ๑๐ หลักอีกครั้งค่ะ")

            elif ivr_state == IVRState.CONFIRMING_PHONE:
                answer = detect_yes_no(user_text)
                if answer == "yes":
                    ivr_state      = IVRState.ASKING_MORE
                    pending_intent = None
                    await speak(IVR_INTENTS["check_points"]["response"])
                    await speak(IVR_ASK_MORE)
                elif answer == "no":
                    ivr_state        = IVRState.ASKING_PHONE
                    pending_phone[0] = None
                    await speak("กรุณาแจ้งเบอร์มือถือใหม่อีกครั้งได้เลยค่ะ")
                else:
                    await speak(f"เบอร์โทรของคุณลูกค้าคือ {phone_to_thai_speech(pending_phone[0])} ถูกต้องไหมคะ")

            elif ivr_state == IVRState.ASKING_MORE:
                if detect_yes_no(user_text) == "yes":
                    ivr_state = IVRState.AWAITING
                    await speak("รับทราบค่ะ กรุณาแจ้งเรื่องที่ต้องการได้เลยค่ะ")
                else:
                    await speak(IVR_GOODBYE)
                    hangup_flag[0] = True

        except asyncio.CancelledError:
            logger.info(f"[IVR {session_id}] Barge-in: task cancelled")
            raise
        except Exception as exc:
            logger.error(f"[IVR {session_id}] Error: {exc}", exc_info=True)

    async def _play_greeting():
        await _send_audio(ws, _greeting_pcm8k)
        bot_speaking_until[0] = time.time() + _BOT_COOLDOWN_SECS

    if _greeting_pcm8k:
        current_task = asyncio.create_task(_play_greeting())

    try:
        while True:
            chunk = await recv_queue.get()
            if chunk is None:
                break
            if not chunk:
                continue

            audio_buf.extend(chunk)
            energy = float(np.abs(np.frombuffer(chunk, dtype=np.int16).astype(np.float32)).mean())

            if energy > _VAD_ENERGY_THRESHOLD:
                if not is_speaking and current_task and not current_task.done():
                    logger.info(f"[IVR {session_id}] Barge-in")
                    current_task.cancel()
                    audio_buf = bytearray(chunk)
                is_speaking    = True
                silence_chunks = 0
            elif is_speaking:
                silence_chunks += 1
            elif (current_task and not current_task.done()) or time.time() < bot_speaking_until[0]:
                audio_buf = bytearray()  # discard echo while bot is speaking or cooling down

            end_of_turn = (
                is_speaking and silence_chunks >= _VAD_SILENCE_CHUNKS
            ) or len(audio_buf) >= _MAX_TURN_BYTES

            if end_of_turn:
                turn_audio     = bytes(audio_buf)
                audio_buf      = bytearray()
                silence_chunks = 0
                is_speaking    = False
                current_task   = asyncio.create_task(process_turn(turn_audio))
                try:
                    await asyncio.shield(current_task)
                except (asyncio.CancelledError, Exception):
                    pass
                if hangup_flag[0]:
                    logger.info(f"[IVR {session_id}] Hanging up")
                    break

    except Exception as exc:
        logger.error(f"[IVR {session_id}] Unexpected: {exc}")
    finally:
        recv_task.cancel()
        if current_task and not current_task.done():
            current_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass
        logger.info(f"[IVR {session_id}] Disconnected")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

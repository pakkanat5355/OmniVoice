#!/usr/bin/env python3
"""
OmniVoice Voicebot — XTTS v2 Thai Edition (server_xtts.py)

  TTS : Coqui XTTS v2  (tts_models/multilingual/multi-dataset/xtts_v2)
        — voice cloning, Thai support, 24kHz, ไม่ต้องใส่ ref text
  ASR : typhoon-ai/typhoon-whisper-turbo  (transformers pipeline)

Install:
    pip install TTS transformers accelerate

Optional (voice cloning):
    วางไฟล์ voicebot/ref_voice.wav (หรือ .mp3 .flac .ogg)
    ถ้าไม่มี → ใช้ default XTTS speaker

Usage:
    cd /app
    uv run python voicebot/server_xtts.py
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

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Reference voice (optional — for voice cloning)
# ---------------------------------------------------------------------------

_VOICE_DIR = os.path.dirname(__file__)

_REF_VOICE_PATH: str | None = next(
    (os.path.join(_VOICE_DIR, f"ref_voice.{ext}")
     for ext in ("wav", "mp3", "flac", "ogg")
     if os.path.exists(os.path.join(_VOICE_DIR, f"ref_voice.{ext}"))),
    None,
)

# XTTS needs a WAV file — convert if ref is mp3/flac/ogg
_REF_VOICE_WAV: str | None = None
if _REF_VOICE_PATH:
    if _REF_VOICE_PATH.endswith(".wav"):
        _REF_VOICE_WAV = _REF_VOICE_PATH
    else:
        # Convert to wav via torchaudio at load time (after torchaudio import)
        _REF_VOICE_WAV = os.path.join(_VOICE_DIR, "_ref_voice_converted.wav")

# ---------------------------------------------------------------------------
# Load ASR — Typhoon Whisper Turbo
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
# Load TTS — XTTS v2
# ---------------------------------------------------------------------------

from TTS.api import TTS as CoquiTTS  # noqa: E402

logger.info("Loading XTTS v2 ...")
_tts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
_tts_model.to(device)
logger.info("XTTS v2 loaded.")

# Convert ref voice to WAV if needed
if _REF_VOICE_PATH and not _REF_VOICE_PATH.endswith(".wav"):
    logger.info(f"Converting {_REF_VOICE_PATH} → {_REF_VOICE_WAV}")
    _wav, _sr = torchaudio.load(_REF_VOICE_PATH)
    torchaudio.save(_REF_VOICE_WAV, _wav, _sr)
    logger.info("Conversion done.")

if _REF_VOICE_WAV:
    logger.info(f"Voice cloning from: {_REF_VOICE_WAV}")
else:
    logger.info("No ref_voice found — XTTS will use default speaker")

# Serialise GPU calls
_gpu_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

def _transcribe_sync(audio_array: np.ndarray, sample_rate: int) -> str:
    result = _asr_pipe(
        {"array": audio_array, "sampling_rate": sample_rate},
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

_TTS_SPEED = 0.85   # applied via resampling after generation
_XTTS_SR   = 24000  # XTTS v2 output sample rate


def _tts_sync(text: str) -> np.ndarray:
    """XTTS v2 inference → float32 ndarray at 24kHz."""
    kwargs = dict(text=text, language="th", split_sentences=True)
    if _REF_VOICE_WAV:
        kwargs["speaker_wav"] = _REF_VOICE_WAV

    wav = _tts_model.tts(**kwargs)
    audio = np.array(wav, dtype=np.float32)

    # Speed adjustment via resampling (pitch shifts slightly — acceptable for IVR)
    if abs(_TTS_SPEED - 1.0) > 0.01:
        t = torch.from_numpy(audio).unsqueeze(0)
        orig_len   = t.shape[-1]
        target_len = int(orig_len / _TTS_SPEED)
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=target_len, mode="linear", align_corners=False
        ).squeeze(0)
        audio = t.squeeze(0).numpy()

    return audio


async def tts(text: str) -> np.ndarray:
    async with _gpu_lock:
        return await asyncio.get_event_loop().run_in_executor(None, _tts_sync, text)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def pcm8k_to_float32(audio_bytes: bytes) -> np.ndarray:
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float32_24k_to_pcm8k_bytes(wav: np.ndarray) -> bytes:
    tensor = torch.from_numpy(wav).unsqueeze(0)
    tensor_8k = torchaudio.functional.resample(tensor, _XTTS_SR, 8000)
    pcm16 = (tensor_8k.squeeze(0).numpy() * 32768).clip(-32768, 32767).astype(np.int16)
    return pcm16.tobytes()


async def _tts_pcm8k(text: str) -> bytes:
    wav = await tts(text)
    return float32_24k_to_pcm8k_bytes(wav)


# ---------------------------------------------------------------------------
# IVR — ABC Call Center
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


_TH_DIGIT       = {"0":"ศูนย์","1":"หนึ่ง","2":"สอง","3":"สาม","4":"สี่",
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
# FastAPI
# ---------------------------------------------------------------------------

_greeting_pcm8k: bytes | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _greeting_pcm8k
    logger.info("Pre-generating greeting audio (XTTS v2)...")
    _greeting_pcm8k = await _tts_pcm8k(IVR_GREETING)
    logger.info(f"Greeting ready: {len(_greeting_pcm8k)} bytes")
    yield


app = FastAPI(title="OmniVoice Voicebot — XTTS v2 Thai", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    text: str


@app.post("/api/tts")
async def tts_endpoint(req: ChatRequest):
    wav = await tts(req.text)
    buf = io.BytesIO()
    sf.write(buf, wav, _XTTS_SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Asterisk WebSocket (/asterisk_ws)
# ---------------------------------------------------------------------------

_VAD_ENERGY_THRESHOLD = 50
_VAD_SILENCE_CHUNKS   = 20
_MAX_TURN_BYTES       = 16000 * 10
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
                logger.info(f"[IVR {session_id}] Hallucination detected, discarding")
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
                audio_buf = bytearray()

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

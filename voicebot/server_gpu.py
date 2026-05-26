#!/usr/bin/env python3
"""
OmniVoice Voicebot — GPU Edition (server_gpu.py)

All inference runs locally on GPU — no external API calls.
  - ASR : typhoon-ai/typhoon-whisper-turbo  (via OmniVoice .transcribe())
  - TTS : OmniVoice k2-fsa/OmniVoice        (via model.generate())

Use this on RunPod / any NVIDIA GPU machine.
For the cheap CPU/API version, use server.py instead.

Usage:
    cd /app
    export GROQ_API_KEY=""   # not needed, but harmless if set
    uv run python voicebot/server_gpu.py
"""

import asyncio
import io
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from omnivoice import OmniVoice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = get_best_device()

# ---------------------------------------------------------------------------
# Voice config (must be before model load)
# ---------------------------------------------------------------------------

_REF_VOICE_PATH = next(
    (p for ext in ("wav", "mp3", "flac", "ogg")
     for p in [os.path.join(os.path.dirname(__file__), f"ref_voice.{ext}")]
     if os.path.exists(p)),
    None,
)
_BOT_VOICE_DESIGN = (
    "Thai female AI assistant, warm and friendly customer support voice, "
    "natural Thai conversational rhythm, soft and polite tone, realistic pauses, "
    "medium-slow pacing, expressive but professional, smooth sentence transitions, not robotic"
)

# ---------------------------------------------------------------------------
# Load OmniVoice (TTS + Typhoon ASR on GPU)
# ---------------------------------------------------------------------------

logger.info(f"Loading OmniVoice + Typhoon Whisper ASR on {device} ...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=device,
    dtype=torch.float16,
    load_asr=True,
    asr_model_name="typhoon-ai/typhoon-whisper-turbo",
)
logger.info("OmniVoice + Typhoon ASR loaded.")

# Pre-compute voice clone prompt if reference audio exists
_voice_clone_prompt = None
if _REF_VOICE_PATH:
    logger.info(f"Loading voice clone prompt from {_REF_VOICE_PATH} ...")
    _voice_clone_prompt = model.create_voice_clone_prompt(_REF_VOICE_PATH)
    logger.info("Voice clone prompt ready.")
else:
    logger.info(f"No ref_voice found — using voice design: {_BOT_VOICE_DESIGN}")

# Serialise GPU calls — OmniVoice generate() is not thread-safe
_gpu_lock = asyncio.Lock()

# Language code → OmniVoice language name
_LANG_MAP = {"th": "Thai", "en": "English"}


# ---------------------------------------------------------------------------
# ASR — Typhoon Whisper Turbo (local GPU)
# ---------------------------------------------------------------------------

def _transcribe_sync(audio_array: np.ndarray, sample_rate: int) -> str:
    """Synchronous transcription — runs in executor."""
    audio_input = {"array": audio_array, "sampling_rate": sample_rate}
    result = model._asr_pipe(
        audio_input,
        generate_kwargs={"language": "th", "task": "transcribe"},
    )
    return result["text"].strip()


async def transcribe_gpu(audio_array: np.ndarray, sample_rate: int) -> str:
    async with _gpu_lock:
        return await asyncio.get_event_loop().run_in_executor(
            None, _transcribe_sync, audio_array, sample_rate
        )


# ---------------------------------------------------------------------------
# TTS — OmniVoice (local GPU)
# ---------------------------------------------------------------------------

_TTS_SPEED = 1.2  # < 1.0 = slower, > 1.0 = faster


def _tts_sync(text: str, lang: str, instruct: str | None) -> np.ndarray:
    """Synchronous TTS — runs in executor. Returns float32 np.ndarray at 24kHz."""
    language = _LANG_MAP.get(lang, "Thai")
    if _voice_clone_prompt is not None:
        audios = model.generate(text=text, language=language,
                                voice_clone_prompt=_voice_clone_prompt, speed=_TTS_SPEED)
    else:
        audios = model.generate(text=text, language=language,
                                instruct=_BOT_VOICE_DESIGN, speed=_TTS_SPEED)
    return audios[0]


async def tts_gpu(text: str, lang: str, instruct: str | None = None) -> np.ndarray:
    async with _gpu_lock:
        return await asyncio.get_event_loop().run_in_executor(
            None, _tts_sync, text, lang, instruct
        )


# ---------------------------------------------------------------------------
# Audio format helpers
# ---------------------------------------------------------------------------

def pcm8k_to_float32(audio_bytes: bytes) -> np.ndarray:
    """8kHz 16-bit PCM bytes → float32 numpy array (still at 8kHz)."""
    pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0


def float32_24k_to_pcm8k_bytes(wav: np.ndarray) -> bytes:
    """Resample 24kHz float32 → 8kHz 16-bit PCM bytes for Asterisk."""
    tensor = torch.from_numpy(wav).unsqueeze(0)
    tensor_8k = torchaudio.functional.resample(tensor, 24000, 8000)
    pcm16 = (tensor_8k.squeeze(0).numpy() * 32768).clip(-32768, 32767).astype(np.int16)
    return pcm16.tobytes()


# ---------------------------------------------------------------------------
# IVR — ABC Call Center
# ---------------------------------------------------------------------------

import re

# Greeting played immediately when call connects
IVR_GREETING = (
    "สวัสดีค่ะ ติดต่อบริษัท ABC ยินดีให้บริการค่ะ "
    "กรุณาแจ้งเรื่องที่ต้องการได้เลยค่ะ "
    "เช่น สอบถามคะแนนสะสม สอบถามโปรโมชั่น แจ้งปัญหา "
    "ตรวจสอบออเดอร์ สมัครสมาชิก หรือติดต่อเจ้าหน้าที่ค่ะ"
)

# 5 intents + transfer
IVR_INTENTS = {
    "check_points": {
        "name": "สอบถามคะแนนสะสม",
        "keywords": ["คะแนน", "แต้ม", "แนน", "point", "reward", "สะสม"],
        "response": (
            "ขณะนี้คะแนนสะสมของคุณลูกค้ามีทั้งหมด หนึ่งพันสองร้อยห้าสิบ คะแนนค่ะ "
            "สามารถนำคะแนนไปแลกรับสิทธิ์พิเศษได้ที่เว็บไซต์ abc.com ค่ะ "
            "มีอะไรให้ช่วยเพิ่มเติมไหมคะ"
        ),
    },
    "promotions": {
        "name": "สอบถามโปรโมชั่น",
        "keywords": ["โปรโมชั่น", "ส่วนลด", "ดีล", "โปร", "promotion", "discount", "offer"],
        "response": (
            "ขณะนี้บริษัท ABC มีโปรโมชั่นพิเศษ ลด ยี่สิบ เปอร์เซ็นต์ "
            "สำหรับสินค้าทุกรายการถึงสิ้นเดือนนี้ค่ะ "
            "สามารถดูรายละเอียดเพิ่มเติมได้ที่เว็บไซต์ abc.com ค่ะ "
            "มีอะไรให้ช่วยเพิ่มเติมไหมคะ"
        ),
    },
    "complaint": {
        "name": "แจ้งปัญหาหรือร้องเรียน",
        "keywords": ["ปัญหา", "ร้องเรียน", "เสีย", "ไม่ได้", "แจ้ง", "complaint", "บกพร่อง", "ผิดพลาด"],
        "response": (
            "รับทราบค่ะ ดิฉันจะบันทึกเรื่องร้องเรียนของคุณลูกค้าไว้ "
            "และทีมงานจะติดต่อกลับภายใน ยี่สิบสี่ ชั่วโมงค่ะ "
            "ขอบคุณที่แจ้งให้ทราบนะคะ"
        ),
    },
    "order_status": {
        "name": "ตรวจสอบสถานะออเดอร์",
        "keywords": ["ออเดอร์", "คำสั่งซื้อ", "สถานะ", "จัดส่ง", "order", "delivery", "tracking", "พัสดุ", "ของ"],
        "response": (
            "กรุณาแจ้งหมายเลขคำสั่งซื้อของคุณลูกค้าได้เลยค่ะ "
            "หรือทีมงานจะส่ง SMS แจ้งสถานะให้ที่หมายเลขที่ลงทะเบียนไว้ค่ะ "
            "มีอะไรให้ช่วยเพิ่มเติมไหมคะ"
        ),
    },
    "membership": {
        "name": "สมัครสมาชิก",
        "keywords": ["สมัคร", "สมาชิก", "member", "register", "ลงทะเบียน", "เปิดบัญชี"],
        "response": (
            "สามารถสมัครสมาชิกได้ง่ายๆ ผ่านเว็บไซต์ abc.com "
            "หรือดาวน์โหลดแอปพลิเคชัน ABC ได้เลยค่ะ "
            "การสมัครใช้เวลาไม่ถึง ห้า นาทีค่ะ "
            "มีอะไรให้ช่วยเพิ่มเติมไหมคะ"
        ),
    },
    "transfer": {
        "name": "ติดต่อเจ้าหน้าที่",
        "keywords": ["เจ้าหน้าที่", "คุยกับคน", "agent", "operator", "โอนสาย", "transfer", "พนักงาน", "คน"],
        "response": (
            "กรุณาถือสายรอสักครู่นะคะ กำลังโอนสายให้เจ้าหน้าที่ค่ะ"
        ),
    },
}

IVR_REPROMPT = (
    "คุณลูกค้าต้องการสอบถามเรื่องอะไรนะคะ รบกวนพูดอีกทีค่ะ"
)
IVR_NOT_CONFIRMED = "รับทราบค่ะ กรุณาแจ้งเรื่องที่ต้องการใหม่อีกครั้งได้เลยค่ะ"
IVR_ASK_MORE = "คุณลูกค้ามีคำถามอื่นเพิ่มเติมไหมคะ"
IVR_GOODBYE  = (
    "ขอบคุณที่ใช้บริการบริษัท ABC ค่ะ "
    "ช่วงนี้อากาศเปลี่ยนแปลงบ่อย ดูแลสุขภาพด้วยนะคะ ขอบคุณค่ะ"
)


class IVRState(Enum):
    AWAITING          = "awaiting"
    CONFIRMING        = "confirming"
    ASKING_PHONE      = "asking_phone"      # รอรับเบอร์มือถือ
    CONFIRMING_PHONE  = "confirming_phone"  # รอยืนยันเบอร์
    ASKING_MORE       = "asking_more"


# Thai digit words for phone readback
_TH_DIGIT = {"0":"ศูนย์","1":"หนึ่ง","2":"สอง","3":"สาม","4":"สี่",
              "5":"ห้า","6":"หก","7":"เจ็ด","8":"แปด","9":"เก้า"}

# Thai word → digit (for ASR output containing Thai words)
_TH_WORD_TO_DIGIT = {
    "ศูนย์":"0","หนึ่ง":"1","สอง":"2","สาม":"3","สี่":"4",
    "ห้า":"5","หก":"6","เจ็ด":"7","แปด":"8","เก้า":"9","เอ็ด":"1",
}


def extract_phone(text: str) -> str:
    """Extract digit string from ASR text (handles Arabic digits + Thai words)."""
    # Replace Thai digit words with Arabic digits
    t = text
    for word, digit in _TH_WORD_TO_DIGIT.items():
        t = t.replace(word, digit)
    # Keep only digits
    digits = re.sub(r"[^\d]", "", t)
    return digits


def phone_to_thai_speech(digits: str) -> str:
    """Convert digit string to Thai speech: '0812345678' → 'ศูนย์ แปด หนึ่ง...'"""
    return " ".join(_TH_DIGIT.get(d, d) for d in digits)


def detect_intent(text: str) -> str | None:
    """Return intent key if found, else None."""
    t = text.lower()
    for key, intent in IVR_INTENTS.items():
        if any(kw in t for kw in intent["keywords"]):
            return key
    return None


def detect_yes_no(text: str) -> str:
    """Return 'yes', 'no', or 'unknown'."""
    t = text.lower()
    yes_kw = ["ใช่", "ถูก", "ครับ", "ค่ะ", "คะ", "ได้", "ok", "yes", "right", "correct", "ยืนยัน"]
    no_kw  = ["ไม่", "ผิด", "no", "เปลี่ยน", "cancel", "ยกเลิก"]
    if any(k in t for k in no_kw):
        return "no"
    if any(k in t for k in yes_kw):
        return "yes"
    return "unknown"


# ---------------------------------------------------------------------------
# FastAPI + startup: pre-generate greeting audio
# ---------------------------------------------------------------------------

_greeting_pcm8k: bytes | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _greeting_pcm8k
    logger.info("Pre-generating greeting audio...")
    audio_24k = await tts_gpu(IVR_GREETING, "th")
    _greeting_pcm8k = float32_24k_to_pcm8k_bytes(audio_24k)
    logger.info(f"Greeting ready: {len(_greeting_pcm8k)} bytes")
    yield


app = FastAPI(title="OmniVoice Voicebot — ABC IVR", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Bot-Text", "X-User-Text", "X-Language", "X-Latency-Ms", "X-ASR-Ms", "X-TTS-Ms"],
)


class ChatRequest(BaseModel):
    text: str
    voice_style: str | None = None


# ---- Text → TTS endpoint (test use) ----

@app.post("/api/tts")
async def tts_endpoint(req: ChatRequest):
    waveform = await tts_gpu(req.text, "th")
    buf = io.BytesIO()
    sf.write(buf, waveform, 24000, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Asterisk WebSocket (/asterisk_ws) — IVR Call Center
# ---------------------------------------------------------------------------

_VAD_ENERGY_THRESHOLD = 50
_VAD_SILENCE_CHUNKS   = 20      # 20 × 20ms = 0.4s silence
_MAX_TURN_BYTES       = 16000 * 10  # 10s fallback
_BOT_COOLDOWN_SECS    = 0.8         # mute mic this long after bot stops speaking
_ECHO_CHECK_BYTES     = 16000 * 3   # audio longer than 3s gets repetition check


def _is_repetitive_echo(audio_bytes: bytes) -> bool:
    """Return True if audio is likely network echo (long + consistent energy)."""
    if len(audio_bytes) < _ECHO_CHECK_BYTES:
        return False
    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    hop = 4000  # 0.5s window at 8kHz
    energies = np.array([
        float(np.abs(pcm[i : i + hop]).mean())
        for i in range(0, len(pcm) - hop, hop)
    ])
    if len(energies) < 5:
        return False
    mean_e = energies.mean()
    cv = energies.std() / (mean_e + 1e-6)
    # High sustained energy + low variance = repetitive echo
    return mean_e > _VAD_ENERGY_THRESHOLD and cv < 0.35


async def _send_audio(ws: WebSocket, pcm8k: bytes) -> None:
    """Stream PCM bytes back to Asterisk in 20ms frames."""
    FRAME = 320
    for i in range(0, len(pcm8k), FRAME):
        try:
            await ws.send_bytes(pcm8k[i : i + FRAME])
        except Exception:
            return
        await asyncio.sleep(0.018)


async def _speak(ws: WebSocket, text: str) -> None:
    """TTS → send to Asterisk. Raises CancelledError on barge-in."""
    audio_24k = await tts_gpu(text, "th")
    out_bytes  = float32_24k_to_pcm8k_bytes(audio_24k)
    await _send_audio(ws, out_bytes)


@app.websocket("/asterisk_ws")
async def asterisk_ws(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[IVR {session_id}] Connected")

    # Dedicated receiver — always reads, independent of send side
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

    # IVR state
    ivr_state       = IVRState.AWAITING
    pending_intent  = None   # intent key waiting for confirmation

    audio_buf      = bytearray()
    silence_chunks = 0
    is_speaking    = False
    current_task   = None   # currently running TTS send task

    hangup_flag  = [False]
    pending_phone = [None]       # phone digits being collected
    bot_speaking_until = [0.0]  # timestamp; mute mic until this time passes

    async def process_turn(audio_bytes: bytes) -> None:
        nonlocal ivr_state, pending_intent

        async def speak(text: str) -> None:
            await _speak(ws, text)
            bot_speaking_until[0] = time.time() + _BOT_COOLDOWN_SECS

        try:
            if _is_repetitive_echo(audio_bytes):
                logger.info(f"[IVR {session_id}] Pre-ASR echo discarded ({len(audio_bytes)//16000:.1f}s)")
                return

            f32 = pcm8k_to_float32(audio_bytes)
            t0  = time.time()
            user_text = await transcribe_gpu(f32, 8000)
            logger.info(f"[IVR {session_id}] ASR {(time.time()-t0)*1000:.0f}ms → '{user_text}'")

            if not user_text.strip():
                return

            # Hallucination filter: same word repeated many times = acoustic echo artifact
            words = user_text.split()
            if len(words) >= 5 and len(set(words)) <= 2:
                logger.info(f"[IVR {session_id}] Hallucination detected, discarding: '{user_text[:50]}...'")
                return

            # ── AWAITING: รอ intent ────────────────────────────────────────
            if ivr_state == IVRState.AWAITING:
                intent_key = detect_intent(user_text)
                if intent_key:
                    pending_intent = intent_key
                    intent_name    = IVR_INTENTS[intent_key]["name"]
                    ivr_state      = IVRState.CONFIRMING
                    await speak(f"ต้องการ{intent_name} ถูกต้องใช่ไหมคะ")
                else:
                    await speak(IVR_REPROMPT)

            # ── CONFIRMING: รอ yes/no ──────────────────────────────────────
            elif ivr_state == IVRState.CONFIRMING:
                answer = detect_yes_no(user_text)
                if answer == "yes":
                    if pending_intent == "check_points":
                        # Special flow: ask phone number first
                        ivr_state = IVRState.ASKING_PHONE
                        await speak("กรุณาแจ้งเบอร์มือถือที่ลงทะเบียนไว้ได้เลยค่ะ")
                    else:
                        response   = IVR_INTENTS[pending_intent]["response"]
                        ivr_state  = IVRState.ASKING_MORE
                        pending_intent = None
                        await speak(response)
                        await speak(IVR_ASK_MORE)
                elif answer == "no":
                    ivr_state      = IVRState.AWAITING
                    pending_intent = None
                    await speak(IVR_NOT_CONFIRMED)
                else:
                    intent_name = IVR_INTENTS[pending_intent]["name"]
                    await speak(f"ต้องการ{intent_name} ถูกต้องใช่ไหมคะ")

            # ── ASKING_PHONE: รอเบอร์มือถือ ───────────────────────────────
            elif ivr_state == IVRState.ASKING_PHONE:
                digits = extract_phone(user_text)
                if len(digits) >= 9:
                    pending_phone[0] = digits
                    spoken = phone_to_thai_speech(digits)
                    ivr_state = IVRState.CONFIRMING_PHONE
                    await speak(f"เบอร์โทรของคุณลูกค้าคือ {spoken} ถูกต้องไหมคะ")
                else:
                    await speak("ขอโทษค่ะ ไม่ได้ยินเบอร์ครบ รบกวนพูดเบอร์มือถือ ๑๐ หลักอีกครั้งค่ะ")

            # ── CONFIRMING_PHONE: รอยืนยันเบอร์ ──────────────────────────
            elif ivr_state == IVRState.CONFIRMING_PHONE:
                answer = detect_yes_no(user_text)
                if answer == "yes":
                    response   = IVR_INTENTS["check_points"]["response"]
                    ivr_state  = IVRState.ASKING_MORE
                    pending_intent = None
                    await speak(response)
                    await speak(IVR_ASK_MORE)
                elif answer == "no":
                    ivr_state = IVRState.ASKING_PHONE
                    pending_phone[0] = None
                    await speak("กรุณาแจ้งเบอร์มือถือใหม่อีกครั้งได้เลยค่ะ")
                else:
                    spoken = phone_to_thai_speech(pending_phone[0])
                    await speak(f"เบอร์โทรของคุณลูกค้าคือ {spoken} ถูกต้องไหมคะ")

            # ── ASKING_MORE: มีอะไรเพิ่มเติมไหม ──────────────────────────
            elif ivr_state == IVRState.ASKING_MORE:
                answer = detect_yes_no(user_text)
                if answer == "yes":
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

    # Play greeting immediately on connect
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
            pcm16  = np.frombuffer(chunk, dtype=np.int16)
            energy = float(np.abs(pcm16.astype(np.float32)).mean())

            bot_is_busy = (current_task and not current_task.done()) or time.time() < bot_speaking_until[0]

            if energy > _VAD_ENERGY_THRESHOLD:
                if bot_is_busy:
                    audio_buf = bytearray()  # suppress echo from bot's own audio
                else:
                    if not is_speaking:
                        is_speaking = True
                        audio_buf = bytearray(chunk)  # start fresh, drop pre-speech noise
                    silence_chunks = 0
            elif is_speaking:
                silence_chunks += 1
            else:
                audio_buf = bytearray()  # discard silence/noise while idle

            end_of_turn = (
                is_speaking and silence_chunks >= _VAD_SILENCE_CHUNKS
            ) or len(audio_buf) >= _MAX_TURN_BYTES

            if end_of_turn:
                turn_audio     = bytes(audio_buf)
                audio_buf      = bytearray()
                silence_chunks = 0
                is_speaking    = False
                current_task   = asyncio.create_task(process_turn(turn_audio))
                # Wait for task, then check if we should hang up
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
# Static files (same web UI)
# ---------------------------------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

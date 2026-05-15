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
_BOT_VOICE_DESIGN = "female, middle-aged, very low pitch"

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

_TTS_SPEED = 0.85  # < 1.0 = slower, > 1.0 = faster


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
    AWAITING     = "awaiting"      # รอ user บอกว่าต้องการอะไร
    CONFIRMING   = "confirming"    # รอ yes/no ยืนยัน intent
    ASKING_MORE  = "asking_more"   # หลัง intent เสร็จ ถามว่ามีอะไรเพิ่มเติม


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


async def _send_audio(ws: WebSocket, pcm8k: bytes) -> None:
    """Stream PCM bytes back to Asterisk in 20ms frames."""
    FRAME = 320
    for i in range(0, len(pcm8k), FRAME):
        await ws.send_bytes(pcm8k[i : i + FRAME])
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

    hangup_flag = [False]  # mutable flag accessible inside nested async

    async def process_turn(audio_bytes: bytes) -> None:
        """IVR pipeline — runs as a cancellable asyncio task."""
        nonlocal ivr_state, pending_intent
        try:
            f32 = pcm8k_to_float32(audio_bytes)
            t0  = time.time()
            user_text = await transcribe_gpu(f32, 8000)
            logger.info(f"[IVR {session_id}] ASR {(time.time()-t0)*1000:.0f}ms → '{user_text}'")

            if not user_text.strip():
                return

            if ivr_state == IVRState.AWAITING:
                intent_key = detect_intent(user_text)
                if intent_key:
                    pending_intent = intent_key
                    intent_name    = IVR_INTENTS[intent_key]["name"]
                    confirm_text   = f"ต้องการ{intent_name} ถูกต้องใช่ไหมคะ"
                    logger.info(f"[IVR {session_id}] Intent={intent_key} → confirm")
                    ivr_state = IVRState.CONFIRMING
                    await _speak(ws, confirm_text)
                else:
                    logger.info(f"[IVR {session_id}] No intent → reprompt")
                    await _speak(ws, IVR_REPROMPT)

            elif ivr_state == IVRState.CONFIRMING:
                answer = detect_yes_no(user_text)
                if answer == "yes":
                    response = IVR_INTENTS[pending_intent]["response"]
                    logger.info(f"[IVR {session_id}] Confirmed {pending_intent} → respond")
                    ivr_state      = IVRState.ASKING_MORE
                    pending_intent = None
                    await _speak(ws, response)
                    await _speak(ws, IVR_ASK_MORE)
                elif answer == "no":
                    logger.info(f"[IVR {session_id}] Not confirmed → re-ask")
                    ivr_state      = IVRState.AWAITING
                    pending_intent = None
                    await _speak(ws, IVR_NOT_CONFIRMED)
                else:
                    intent_name  = IVR_INTENTS[pending_intent]["name"]
                    await _speak(ws, f"ต้องการ{intent_name} ถูกต้องใช่ไหมคะ")

            elif ivr_state == IVRState.ASKING_MORE:
                answer = detect_yes_no(user_text)
                if answer == "yes":
                    logger.info(f"[IVR {session_id}] More questions → back to AWAITING")
                    ivr_state = IVRState.AWAITING
                    await _speak(ws, "รับทราบค่ะ กรุณาแจ้งเรื่องที่ต้องการได้เลยค่ะ")
                else:
                    # no / unknown → goodbye and hangup
                    logger.info(f"[IVR {session_id}] No more → goodbye")
                    await _speak(ws, IVR_GOODBYE)
                    hangup_flag[0] = True

        except asyncio.CancelledError:
            logger.info(f"[IVR {session_id}] Barge-in: task cancelled")
            raise
        except Exception as exc:
            logger.error(f"[IVR {session_id}] Error: {exc}", exc_info=True)

    # Play greeting immediately on connect
    if _greeting_pcm8k:
        current_task = asyncio.create_task(_send_audio(ws, _greeting_pcm8k))

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

            if energy > _VAD_ENERGY_THRESHOLD:
                if not is_speaking and current_task and not current_task.done():
                    logger.info(f"[IVR {session_id}] Barge-in")
                    current_task.cancel()
                    audio_buf = bytearray(chunk)
                is_speaking    = True
                silence_chunks = 0
            elif is_speaking:
                silence_chunks += 1
            elif current_task and not current_task.done():
                audio_buf = bytearray()  # discard echo while bot is speaking

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

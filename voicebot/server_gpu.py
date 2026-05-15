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
from datetime import datetime

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

def _tts_sync(text: str, lang: str, instruct: str | None) -> np.ndarray:
    """Synchronous TTS — runs in executor. Returns float32 np.ndarray at 24kHz."""
    language = _LANG_MAP.get(lang, "Thai")
    if _voice_clone_prompt is not None:
        # Voice cloning mode — use reference audio voice
        audios = model.generate(text=text, language=language, voice_clone_prompt=_voice_clone_prompt)
    else:
        # Voice design fallback
        kwargs = dict(text=text, language=language, instruct=_BOT_VOICE_DESIGN)
        audios = model.generate(**kwargs)
    return audios[0]  # np.ndarray float32 24kHz


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
# Chatbot logic (same as server.py)
# ---------------------------------------------------------------------------

import random
import re

GREETINGS = {
    "th": ["สวัสดี", "หวัดดี", "ดีครับ", "ดีค่ะ", "สวัสดีครับ", "สวัสดีค่ะ", "hello", "hi", "hey"],
    "en": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
}

RESPONSES = {
    "greeting_th": [
        "สวัสดีครับ! ยินดีต้อนรับสู่ระบบ Voice Bot ของเรา มีอะไรให้ช่วยไหมครับ?",
        "สวัสดีครับ! วันนี้มีอะไรให้ผมช่วยบ้างครับ?",
        "หวัดดีครับ! ผมพร้อมช่วยเหลือคุณแล้วครับ",
    ],
    "greeting_en": [
        "Hello! Welcome to our Voice Bot. How can I help you today?",
        "Hi there! I'm your voice assistant. What can I do for you?",
    ],
    "about_th": "ผมเป็น Voice Bot ต้นแบบ สร้างด้วย OmniVoice ซึ่งเป็นระบบ Text to Speech ที่รองรับมากกว่า 600 ภาษา สามารถโคลนเสียงและออกแบบเสียงได้ตามต้องการครับ",
    "about_en": "I'm a prototype Voice Bot built with OmniVoice, supporting over 600 languages with voice cloning and voice design.",
    "capabilities_th": "ผมสามารถพูดได้มากกว่า 600 ภาษา โคลนเสียงจากตัวอย่างเสียงสั้นๆ และออกแบบเสียงตามที่คุณต้องการได้ครับ",
    "capabilities_en": "I can speak in over 600 languages, clone voices from short audio samples, and design custom voices.",
    "fallback_th": "ขอโทษครับ ผมยังไม่เข้าใจคำถามนี้ ลองถามว่า 'ทำอะไรได้บ้าง' ดูครับ",
    "fallback_en": "I'm sorry, I didn't quite understand that. Try asking 'what can you do?'",
}

_THAI_DIGITS = ["ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]


def _number_to_thai(n: int) -> str:
    if n == 0:
        return "ศูนย์"
    if n < 0:
        return "ลบ" + _number_to_thai(-n)
    result = ""
    s = str(n)
    length = len(s)
    for i, ch in enumerate(s):
        digit = int(ch)
        pos = length - i - 1
        if digit == 0:
            continue
        if pos >= 7:
            millions = n // 1_000_000
            remainder = n % 1_000_000
            result = _number_to_thai(millions) + "ล้าน"
            if remainder > 0:
                result += _number_to_thai(remainder)
            return result
        if pos == 1:
            if digit == 1:
                result += "สิบ"
            elif digit == 2:
                result += "ยี่สิบ"
            else:
                result += _THAI_DIGITS[digit] + "สิบ"
        elif pos == 0:
            if digit == 1 and length > 1:
                result += "เอ็ด"
            else:
                result += _THAI_DIGITS[digit]
        else:
            result += _THAI_DIGITS[digit] + ["", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน", "ล้าน"][pos]
    return result


def format_thai_time(hour: int, minute: int) -> str:
    min_text = _number_to_thai(minute) + "นาที" if minute > 0 else ""
    if hour == 0:
        base = "เที่ยงคืน"
    elif 1 <= hour <= 5:
        base = "ตี" + _number_to_thai(hour)
    elif 6 <= hour <= 11:
        base = _number_to_thai(hour) + "โมงเช้า"
    elif hour == 12:
        base = "เที่ยงวัน"
    elif 13 <= hour <= 17:
        base = "บ่าย" + _number_to_thai(hour - 12) + "โมง"
    elif 18 <= hour <= 23:
        h = hour - 18
        base = "หกโมงเย็น" if h == 0 else _number_to_thai(h) + "ทุ่ม"
    else:
        base = _number_to_thai(hour) + "นาฬิกา"
    return base + min_text


def normalize_for_tts(text: str, lang: str = "th") -> str:
    if lang == "th":
        def _replace_time(m):
            return format_thai_time(int(m.group(1)), int(m.group(2)))
        text = re.sub(r"(\d{1,2}):(\d{2})", _replace_time, text)
        text = text.replace(" น.", " นาฬิกา")
        text = re.sub(r"\d+", lambda m: _number_to_thai(int(m.group(0))), text)
    return text


def detect_language(text: str) -> str:
    thai_chars = sum(1 for c in text if "฀" <= c <= "๿")
    return "th" if thai_chars > len(text) * 0.2 else "en"


def generate_response(user_text: str) -> tuple[str, str]:
    text_lower = user_text.lower().strip()
    lang = detect_language(user_text)

    for g in GREETINGS.get(lang, GREETINGS["en"]):
        if g in text_lower:
            return random.choice(RESPONSES[f"greeting_{lang}"]), lang

    about_kw = ["คุณเป็นใคร", "เป็นใคร", "คือใคร", "แนะนำตัว"] if lang == "th" else ["who are you", "about you", "introduce yourself"]
    for kw in about_kw:
        if kw in text_lower:
            return RESPONSES[f"about_{lang}"], lang

    cap_kw = ["ทำอะไรได้", "ความสามารถ", "ทำได้บ้าง"] if lang == "th" else ["what can you do", "capabilities"]
    for kw in cap_kw:
        if kw in text_lower:
            return RESPONSES[f"capabilities_{lang}"], lang

    time_kw = ["กี่โมง", "เวลา"] if lang == "th" else ["what time", "time is it"]
    for kw in time_kw:
        if kw in text_lower:
            now = datetime.now()
            if lang == "th":
                return f"ตอนนี้เวลา{format_thai_time(now.hour, now.minute)}ครับ", lang
            else:
                h = now.hour % 12 or 12
                return f"The current time is {h}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}.", lang

    if lang == "th":
        return f"คุณพูดว่า '{user_text}' ใช่ไหมครับ? ลองถามว่า 'ทำอะไรได้บ้าง' ดูครับ", lang
    else:
        return f"You said '{user_text}'. Try asking 'what can you do?'", lang


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="OmniVoice Voicebot — GPU Edition")

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


# ---- Text → TTS endpoint ----

@app.post("/api/tts")
async def tts_endpoint(req: ChatRequest):
    lang = detect_language(req.text)
    waveform = await tts_gpu(normalize_for_tts(req.text, lang), lang, req.voice_style)
    buf = io.BytesIO()
    sf.write(buf, waveform, 24000, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


# ---- Voice chat (upload audio → response audio) ----

@app.post("/api/voice-chat")
async def voice_chat(
    audio_file: UploadFile = File(...),
    voice_style: str = Form(default=""),
):
    t0 = time.time()
    audio_bytes = await audio_file.read()

    suffix = ".webm"
    if audio_file.content_type:
        for ext in ["wav", "mp3", "ogg"]:
            if ext in (audio_file.content_type or ""):
                suffix = f".{ext}"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        t_asr = time.time()
        async with _gpu_lock:
            user_text = await asyncio.get_event_loop().run_in_executor(
                None, lambda: model._asr_pipe(
                    tmp_path,
                    generate_kwargs={"language": "th", "task": "transcribe"},
                )["text"].strip()
            )
        asr_ms = (time.time() - t_asr) * 1000
    finally:
        os.unlink(tmp_path)

    logger.info(f"ASR: '{user_text}' ({asr_ms:.0f}ms)")

    if not user_text.strip():
        buf = io.BytesIO()
        sf.write(buf, np.zeros(2400, dtype=np.float32), 24000, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav", headers={"X-Bot-Text": "(empty)", "X-Language": "th"})

    bot_text, lang = generate_response(user_text)
    t_tts = time.time()
    waveform = await tts_gpu(normalize_for_tts(bot_text, lang), lang, voice_style or None)
    tts_ms = (time.time() - t_tts) * 1000

    buf = io.BytesIO()
    sf.write(buf, waveform, 24000, format="WAV", subtype="PCM_16")
    buf.seek(0)

    total_ms = (time.time() - t0) * 1000
    logger.info(f"VoiceChat: '{user_text}' → '{bot_text}' (ASR={asr_ms:.0f}ms TTS={tts_ms:.0f}ms Total={total_ms:.0f}ms)")

    headers = {
        "X-Bot-Text": bot_text.encode("utf-8").decode("latin-1", errors="replace"),
        "X-User-Text": user_text.encode("utf-8").decode("latin-1", errors="replace"),
        "X-Language": lang,
        "X-Latency-Ms": str(round(total_ms, 1)),
        "X-ASR-Ms": str(round(asr_ms, 1)),
        "X-TTS-Ms": str(round(tts_ms, 1)),
    }
    return StreamingResponse(buf, media_type="audio/wav", headers=headers)


# ---------------------------------------------------------------------------
# Asterisk WebSocket (/asterisk_ws)
# Receives raw 8kHz 16-bit PCM, returns the same format
# ---------------------------------------------------------------------------

_VAD_ENERGY_THRESHOLD = 50      # SIP phone 8kHz audio has low amplitude — keep this low
_VAD_SILENCE_CHUNKS   = 20     # 20 × 20ms = 0.4s silence
_MAX_TURN_BYTES       = 16000 * 10  # 10s fallback (was 30s — too long to wait)

# Voice cloning — reference audio (wav/mp3/flac/ogg all supported).
# If None, falls back to voice design mode (_BOT_VOICE_DESIGN).
_REF_VOICE_PATH = next(
    (p for ext in ("wav", "mp3", "flac", "ogg")
     for p in [os.path.join(os.path.dirname(__file__), f"ref_voice.{ext}")]
     if os.path.exists(p)),
    None,
)
_BOT_VOICE_DESIGN = "female, middle-aged, very low pitch"  # fallback if no ref audio


async def _asterisk_process_turn(ws: WebSocket, session_id: str, audio_bytes: bytes) -> None:
    try:
        f32_8k = pcm8k_to_float32(audio_bytes)

        t_asr = time.time()
        user_text = await transcribe_gpu(f32_8k, 8000)
        logger.info(f"[Asterisk {session_id}] ASR ({(time.time()-t_asr)*1000:.0f}ms) → '{user_text}'")

        if not user_text:
            return

        bot_text, lang = generate_response(user_text)
        tts_text = normalize_for_tts(bot_text, lang)
        logger.info(f"[Asterisk {session_id}] Bot → '{bot_text}'")

        t_tts = time.time()
        audio_24k = await tts_gpu(tts_text, lang)
        logger.info(f"[Asterisk {session_id}] TTS ({(time.time()-t_tts)*1000:.0f}ms)")

        out_bytes = float32_24k_to_pcm8k_bytes(audio_24k)

        # Stream in 20ms frames — matches Asterisk's RTP cadence
        FRAME = 320
        for i in range(0, len(out_bytes), FRAME):
            await ws.send_bytes(out_bytes[i : i + FRAME])
            await asyncio.sleep(0.018)

        logger.info(f"[Asterisk {session_id}] Done — sent {len(out_bytes)} bytes")

    except asyncio.CancelledError:
        logger.info(f"[Asterisk {session_id}] Barge-in: send cancelled")
        raise
    except Exception as exc:
        logger.error(f"[Asterisk {session_id}] Pipeline error: {exc}", exc_info=True)


@app.websocket("/asterisk_ws")
async def asterisk_ws(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[Asterisk {session_id}] Connected")

    # Dedicated receiver task — always reading, independent of send side
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

    audio_buf: bytearray = bytearray()
    silence_chunks = 0
    is_speaking    = False
    current_task   = None
    _log_energy_count = 0

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

            if _log_energy_count < 50:
                logger.info(f"[Asterisk {session_id}] energy={energy:.1f} chunk={len(chunk)}B")
                _log_energy_count += 1

            if energy > _VAD_ENERGY_THRESHOLD:
                if not is_speaking and current_task and not current_task.done():
                    # Barge-in: cancel bot and reset buffer to just this chunk
                    logger.info(f"[Asterisk {session_id}] Barge-in detected")
                    current_task.cancel()
                    audio_buf = bytearray(chunk)
                is_speaking    = True
                silence_chunks = 0
            elif is_speaking:
                silence_chunks += 1
            elif current_task and not current_task.done():
                # Bot is speaking, user is silent → discard echo accumulation
                audio_buf = bytearray()

            end_of_turn = (
                is_speaking and silence_chunks >= _VAD_SILENCE_CHUNKS
            ) or len(audio_buf) >= _MAX_TURN_BYTES

            if end_of_turn:
                turn_audio     = bytes(audio_buf)
                audio_buf      = bytearray()
                silence_chunks = 0
                is_speaking    = False
                current_task   = asyncio.create_task(
                    _asterisk_process_turn(ws, session_id, turn_audio)
                )

    except Exception as exc:
        logger.error(f"[Asterisk {session_id}] Unexpected error: {exc}")
    finally:
        recv_task.cancel()
        if current_task and not current_task.done():
            current_task.cancel()
        logger.info(f"[Asterisk {session_id}] Disconnected")


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

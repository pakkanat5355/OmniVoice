#!/usr/bin/env python3
"""
OmniVoice Voicebot — JamAI Edition (server_gpu_jamai.py)

  - ASR : typhoon-ai/typhoon-whisper-turbo  (local GPU)
  - LLM : JamAI API at localhost:8989        (external)
  - TTS : OmniVoice k2-fsa/OmniVoice        (local GPU)

Flow: caller speaks → ASR → JamAI /api/llm → TTS → caller hears

Usage:
    cd /app
    uv run python voicebot/server_gpu_jamai.py
"""

import asyncio
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
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
# Voice config
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

_n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
_device_map = "auto" if _n_gpu >= 2 else device
logger.info(f"Loading OmniVoice + Typhoon Whisper ASR (device_map={_device_map}, {_n_gpu} GPU(s)) ...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=_device_map,
    dtype=torch.float16,
    load_asr=True,
    asr_model_name="typhoon-ai/typhoon-whisper-turbo",
)
logger.info("OmniVoice + Typhoon ASR loaded.")

_voice_clone_prompt = None
if _REF_VOICE_PATH:
    logger.info(f"Loading voice clone prompt from {_REF_VOICE_PATH} ...")
    _voice_clone_prompt = model.create_voice_clone_prompt(_REF_VOICE_PATH)
    logger.info("Voice clone prompt ready.")
else:
    logger.info(f"No ref_voice found — using voice design: {_BOT_VOICE_DESIGN}")

_gpu_lock = asyncio.Lock()
_LANG_MAP = {"th": "Thai", "en": "English"}


# ---------------------------------------------------------------------------
# ASR — Typhoon Whisper Turbo (local GPU)
# ---------------------------------------------------------------------------

def _transcribe_sync(audio_array: np.ndarray, sample_rate: int) -> str:
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

_TTS_SPEED = 1.2


def _tts_sync(text: str, lang: str) -> np.ndarray:
    language = _LANG_MAP.get(lang, "Thai")
    if _voice_clone_prompt is not None:
        audios = model.generate(text=text, language=language,
                                voice_clone_prompt=_voice_clone_prompt, speed=_TTS_SPEED)
    else:
        audios = model.generate(text=text, language=language,
                                instruct=_BOT_VOICE_DESIGN, speed=_TTS_SPEED)
    return audios[0]


async def tts_gpu(text: str, lang: str = "th") -> np.ndarray:
    async with _gpu_lock:
        return await asyncio.get_event_loop().run_in_executor(
            None, _tts_sync, text, lang
        )


# ---------------------------------------------------------------------------
# Audio format helpers
# ---------------------------------------------------------------------------

def pcm8k_to_float32(audio_bytes: bytes) -> np.ndarray:
    pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0


def float32_24k_to_pcm8k_bytes(wav: np.ndarray) -> bytes:
    tensor = torch.from_numpy(wav).unsqueeze(0)
    tensor_8k = torchaudio.functional.resample(tensor, 24000, 8000)
    pcm16 = (tensor_8k.squeeze(0).numpy() * 32768).clip(-32768, 32767).astype(np.int16)
    return pcm16.tobytes()


# ---------------------------------------------------------------------------
# JamAI API
# ---------------------------------------------------------------------------

JAMAI_API_URL = "http://localhost:8989/api/llm"
JAMAI_TIMEOUT  = 30.0


async def call_jamai(user_text: str) -> tuple[str, int]:
    """POST to JamAI, return (ai_response_text, execute_ms).

    Response path: result.rows[0].columns.ai.choices[0].message.content
    """
    t0 = time.time()
    async with httpx.AsyncClient(timeout=JAMAI_TIMEOUT) as client:
        resp = await client.post(
            JAMAI_API_URL,
            json={"text": user_text},
            headers={"accept": "application/json", "content-type": "application/json"},
        )
        resp.raise_for_status()
    execute_ms = int((time.time() - t0) * 1000)

    data = resp.json()
    content = (
        data["result"]["rows"][0]["columns"]["ai"]
        ["choices"][0]["message"]["content"]
    )
    # Clean newlines for TTS
    content = content.replace("\n\n", " ").replace("\n", " ").strip()
    return content, execute_ms


# ---------------------------------------------------------------------------
# Greeting
# ---------------------------------------------------------------------------

IVR_GREETING = "สวัสดีค่ะ ยินดีให้บริการค่ะ กรุณาพูดเรื่องที่ต้องการได้เลยค่ะ"

# ---------------------------------------------------------------------------
# VAD / echo suppression constants
# ---------------------------------------------------------------------------

_VAD_ENERGY_THRESHOLD = 50
_VAD_SILENCE_CHUNKS   = 20       # 20 × 20ms = 0.4s silence
_MAX_TURN_BYTES       = 16000 * 10
_BOT_COOLDOWN_SECS    = 0.8
_ECHO_CHECK_BYTES     = 16000 * 3  # audio > 3s triggers repetition check


def _is_repetitive_echo(audio_bytes: bytes) -> bool:
    if len(audio_bytes) < _ECHO_CHECK_BYTES:
        return False
    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    hop = 4000
    energies = np.array([
        float(np.abs(pcm[i : i + hop]).mean())
        for i in range(0, len(pcm) - hop, hop)
    ])
    if len(energies) < 5:
        return False
    mean_e = energies.mean()
    cv = energies.std() / (mean_e + 1e-6)
    return mean_e > _VAD_ENERGY_THRESHOLD and cv < 0.35


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_greeting_pcm8k: bytes | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _greeting_pcm8k
    logger.info("Pre-generating greeting audio...")
    audio_24k = await tts_gpu(IVR_GREETING)
    _greeting_pcm8k = float32_24k_to_pcm8k_bytes(audio_24k)
    logger.info(f"Greeting ready: {len(_greeting_pcm8k)} bytes")
    yield


app = FastAPI(title="OmniVoice Voicebot — JamAI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Bot-Text", "X-User-Text", "X-Language", "X-Latency-Ms", "X-ASR-Ms", "X-TTS-Ms"],
)


class ChatRequest(BaseModel):
    text: str


def _safe_header(text: str) -> str:
    """Encode Thai text into latin-1-safe header value."""
    return text.encode("utf-8").decode("latin-1", errors="replace")


@app.post("/api/tts")
async def tts_endpoint(req: ChatRequest):
    waveform = await tts_gpu(req.text)
    buf = io.BytesIO()
    sf.write(buf, waveform, 24000, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


@app.post("/api/chat-audio")
async def chat_audio_endpoint(req: ChatRequest):
    """Text → JamAI → TTS → WAV (for web chat text input)."""
    t0 = time.time()

    try:
        ai_text, _ = await call_jamai(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JamAI error: {e}")

    t_tts = time.time()
    waveform = await tts_gpu(ai_text)
    tts_ms = int((time.time() - t_tts) * 1000)
    total_ms = int((time.time() - t0) * 1000)

    buf = io.BytesIO()
    sf.write(buf, waveform, 24000, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "X-Bot-Text":    _safe_header(ai_text),
            "X-Language":    "th",
            "X-Latency-Ms":  str(total_ms),
            "X-TTS-Ms":      str(tts_ms),
        },
    )


@app.post("/api/voice-chat")
async def voice_chat_endpoint(
    audio_file: UploadFile = File(...),
    voice_style: str = Form(""),
):
    """Audio upload → ASR → JamAI → TTS → WAV (for web chat mic input)."""
    t0 = time.time()

    # Read + decode uploaded audio
    audio_data = await audio_file.read()
    try:
        audio_np, sr = sf.read(io.BytesIO(audio_data))
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        audio_np = audio_np.astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")

    # ASR
    t_asr = time.time()
    user_text = await transcribe_gpu(audio_np, sr)
    asr_ms = int((time.time() - t_asr) * 1000)
    logger.info(f"[web] ASR {asr_ms}ms → '{user_text}'")

    if not user_text.strip():
        fallback = "ไม่ได้ยินเสียง กรุณาลองพูดใหม่ค่ะ"
        waveform = await tts_gpu(fallback)
        buf = io.BytesIO()
        sf.write(buf, waveform, 24000, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav", headers={
            "X-User-Text":  "(empty)",
            "X-Bot-Text":   _safe_header(fallback),
            "X-Language":   "th",
            "X-Latency-Ms": str(int((time.time() - t0) * 1000)),
            "X-ASR-Ms":     str(asr_ms),
            "X-TTS-Ms":     "0",
        })

    # JamAI
    try:
        ai_text, _ = await call_jamai(user_text)
        logger.info(f"[web] JamAI → '{ai_text[:80]}'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JamAI error: {e}")

    # TTS
    t_tts = time.time()
    waveform = await tts_gpu(ai_text)
    tts_ms = int((time.time() - t_tts) * 1000)
    total_ms = int((time.time() - t0) * 1000)

    buf = io.BytesIO()
    sf.write(buf, waveform, 24000, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "X-User-Text":  _safe_header(user_text),
            "X-Bot-Text":   _safe_header(ai_text),
            "X-Language":   "th",
            "X-Latency-Ms": str(total_ms),
            "X-ASR-Ms":     str(asr_ms),
            "X-TTS-Ms":     str(tts_ms),
        },
    )


# ---------------------------------------------------------------------------
# Asterisk WebSocket (/asterisk_ws)
# ---------------------------------------------------------------------------

async def _send_audio(ws: WebSocket, pcm8k: bytes) -> None:
    FRAME = 320
    for i in range(0, len(pcm8k), FRAME):
        try:
            await ws.send_bytes(pcm8k[i : i + FRAME])
        except Exception:
            return
        await asyncio.sleep(0.018)


async def _speak(ws: WebSocket, text: str) -> None:
    audio_24k = await tts_gpu(text)
    out_bytes  = float32_24k_to_pcm8k_bytes(audio_24k)
    await _send_audio(ws, out_bytes)


@app.websocket("/asterisk_ws")
async def asterisk_ws(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[BOT {session_id}] Connected")

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

    audio_buf          = bytearray()
    silence_chunks     = 0
    is_speaking        = False
    current_task       = None
    bot_speaking_until = [0.0]

    async def process_turn(audio_bytes: bytes) -> None:
        async def speak(text: str) -> None:
            await _speak(ws, text)
            bot_speaking_until[0] = time.time() + _BOT_COOLDOWN_SECS

        try:
            # Pre-ASR echo check
            if _is_repetitive_echo(audio_bytes):
                logger.info(f"[BOT {session_id}] Pre-ASR echo discarded ({len(audio_bytes)//16000:.1f}s)")
                return

            # ASR
            f32 = pcm8k_to_float32(audio_bytes)
            t0  = time.time()
            user_text = await transcribe_gpu(f32, 8000)
            asr_ms = int((time.time() - t0) * 1000)
            logger.info(f"[BOT {session_id}] ASR {asr_ms}ms → '{user_text}'")

            if not user_text.strip():
                return

            # Hallucination filter
            words = user_text.split()
            if len(words) >= 5 and len(set(words)) <= 2:
                logger.info(f"[BOT {session_id}] Hallucination detected, discarding")
                return

            # JamAI API
            try:
                ai_text, execute_ms = await call_jamai(user_text)
                logger.info(f"[BOT {session_id}] JamAI {execute_ms}ms → '{ai_text[:100]}'")
            except Exception as e:
                logger.error(f"[BOT {session_id}] JamAI error: {e}")
                await speak("ขออภัยค่ะ ระบบขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้งค่ะ")
                return

            await speak(ai_text)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"[BOT {session_id}] Error: {exc}", exc_info=True)

    # Play greeting on connect
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
                    audio_buf = bytearray()
                else:
                    if not is_speaking:
                        is_speaking = True
                        audio_buf = bytearray(chunk)
                    silence_chunks = 0
            elif is_speaking:
                silence_chunks += 1
            else:
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

    except Exception as exc:
        logger.error(f"[BOT {session_id}] Unexpected: {exc}")
    finally:
        recv_task.cancel()
        if current_task and not current_task.done():
            current_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass
        logger.info(f"[BOT {session_id}] Disconnected")


# ---------------------------------------------------------------------------
# Static files
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

#!/usr/bin/env python3
"""
Voice Transcription Service (voice-transcription/server.py)

Flow:
  trackId + message
      → GET ePro API → download URL
      → download MP3 → temp_audio/{trackId}.mp3
      → Typhoon Whisper Turbo ASR → transcription text
      → compare with message (Typhoon LLM if API key set, else similarity score)
      → return { transcription, result: pass|fail }

Usage:
    cd /app
    uv run python voice-transcription/server.py

    # with Typhoon LLM semantic compare (recommended for Thai):
    TYPHOON_API_KEY=sk-... uv run python voice-transcription/server.py
"""

import asyncio
import logging
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import httpx
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------

EPRO_API_BASE        = os.getenv("EPRO_API_BASE",        "http://172.18.72.80:7000")
TYPHOON_API_KEY      = os.getenv("TYPHOON_API_KEY",       "")
TYPHOON_API_BASE     = os.getenv("TYPHOON_API_BASE",      "https://api.opentyphoon.ai/v1")
TYPHOON_LLM_MODEL    = os.getenv("TYPHOON_LLM_MODEL",     "typhoon-v2-70b-instruct")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
KEEP_TEMP_AUDIO      = os.getenv("KEEP_TEMP_AUDIO", "1") == "1"   # keep files for debug

TEMP_AUDIO_DIR = Path(os.path.dirname(__file__)) / "temp_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = _get_device()
logger.info(f"Device: {device}")

# ---------------------------------------------------------------------------
# Typhoon Whisper Turbo — ASR (Thai speech → text)
# ---------------------------------------------------------------------------

logger.info("Loading typhoon-ai/typhoon-whisper-turbo ...")
_asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="typhoon-ai/typhoon-whisper-turbo",
    device=0 if device == "cuda" else -1,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
)
logger.info("Typhoon Whisper Turbo ready.")

_asr_lock = asyncio.Lock()


def _transcribe_sync(audio_path: Path) -> str:
    # torchaudio handles mp3/wav/etc via ffmpeg backend
    waveform, sample_rate = torchaudio.load(str(audio_path))
    audio_array = waveform.mean(dim=0).numpy().astype(np.float32)

    result = _asr_pipe(
        {"array": audio_array, "sampling_rate": sample_rate},
        generate_kwargs={"language": "th", "task": "transcribe"},
    )
    return result["text"].strip()


async def transcribe(audio_path: Path) -> str:
    async with _asr_lock:
        return await asyncio.get_event_loop().run_in_executor(
            None, _transcribe_sync, audio_path
        )

# ---------------------------------------------------------------------------
# Text normalization + similarity (fallback comparison)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # Keep Thai chars, ASCII alphanumeric only — strip punctuation/spaces
    text = re.sub(r"[^฀-๿a-zA-Z0-9]", "", text)
    return text.lower()


def _similarity(a: str, b: str) -> float:
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()

# ---------------------------------------------------------------------------
# Typhoon LLM — semantic comparison (Thai-optimized)
# ---------------------------------------------------------------------------

_COMPARE_PROMPT = """\
เปรียบเทียบ "ข้อความต้นฉบับ" กับ "ข้อความที่ถอดเสียง" แล้วตอบด้วยคำว่า pass หรือ fail เท่านั้น

ข้อความต้นฉบับ: {expected}
ข้อความที่ถอดเสียง: {transcription}

เกณฑ์:
- pass  : ความหมายตรงกัน หรือมีคำสำคัญครบ แม้สำเนียง/คำเชื่อมต่างกันเล็กน้อย
- fail  : ความหมายต่างกัน หรือขาดคำสำคัญ

ตอบ (pass/fail):"""


async def _llm_compare(transcription: str, expected: str) -> str:
    """Return 'pass' or 'fail' via Typhoon LLM semantic comparison."""
    prompt = _COMPARE_PROMPT.format(
        expected=expected, transcription=transcription
    )
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{TYPHOON_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {TYPHOON_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": TYPHOON_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"].strip().lower()
    return "pass" if "pass" in answer else "fail"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Voice Transcription Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionRequest(BaseModel):
    trackId: int
    message: str


class TranscriptionResponse(BaseModel):
    trackId:       int
    transcription: str
    message:       str
    similarity:    float
    result:        str   # "pass" | "fail"
    method:        str   # "llm" | "similarity"
    elapsed_ms:    int


@app.post("/api/voice-transcription", response_model=TranscriptionResponse)
async def voice_transcription(req: TranscriptionRequest):
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1 — Get download URL from ePro API
    # ------------------------------------------------------------------
    epro_url = (
        f"{EPRO_API_BASE}/epro/api/convertvoice/voiceDownload"
        f"?id={req.trackId}&type=.wav"
    )
    logger.info(f"[{req.trackId}] GET {epro_url}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.get(epro_url)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"ePro API error: {e}")

    # Response body is the download URL (plain string or JSON string)
    download_url = r.text.strip().strip('"')
    if not download_url.startswith("http"):
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected ePro response: {download_url[:120]}",
        )
    logger.info(f"[{req.trackId}] Download URL: {download_url}")

    # ------------------------------------------------------------------
    # Step 2 — Download MP3 to temp_audio/
    # ------------------------------------------------------------------
    audio_path = TEMP_AUDIO_DIR / f"{req.trackId}.mp3"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.get(download_url)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Audio download error: {e}")

    audio_path.write_bytes(r.content)
    logger.info(f"[{req.trackId}] Saved {len(r.content):,} bytes → {audio_path.name}")

    # ------------------------------------------------------------------
    # Step 3 — Transcribe (Typhoon Whisper Turbo)
    # ------------------------------------------------------------------
    try:
        transcription = await transcribe(audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")
    finally:
        if not KEEP_TEMP_AUDIO:
            audio_path.unlink(missing_ok=True)

    logger.info(f"[{req.trackId}] Transcription: '{transcription}'")

    # ------------------------------------------------------------------
    # Step 4 — Compare transcription vs expected message
    # ------------------------------------------------------------------
    similarity = _similarity(transcription, req.message)

    if TYPHOON_API_KEY:
        try:
            result = await _llm_compare(transcription, req.message)
            method = "llm"
            logger.info(
                f"[{req.trackId}] LLM compare → {result}  (similarity={similarity:.3f})"
            )
        except Exception as e:
            logger.warning(
                f"[{req.trackId}] LLM compare failed ({e}) — using similarity fallback"
            )
            result = "pass" if similarity >= SIMILARITY_THRESHOLD else "fail"
            method = "similarity"
    else:
        result = "pass" if similarity >= SIMILARITY_THRESHOLD else "fail"
        method = "similarity"
        logger.info(
            f"[{req.trackId}] Similarity compare → {result}  ({similarity:.3f})"
        )

    elapsed_ms = int((time.time() - t0) * 1000)
    logger.info(f"[{req.trackId}] Done: {result} in {elapsed_ms}ms")

    return TranscriptionResponse(
        trackId=req.trackId,
        transcription=transcription,
        message=req.message,
        similarity=round(similarity, 4),
        result=result,
        method=method,
        elapsed_ms=elapsed_ms,
    )


@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "device":     device,
        "llm_mode":   bool(TYPHOON_API_KEY),
        "llm_model":  TYPHOON_LLM_MODEL if TYPHOON_API_KEY else None,
        "threshold":  SIMILARITY_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

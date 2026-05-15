#!/usr/bin/env python3
"""
OmniVoice Voicebot — Backend Server

A simple voicebot prototype that:
1. Receives user audio from browser microphone
2. Transcribes it using Typhoon Whisper Turbo ASR (SCB10X — optimized for Thai)
3. Generates a chatbot response (rule-based)
4. Converts the response to speech via OmniVoice TTS
5. Returns the audio to the browser

Usage:
    cd /path/to/omnivoice
    uv run python voicebot/server.py
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
from groq import Groq
from gtts import gTTS

from omnivoice import OmniVoice

# Initialize Groq Client for fast ASR
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

def generate_tts_google(text: str) -> np.ndarray:
    """Generate 24kHz float32 waveform using Google Translate TTS (gTTS)."""
    tts = gTTS(text, lang='th')
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
        tmp_path = fp.name
        tts.save(tmp_path)
    
    try:
        waveform, sample_rate = torchaudio.load(tmp_path)
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Resample to 24000Hz for compatibility with rest of the app
        if sample_rate != 24000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 24000)
            
        return waveform.squeeze(0).numpy()
    finally:
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def transcribe_with_groq(file_path: str) -> str:
    with open(file_path, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(os.path.basename(file_path), file.read()),
            model="whisper-large-v3",
            language="th",
            prompt="สวัสดีครับ ยินดีให้บริการ" # Prompt to guide Thai transcription
        )
        return transcription.text


# ── μ-law codec (try audioop first, fallback to numpy) ───────────────────────
try:
    import audioop as _ao
    def _ulaw_to_pcm16_bytes(data: bytes) -> bytes:
        return _ao.ulaw2lin(data, 2)
    def _pcm16_to_ulaw_bytes(data: bytes) -> bytes:
        return _ao.lin2ulaw(data, 2)
except ImportError:
    _EXP_LUT = np.array([0, 132, 396, 924, 1980, 4092, 8316, 16764], dtype=np.int32)

    def _ulaw_to_pcm16_bytes(data: bytes) -> bytes:
        u = (~np.frombuffer(data, dtype=np.uint8)).astype(np.int32)
        sign = u & 0x80
        exp  = (u >> 4) & 0x07
        man  = u & 0x0F
        mag  = _EXP_LUT[exp] + (man << (exp + 3))
        return np.where(sign > 0, -mag, mag).clip(-32768, 32767).astype(np.int16).tobytes()

    def _pcm16_to_ulaw_bytes(data: bytes) -> bytes:
        BIAS, CLIP = 0x84, 32635
        s = np.frombuffer(data, dtype=np.int16).astype(np.int32)
        sign = (s < 0).astype(np.uint8) * 0x80
        s = np.minimum(np.abs(s), CLIP) + BIAS
        # Exponent = floor(log2(s)) - 3, clamped 0..7
        exp = np.clip((np.floor(np.log2(s + 1)) - 3).astype(np.int32), 0, 7)
        man = ((s >> (exp + 3)) & 0x0F).astype(np.uint8)
        return (~(sign | (exp.astype(np.uint8) << 4) | man)).astype(np.uint8).tobytes()


def ulaw_to_float32_16k(data: bytes) -> np.ndarray:
    """Decode PCMU 8 kHz → float32 16 kHz (for Whisper)."""
    pcm16 = np.frombuffer(_ulaw_to_pcm16_bytes(data), dtype=np.int16)
    f32   = pcm16.astype(np.float32) / 32768.0
    tensor = torch.from_numpy(f32).unsqueeze(0)
    tensor = torchaudio.functional.resample(tensor, 8000, 16000)
    return tensor.squeeze(0).numpy()


def float32_24k_to_ulaw_8k(wav: np.ndarray) -> bytes:
    """Resample TTS output (float32 24 kHz) → PCMU 8 kHz for Genesys."""
    tensor = torch.from_numpy(wav).unsqueeze(0)
    tensor = torchaudio.functional.resample(tensor, 24000, 8000)
    pcm16  = (tensor.squeeze(0).numpy() * 32768).clip(-32768, 32767).astype(np.int16)
    return _pcm16_to_ulaw_bytes(pcm16.tobytes())


SERVER_UUID = str(uuid.uuid4())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-detect device
# ---------------------------------------------------------------------------

def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Load OmniVoice model (global singleton) — with Typhoon Whisper ASR (SCB10X)
# ---------------------------------------------------------------------------

ASR_MODEL = "typhoon-ai/typhoon-whisper-turbo"  # SCB10X — fine-tuned for Thai

device = get_best_device()
logger.info(f"Loading OmniVoice on device={device} (with Typhoon ASR) ...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=device,
    dtype=torch.float16,
    load_asr=False,  # ❌ Disabled local ASR to use Groq API instead

    asr_model_name=ASR_MODEL,
)
logger.info(f"OmniVoice + Typhoon ASR ({ASR_MODEL}) loaded.")

# ---------------------------------------------------------------------------
# Simple chatbot logic
# ---------------------------------------------------------------------------

GREETINGS = {
    "th": [
        "สวัสดี", "หวัดดี", "ดีครับ", "ดีค่ะ", "สวัสดีครับ", "สวัสดีค่ะ",
        "hello", "hi", "hey",
    ],
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
        "Hey! Nice to meet you. How can I assist you?",
    ],
    "about_th": "ผมเป็น Voice Bot ต้นแบบ สร้างด้วย OmniVoice ซึ่งเป็นระบบ Text to Speech ที่รองรับมากกว่า 600 ภาษา สามารถโคลนเสียงและออกแบบเสียงได้ตามต้องการครับ",
    "about_en": "I'm a prototype Voice Bot built with OmniVoice, a state-of-the-art text-to-speech system supporting over 600 languages with voice cloning and voice design capabilities.",
    "capabilities_th": "ผมสามารถพูดได้มากกว่า 600 ภาษา, โคลนเสียงจากตัวอย่างเสียงสั้นๆ, และออกแบบเสียงตามที่คุณต้องการได้ครับ เช่น เสียงผู้ชาย ผู้หญิง เด็ก หรือผู้สูงอายุ",
    "capabilities_en": "I can speak in over 600 languages, clone voices from short audio samples, and design custom voices with specific attributes like gender, age, pitch, and accent.",
    "fallback_th": "ขอโทษครับ ผมยังไม่เข้าใจคำถามนี้ แต่ผมสามารถบอกเกี่ยวกับความสามารถของ OmniVoice ได้ ลองถามว่า 'ทำอะไรได้บ้าง' ดูครับ",
    "fallback_en": "I'm sorry, I didn't quite understand that. But I can tell you about OmniVoice's capabilities. Try asking 'what can you do?'",
}

import random
import re


# ---------------------------------------------------------------------------
# Thai text normalization — convert numbers to words for TTS
# ---------------------------------------------------------------------------

_THAI_DIGITS = ["ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]
_THAI_UNITS = ["", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน", "ล้าน"]


def _number_to_thai(n: int) -> str:
    """Convert an integer (0-999999) to Thai words."""
    if n == 0:
        return "ศูนย์"
    if n < 0:
        return "ลบ" + _number_to_thai(-n)

    result = ""
    s = str(n)
    length = len(s)

    for i, ch in enumerate(s):
        digit = int(ch)
        pos = length - i - 1  # position from right (0=ones, 1=tens, etc.)

        if digit == 0:
            continue

        if pos >= 7:
            # For numbers >= 10 million, recursively handle
            millions = n // 1_000_000
            remainder = n % 1_000_000
            result = _number_to_thai(millions) + "ล้าน"
            if remainder > 0:
                result += _number_to_thai(remainder)
            return result

        if pos == 1:  # tens place
            if digit == 1:
                result += "สิบ"
            elif digit == 2:
                result += "ยี่สิบ"
            else:
                result += _THAI_DIGITS[digit] + "สิบ"
        elif pos == 0:  # ones place
            if digit == 1 and length > 1:
                result += "เอ็ด"
            else:
                result += _THAI_DIGITS[digit]
        else:
            result += _THAI_DIGITS[digit] + _THAI_UNITS[pos]

    return result


def format_thai_time(hour: int, minute: int) -> str:
    """Format time in natural Thai speech.

    Uses colloquial Thai time system:
    - 00:00-05:59 → ตีหนึ่ง-ตีห้า / เที่ยงคืน
    - 06:00-11:59 → หกโมงเช้า-สิบเอ็ดโมงเช้า
    - 12:00 → เที่ยงวัน
    - 13:00-17:59 → บ่ายโมง-ห้าโมงเย็น
    - 18:00-23:59 → หกโมงเย็น-ห้าทุ่ม
    """
    min_text = ""
    if minute > 0:
        min_text = _number_to_thai(minute) + "นาที"

    if hour == 0:
        base = "เที่ยงคืน"
    elif 1 <= hour <= 5:
        base = "ตี" + _number_to_thai(hour)
    elif 6 <= hour <= 11:
        base = _number_to_thai(hour) + "โมงเช้า"
    elif hour == 12:
        base = "เที่ยงวัน"
    elif 13 <= hour <= 17:
        h = hour - 12
        base = "บ่าย" + _number_to_thai(h) + "โมง"
    elif 18 <= hour <= 23:
        h = hour - 18
        if h == 0:
            base = "หกโมงเย็น"
        else:
            base = _number_to_thai(h) + "ทุ่ม"
    else:
        base = _number_to_thai(hour) + "นาฬิกา"

    return base + min_text


def normalize_for_tts(text: str, lang: str = "th") -> str:
    """Normalize text for TTS — replace numbers with words.

    OmniVoice struggles with Arabic numerals, so we convert them to
    natural language before sending to the TTS engine.
    """
    if lang == "th":
        # Handle time patterns like "12:30" or "01:26"
        def _replace_time(m):
            h, mins = int(m.group(1)), int(m.group(2))
            return format_thai_time(h, mins)

        text = re.sub(r"(\d{1,2}):(\d{2})", _replace_time, text)

        # Handle "น." abbreviation
        text = text.replace(" น.", " นาฬิกา")

        # Handle decimal numbers like "3.14"
        def _replace_decimal(m):
            whole = int(m.group(1))
            frac = m.group(2)
            result = _number_to_thai(whole) + "จุด"
            for digit in frac:
                result += _THAI_DIGITS[int(digit)]
            return result

        text = re.sub(r"(\d+)\.(\d+)", _replace_decimal, text)

        # Handle remaining integers
        def _replace_int(m):
            return _number_to_thai(int(m.group(0)))

        text = re.sub(r"\d+", _replace_int, text)

    elif lang == "en":
        # Handle time patterns
        def _replace_time_en(m):
            h, mins = int(m.group(1)), int(m.group(2))
            h_str = _int_to_english(h)
            m_str = _int_to_english(mins) if mins > 0 else ""
            if mins > 0 and mins < 10:
                m_str = "oh " + m_str
            return f"{h_str} {m_str}".strip()

        text = re.sub(r"(\d{1,2}):(\d{2})", _replace_time_en, text)

        # Handle remaining integers
        def _replace_int_en(m):
            return _int_to_english(int(m.group(0)))

        text = re.sub(r"\d+", _replace_int_en, text)

    return text


def _int_to_english(n: int) -> str:
    """Convert integer to English words (simple, up to 9999)."""
    if n == 0:
        return "zero"
    ones = ["", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
            "eighty", "ninety"]

    if n < 20:
        return ones[n]
    elif n < 100:
        return tens[n // 10] + (" " + ones[n % 10] if n % 10 else "")
    elif n < 1000:
        return ones[n // 100] + " hundred" + (" " + _int_to_english(n % 100) if n % 100 else "")
    else:
        return _int_to_english(n // 1000) + " thousand" + (" " + _int_to_english(n % 1000) if n % 1000 else "")


def detect_language(text: str) -> str:
    """Simple language detection based on character ranges."""
    thai_chars = sum(1 for c in text if "\u0e00" <= c <= "\u0e7f")
    return "th" if thai_chars > len(text) * 0.2 else "en"


def generate_response(user_text: str) -> tuple[str, str]:
    """
    Generate a chatbot response.
    Returns (response_text, language).
    """
    text_lower = user_text.lower().strip()
    lang = detect_language(user_text)

    # Greeting detection
    for g in GREETINGS.get(lang, GREETINGS["en"]):
        if g in text_lower:
            responses = RESPONSES[f"greeting_{lang}"]
            return random.choice(responses), lang

    # About / who are you
    about_keywords_th = ["คุณเป็นใคร", "เป็นใคร", "คือใคร", "แนะนำตัว", "บอกเกี่ยวกับ"]
    about_keywords_en = ["who are you", "about you", "introduce yourself", "what are you"]
    for kw in (about_keywords_th if lang == "th" else about_keywords_en):
        if kw in text_lower:
            return RESPONSES[f"about_{lang}"], lang

    # Capabilities
    cap_keywords_th = ["ทำอะไรได้", "ความสามารถ", "ทำได้บ้าง", "ช่วยอะไร", "ฟีเจอร์"]
    cap_keywords_en = ["what can you do", "capabilities", "features", "help me with"]
    for kw in (cap_keywords_th if lang == "th" else cap_keywords_en):
        if kw in text_lower:
            return RESPONSES[f"capabilities_{lang}"], lang

    # Time
    time_keywords_th = ["กี่โมง", "เวลา"]
    time_keywords_en = ["what time", "current time", "time is it"]
    for kw in (time_keywords_th if lang == "th" else time_keywords_en):
        if kw in text_lower:
            now = datetime.now()
            if lang == "th":
                time_str = format_thai_time(now.hour, now.minute)
                return f"ตอนนี้เวลา{time_str}ครับ", lang
            else:
                h = now.hour % 12 or 12
                period = "AM" if now.hour < 12 else "PM"
                time_str = _int_to_english(h)
                if now.minute > 0:
                    if now.minute < 10:
                        time_str += " oh " + _int_to_english(now.minute)
                    else:
                        time_str += " " + _int_to_english(now.minute)
                return f"The current time is {time_str} {period}.", lang

    # Echo mode - repeat what user said with a prefix
    if lang == "th":
        return f"คุณพูดว่า '{user_text}' ใช่ไหมครับ? ถ้าต้องการความช่วยเหลือ ลองถามว่า 'ทำอะไรได้บ้าง' ดูครับ", lang
    else:
        return f"You said '{user_text}'. If you need help, try asking 'what can you do?'", lang


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="OmniVoice Voicebot")

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


# ---- Text-based chat (fallback) ----

@app.post("/api/chat-audio")
async def chat_audio(req: ChatRequest):
    """Text-based chat — returns WAV audio response with metadata headers."""
    t0 = time.time()

    bot_text, lang = generate_response(req.text)

    # Normalize numbers to words for TTS
    tts_text = normalize_for_tts(bot_text, lang)
    kw = {"text": tts_text}
    if req.voice_style:
        kw["instruct"] = req.voice_style

    waveform = generate_tts_elevenlabs(tts_text)

    buf = io.BytesIO()
    sf.write(buf, waveform, 24000, format="WAV", subtype="PCM_16")
    buf.seek(0)

    latency = (time.time() - t0) * 1000
    logger.info(f"ChatAudio: '{req.text}' -> '{bot_text}' ({latency:.0f}ms)")

    headers = {
        "X-Bot-Text": bot_text.encode("utf-8").decode("latin-1", errors="replace"),
        "X-User-Text": req.text.encode("utf-8").decode("latin-1", errors="replace"),
        "X-Language": lang,
        "X-Latency-Ms": str(round(latency, 1)),
        "X-ASR-Ms": "0",
        "X-TTS-Ms": str(round(latency, 1)),
    }
    return StreamingResponse(buf, media_type="audio/wav", headers=headers)


# ---- Voice-based chat (Whisper ASR + Chatbot + TTS) ----

@app.post("/api/voice-chat")
async def voice_chat(
    audio_file: UploadFile = File(...),
    voice_style: str = Form(default=""),
):
    """
    Full voice pipeline:
    1. Receive audio from browser microphone
    2. Transcribe with Whisper ASR
    3. Generate chatbot response
    4. Synthesize speech with OmniVoice TTS
    5. Return WAV audio + metadata headers
    """
    t0 = time.time()

    # Step 1: Read the uploaded audio
    audio_bytes = await audio_file.read()

    # Step 2: Transcribe with Whisper ASR
    t_asr_start = time.time()

    # Save to temp file for Whisper (supports webm/wav/mp3 etc.)
    suffix = ".webm"
    if audio_file.content_type:
        if "wav" in audio_file.content_type:
            suffix = ".wav"
        elif "mp3" in audio_file.content_type:
            suffix = ".mp3"
        elif "ogg" in audio_file.content_type:
            suffix = ".ogg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        user_text = transcribe_with_groq(tmp_path)
    finally:
        import os
        os.unlink(tmp_path)

    asr_ms = (time.time() - t_asr_start) * 1000
    logger.info(f"ASR: '{user_text}' ({asr_ms:.0f}ms)")

    if not user_text.strip():
        # Empty transcription
        buf = io.BytesIO()
        sf.write(buf, np.zeros(2400, dtype=np.float32), 24000, format="WAV", subtype="PCM_16")
        buf.seek(0)
        headers = {
            "X-Bot-Text": "ขอโทษครับ ผมไม่ได้ยินเสียง ลองพูดใหม่อีกครั้งครับ".encode("utf-8").decode("latin-1", errors="replace"),
            "X-User-Text": "(empty)",
            "X-Language": "th",
            "X-Latency-Ms": str(round((time.time() - t0) * 1000, 1)),
            "X-ASR-Ms": str(round(asr_ms, 1)),
            "X-TTS-Ms": "0",
        }
        return StreamingResponse(buf, media_type="audio/wav", headers=headers)

    # Step 3: Generate chatbot response
    bot_text, lang = generate_response(user_text)

    # Step 4: Synthesize speech with OmniVoice (normalize numbers first)
    t_tts_start = time.time()

    tts_text = normalize_for_tts(bot_text, lang)
    kw = {"text": tts_text}
    if voice_style:
        kw["instruct"] = voice_style

    waveform = generate_tts_elevenlabs(tts_text)

    tts_ms = (time.time() - t_tts_start) * 1000

    # Step 5: Return WAV audio
    buf = io.BytesIO()
    sf.write(buf, waveform, 24000, format="WAV", subtype="PCM_16")
    buf.seek(0)

    total_ms = (time.time() - t0) * 1000
    logger.info(
        f"VoiceChat: ASR='{user_text}' -> Bot='{bot_text}' "
        f"(ASR={asr_ms:.0f}ms, TTS={tts_ms:.0f}ms, Total={total_ms:.0f}ms)"
    )

    headers = {
        "X-Bot-Text": bot_text.encode("utf-8").decode("latin-1", errors="replace"),
        "X-User-Text": user_text.encode("utf-8").decode("latin-1", errors="replace"),
        "X-Language": lang,
        "X-Latency-Ms": str(round(total_ms, 1)),
        "X-ASR-Ms": str(round(asr_ms, 1)),
        "X-TTS-Ms": str(round(tts_ms, 1)),
    }
    return StreamingResponse(buf, media_type="audio/wav", headers=headers)


# ---- Standalone ASR endpoint ----

@app.post("/api/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Transcribe audio using Whisper ASR only."""
    audio_bytes = await audio_file.read()

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    t0 = time.time()
    try:
        text = transcribe_with_groq(tmp_path)
    finally:
        import os
        os.unlink(tmp_path)

    latency = (time.time() - t0) * 1000
    return {"text": text, "latency_ms": round(latency, 1)}


# ---- TTS-only endpoint ----

@app.post("/api/tts")
async def tts(req: ChatRequest):
    """Direct TTS endpoint — returns WAV audio."""
    kw = {"text": req.text}
    if req.voice_style:
        kw["instruct"] = req.voice_style

    waveform = generate_tts_google(req.text)

    buf = io.BytesIO()
    sf.write(buf, waveform, 24000, format="WAV", subtype="PCM_16")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Genesys AudioHook WebSocket  (/audiohook)
# Protocol ref: https://developer.genesys.cloud/devapps/audiohook
# ---------------------------------------------------------------------------

# VAD tunables
_VAD_ENERGY_THRESHOLD = 400   # เพิ่มจาก 150 เป็น 400 เพื่อตัดเสียงรบกวนรอบข้าง
_VAD_SILENCE_CHUNKS   = 40    # ลดจาก 50 เหลือ 40 (รอเงียบแค่ 0.8 วินาทีแล้วตอบเลย)
_MAX_TURN_BYTES       = 8000 * 30  # hard cap: 30 s at 8 kHz/1 B per sample


async def _audiohook_process_turn(
    ws: WebSocket,
    session_id: str,
    audio_bytes: bytes,
) -> None:
    """Run ASR → bot → TTS and push PCMU audio back to Genesys."""
    try:
        f32_16k = await asyncio.get_event_loop().run_in_executor(
            None, ulaw_to_float32_16k, audio_bytes
        )

        # Write temp WAV for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, f32_16k, 16000, subtype="PCM_16")
            tmp_path = tmp.name

        try:
            user_text = await asyncio.get_event_loop().run_in_executor(
                None, transcribe_with_groq, tmp_path
            )
        finally:
            os.unlink(tmp_path)

        user_text = (user_text or "").strip()
        logger.info(f"[AudioHook {session_id}] ASR → '{user_text}'")
        if not user_text:
            return

        bot_text, lang = generate_response(user_text)
        tts_text = normalize_for_tts(bot_text, lang)
        logger.info(f"[AudioHook {session_id}] Bot → '{bot_text}'")

        audio_out = await asyncio.get_event_loop().run_in_executor(
            None, generate_tts_google, tts_text
        )
        ulaw_out = await asyncio.get_event_loop().run_in_executor(
            None, float32_24k_to_ulaw_8k, audio_out
        )

        logger.info(f"[AudioHook {session_id}] Sending {len(ulaw_out)} bytes of audio to Genesys...")
        # Genesys rate-limits WebSocket messages. Send in large chunks (1 second = 8000 bytes)
        CHUNK = 8000
        for i in range(0, len(ulaw_out), CHUNK):
            chunk_bytes = ulaw_out[i : i + CHUNK]
            await ws.send_bytes(chunk_bytes)
            # Sleep slightly less than the chunk's duration to keep Genesys buffer full
            await asyncio.sleep(len(chunk_bytes) / 8000.0 * 0.8)
        logger.info(f"[AudioHook {session_id}] Finished sending audio.")

    except Exception as exc:
        logger.error(f"[AudioHook {session_id}] Pipeline error: {exc}")


@app.get("/audiohook")
async def audiohook_health():
    """Health check endpoint for Genesys."""
    return {"status": "ok"}


@app.websocket("/audiohook")
@app.websocket("/audiohook/{session_id}")
async def audiohook(ws: WebSocket, session_id: str = None):
    """
    Genesys AudioHook v2 endpoint.
    Configure in Genesys Admin → Telephony → Audio Hooks.
    URI: wss://<ngrok-host>/audiohook
    """
    await ws.accept()
    
    # Genesys sends the true session ID in the headers
    true_session_id = ws.headers.get("audiohook-session-id")
    if true_session_id:
        session_id = true_session_id
    elif not session_id:
        session_id = str(uuid.uuid4())
        
    logger.info(f"[AudioHook {session_id}] Connected (True Session ID: {true_session_id})")

    audio_buf: bytearray = bytearray()
    silence_chunks = 0
    is_speaking    = False
    position       = 0
    server_seq     = 1

    try:
        while True:
            msg = await ws.receive()

            # ── Control messages (JSON) ──────────────────────────────────
            if "text" in msg:
                logger.info(f"[AudioHook] Incoming msg: {msg['text']}")
                data     = json.loads(msg['text'])
                msg_type = data.get("type", "")
                msg_id   = data.get("id", str(uuid.uuid4()))
                pos_str  = data.get("position", "0")

                if msg_type == "open":
                    media = data.get("parameters", {}).get("media") or [
                        {"type": "audio", "format": "PCMU", "rate": 8000, "channels": "mono"}
                    ]
                    await ws.send_text(json.dumps({
                        "version":  "2",
                        "type":     "opened",
                        "seq":      server_seq,
                        "clientseq": data.get("seq", 1),
                        "id":       session_id,
                        "parameters": {"media": media}
                    }))
                    server_seq += 1
                    logger.info(f"[AudioHook {session_id}] Session opened")

                elif msg_type == "ping":
                    await ws.send_text(json.dumps({
                        "version":  "2",
                        "type":     "pong",
                        "seq":      server_seq,
                        "clientseq": data.get("seq", 0),
                        "id":       session_id
                    }))
                    server_seq += 1

                elif msg_type == "close":
                    await ws.send_text(json.dumps({
                        "version":  "2",
                        "type":     "closed",
                        "seq":      server_seq,
                        "clientseq": data.get("seq", 0),
                        "id":       session_id
                    }))
                    server_seq += 1
                    break

                elif msg_type == "discard":
                    audio_buf.clear()
                    silence_chunks = 0
                    is_speaking    = False

            # ── Binary audio frames (PCMU 8 kHz) ────────────────────────
            elif "bytes" in msg:
                chunk = msg["bytes"]
                if not chunk:
                    continue

                audio_buf.extend(chunk)
                position += len(chunk)

                # Energy-based VAD on this chunk
                pcm16  = np.frombuffer(
                    _ulaw_to_pcm16_bytes(bytes(chunk)), dtype=np.int16
                )
                energy = float(np.abs(pcm16.astype(np.float32)).mean())

                if energy > _VAD_ENERGY_THRESHOLD:
                    is_speaking    = True
                    silence_chunks = 0
                elif is_speaking:
                    silence_chunks += 1

                # Fire pipeline when silence detected OR buffer too large
                end_of_turn = (
                    is_speaking and silence_chunks >= _VAD_SILENCE_CHUNKS
                ) or len(audio_buf) >= _MAX_TURN_BYTES

                if end_of_turn:
                    turn_audio     = bytes(audio_buf)
                    audio_buf      = bytearray()
                    silence_chunks = 0
                    is_speaking    = False
                    asyncio.create_task(
                        _audiohook_process_turn(ws, session_id, turn_audio)
                    )

            elif msg.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(f"[AudioHook {session_id}] Unexpected error: {exc}")
    finally:
        logger.info(f"[AudioHook {session_id}] Disconnected")


# ---------------------------------------------------------------------------
# Asterisk Generic WebSocket (/asterisk_ws)
# Accepts raw 8kHz 16-bit PCM (slin8) in binary frames
# ---------------------------------------------------------------------------

async def _asterisk_process_turn(
    ws: WebSocket,
    session_id: str,
    audio_bytes: bytes,
) -> None:
    """Run ASR → bot → TTS and push raw PCM audio back to Asterisk."""
    try:
        # audio_bytes is 8kHz 16-bit PCM. Convert to float32 16kHz for Whisper
        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
        f32 = pcm16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(f32).unsqueeze(0)
        tensor_16k = torchaudio.functional.resample(tensor, 8000, 16000)
        f32_16k = tensor_16k.squeeze(0).numpy()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, f32_16k, 16000, subtype="PCM_16")
            tmp_path = tmp.name

        try:
            user_text = await asyncio.get_event_loop().run_in_executor(
                None, transcribe_with_groq, tmp_path
            )
        finally:
            import os
            os.unlink(tmp_path)

        user_text = (user_text or "").strip()
        logger.info(f"[Asterisk {session_id}] ASR → '{user_text}'")
        if not user_text:
            return

        bot_text, lang = generate_response(user_text)
        tts_text = normalize_for_tts(bot_text, lang)
        logger.info(f"[Asterisk {session_id}] Bot → '{bot_text}'")

        audio_out = await asyncio.get_event_loop().run_in_executor(
            None, generate_tts_google, tts_text
        )
        
        # audio_out is float32 24kHz. Convert to 16-bit PCM 8kHz for Asterisk
        tensor_out = torch.from_numpy(audio_out).unsqueeze(0)
        tensor_out_8k = torchaudio.functional.resample(tensor_out, 24000, 8000)
        pcm16_out = (tensor_out_8k.squeeze(0).numpy() * 32768).clip(-32768, 32767).astype(np.int16)
        out_bytes = pcm16_out.tobytes()

        logger.info(f"[Asterisk {session_id}] Sending {len(out_bytes)} bytes of PCM to Asterisk...")
        CHUNK = 8000 # 0.5 sec of 8kHz 16-bit audio
        for i in range(0, len(out_bytes), CHUNK):
            chunk_bytes = out_bytes[i : i + CHUNK]
            await ws.send_bytes(chunk_bytes)
            await asyncio.sleep(len(chunk_bytes) / 16000.0 * 0.8) # 16000 bytes/sec
        logger.info(f"[Asterisk {session_id}] Finished sending audio.")

    except Exception as exc:
        logger.error(f"[Asterisk {session_id}] Pipeline error: {exc}")


@app.websocket("/asterisk_ws")
async def asterisk_ws(ws: WebSocket):
    """
    WebSocket endpoint for Asterisk (via local proxy).
    Expects raw 8kHz 16-bit signed linear PCM in binary frames.
    """
    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[Asterisk {session_id}] Connected")

    audio_buf: bytearray = bytearray()
    silence_chunks = 0
    is_speaking    = False
    current_task   = None  # Track the current bot response task for Barge-in

    try:
        while True:
            msg = await ws.receive()

            if "bytes" in msg:
                chunk = msg["bytes"]
                if not chunk:
                    continue

                audio_buf.extend(chunk)

                # Energy-based VAD (16-bit PCM)
                pcm16 = np.frombuffer(chunk, dtype=np.int16)
                energy = float(np.abs(pcm16.astype(np.float32)).mean())

                if energy > _VAD_ENERGY_THRESHOLD:
                    if not is_speaking:
                        # User just started speaking (Barge-in detected!)
                        if current_task and not current_task.done():
                            logger.info(f"[Asterisk {session_id}] 🛑 Barge-in detected! Interrupting bot...")
                            current_task.cancel()
                    
                    is_speaking = True
                    silence_chunks = 0
                elif is_speaking:
                    silence_chunks += 1

                # Fire pipeline when silence detected OR buffer too large
                end_of_turn = (
                    is_speaking and silence_chunks >= _VAD_SILENCE_CHUNKS
                ) or len(audio_buf) >= (_MAX_TURN_BYTES * 2) # x2 because 16-bit

                if end_of_turn:
                    turn_audio = bytes(audio_buf)
                    audio_buf = bytearray()
                    silence_chunks = 0
                    is_speaking = False
                    
                    # Start new bot response task
                    current_task = asyncio.create_task(
                        _asterisk_process_turn(ws, session_id, turn_audio)
                    )

            elif msg.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(f"[Asterisk {session_id}] Unexpected error: {exc}")
    finally:
        if current_task and not current_task.done():
            current_task.cancel()
        logger.info(f"[Asterisk {session_id}] Disconnected")


# ---------------------------------------------------------------------------
# Serve frontend static files
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

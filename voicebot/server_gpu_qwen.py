#!/usr/bin/env python3
"""
OmniVoice Voicebot — Qwen3 Edition (server_gpu_qwen.py)

  - ASR : typhoon-ai/typhoon-whisper-turbo       (local GPU, in-process)
  - LLM : Qwen/Qwen3-4B-Instruct-2507 via transformers (local GPU, in-process)
  - TTS : OmniVoice k2-fsa/OmniVoice             (local GPU, in-process)

Flow: caller speaks → ASR → Qwen3 → TTS → caller hears

Everything runs in ONE Python process managed by uv — just like the Typhoon
ASR model. No external service (no Ollama, no Docker). The LLM is loaded with
transformers at startup and generates in-process on the GPU.

LLM: Qwen3-4B-Instruct-2507 (bf16). The dedicated instruct (non-thinking)
model — follows the system prompt / scope rules more reliably than base
Qwen3-4B and replies without a reasoning block, so it is lower latency. Loads
natively in transformers (no quant library). Uses ~8 GB VRAM; on the 2× NVIDIA
A2 (15 GB) box it sits on cuda:1 next to TTS model_1.

Use case: outbound/inbound debt-collection ("ติดตามหนี้") assistant.
The system prompt keeps Qwen3 on-topic — it follows up on overdue payment,
offers payment options, and politely refuses / redirects off-topic questions.

Conversation memory: per WebSocket call session (one Asterisk call = one history).

Usage:
    cd /app
    uv run python voicebot/server_gpu_qwen.py

Env overrides:
    QWEN_MODEL    (default Qwen/Qwen3-4B; HF id or local path)
    QWEN_DEVICE   (default cuda:1 if 2 GPUs else cuda:0)
"""

import asyncio
import io
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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
# GPU 0 : TTS model_0 + ASR (Typhoon Whisper)
# GPU 1 : TTS model_1  (if 2nd GPU available → parallel sentence TTS)
# ---------------------------------------------------------------------------

_n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
_LANG_MAP = {"th": "Thai", "en": "English"}

logger.info("Loading OmniVoice + Typhoon Whisper ASR on cuda:0 ...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0" if torch.cuda.is_available() else device,
    dtype=torch.float16,
    load_asr=True,
    asr_model_name="typhoon-ai/typhoon-whisper-turbo",
)
logger.info("OmniVoice (GPU 0) + Typhoon ASR loaded.")

model_1 = None
if _n_gpu >= 2:
    logger.info("Loading second OmniVoice instance on cuda:1 for parallel TTS ...")
    model_1 = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:1",
        dtype=torch.float16,
        load_asr=False,
    )
    logger.info("OmniVoice (GPU 1) loaded.")

_voice_clone_prompt = None
if _REF_VOICE_PATH:
    logger.info(f"Loading voice clone prompt from {_REF_VOICE_PATH} ...")
    _voice_clone_prompt = model.create_voice_clone_prompt(_REF_VOICE_PATH)
    logger.info("Voice clone prompt ready.")
else:
    logger.info(f"No ref_voice found — using voice design: {_BOT_VOICE_DESIGN}")

_gpu_lock   = asyncio.Lock()   # GPU 0 (ASR + TTS model_0)
_gpu_lock_1 = asyncio.Lock()   # GPU 1 (TTS model_1)


# ---------------------------------------------------------------------------
# Load Qwen3 LLM (transformers, in-process — same style as Typhoon ASR)
# Qwen3-4B-Instruct-2507 in bf16: plain transformers load, no quant backend.
# Placed on cuda:1 by default so it shares the 2nd GPU with TTS model_1 and
# leaves cuda:0 (ASR + main TTS) lighter.
# ---------------------------------------------------------------------------

QWEN_MODEL  = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
QWEN_DEVICE = os.environ.get(
    "QWEN_DEVICE",
    "cuda:1" if _n_gpu >= 2 else ("cuda:0" if torch.cuda.is_available() else device),
)

logger.info(f"Loading LLM {QWEN_MODEL} on {QWEN_DEVICE} ...")
_qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
_qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    dtype=torch.float16,
    device_map=QWEN_DEVICE,
)
_qwen_model.eval()
logger.info("Qwen3 LLM loaded.")

_llm_lock = asyncio.Lock()   # serialize LLM generation


# ---------------------------------------------------------------------------
# ASR — Typhoon Whisper Turbo (GPU 0)
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
# TTS — OmniVoice parallel sentence synthesis
# ---------------------------------------------------------------------------

_TTS_SPEED = 1.2
_SENTENCE_SILENCE = 0.08  # seconds of silence to insert between sentences
# OmniVoice diffusion steps (model default 32). Fewer steps = much faster on
# the A2 GPUs, at a small quality cost — the main TTS latency lever.
_TTS_NUM_STEP = int(os.environ.get("TTS_NUM_STEP", "16"))


def _split_sentences(text: str) -> list[str]:
    """Split Thai text into clauses on sentence-ending particles / punctuation."""
    parts = re.split(r'(?<=ค่ะ)\s+|(?<=ครับ)\s+|(?<=คะ)\s+|(?<=นะ)\s+|[.!?]\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    # Merge very short trailing fragments into the previous sentence
    merged: list[str] = []
    for p in parts:
        if merged and len(p) < 8:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged if merged else [text]


# Spell digits out as Thai words before TTS — OmniVoice mispronounces raw
# digits (e.g. "5,200", "2568"), so convert them to "ห้าพันสองร้อย" etc.
try:
    from pythainlp.util import num_to_thaiword as _num2thai
except Exception:  # pythainlp missing → leave numbers as-is
    _num2thai = None

_NUM_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')
_THAI_DIGIT = {
    "0": "ศูนย์", "1": "หนึ่ง", "2": "สอง", "3": "สาม", "4": "สี่",
    "5": "ห้า", "6": "หก", "7": "เจ็ด", "8": "แปด", "9": "เก้า",
}


def _num_token_to_thai(tok: str) -> str:
    s = tok.replace(",", "")
    try:
        if "." in s:
            intp, decp = s.split(".", 1)
            words = _num2thai(int(intp)) if intp else "ศูนย์"
            return words + "จุด" + "".join(_THAI_DIGIT.get(d, d) for d in decp)
        return _num2thai(int(s))
    except Exception:
        return tok


def _spell_numbers_th(text: str) -> str:
    """Replace Arabic-digit numbers with Thai reading words for clean TTS."""
    if _num2thai is None:
        return text
    return _NUM_RE.sub(lambda m: _num_token_to_thai(m.group(0)), text)


def _tts_sync(text: str, lang: str, m=None) -> np.ndarray:
    m = m or model
    language = _LANG_MAP.get(lang, "Thai")
    if _voice_clone_prompt is not None:
        audios = m.generate(text=text, language=language,
                            voice_clone_prompt=_voice_clone_prompt, speed=_TTS_SPEED,
                            num_step=_TTS_NUM_STEP)
    else:
        audios = m.generate(text=text, language=language,
                            instruct=_BOT_VOICE_DESIGN, speed=_TTS_SPEED,
                            num_step=_TTS_NUM_STEP)
    return audios[0]


async def _tts_sentence(text: str, lang: str, gpu_id: int) -> np.ndarray:
    if gpu_id == 1 and model_1 is not None:
        async with _gpu_lock_1:
            return await asyncio.get_event_loop().run_in_executor(
                None, _tts_sync, text, lang, model_1
            )
    async with _gpu_lock:
        return await asyncio.get_event_loop().run_in_executor(
            None, _tts_sync, text, lang, model
        )


async def tts_gpu(text: str, lang: str = "th") -> np.ndarray:
    text = _spell_numbers_th(text)
    sentences = _split_sentences(text)

    if len(sentences) <= 1 or model_1 is None:
        # Single GPU path (no 2nd GPU or single sentence)
        return await _tts_sentence(text, lang, 0)

    # Parallel path — odd sentences → GPU 0, even → GPU 1
    tasks = [
        _tts_sentence(sent, lang, i % 2)
        for i, sent in enumerate(sentences)
    ]
    logger.info(f"[TTS] {len(sentences)} sentences → 2 GPUs parallel")
    audio_parts = await asyncio.gather(*tasks)

    silence = np.zeros(int(_SENTENCE_SILENCE * 24000), dtype=np.float32)
    result = audio_parts[0]
    for part in audio_parts[1:]:
        result = np.concatenate([result, silence, part])
    return result


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
# Qwen3 — debt-collection agent (in-process generation)
# ---------------------------------------------------------------------------

# --- Debt account context -------------------------------------------------
# In production, fill these per call (e.g. look up by phone number before the
# WebSocket opens). Kept as module defaults here so the bot has something to
# work with out of the box.
DEBT_CONTEXT = {
    "company":      "บริษัท ตัวอย่าง จำกัด",
    "customer":     "คุณลูกค้า",
    "amount":       "5,200 บาท",
    "due_date":     "5 มิถุนายน 2568",
    "days_overdue": "14",
    "channels":     "โอนผ่านแอปธนาคาร, เคาน์เตอร์เซอร์วิส, หรือ QR พร้อมเพย์",
}


def _build_system_prompt(ctx: dict) -> str:
    """Strong, scope-constrained Thai system prompt for debt collection."""
    return f"""คุณคือ "น้องใจดี" เจ้าหน้าที่ AI ฝ่ายติดตามชำระหนี้ของ{ctx['company']}
หน้าที่ของคุณคือโทรติดตามยอดค้างชำระกับลูกค้าอย่างสุภาพ เป็นมิตร และเป็นมืออาชีพ

ข้อมูลบัญชีของลูกค้า (ใช้อ้างอิงเท่านั้น ห้ามแต่งข้อมูลเพิ่ม):
- ชื่อลูกค้า: {ctx['customer']}
- ยอดค้างชำระ: {ctx['amount']}
- ครบกำหนดชำระ: {ctx['due_date']} (เกินกำหนดมาแล้ว {ctx['days_overdue']} วัน)
- ช่องทางชำระเงิน: {ctx['channels']}

เป้าหมายการสนทนา (ทำตามลำดับ แต่ยืดหยุ่นตามจังหวะลูกค้า):
1. ทักทาย แนะนำตัวว่าโทรจาก{ctx['company']} และยืนยันว่ากำลังคุยกับเจ้าของบัญชี
2. แจ้งยอดค้างชำระและวันครบกำหนดอย่างนุ่มนวล
3. สอบถามว่าลูกค้าสะดวกชำระเมื่อไหร่ หรือมีปัญหาอะไรที่ทำให้ยังไม่ได้ชำระ
4. เสนอช่องทางการชำระเงิน และช่วยนัดวันชำระ
5. สรุปข้อตกลงและกล่าวขอบคุณก่อนวางสาย

กฎสำคัญ (ห้ามฝ่าฝืน):
- พูดภาษาไทยล้วน สุภาพ ลงท้าย "ค่ะ" ใช้น้ำเสียงเป็นมิตร ไม่กดดัน ไม่ข่มขู่ ไม่ดูถูก
- ตอบสั้น กระชับ เป็นธรรมชาติเหมือนคนคุยโทรศัพท์ (1-3 ประโยคต่อครั้ง) เพราะคำตอบจะถูกอ่านออกเสียง
- คุยเฉพาะเรื่องการชำระหนี้และบัญชีนี้เท่านั้น ถ้าลูกค้าถามนอกเรื่อง (เช่น ข่าว สูตรอาหาร เขียนโค้ด ดูดวง เรื่องทั่วไป) ให้ปฏิเสธอย่างสุภาพแล้วดึงกลับเข้าเรื่อง เช่น "ขออภัยค่ะ ส่วนนี้น้องใจดีช่วยไม่ได้นะคะ ขอกลับมาเรื่องการชำระยอดค้างนะคะ"
- ห้ามให้คำปรึกษากฎหมาย การเงิน การลงทุน หรือสัญญาว่าจะลดหนี้/ยกหนี้/คิดดอกเบี้ยเอง หากลูกค้าขอลดยอดหรือผ่อนผัน ให้บอกว่าจะบันทึกเรื่องส่งให้เจ้าหน้าที่ติดต่อกลับ
- ห้ามเปิดเผยข้อมูลบัญชีนี้ให้คนที่ไม่ใช่เจ้าของบัญชี ถ้าคุยอยู่กับคนอื่น ให้ขอช่องทางติดต่อเจ้าของบัญชีแทน
- ห้ามแต่งตัวเลข ยอดเงิน วันที่ หรือเงื่อนไขที่ไม่มีในข้อมูลข้างต้น ถ้าไม่รู้ให้บอกว่าจะให้เจ้าหน้าที่ตรวจสอบและติดต่อกลับ
- ถ้าลูกค้าโมโห ขอให้หยุดติดต่อ หรือบอกว่าจ่ายแล้ว ให้รับเรื่องอย่างสุภาพ ขอโทษ และแจ้งว่าจะบันทึกเรื่องให้เจ้าหน้าที่ตรวจสอบ
- ห้ามแสดงกระบวนการคิด ให้ตอบเฉพาะสิ่งที่จะพูดกับลูกค้าเท่านั้น"""


SYSTEM_PROMPT = _build_system_prompt(DEBT_CONTEXT)

# Qwen3 emits <think>...</think> when reasoning is on; strip it just in case.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_llm_text(content: str) -> str:
    content = _THINK_RE.sub("", content)
    content = content.replace("\n\n", " ").replace("\n", " ").strip()
    return content


_QWEN_MAX_NEW_TOKENS = 200   # keep replies short for TTS


def _qwen_generate_sync(messages: list[dict]) -> str:
    """Run Qwen3 generation in-process. `messages` = system + history + user."""
    # enable_thinking=False → Qwen3 skips its reasoning block for low latency
    prompt = _qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = _qwen_tokenizer(prompt, return_tensors="pt").to(_qwen_model.device)
    with torch.no_grad():
        generated = _qwen_model.generate(
            **inputs,
            max_new_tokens=_QWEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=_qwen_tokenizer.eos_token_id,
        )
    # Slice off the prompt tokens, decode only the new completion
    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    content = _qwen_tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _clean_llm_text(content)


async def call_qwen(messages: list[dict]) -> tuple[str, int]:
    """Generate an assistant reply in-process, return (assistant_text, execute_ms).

    `messages` is a chat list (system + history + latest user). Serialized via
    _llm_lock and run in an executor so the event loop stays responsive.
    """
    t0 = time.time()
    async with _llm_lock:
        content = await asyncio.get_event_loop().run_in_executor(
            None, _qwen_generate_sync, messages
        )
    execute_ms = int((time.time() - t0) * 1000)
    return content, execute_ms


# Per-call conversation history (WebSocket). Max user/assistant turns kept.
_MAX_HISTORY_TURNS = 8


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep only the last N user/assistant exchanges (system added separately)."""
    if len(history) <= _MAX_HISTORY_TURNS * 2:
        return history
    return history[-_MAX_HISTORY_TURNS * 2:]


# ---------------------------------------------------------------------------
# Greeting
# ---------------------------------------------------------------------------

IVR_GREETING = (
    f"สวัสดีค่ะ น้องใจดีติดต่อจาก{DEBT_CONTEXT['company']}นะคะ "
    f"ขอเรียนสายกับ{DEBT_CONTEXT['customer']}ได้ไหมคะ"
)

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


app = FastAPI(title="OmniVoice Voicebot — Qwen3 Debt Collection", lifespan=lifespan)
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
    """Text → Qwen3 → TTS → WAV (for web chat text input). Stateless single turn."""
    t0 = time.time()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.text},
    ]
    try:
        ai_text, _ = await call_qwen(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qwen error: {e}")

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
    """Audio upload → ASR → Qwen3 → TTS → WAV (for web chat mic input). Stateless."""
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

    # Qwen3
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    try:
        ai_text, _ = await call_qwen(messages)
        logger.info(f"[web] Qwen → '{ai_text[:80]}'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qwen error: {e}")

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
    """Stream TTS to the caller sentence-by-sentence.

    Sentences are synthesized in parallel across the two GPUs but sent in order
    as each finishes, so the caller hears the first sentence almost immediately
    instead of waiting for the whole reply to render.
    """
    text = _spell_numbers_th(text)
    sentences = _split_sentences(text)

    if len(sentences) <= 1 or model_1 is None:
        audio = await _tts_sentence(text, "th", 0)
        await _send_audio(ws, float32_24k_to_pcm8k_bytes(audio))
        return

    # Kick off every sentence now (odd → GPU 0, even → GPU 1) ...
    tasks = [
        asyncio.ensure_future(_tts_sentence(sent, "th", i % 2))
        for i, sent in enumerate(sentences)
    ]
    logger.info(f"[TTS] streaming {len(sentences)} sentences → 2 GPUs parallel")
    # ... then play them back in order as soon as each is ready.
    for t in tasks:
        audio = await t
        await _send_audio(ws, float32_24k_to_pcm8k_bytes(audio))


@app.websocket("/asterisk_ws")
async def asterisk_ws(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[BOT {session_id}] Connected")

    # Per-call conversation memory (user/assistant turns; system added at call time)
    history: list[dict] = []

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

            # Qwen3 with per-call history
            history.append({"role": "user", "content": user_text})
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _trim_history(history)
            try:
                ai_text, execute_ms = await call_qwen(messages)
                logger.info(f"[BOT {session_id}] Qwen {execute_ms}ms → '{ai_text[:100]}'")
            except Exception as e:
                logger.error(f"[BOT {session_id}] Qwen error: {e}")
                history.pop()  # drop the unanswered user turn
                await speak("ขออภัยค่ะ ระบบขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้งค่ะ")
                return

            history.append({"role": "assistant", "content": ai_text})
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
        history.append({"role": "assistant", "content": IVR_GREETING})
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
        logger.info(f"[BOT {session_id}] Disconnected ({len(history)} turns)")


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

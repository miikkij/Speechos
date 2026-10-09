"""Speech-to-text transcription endpoints."""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
import wave
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.audio import TARGET_SAMPLE_RATE, float32_to_wav_bytes
from src.config import ModelConfig
from src.models import ModelNotConfiguredError
from src.routers._common import decode_for_model, parse_language, read_audio_upload
from src.server import get_config, get_models

logger = logging.getLogger(__name__)
router = APIRouter(tags=["transcription"])


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
):
    """Transcribe an audio file to text.

    ``language`` is an ISO 639-1 code (e.g. ``fi``); omit it or send ``auto`` to
    let the model detect the language. Returns text, segments and the language.
    """
    config = get_config()
    data = await read_audio_upload(file, config)
    lang = parse_language(language)
    audio = await decode_for_model(data)
    duration = len(audio) / TARGET_SAMPLE_RATE

    t0 = time.perf_counter()
    try:
        result = await get_models().run(
            "stt", lambda stt, cfg: transcribe_array(stt, cfg, audio, lang), language=lang
        )
    except ModelNotConfiguredError as e:
        raise HTTPException(503, str(e)) from e
    except (ImportError, FileNotFoundError, OSError) as e:
        raise HTTPException(503, f"STT engine not available: {e}") from e
    elapsed = time.perf_counter() - t0

    return {
        **result,
        "duration": round(duration, 3),
        "processing_time": round(elapsed, 3),
    }


def transcribe_array(stt: Any, cfg: ModelConfig, audio: np.ndarray, language: str | None) -> dict[str, Any]:
    """Run one STT engine on 16 kHz mono float32 audio (blocking — call in a thread)."""
    engine = cfg.engine

    if engine == "faster-whisper":
        segments, info = stt.transcribe(audio, language=language, beam_size=5, vad_filter=True)
        result_segments = [
            {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip()}
            for seg in segments  # generator — decoding happens here
        ]
        return {
            "text": " ".join(s["text"] for s in result_segments),
            "segments": result_segments,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
        }

    if engine == "whisperx":
        result = stt.transcribe(audio, language=language, batch_size=16)
        result_segments = [
            {
                "start": round(seg.get("start", 0), 3),
                "end": round(seg.get("end", 0), 3),
                "text": seg.get("text", "").strip(),
            }
            for seg in result.get("segments", [])
        ]
        return {
            "text": " ".join(s["text"] for s in result_segments),
            "segments": result_segments,
            "language": result.get("language", language or "en"),
            "language_probability": 1.0,
        }

    if engine == "vosk":
        from vosk import KaldiRecognizer

        wf = wave.open(io.BytesIO(float32_to_wav_bytes(audio)), "rb")
        rec = KaldiRecognizer(stt, wf.getframerate())
        rec.SetWords(True)
        while chunk := wf.readframes(4000):
            rec.AcceptWaveform(chunk)
        text = json.loads(rec.FinalResult()).get("text", "")

    elif engine == "nemo":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, TARGET_SAMPLE_RATE)
            tmp_path = tmp.name
        try:
            transcriptions = stt.transcribe([tmp_path])
        finally:
            os.unlink(tmp_path)
        # NeMo returns list of strings or list of Hypothesis objects
        if transcriptions and isinstance(transcriptions[0], str):
            text = transcriptions[0]
        elif transcriptions and hasattr(transcriptions[0], "text"):
            text = transcriptions[0].text
        else:
            text = str(transcriptions[0]) if transcriptions else ""

    elif engine == "moonshine":
        out = stt.transcribe(audio)
        text = out if isinstance(out, str) else str(out)

    elif engine == "wav2vec2-stt":
        text = stt({"raw": audio, "sampling_rate": TARGET_SAMPLE_RATE}).get("text", "")

    elif engine == "granite-speech":
        # IBM Granite-Speech — chat-template prompt with an <|audio|> placeholder
        import torch

        processor, model, device = stt["processor"], stt["model"], stt["device"]
        chat = [{"role": "user", "content": "<|audio|>can you transcribe the speech into a written format?"}]
        prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        inputs = processor(prompt, audio_tensor, device=device, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        text = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    else:
        raise HTTPException(500, f"No handler for STT engine: {engine}")

    # Engines without language detection report the requested language
    return {
        "text": text.strip(),
        "segments": [],
        "language": language or "en",
        "language_probability": 1.0,
    }

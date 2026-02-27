"""Speech-to-text transcription endpoints."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.audio import audio_duration, prepare_for_model, validate_audio_file
from src.server import get_config, get_models

logger = logging.getLogger(__name__)
router = APIRouter(tags=["transcription"])


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
):
    """Transcribe an audio file to text.

    Returns transcription with timestamps and detected language.
    """
    data = await file.read()
    if not validate_audio_file(data):
        raise HTTPException(400, "Invalid audio file format")

    config = get_config()
    max_bytes = config.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {config.max_upload_mb}MB limit")

    models = get_models()
    stt = models.get_stt()
    audio = prepare_for_model(data)
    duration = len(audio) / 16_000

    t0 = time.perf_counter()

    stt_cfg = config.stt
    if stt_cfg and stt_cfg.engine == "faster-whisper":
        segments, info = stt.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        result_segments = []
        full_text_parts = []
        for seg in segments:
            result_segments.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

        elapsed = time.perf_counter() - t0
        return {
            "text": " ".join(full_text_parts),
            "segments": result_segments,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(duration, 3),
            "processing_time": round(elapsed, 3),
        }

    if stt_cfg and stt_cfg.engine == "vosk":
        import json
        import wave
        import io
        from vosk import KaldiRecognizer
        from src.audio import float32_to_wav_bytes

        wav_bytes = float32_to_wav_bytes(audio)
        wf = wave.open(io.BytesIO(wav_bytes), "rb")
        rec = KaldiRecognizer(stt, wf.getframerate())
        rec.SetWords(True)

        results = []
        while True:
            chunk = wf.readframes(4000)
            if len(chunk) == 0:
                break
            rec.AcceptWaveform(chunk)
        final = json.loads(rec.FinalResult())

        elapsed = time.perf_counter() - t0
        return {
            "text": final.get("text", ""),
            "segments": [],
            "language": language or "en",
            "language_probability": 1.0,
            "duration": round(duration, 3),
            "processing_time": round(elapsed, 3),
        }

    raise HTTPException(500, "No STT engine configured")

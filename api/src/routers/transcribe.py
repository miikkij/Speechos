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
    logger.info(
        "Transcribe request: filename=%s content_type=%s size=%d bytes",
        file.filename, file.content_type, len(data),
    )
    if not validate_audio_file(data):
        raise HTTPException(400, "Invalid audio file format")

    config = get_config()
    max_bytes = config.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {config.max_upload_mb}MB limit")

    # Docker STT engines: convert to WAV first (browser sends WebM/Opus)
    from src.docker_stt import is_docker_stt_engine, docker_transcribe
    from src.audio import read_audio, float32_to_wav_bytes
    if config.stt and is_docker_stt_engine(config.stt.engine):
        t0 = time.perf_counter()
        # Convert any format (WebM, OGG, MP3, etc.) to 16kHz mono WAV
        audio_np, sr = read_audio(data)
        from src.audio import resample_to_16k
        audio_16k = resample_to_16k(audio_np, sr)
        wav_data = float32_to_wav_bytes(audio_16k)
        import numpy as np
        rms = float(np.sqrt(np.mean(audio_16k**2)))
        duration_s = len(audio_16k) / 16_000
        logger.info(
            "Docker STT: engine=%s duration=%.2fs rms=%.6f wav_size=%d",
            config.stt.engine, duration_s, rms, len(wav_data),
        )
        try:
            result = await docker_transcribe(
                config.stt.engine, wav_data, "audio.wav", language,
            )
        except (ValueError, RuntimeError) as e:
            logger.error("Docker STT error: %s", e)
            raise HTTPException(503, f"Docker STT engine error: {e}")
        elapsed = time.perf_counter() - t0
        duration = len(audio_16k) / 16_000
        text = result.get("text", "")
        logger.info(
            "Docker STT result: text_len=%d duration=%.2fs time=%.2fs text=%s",
            len(text), duration, elapsed, text[:200],
        )
        return {
            "text": text,
            "segments": result.get("segments", []),
            "language": result.get("language", language or "en"),
            "language_probability": 1.0,
            "duration": round(duration, 3),
            "processing_time": round(elapsed, 3),
        }

    models = get_models()
    try:
        stt = models.get_stt()
    except (ImportError, FileNotFoundError, OSError, RuntimeError) as e:
        raise HTTPException(503, f"STT engine not available: {e}")
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
        import json as json_mod
        import wave
        import io
        from vosk import KaldiRecognizer
        from src.audio import float32_to_wav_bytes

        wav_bytes = float32_to_wav_bytes(audio)
        wf = wave.open(io.BytesIO(wav_bytes), "rb")
        rec = KaldiRecognizer(stt, wf.getframerate())
        rec.SetWords(True)

        while True:
            chunk = wf.readframes(4000)
            if len(chunk) == 0:
                break
            rec.AcceptWaveform(chunk)
        final = json_mod.loads(rec.FinalResult())

        elapsed = time.perf_counter() - t0
        return {
            "text": final.get("text", ""),
            "segments": [],
            "language": language or "en",
            "language_probability": 1.0,
            "duration": round(duration, 3),
            "processing_time": round(elapsed, 3),
        }

    if stt_cfg and stt_cfg.engine == "nemo":
        # NeMo ASRModel: transcribe from numpy array
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, 16_000)
            tmp_path = tmp.name
        try:
            transcriptions = stt.transcribe([tmp_path])
        finally:
            import os
            os.unlink(tmp_path)

        # NeMo returns list of strings or list of Hypothesis objects
        if transcriptions and isinstance(transcriptions[0], str):
            text = transcriptions[0]
        elif transcriptions and hasattr(transcriptions[0], 'text'):
            text = transcriptions[0].text
        else:
            text = str(transcriptions[0]) if transcriptions else ""

        elapsed = time.perf_counter() - t0
        return {
            "text": text,
            "segments": [],
            "language": language or "en",
            "language_probability": 1.0,
            "duration": round(duration, 3),
            "processing_time": round(elapsed, 3),
        }

    if stt_cfg and stt_cfg.engine == "whisperx":
        result = stt.transcribe(audio, language=language, batch_size=16)
        result_segments = []
        full_text_parts = []
        for seg in result.get("segments", []):
            result_segments.append({
                "start": round(seg.get("start", 0), 3),
                "end": round(seg.get("end", 0), 3),
                "text": seg.get("text", "").strip(),
            })
            full_text_parts.append(seg.get("text", "").strip())

        elapsed = time.perf_counter() - t0
        return {
            "text": " ".join(full_text_parts),
            "segments": result_segments,
            "language": result.get("language", language or "en"),
            "language_probability": 1.0,
            "duration": round(duration, 3),
            "processing_time": round(elapsed, 3),
        }

    if stt_cfg and stt_cfg.engine == "moonshine":
        text = stt.transcribe(audio)
        elapsed = time.perf_counter() - t0
        return {
            "text": text if isinstance(text, str) else str(text),
            "segments": [],
            "language": language or "en",
            "language_probability": 1.0,
            "duration": round(duration, 3),
            "processing_time": round(elapsed, 3),
        }

    if stt_cfg and stt_cfg.engine == "wav2vec2-stt":
        # Transformers ASR pipeline
        result = stt({"raw": audio, "sampling_rate": 16_000})
        elapsed = time.perf_counter() - t0
        return {
            "text": result.get("text", ""),
            "segments": [],
            "language": language or "en",
            "language_probability": 1.0,
            "duration": round(duration, 3),
            "processing_time": round(elapsed, 3),
        }

    raise HTTPException(500, f"No handler for STT engine: {stt_cfg.engine if stt_cfg else 'none'}")

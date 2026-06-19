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
        # NeMo ASRModel — transcribe from numpy array
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

    if stt_cfg and stt_cfg.engine == "granite-speech":
        # IBM Granite-Speech — chat-template prompt with an <|audio|> placeholder
        import torch

        processor = stt["processor"]
        model = stt["model"]
        device = stt["device"]
        chat = [{
            "role": "user",
            "content": "<|audio|>can you transcribe the speech into a written format?",
        }]
        prompt = processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        inputs = processor(prompt, audio_tensor, device=device, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        text = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        elapsed = time.perf_counter() - t0
        return {
            "text": text.strip(),
            "segments": [],
            "language": language or "en",
            "language_probability": 1.0,
            "duration": round(duration, 3),
            "processing_time": round(elapsed, 3),
        }

    raise HTTPException(500, f"No handler for STT engine: {stt_cfg.engine if stt_cfg else 'none'}")

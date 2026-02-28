"""Audio analysis endpoints: emotion detection and acoustic features."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.analysis import analyze_audio
from src.audio import prepare_for_model, validate_audio_file
from src.server import get_config, get_models

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze an audio file for emotion and acoustic features.

    Returns emotion predictions (if model available) and acoustic features
    (pitch, energy, tempo, spectral, speaking rate, MFCCs).
    """
    data = await file.read()
    if not validate_audio_file(data):
        raise HTTPException(400, "Invalid audio file format")

    config = get_config()
    max_bytes = config.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {config.max_upload_mb}MB limit")

    audio = prepare_for_model(data)
    models = get_models()

    # Try to load emotion model (returns None if not configured)
    emotion_model = None
    try:
        emotion_model = models.get_emotion()
    except Exception as e:
        logger.warning("Emotion model unavailable: %s", e)

    # Try to load diarization model (returns None if not configured)
    diarization_model = None
    try:
        diarization_model = models.get_diarization()
    except Exception as e:
        logger.warning("Diarization model unavailable: %s", e)

    t0 = time.perf_counter()
    result = analyze_audio(audio, sr=16_000, emotion_model=emotion_model, diarization_model=diarization_model)
    elapsed = time.perf_counter() - t0

    result["processing_time"] = round(elapsed, 3)
    result["emotion_available"] = emotion_model is not None
    result["diarization_available"] = diarization_model is not None

    return result

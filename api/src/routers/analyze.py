"""Audio analysis endpoints — emotion, diarization, VAD and acoustic features."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, File, UploadFile

from src.analysis import detect_speech, extract_audio_features, predict_emotion, run_diarization
from src.audio import TARGET_SAMPLE_RATE
from src.routers._common import decode_for_model, read_audio_upload
from src.server import get_config, get_models

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])

SR = TARGET_SAMPLE_RATE


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze an audio file for emotion, speakers, speech activity and acoustic features.

    Acoustic features (pitch, energy, tempo, spectral, speaking rate, MFCCs) are
    always returned; emotion, diarization and VAD only when a model is configured.
    The model-backed parts run concurrently, each under its own model lock.
    """
    config = get_config()
    data = await read_audio_upload(file, config)
    audio = await decode_for_model(data)
    models = get_models()

    async def optional(category: str, fn: Any) -> tuple[Any, str | None]:
        """(result, error). Result None + error None → model not configured."""
        if getattr(config, category) is None:
            return None, None
        try:
            return await models.run(category, fn), None
        except Exception as e:
            logger.warning("%s unavailable/failed: %s", category, e)
            return None, str(e)

    t0 = time.perf_counter()
    features, (emotions, emotion_error), (diar, diar_error), (speech, vad_error) = await asyncio.gather(
        asyncio.to_thread(extract_audio_features, audio, SR),
        optional("emotion", lambda m, c: predict_emotion(audio, SR, m)),
        optional("diarization", lambda m, c: run_diarization(audio, SR, m, engine=c.engine)),
        optional("vad", lambda m, c: detect_speech(audio, SR, m, engine=c.engine)),
    )
    elapsed = time.perf_counter() - t0

    result: dict[str, Any] = {"features": features, "emotions": emotions or []}
    if emotions:
        result["primary_emotion"] = emotions[0]["label"]
    if emotion_error:
        result["emotion_error"] = emotion_error

    if diar is not None:
        segments, summary = diar
        result["diarization"] = {"segments": segments, "summary": summary}
    else:
        result["diarization"] = {"segments": [], "summary": {"num_speakers": 0, "speakers": {}}}
        if diar_error:
            result["diarization"]["error"] = diar_error

    result["speech"] = speech
    if vad_error:
        result["vad_error"] = vad_error

    result["processing_time"] = round(elapsed, 3)
    result["emotion_available"] = emotions is not None
    result["diarization_available"] = diar is not None
    result["vad_available"] = speech is not None
    return result

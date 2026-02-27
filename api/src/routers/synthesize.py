"""Text-to-speech synthesis endpoints."""

from __future__ import annotations

import io
import logging
import time
import wave

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.server import get_config, get_models

logger = logging.getLogger(__name__)
router = APIRouter(tags=["synthesis"])


class SynthesizeRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float = 1.0


@router.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Synthesize speech from text, returns WAV audio."""
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")
    if len(req.text) > 10_000:
        raise HTTPException(400, "Text exceeds 10,000 character limit")

    config = get_config()
    models = get_models()
    tts = models.get_tts()

    t0 = time.perf_counter()

    tts_cfg = config.tts
    if tts_cfg and tts_cfg.engine == "piper":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            tts.synthesize(req.text, wf)
        buf.seek(0)

        elapsed = time.perf_counter() - t0
        logger.info("TTS completed in %.2fs for %d chars", elapsed, len(req.text))

        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={
                "X-Processing-Time": str(round(elapsed, 3)),
                "Content-Disposition": "inline; filename=speech.wav",
            },
        )

    raise HTTPException(500, "No TTS engine configured")

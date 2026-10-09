"""Helpers shared by the routers: upload validation, language parsing, decoding."""

from __future__ import annotations

import asyncio

import numpy as np
from fastapi import HTTPException, UploadFile

from src.audio import prepare_for_model, validate_audio_file
from src.config import AppConfig
from src.lang import normalize_language


async def read_audio_upload(file: UploadFile, config: AppConfig) -> bytes:
    max_bytes = config.max_upload_mb * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(413, f"File exceeds {config.max_upload_mb}MB limit")
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {config.max_upload_mb}MB limit")
    if not validate_audio_file(data):
        raise HTTPException(400, "Invalid audio file format")
    return data


async def decode_for_model(data: bytes) -> np.ndarray:
    """Decode + resample to 16 kHz mono float32 in a worker thread."""
    try:
        return await asyncio.to_thread(prepare_for_model, data)
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio: {e}") from e


def parse_language(value: str | None) -> str | None:
    try:
        return normalize_language(value)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

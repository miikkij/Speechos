"""Recording file management endpoints.

Every upload is decoded and re-encoded as 16 kHz mono 16-bit PCM WAV before it
is stored — the browser's container (usually WebM/Opus) is never trusted.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.audio import audio_duration, to_wav_16k_mono
from src.routers._common import read_audio_upload
from src.server import get_config

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recordings"])


def _recording_path(recording_id: str) -> Path:
    # Only allow alphanumeric, underscore, hyphen (no path traversal)
    safe_id = "".join(c for c in recording_id if c.isalnum() or c in ("_", "-"))
    if not safe_id or safe_id != recording_id:
        raise HTTPException(400, "Invalid recording ID")
    return Path(get_config().recordings_dir) / f"{safe_id}.wav"


def _list_recordings_sync(rec_dir: Path) -> list[dict]:
    recordings = []
    for f in sorted(rec_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = f.stat()
        try:
            duration = audio_duration(f)
        except Exception:
            duration = 0
        recordings.append({
            "id": f.stem,
            "filename": f.name,
            "size_bytes": stat.st_size,
            "duration": round(duration, 3),
            "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
        })
    return recordings


@router.get("/recordings")
async def list_recordings():
    """List all saved recordings."""
    rec_dir = Path(get_config().recordings_dir)
    if not rec_dir.exists():
        return {"recordings": []}
    return {"recordings": await asyncio.to_thread(_list_recordings_sync, rec_dir)}


@router.post("/recordings")
async def upload_recording(file: UploadFile = File(...)):
    """Upload a recording (any supported format) and store it as 16 kHz mono WAV."""
    config = get_config()
    data = await read_audio_upload(file, config)
    try:
        wav_bytes, duration = await asyncio.to_thread(to_wav_16k_mono, data)
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio: {e}") from e

    rec_dir = Path(config.recordings_dir)
    rec_dir.mkdir(parents=True, exist_ok=True)
    recording_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    filepath = rec_dir / f"{recording_id}.wav"

    async with aiofiles.open(filepath, "wb") as f:
        await f.write(wav_bytes)

    return {
        "id": recording_id,
        "filename": filepath.name,
        "size_bytes": len(wav_bytes),
        "duration": round(duration, 3),
    }


@router.delete("/recordings/{recording_id}")
async def delete_recording(recording_id: str):
    """Delete a recording by ID."""
    filepath = _recording_path(recording_id)
    if not filepath.exists():
        raise HTTPException(404, "Recording not found")
    filepath.unlink()
    return {"deleted": recording_id}


@router.get("/recordings/{recording_id}/audio")
async def get_recording_audio(recording_id: str):
    """Stream a recording's audio file."""
    filepath = _recording_path(recording_id)
    if not filepath.exists():
        raise HTTPException(404, "Recording not found")
    return FileResponse(filepath, media_type="audio/wav", filename=filepath.name)

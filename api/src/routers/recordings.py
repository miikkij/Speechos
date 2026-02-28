"""Recording file management endpoints."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.audio import audio_duration, validate_audio_file
from src.server import get_config

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recordings"])


@router.get("/recordings")
async def list_recordings():
    """List all saved recordings."""
    config = get_config()
    rec_dir = Path(config.recordings_dir)
    if not rec_dir.exists():
        return {"recordings": []}

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
    return {"recordings": recordings}


@router.post("/recordings")
async def upload_recording(file: UploadFile = File(...)):
    """Upload and save a recording."""
    data = await file.read()
    if not validate_audio_file(data):
        raise HTTPException(400, "Invalid audio file")

    config = get_config()
    max_bytes = config.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {config.max_upload_mb}MB limit")

    rec_dir = Path(config.recordings_dir)
    recording_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    filepath = rec_dir / f"{recording_id}.wav"

    async with aiofiles.open(filepath, "wb") as f:
        await f.write(data)

    duration = audio_duration(filepath)
    return {
        "id": recording_id,
        "filename": filepath.name,
        "size_bytes": len(data),
        "duration": round(duration, 3),
    }


@router.delete("/recordings/{recording_id}")
async def delete_recording(recording_id: str):
    """Delete a recording by ID."""
    config = get_config()
    # Sanitize: only allow alphanumeric, underscore, hyphen
    safe_id = "".join(c for c in recording_id if c.isalnum() or c in ("_", "-"))
    if safe_id != recording_id:
        raise HTTPException(400, "Invalid recording ID")

    filepath = Path(config.recordings_dir) / f"{safe_id}.wav"
    if not filepath.exists():
        raise HTTPException(404, "Recording not found")
    filepath.unlink()
    return {"deleted": recording_id}


@router.get("/recordings/{recording_id}/audio")
async def get_recording_audio(recording_id: str):
    """Stream a recording's audio file."""
    from fastapi.responses import FileResponse

    config = get_config()
    safe_id = "".join(c for c in recording_id if c.isalnum() or c in ("_", "-"))
    if safe_id != recording_id:
        raise HTTPException(400, "Invalid recording ID")

    filepath = Path(config.recordings_dir) / f"{safe_id}.wav"
    if not filepath.exists():
        raise HTTPException(404, "Recording not found")

    return FileResponse(
        filepath,
        media_type="audio/wav",
        filename=f"{safe_id}.wav",
    )

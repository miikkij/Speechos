"""Docker container management for STT engines running in WSL2.

Each Docker STT engine runs as an isolated container with GPU passthrough.
Only one GPU-heavy STT container should run at a time to avoid OOM.
TTS containers are NOT stopped: they can coexist with STT containers.
"""

from __future__ import annotations

import logging
import time

import httpx

from src.docker_utils import get_compose_file, run_docker_cmd

logger = logging.getLogger(__name__)


# ── Response parsers ────────────────────────────────────────


def _parse_text_response(resp: httpx.Response) -> dict:
    """Parse plain-text or JSON {text: ...} response (Speaches, Whisper ASR, whisper.cpp)."""
    ct = resp.headers.get("content-type", "")
    if "application/json" in ct:
        data = resp.json()
        text = data.get("text", "") if isinstance(data, dict) else str(data)
    else:
        text = resp.text.strip()
    return {"text": text, "segments": [], "language": "en"}


def _parse_linto_response(resp: httpx.Response) -> dict:
    """Parse LinTO NeMo / LinTO Whisper JSON response."""
    import json
    data = resp.json()
    # LinTO double-encodes: response is a JSON string containing JSON
    if isinstance(data, str):
        data = json.loads(data)
    text = data.get("text", "")
    # LinTO uses "words" (word-level) or "segments" (sentence-level)
    segments = []
    for seg in data.get("segments", data.get("words", [])):
        segments.append({
            "start": round(seg.get("start", 0), 3),
            "end": round(seg.get("end", 0), 3),
            "text": seg.get("text", seg.get("word", "")).strip(),
        })
    return {
        "text": text,
        "segments": segments,
        "language": data.get("language", "en"),
    }


def _parse_whisperx_response(resp: httpx.Response) -> dict:
    """Parse WhisperX API JSON response."""
    data = resp.json()
    segments = []
    full_text_parts = []
    for seg in data.get("segments", []):
        segments.append({
            "start": round(seg.get("start", 0), 3),
            "end": round(seg.get("end", 0), 3),
            "text": seg.get("text", "").strip(),
        })
        full_text_parts.append(seg.get("text", "").strip())
    text = data.get("text", " ".join(full_text_parts))
    return {"text": text, "segments": segments, "language": data.get("language", "en")}


# ── Engine registry ─────────────────────────────────────────

DOCKER_STT_ENGINES: dict[str, dict] = {
    # Tested 2026-02-28: 9s startup, returns JSON {text: "..."}
    "speaches": {
        "service": "speaches",
        "port": 36320,
        "health_url": "http://localhost:36320/health",
        "transcribe_url": "http://localhost:36320/v1/audio/transcriptions",
        "file_field": "file",
        "extra_fields": {"model": "Systran/faster-whisper-small"},
        "headers": {},
        "parser": _parse_text_response,
    },
    # Tested 2026-02-28: 51s startup, returns plain text
    "whisper-asr": {
        "service": "whisper-asr",
        "port": 36321,
        "health_url": "http://localhost:36321/",
        "transcribe_url": "http://localhost:36321/asr",
        "file_field": "audio_file",
        "extra_fields": {},
        "headers": {},
        "parser": _parse_text_response,
    },
    # Tested 2026-02-28: 57s startup, double-encoded JSON, word timestamps
    "linto-nemo": {
        "service": "linto-nemo",
        "port": 36323,
        "health_url": "http://localhost:36323/healthcheck",
        "transcribe_url": "http://localhost:36323/transcribe",
        "file_field": "file",
        "extra_fields": {},
        "headers": {"Accept": "application/json"},
        "parser": _parse_linto_response,
    },
    # Tested 2026-02-28: 42s startup, double-encoded JSON, word timestamps + confidence
    "linto-whisper": {
        "service": "linto-whisper",
        "port": 36324,
        "health_url": "http://localhost:36324/healthcheck",
        "transcribe_url": "http://localhost:36324/transcribe",
        "file_field": "file",
        "extra_fields": {},
        "headers": {"Accept": "application/json"},
        "parser": _parse_linto_response,
    },
    # Tested 2026-02-28: 75s startup, double-encoded JSON, lowercase only (no caps/punct)
    "linto-nemo-1.1b": {
        "service": "linto-nemo-1.1b",
        "port": 36327,
        "health_url": "http://localhost:36327/healthcheck",
        "transcribe_url": "http://localhost:36327/transcribe",
        "file_field": "file",
        "extra_fields": {},
        "headers": {"Accept": "application/json"},
        "parser": _parse_linto_response,
    },
}
# REMOVED: whisper-cpp: requires manual model download (ggml-large-v3.bin), not auto-startable
# REMOVED: whisperx-api: upstream bug: VAD model download URL returns 301, container crashes

_COMPOSE_FILE = get_compose_file("stt-engines.yml")


# ── Public API ──────────────────────────────────────────────


def is_docker_stt_engine(engine: str) -> bool:
    """Check if an STT engine is Docker-backed."""
    return engine in DOCKER_STT_ENGINES


def get_docker_stt_config(engine: str) -> dict | None:
    """Get Docker configuration for an STT engine."""
    return DOCKER_STT_ENGINES.get(engine)


def is_stt_container_running(engine: str) -> bool:
    """Check if a Docker STT container is running."""
    cfg = DOCKER_STT_ENGINES.get(engine)
    if not cfg:
        return False
    try:
        result = run_docker_cmd(_COMPOSE_FILE, ["ps", "-q", cfg["service"]], timeout=10)
        return bool(result.stdout.strip())
    except Exception as e:
        logger.warning("Failed to check STT container status for %s: %s", engine, e)
        return False


def start_stt_container(engine: str, wait_ready: bool = True, timeout: int = 180) -> bool:
    """Start a Docker STT container. Returns True if ready."""
    cfg = DOCKER_STT_ENGINES.get(engine)
    if not cfg:
        logger.error("Unknown Docker STT engine: %s", engine)
        return False

    logger.info("Starting Docker STT container: %s", engine)

    # Stop other STT containers (GPU sharing), but NOT TTS containers
    stop_all_stt_containers(exclude=engine)

    result = run_docker_cmd(_COMPOSE_FILE, ["up", "-d", cfg["service"]], timeout=120)
    if result.returncode != 0:
        logger.error("Failed to start %s: %s", engine, result.stderr)
        return False

    if not wait_ready:
        return True

    health_url = cfg["health_url"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(health_url, timeout=5)
            if resp.status_code < 500:
                logger.info("Docker STT container %s is ready", engine)
                return True
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            pass
        time.sleep(3)

    logger.warning("Docker STT container %s did not become ready within %ds", engine, timeout)
    return False


def stop_stt_container(engine: str) -> None:
    """Stop a Docker STT container."""
    cfg = DOCKER_STT_ENGINES.get(engine)
    if not cfg:
        return
    logger.info("Stopping Docker STT container: %s", engine)
    run_docker_cmd(_COMPOSE_FILE, ["stop", cfg["service"]], timeout=30)


def stop_all_stt_containers(exclude: str | None = None) -> None:
    """Stop all Docker STT containers except the excluded one."""
    for engine in DOCKER_STT_ENGINES:
        if engine != exclude:
            if is_stt_container_running(engine):
                stop_stt_container(engine)


def _is_stt_container_healthy(engine: str) -> bool:
    """Check if a Docker STT container is running AND the model is loaded."""
    cfg = DOCKER_STT_ENGINES.get(engine)
    if not cfg:
        return False
    try:
        resp = httpx.get(cfg["health_url"], timeout=5)
        return resp.status_code < 500
    except (httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout, httpx.ConnectTimeout):
        return False


def ensure_stt_ready(engine: str) -> bool:
    """Ensure a Docker STT container is running and healthy. Public for pre-warming."""
    if _is_stt_container_healthy(engine):
        return True
    # Container not healthy: (re)start it
    return start_stt_container(engine)


async def docker_transcribe(
    engine: str,
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language: str | None = None,
) -> dict:
    """Transcribe audio via a Docker STT container.

    Returns dict with keys: text, segments, language.
    """
    cfg = DOCKER_STT_ENGINES.get(engine)
    if not cfg:
        raise ValueError(f"Unknown Docker STT engine: {engine}")

    # Ensure container is running AND healthy (model loaded)
    if not ensure_stt_ready(engine):
        raise RuntimeError(f"Failed to start Docker STT container: {engine}")

    # Build multipart upload
    files = {cfg["file_field"]: (filename, audio_bytes, "audio/wav")}
    data = dict(cfg["extra_fields"])
    if language:
        data["language"] = language
    headers = dict(cfg["headers"])

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                cfg["transcribe_url"],
                files=files,
                data=data,
                headers=headers,
            )
    except (httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
        raise RuntimeError(
            f"Docker STT {engine} connection failed (container may still be loading): {e}"
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Docker STT {engine} returned {resp.status_code}: {resp.text[:500]}"
        )

    return cfg["parser"](resp)

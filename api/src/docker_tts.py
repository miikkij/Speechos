"""Docker container management for TTS engines running in WSL2.

Each Docker TTS engine runs as an isolated container with GPU passthrough.
Only one GPU-heavy TTS container should run at a time to avoid OOM on the 24GB GPU.
"""

from __future__ import annotations

import logging
import time

import httpx

from src.docker_utils import get_compose_file, run_docker_cmd

logger = logging.getLogger(__name__)

# Map engine names to docker-compose service names and their API details
DOCKER_TTS_ENGINES: dict[str, dict] = {
    "xtts": {
        "service": "xtts",
        "port": 36310,
        "health_url": "http://localhost:36310/",
        "synth_url": "http://localhost:36310/api/tts",
        "synth_method": "GET",
        "synth_params": lambda text, voice: {"text": text, "language_id": voice or "en"},
        "response_type": "wav_binary",
    },
    "chattts": {
        "service": "chattts",
        "port": 36311,
        "health_url": "http://localhost:36311/",
        "synth_url": "http://localhost:36311/generate",
        "synth_method": "POST",
        "synth_params": lambda text, voice: {"text": text},
        "response_type": "wav_binary",
    },
    "melotts": {
        "service": "melotts",
        "port": 36312,
        "health_url": "http://localhost:36312/",
        "synth_url": "http://localhost:36312/convert/tts",
        "synth_method": "POST",
        "synth_params": lambda text, voice: {
            "text": text,
            "language": "EN",
            "speaker_id": voice or "EN-US",
        },
        "response_type": "wav_binary",
    },
    "orpheus": {
        "service": "orpheus",
        "port": 36313,
        "health_url": "http://localhost:36313/",
        "synth_url": "http://localhost:36313/api/generate",
        "synth_method": "POST",
        "synth_params": lambda text, voice: {"text": text},
        "response_type": "wav_binary",
    },
    "fish-speech": {
        "service": "fish-speech",
        "port": 36314,
        "health_url": "http://localhost:36314/",
        "synth_url": "http://localhost:36314/v1/tts",
        "synth_method": "POST",
        "synth_params": lambda text, voice: {"text": text},
        "response_type": "wav_binary",
    },
    # cosyvoice: DISABLED: Docker image catcto/cosyvoice:latest doesn't exist on Docker Hub.
    # Needs local build from https://github.com/catcto/CosyVoiceDocker or alternative image.
    "qwen3-tts": {
        "service": "qwen3-tts",
        "port": 36316,
        "health_url": "http://localhost:36316/health",
        "synth_url": "http://localhost:36316/v1/audio/speech",
        "synth_method": "POST",
        "synth_params": lambda text, voice: {
            "input": text,
            "voice": voice or "serena",
            "response_format": "wav",
        },
        "response_type": "wav_binary",
    },
    "parler": {
        "service": "parler",
        "port": 36317,
        "health_url": "http://localhost:36317/health",
        "synth_url": "http://localhost:36317/v1/audio/speech",
        "synth_method": "POST",
        "synth_params": lambda text, voice: {
            "input": text,
            "voice": voice or "A female speaker delivers a clear, natural speech.",
            "response_type": "wav",
        },
        "response_type": "wav_binary",
    },
}

_COMPOSE_FILE = get_compose_file("tts-engines.yml")


def is_docker_engine(engine: str) -> bool:
    """Check if a TTS engine is Docker-backed."""
    return engine in DOCKER_TTS_ENGINES


def get_docker_config(engine: str) -> dict | None:
    """Get Docker configuration for a TTS engine."""
    return DOCKER_TTS_ENGINES.get(engine)


def is_container_running(engine: str) -> bool:
    """Check if a Docker TTS container is running."""
    cfg = DOCKER_TTS_ENGINES.get(engine)
    if not cfg:
        return False
    try:
        result = run_docker_cmd(_COMPOSE_FILE, ["ps", "-q", cfg["service"]], timeout=10)
        return bool(result.stdout.strip())
    except Exception as e:
        logger.warning("Failed to check container status for %s: %s", engine, e)
        return False


def start_container(engine: str, wait_ready: bool = True, timeout: int = 180) -> bool:
    """Start a Docker TTS container. Returns True if ready."""
    cfg = DOCKER_TTS_ENGINES.get(engine)
    if not cfg:
        logger.error("Unknown Docker TTS engine: %s", engine)
        return False

    logger.info("Starting Docker TTS container: %s", engine)

    # Stop any other Docker TTS containers first (GPU memory sharing)
    stop_all_containers(exclude=engine)

    result = run_docker_cmd(_COMPOSE_FILE, ["up", "-d", cfg["service"]], timeout=120)
    if result.returncode != 0:
        logger.error("Failed to start %s: %s", engine, result.stderr)
        return False

    if not wait_ready:
        return True

    # Wait for the container to be healthy
    health_url = cfg["health_url"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(health_url, timeout=5)
            if resp.status_code < 500:
                logger.info("Docker TTS container %s is ready", engine)
                return True
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            pass
        time.sleep(3)

    logger.warning("Docker TTS container %s did not become ready within %ds", engine, timeout)
    return False


def stop_container(engine: str) -> None:
    """Stop a Docker TTS container."""
    cfg = DOCKER_TTS_ENGINES.get(engine)
    if not cfg:
        return
    logger.info("Stopping Docker TTS container: %s", engine)
    run_docker_cmd(_COMPOSE_FILE, ["stop", cfg["service"]], timeout=30)


def stop_all_containers(exclude: str | None = None) -> None:
    """Stop all Docker TTS containers except the excluded one."""
    for engine in DOCKER_TTS_ENGINES:
        if engine != exclude:
            if is_container_running(engine):
                stop_container(engine)


async def docker_synthesize(engine: str, text: str, voice: str | None = None) -> bytes:
    """Synthesize speech via a Docker TTS container. Returns WAV bytes."""
    cfg = DOCKER_TTS_ENGINES.get(engine)
    if not cfg:
        raise ValueError(f"Unknown Docker TTS engine: {engine}")

    # Ensure container is running
    if not is_container_running(engine):
        if not start_container(engine):
            raise RuntimeError(f"Failed to start Docker TTS container: {engine}")

    params = cfg["synth_params"](text, voice)
    method = cfg["synth_method"]
    url = cfg["synth_url"]

    async with httpx.AsyncClient(timeout=120) as client:
        if method == "GET":
            resp = await client.get(url, params=params)
        else:
            resp = await client.post(url, json=params)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Docker TTS {engine} returned {resp.status_code}: {resp.text[:500]}"
        )

    return resp.content

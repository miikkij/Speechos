"""Docker container management for TTS engines running in WSL2.

Each Docker TTS engine runs as an isolated container with GPU passthrough.
Only one GPU-heavy TTS container should run at a time to avoid OOM on the 24GB GPU.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Map engine names to docker-compose service names and their API details
DOCKER_TTS_ENGINES: dict[str, dict] = {
    "xtts": {
        "service": "xtts",
        "port": 36310,
        "health_url": "http://localhost:36310/",
        "synth_url": "http://localhost:36310/api/tts",
        "synth_method": "GET",
        "synth_params": lambda text, voice, lang: {"text": text, "language_id": lang or "en"},
        "response_type": "wav_binary",
    },
    "chattts": {
        "service": "chattts",
        "port": 36311,
        "health_url": "http://localhost:36311/",
        "synth_url": "http://localhost:36311/generate",
        "synth_method": "POST",
        "synth_params": lambda text, voice, lang: {"text": text},
        "response_type": "wav_binary",
    },
    "melotts": {
        "service": "melotts",
        "port": 36312,
        "health_url": "http://localhost:36312/",
        "synth_url": "http://localhost:36312/tts/convert/tts",
        "synth_method": "POST",
        "synth_params": lambda text, voice, lang: {
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
        "synth_url": "http://localhost:36313/api/synthesize",
        "synth_method": "POST",
        "synth_params": lambda text, voice, lang: {"text": text},
        "response_type": "wav_binary",
    },
    "fish-speech": {
        "service": "fish-speech",
        "port": 36314,
        "health_url": "http://localhost:36314/",
        "synth_url": "http://localhost:36314/api/synthesize",
        "synth_method": "POST",
        "synth_params": lambda text, voice, lang: {"text": text},
        "response_type": "wav_binary",
    },
    "cosyvoice": {
        "service": "cosyvoice",
        "port": 36315,
        "health_url": "http://localhost:36315/",
        "synth_url": "http://localhost:36315/api/synthesize",
        "synth_method": "POST",
        "synth_params": lambda text, voice, lang: {"text": text},
        "response_type": "wav_binary",
    },
    "qwen3-tts": {
        "service": "qwen3-tts",
        "port": 36316,
        "health_url": "http://localhost:36316/",
        "synth_url": "http://localhost:36316/base_tts/",
        "synth_method": "GET",
        "synth_params": lambda text, voice, lang: {"text": text},
        "response_type": "wav_binary",
    },
    "parler": {
        "service": "parler",
        "port": 36317,
        "health_url": "http://localhost:36317/health",
        "synth_url": "http://localhost:36317/v1/audio/speech",
        "synth_method": "POST",
        "synth_params": lambda text, voice, lang: {
            "input": text,
            "voice": voice or "A female speaker delivers a clear, natural speech.",
        },
        "response_type": "wav_binary",
    },
    # VibeVoice — long-form, multi-speaker podcast TTS (community vibevoice-api,
    # OpenAI-compatible /v1/audio/speech). Use "Speaker 0:" / "Speaker 1:" prefixes
    # in the text for multi-speaker scripts. Output is 24kHz WAV.
    "vibevoice": {
        "service": "vibevoice",
        "port": 36318,
        "health_url": "http://localhost:36318/health",
        "synth_url": "http://localhost:36318/v1/audio/speech",
        "synth_method": "POST",
        "synth_params": lambda text, voice, lang: {
            "model": "vibevoice/VibeVoice-1.5B",
            "input": text,
            "voice": voice or "Frank",
            "response_format": "wav",
        },
        "response_type": "wav_binary",
    },
}

# Resolve path to tts-engines.yml (relative to this file)
_COMPOSE_FILE = Path(__file__).resolve().parent.parent.parent / "docker" / "tts-engines.yml"


def _wsl_compose_path() -> str:
    """Convert Windows path to WSL2 path for docker compose."""
    path = str(_COMPOSE_FILE).replace("\\", "/")
    # E:/dev/... -> /mnt/e/dev/...
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path = f"/mnt/{drive}{path[2:]}"
    return path




@dataclass
class DockerCmdResult:
    returncode: int
    stdout: str
    stderr: str


async def _run_docker_cmd(args: list[str], timeout: float = 60) -> DockerCmdResult:
    """Run a docker compose command via WSL2 bash without blocking the event loop."""
    compose_path = _wsl_compose_path()
    cmd_str = " ".join(shlex.quote(a) for a in ["docker", "compose", "-f", compose_path, *args])
    logger.info("Docker command: %s", cmd_str)
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", cmd_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except NotImplementedError:
        # Windows SelectorEventLoop (uvicorn with workers > 1) has no subprocess support
        done = await asyncio.to_thread(
            subprocess.run, ["bash", "-c", cmd_str], capture_output=True, text=True, timeout=timeout
        )
        return DockerCmdResult(done.returncode, done.stdout, done.stderr)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return DockerCmdResult(
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


def is_docker_engine(engine: str) -> bool:
    """Check if a TTS engine is Docker-backed."""
    return engine in DOCKER_TTS_ENGINES


def get_docker_config(engine: str) -> dict | None:
    """Get Docker configuration for a TTS engine."""
    return DOCKER_TTS_ENGINES.get(engine)


async def is_container_running(engine: str) -> bool:
    """Check if a Docker TTS container is running."""
    cfg = DOCKER_TTS_ENGINES.get(engine)
    if not cfg:
        return False
    try:
        result = await _run_docker_cmd(["ps", "-q", cfg["service"]], timeout=10)
        return bool(result.stdout.strip())
    except Exception as e:
        logger.warning("Failed to check container status for %s: %s", engine, e)
        return False


async def start_container(engine: str, wait_ready: bool = True, timeout: float = 120) -> bool:
    """Start a Docker TTS container. Returns True if ready."""
    cfg = DOCKER_TTS_ENGINES.get(engine)
    if not cfg:
        logger.error("Unknown Docker TTS engine: %s", engine)
        return False

    logger.info("Starting Docker TTS container: %s", engine)

    # Stop any other Docker TTS containers first (GPU memory sharing)
    await stop_all_containers(exclude=engine)

    result = await _run_docker_cmd(["up", "-d", cfg["service"]], timeout=120)
    if result.returncode != 0:
        logger.error("Failed to start %s: %s", engine, result.stderr)
        return False

    if not wait_ready:
        return True

    # Wait for the container's HTTP endpoint to answer
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=5) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(cfg["health_url"])
                if resp.status_code < 500:
                    logger.info("Docker TTS container %s is ready", engine)
                    return True
            except httpx.TransportError:
                pass
            await asyncio.sleep(3)

    logger.warning("Docker TTS container %s did not become ready within %ds", engine, timeout)
    return False


async def stop_container(engine: str) -> None:
    """Stop a Docker TTS container."""
    cfg = DOCKER_TTS_ENGINES.get(engine)
    if not cfg:
        return
    logger.info("Stopping Docker TTS container: %s", engine)
    await _run_docker_cmd(["stop", cfg["service"]], timeout=30)


async def stop_all_containers(exclude: str | None = None) -> None:
    """Stop all Docker TTS containers except the excluded one."""
    others = [e for e in DOCKER_TTS_ENGINES if e != exclude]
    running = await asyncio.gather(*(is_container_running(e) for e in others))
    await asyncio.gather(*(stop_container(e) for e, up in zip(others, running) if up))


async def docker_synthesize(
    engine: str, text: str, voice: str | None = None, language: str | None = None
) -> bytes:
    """Synthesize speech via a Docker TTS container. Returns WAV bytes."""
    cfg = DOCKER_TTS_ENGINES.get(engine)
    if not cfg:
        raise ValueError(f"Unknown Docker TTS engine: {engine}")

    # Ensure container is running
    if not await is_container_running(engine):
        if not await start_container(engine):
            raise RuntimeError(f"Failed to start Docker TTS container: {engine}")

    params = cfg["synth_params"](text, voice, language)
    url = cfg["synth_url"]

    async with httpx.AsyncClient(timeout=120) as client:
        if cfg["synth_method"] == "GET":
            resp = await client.get(url, params=params)
        else:
            resp = await client.post(url, json=params)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Docker TTS {engine} returned {resp.status_code}: {resp.text[:500]}"
        )

    return resp.content

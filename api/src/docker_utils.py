"""Shared Docker helpers for WSL2 compose commands.

Used by docker_tts.py, docker_stt.py, and future docker_analysis.py.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_compose_file(filename: str) -> Path:
    """Return the absolute path to a compose file in docker/."""
    return _PROJECT_ROOT / "docker" / filename


def wsl_compose_path(compose_file: Path) -> str:
    """Convert Windows path to WSL2 path for docker compose."""
    path = str(compose_file).replace("\\", "/")
    # E:/dev/... -> /mnt/e/dev/...
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path = f"/mnt/{drive}{path[2:]}"
    return path


def run_docker_cmd(
    compose_file: Path, args: list[str], timeout: int = 60
) -> subprocess.CompletedProcess:
    """Run a docker compose command via bash."""
    compose_path = wsl_compose_path(compose_file)
    cmd_str = " ".join(["docker", "compose", "-f", compose_path] + args)
    full_cmd = ["bash", "-c", cmd_str]
    logger.info("Docker command: %s", cmd_str)
    return subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

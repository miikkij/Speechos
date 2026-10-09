"""Docker TTS helpers use non-blocking subprocesses (docker itself is mocked)."""

from __future__ import annotations

import asyncio

import pytest

import src.docker_tts as docker_tts


class FakeProc:
    def __init__(self, stdout: bytes = b"", returncode: int = 0, delay: float = 0.0) -> None:
        self._stdout = stdout
        self.returncode = returncode
        self.delay = delay
        self.killed = False

    async def communicate(self):
        await asyncio.sleep(self.delay)
        return self._stdout, b""

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


@pytest.fixture
def fake_exec(monkeypatch):
    calls: list[tuple] = []
    procs: dict[str, FakeProc] = {}

    async def create_subprocess_exec(*args, **kwargs):
        calls.append(args)
        cmd = args[-1]
        for needle, proc in procs.items():
            if needle in cmd:
                return proc
        return FakeProc()

    monkeypatch.setattr(docker_tts.asyncio, "create_subprocess_exec", create_subprocess_exec)
    return calls, procs


async def test_is_container_running(fake_exec):
    calls, procs = fake_exec
    procs["ps -q xtts"] = FakeProc(stdout=b"abc123\n")
    assert await docker_tts.is_container_running("xtts") is True
    assert await docker_tts.is_container_running("melotts") is False
    assert await docker_tts.is_container_running("not-an-engine") is False
    assert calls[0][:2] == ("bash", "-c")
    assert "docker compose -f" in calls[0][2]


async def test_timeout_kills_process(fake_exec):
    _, procs = fake_exec
    procs["ps -q"] = slow = FakeProc(delay=5)
    with pytest.raises(TimeoutError):
        await docker_tts._run_docker_cmd(["ps", "-q", "xtts"], timeout=0.05)
    assert slow.killed


async def test_docker_status_does_not_block_loop(fake_exec):
    _, procs = fake_exec
    procs["ps -q"] = FakeProc(delay=0.3)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    t = asyncio.create_task(ticker())
    await docker_tts.stop_all_containers()
    t.cancel()
    assert ticks > 10  # loop kept running while "docker" was busy


def test_synth_params_use_language():
    params = docker_tts.DOCKER_TTS_ENGINES["xtts"]["synth_params"]("moi", None, "fi")
    assert params == {"text": "moi", "language_id": "fi"}


def test_wsl_path():
    path = docker_tts._wsl_compose_path()
    assert path.endswith("/docker/tts-engines.yml")
    assert ":" not in path

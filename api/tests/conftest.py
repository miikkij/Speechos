"""Shared fixtures. Nothing here loads a real model or needs a GPU / torch."""

from __future__ import annotations

import io
import sys
import types
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.server as server
from src.config import AppConfig, ComputeMode, HardwareProfile, ModelConfig
from src.models import ModelManager

SR = 16_000
REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Audio generators ────────────────────────────────────────


def sine(duration: float = 1.0, sr: int = SR, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def encode_wav(audio: np.ndarray, sr: int = SR, subtype: str = "PCM_16") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def encode_webm_opus(audio: np.ndarray, sr: int = 48_000, channels: int = 2) -> bytes:
    """What Chrome's MediaRecorder produces: WebM container, Opus codec, 48 kHz."""
    import av

    frames = np.ascontiguousarray(np.stack([audio] * channels))
    buf = io.BytesIO()
    container = av.open(buf, "w", format="webm")
    stream = container.add_stream("libopus", rate=sr)
    stream.layout = "stereo" if channels == 2 else "mono"
    frame_size = 960
    for i in range(0, frames.shape[1] - frame_size + 1, frame_size):
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(frames[:, i:i + frame_size]), format="fltp", layout=stream.layout.name
        )
        frame.sample_rate = sr
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return buf.getvalue()


@pytest.fixture
def make_wav() -> Callable[..., bytes]:
    """make_wav(duration=1.0, sr=16000, channels=1, freq=220.0) → WAV bytes."""

    def _make(duration: float = 1.0, sr: int = SR, channels: int = 1, freq: float = 220.0) -> bytes:
        mono = sine(duration, sr, freq)
        audio = mono if channels == 1 else np.stack([mono] * channels, axis=1)
        return encode_wav(audio, sr)

    return _make


@pytest.fixture
def wav_bytes(make_wav: Callable[..., bytes]) -> bytes:
    return make_wav()


@pytest.fixture
def webm_bytes() -> bytes:
    return encode_webm_opus(sine(1.0, 48_000, 440.0))


# ── Fake engines ────────────────────────────────────────────


@dataclass
class _Seg:
    start: float
    end: float
    text: str


@dataclass
class _Info:
    language: str
    language_probability: float


class FakeWhisper:
    """Mimics faster_whisper.WhisperModel.transcribe()."""

    def __init__(self, text: str = "hei maailma", delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: np.ndarray, language: str | None = None, **kwargs: Any):
        import time

        self.calls.append({"samples": len(audio), "language": language, **kwargs})
        if self.delay:
            time.sleep(self.delay)  # blocking, like real CTranslate2 inference
        segs = iter([_Seg(0.0, 1.0, f" {self.text} ")])
        return segs, _Info(language=language or "fi", language_probability=0.97)


@dataclass
class _PiperChunk:
    sample_rate: int
    audio_float_array: np.ndarray


class FakePiperVoice:
    """Mimics piper.PiperVoice.synthesize(text, syn_config=...)."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def synthesize(self, text: str, syn_config: Any = None):
        self.calls.append({"text": text, "syn_config": syn_config})
        yield _PiperChunk(22050, sine(0.25, 22050))


@pytest.fixture
def fake_piper_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Stand-in for the `piper` package (only SynthesisConfig is imported at synth time)."""

    @dataclass
    class SynthesisConfig:
        speaker_id: int | None = None
        length_scale: float | None = None
        noise_scale: float | None = None
        noise_w_scale: float | None = None

    mod = types.ModuleType("piper")
    mod.SynthesisConfig = SynthesisConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", mod)
    return mod


# ── App fixtures ────────────────────────────────────────────


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    hw = HardwareProfile(
        mode=ComputeMode.CPU, cpu_cores=4, ram_gb=8.0, gpu_available=False,
        gpu_name=None, vram_gb=0.0, tier="cpu-8gb",
    )
    return AppConfig(
        hardware=hw,
        stt=ModelConfig(engine="faster-whisper", model="tiny", device="cpu", compute_type="int8"),
        tts=ModelConfig(engine="piper", model="fi_FI-harri-medium", device="cpu"),
        emotion=None,
        diarization=None,
        vad=None,
        model_dir=str(tmp_path / "models"),
        recordings_dir=str(tmp_path / "recordings"),
        samples_dir=str(tmp_path / "samples"),
    )


@pytest.fixture
def manager(app_config: AppConfig) -> ModelManager:
    return ModelManager(app_config)


@pytest.fixture
def fake_whisper(manager: ModelManager, monkeypatch: pytest.MonkeyPatch) -> FakeWhisper:
    fake = FakeWhisper()
    monkeypatch.setattr(manager, "_load_stt", lambda cfg: fake)
    return fake


@pytest.fixture
def piper_voices(
    manager: ModelManager, monkeypatch: pytest.MonkeyPatch, fake_piper_module: types.ModuleType
) -> dict[str, FakePiperVoice]:
    """Fake Piper loader; returns {model_name: voice} for every voice loaded."""
    voices: dict[str, FakePiperVoice] = {}

    def load(cfg: ModelConfig) -> Any:
        if cfg.engine != "piper":
            raise AssertionError(f"unexpected engine {cfg.engine}")
        voices[cfg.model] = FakePiperVoice(cfg.model)
        return voices[cfg.model]

    monkeypatch.setattr(manager, "_load_tts", load)
    return voices


@pytest.fixture
def app(app_config: AppConfig, manager: ModelManager) -> Iterator[FastAPI]:
    yield server.create_app(app_config, manager)
    server.config = None
    server.model_manager = None


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
async def aclient(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

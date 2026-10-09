"""Tier detection and config overrides (psutil and torch.cuda mocked)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from src import config as config_mod
from src.config import TIER_CONFIGS, ComputeMode, detect_hardware, load_config

GB = 1024**3


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("SPEECHOS_") or key == "HF_TOKEN":
            monkeypatch.delenv(key)
    # No stray config.yaml from the developer's checkout; dirs go to tmp
    monkeypatch.chdir(tmp_path)


def fake_hw(monkeypatch: pytest.MonkeyPatch, ram_gb: float, vram_gb: float | None, cores: int = 8) -> None:
    """Pretend the machine has `ram_gb` RAM and (if vram_gb) one CUDA GPU."""
    monkeypatch.setattr(config_mod.psutil, "cpu_count", lambda logical=False: cores)
    monkeypatch.setattr(
        config_mod.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=int(ram_gb * GB))
    )
    cuda = types.SimpleNamespace(
        is_available=lambda: vram_gb is not None,
        device_count=lambda: 1 if vram_gb is not None else 0,
        get_device_name=lambda i: "Fake RTX",
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=int((vram_gb or 0) * GB)),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))


@pytest.mark.parametrize(
    ("ram", "vram", "tier", "mode"),
    [
        (2, None, "cpu-2gb", ComputeMode.CPU),
        (4, None, "cpu-4gb", ComputeMode.CPU),
        (7.8, None, "cpu-8gb", ComputeMode.CPU),
        (16, None, "cpu-16gb", ComputeMode.CPU),
        (64, None, "cpu-32gb", ComputeMode.CPU),
        (4, 8, "gpu-8gb", ComputeMode.GPU),  # < 8 GB RAM → GPU-only mode
        (32, 6, "hybrid-8gb-gpu", ComputeMode.HYBRID),
        (32, 12, "hybrid-12gb-gpu", ComputeMode.HYBRID),
        (64, 24, "hybrid-24gb-gpu", ComputeMode.HYBRID),
        (64, 32, "hybrid-32gb-gpu", ComputeMode.HYBRID),
    ],
)
def test_tier_detection(monkeypatch, ram, vram, tier, mode):
    fake_hw(monkeypatch, ram, vram)
    hw = detect_hardware()
    assert hw.tier == tier
    assert hw.mode == mode
    assert hw.gpu_available is (vram is not None)


def test_no_torch_means_cpu(monkeypatch):
    fake_hw(monkeypatch, 16, None)
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch → ImportError
    hw = detect_hardware()
    assert hw.tier == "cpu-16gb"
    assert hw.gpu_available is False


def test_broken_cuda_falls_back_to_cpu(monkeypatch):
    """e.g. CUDA_VISIBLE_DEVICES="" — is_available() is True but device 0 is invalid."""
    fake_hw(monkeypatch, 16, 24)

    def invalid(i):
        raise AssertionError("Invalid device id")

    monkeypatch.setattr(sys.modules["torch"].cuda, "get_device_name", invalid)
    hw = detect_hardware()
    assert hw.gpu_available is False
    assert hw.tier == "cpu-16gb"


def test_compute_mode_env_forces_cpu_on_gpu_machine(monkeypatch):
    fake_hw(monkeypatch, 64, 24)
    monkeypatch.setenv("SPEECHOS_COMPUTE_MODE", "cpu")
    hw = detect_hardware()
    assert hw.mode == ComputeMode.CPU
    assert hw.tier == "cpu-32gb"


def test_tier_env_override_and_validation(monkeypatch):
    fake_hw(monkeypatch, 8, None)
    monkeypatch.setenv("SPEECHOS_TIER", "gpu-24gb")
    assert detect_hardware().tier == "gpu-24gb"
    monkeypatch.setenv("SPEECHOS_TIER", "gpu-99gb")
    with pytest.raises(ValueError, match="Unknown tier"):
        detect_hardware()


def test_every_tier_defaults_to_finnish_piper():
    for name, tier in TIER_CONFIGS.items():
        assert tier["tts"]["engine"] == "piper", name
        assert tier["tts"]["model"] == "fi_FI-harri-medium", name
        assert tier["stt"]["engine"] == "faster-whisper", name  # multilingual everywhere


@pytest.mark.parametrize("tier", ["gpu-12gb", "gpu-24gb", "gpu-32gb", "hybrid-12gb-gpu", "hybrid-32gb-gpu"])
def test_big_gpu_tiers_use_pyannote_with_hf_token(monkeypatch, tier):
    fake_hw(monkeypatch, 64, 24)
    monkeypatch.setenv("SPEECHOS_TIER", tier)
    monkeypatch.setattr(config_mod.importlib.util, "find_spec", lambda name: object())
    assert load_config().diarization.engine == "resemblyzer"  # no HF_TOKEN yet

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    cfg = load_config()
    assert cfg.diarization.engine == "pyannote"
    assert cfg.diarization.model == "pyannote/speaker-diarization-community-1"


def test_pyannote_needs_package_too(monkeypatch):
    fake_hw(monkeypatch, 64, 24)
    monkeypatch.setenv("SPEECHOS_TIER", "gpu-24gb")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(config_mod.importlib.util, "find_spec", lambda name: None)
    assert load_config().diarization.engine == "resemblyzer"


@pytest.mark.parametrize("tier", ["cpu-8gb", "gpu-8gb", "hybrid-8gb-gpu"])
def test_small_tiers_have_no_diarization(monkeypatch, tier):
    fake_hw(monkeypatch, 8, None)
    monkeypatch.setenv("SPEECHOS_TIER", tier)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    assert load_config().diarization is None


def test_model_env_overrides(monkeypatch):
    fake_hw(monkeypatch, 16, None)
    monkeypatch.setenv("SPEECHOS_STT_MODEL", "large-v3")
    monkeypatch.setenv("SPEECHOS_STT_COMPUTE_TYPE", "int8_float16")
    monkeypatch.setenv("SPEECHOS_TTS_ENGINE", "kokoro")
    monkeypatch.setenv("SPEECHOS_TTS_DEVICE", "cuda")
    monkeypatch.setenv("SPEECHOS_EMOTION_ENGINE", "emotion2vec")
    monkeypatch.setenv("SPEECHOS_EMOTION_MODEL", "iic/emotion2vec_plus_large")
    monkeypatch.setenv("SPEECHOS_VAD_ENGINE", "none")
    cfg = load_config()
    assert (cfg.stt.engine, cfg.stt.model, cfg.stt.compute_type) == ("faster-whisper", "large-v3", "int8_float16")
    # New engine → old engine's model name is not inherited
    assert (cfg.tts.engine, cfg.tts.model, cfg.tts.device) == ("kokoro", "default", "cuda")
    assert (cfg.emotion.engine, cfg.emotion.model) == ("emotion2vec", "iic/emotion2vec_plus_large")
    assert cfg.vad is None


def test_server_env_overrides(monkeypatch, tmp_path):
    fake_hw(monkeypatch, 16, None)
    monkeypatch.setenv("SPEECHOS_PORT", "36500")
    monkeypatch.setenv("SPEECHOS_MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("SPEECHOS_RECORDINGS_DIR", str(tmp_path / "r"))
    monkeypatch.setenv("SPEECHOS_CORS_ORIGINS", "http://a.test, http://b.test")
    cfg = load_config()
    assert cfg.port == 36500
    assert cfg.cors_origins == ["http://a.test", "http://b.test"]
    assert (tmp_path / "m").is_dir() and (tmp_path / "r").is_dir()


def test_yaml_overrides(monkeypatch, tmp_path):
    fake_hw(monkeypatch, 64, 24)
    path = tmp_path / "config.yaml"
    path.write_text(
        """
compute_mode: cpu
tier: cpu-4gb
stt:
  engine: faster-whisper
  model: small
emotion:
  engine: none
server:
  port: 40000
  cors_origins: ["http://ui.test"]
storage:
  max_upload_mb: 5
""",
        encoding="utf-8",
    )
    cfg = load_config(str(path))
    assert cfg.hardware.mode == ComputeMode.CPU
    assert cfg.hardware.tier == "cpu-4gb"
    assert cfg.stt.model == "small"
    assert cfg.emotion is None
    assert cfg.port == 40000
    assert cfg.cors_origins == ["http://ui.test"]
    assert cfg.max_upload_mb == 5


def test_env_beats_yaml(monkeypatch, tmp_path):
    fake_hw(monkeypatch, 16, None)
    path = tmp_path / "config.yaml"
    path.write_text("stt:\n  engine: faster-whisper\n  model: small\n", encoding="utf-8")
    monkeypatch.setenv("SPEECHOS_STT_MODEL", "medium")
    assert load_config(str(path)).stt.model == "medium"


def test_cors_origins_reach_middleware(app_config):
    from fastapi.testclient import TestClient

    import src.server as server
    from src.models import ModelManager

    app_config.cors_origins = ["http://allowed.test"]
    app = server.create_app(app_config, ModelManager(app_config))
    with TestClient(app) as c:
        ok = c.get("/health", headers={"Origin": "http://allowed.test"})
        bad = c.get("/health", headers={"Origin": "http://evil.test"})
    assert ok.headers.get("access-control-allow-origin") == "http://allowed.test"
    assert "access-control-allow-origin" not in bad.headers

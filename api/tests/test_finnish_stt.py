"""Finnish requests use the Finnish Whisper (stt_fi) when it is on disk; other languages keep large-v3."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src import config as config_mod
from src.config import FINNISH_WHISPER_DIR, ModelConfig, load_config
from src.models import ModelManager

from .conftest import FakeWhisper


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for key in list(os.environ):
        if key.startswith("SPEECHOS_") or key == "HF_TOKEN":
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    # Never pick up a real converted model from the developer's checkout
    monkeypatch.setattr(config_mod, "_REPO_ROOT", tmp_path / "repo")
    model_dir = tmp_path / "models"
    monkeypatch.setenv("SPEECHOS_MODEL_DIR", str(model_dir))
    return model_dir


def install_finnish_whisper(model_dir: Path) -> Path:
    path = model_dir / "whisper-ct2" / FINNISH_WHISPER_DIR
    path.mkdir(parents=True)
    (path / "model.bin").write_bytes(b"\0")
    return path


def test_enabled_on_large_v3_gpu_tier_when_model_exists(clean_env, monkeypatch):
    path = install_finnish_whisper(clean_env)
    monkeypatch.setenv("SPEECHOS_TIER", "gpu-24gb")

    cfg = load_config()

    assert cfg.stt_fi == ModelConfig(engine="faster-whisper", model=str(path), device="cuda", compute_type="float16")
    assert cfg.stt is not None and cfg.stt.model == "large-v3"  # the default stays multilingual


def test_follows_tier_compute_type(clean_env, monkeypatch):
    install_finnish_whisper(clean_env)
    monkeypatch.setenv("SPEECHOS_TIER", "gpu-12gb")

    cfg = load_config()

    assert cfg.stt_fi is not None and cfg.stt_fi.compute_type == "int8_float16"


def test_not_enabled_without_the_model(clean_env, monkeypatch):
    monkeypatch.setenv("SPEECHOS_TIER", "gpu-24gb")
    assert load_config().stt_fi is None


@pytest.mark.parametrize("tier", ["cpu-32gb", "gpu-8gb"])
def test_not_enabled_on_tiers_without_large_v3_on_gpu(clean_env, monkeypatch, tier):
    install_finnish_whisper(clean_env)
    monkeypatch.setenv("SPEECHOS_TIER", tier)
    assert load_config().stt_fi is None


def test_env_can_disable_it(clean_env, monkeypatch):
    install_finnish_whisper(clean_env)
    monkeypatch.setenv("SPEECHOS_TIER", "gpu-24gb")
    monkeypatch.setenv("SPEECHOS_STT_FI_ENGINE", "none")
    assert load_config().stt_fi is None


@pytest.mark.parametrize(("language", "slot"), [("fi", "stt_fi"), ("fi-FI", "stt_fi"), ("en", "stt"), (None, "stt")])
def test_resolve_routes_finnish_only(manager: ModelManager, language, slot):
    manager.config.stt_fi = ModelConfig(engine="faster-whisper", model="/models/fi-ct2", device="cpu")
    key, cfg = manager.resolve("stt", language)
    assert key == slot
    assert cfg.model == ("/models/fi-ct2" if slot == "stt_fi" else "tiny")


def test_transcribe_uses_finnish_model_for_fi(client, manager: ModelManager, monkeypatch, wav_bytes):
    multilingual, finnish = FakeWhisper("hello world"), FakeWhisper("hei maailma")
    manager.config.stt_fi = ModelConfig(engine="faster-whisper", model="/models/fi-ct2", device="cpu")
    monkeypatch.setattr(manager, "_load_stt", lambda cfg: finnish if cfg.model == "/models/fi-ct2" else multilingual)

    def post(language: str) -> str:
        files = {"file": ("a.wav", wav_bytes, "audio/wav")}
        r = client.post("/api/transcribe", files=files, data={"language": language})
        assert r.status_code == 200, r.text
        return r.json()["text"]

    assert post("fi") == "hei maailma"
    assert post("en") == "hello world"
    assert post("auto") == "hello world"

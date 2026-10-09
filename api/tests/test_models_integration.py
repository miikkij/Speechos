"""Real-model smoke tests (opt-in): ``uv run pytest -m models``.

Run on CPU, download weights on first use into ``SPEECHOS_MODEL_DIR`` (default
./models) and are skipped by default. Salvaged from the old hand-run scripts
(test_integration.py, test_vad_stt.py, test_all_models.py).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.audio import prepare_for_model
from src.config import ModelConfig
from src.models import ModelManager
from src.routers.synthesize import SynthesisInput, synthesize_wav
from src.routers.transcribe import transcribe_array

from .conftest import REPO_ROOT

pytestmark = pytest.mark.models

FI_SAMPLE = REPO_ROOT / "samples" / "fi_harri_sample.wav"
FI_TEXT = "Hyvää huomenta. Tämä on suomenkielinen testilause puheentunnistukselle."
STT_MODEL = os.environ.get("SPEECHOS_TEST_STT_MODEL", "small")


@pytest.fixture
def real_manager(app_config) -> ModelManager:
    app_config.model_dir = os.environ.get("SPEECHOS_MODEL_DIR", "./models")
    app_config.stt = ModelConfig(engine="faster-whisper", model=STT_MODEL, device="cpu", compute_type="int8")
    app_config.vad = ModelConfig(engine="silero", model="v5")
    return ModelManager(app_config)


def _synth_fi(manager: ModelManager, text: str = FI_TEXT) -> np.ndarray:
    key, cfg = manager.resolve("tts", "fi")
    inp = SynthesisInput(text=text, voice=None, speed=1.0, language="fi")
    wav = manager.run_sync(key, cfg, lambda tts, c: synthesize_wav(tts, c, inp))
    return prepare_for_model(wav)


def test_piper_finnish_voice(real_manager):
    audio = _synth_fi(real_manager)
    assert len(audio) / 16_000 > 2.0
    assert np.abs(audio).max() > 0.05


def test_whisper_transcribes_finnish(real_manager):
    audio = prepare_for_model(FI_SAMPLE) if FI_SAMPLE.exists() else _synth_fi(real_manager)
    key, cfg = real_manager.resolve("stt")
    result = real_manager.run_sync(key, cfg, lambda stt, c: transcribe_array(stt, c, audio, "fi"))
    text = result["text"].lower()
    assert result["language"] == "fi"
    assert "huomenta" in text, text


def test_whisper_detects_finnish(real_manager):
    audio = _synth_fi(real_manager)
    key, cfg = real_manager.resolve("stt")
    result = real_manager.run_sync(key, cfg, lambda stt, c: transcribe_array(stt, c, audio, None))
    assert result["language"] == "fi"


def test_silero_vad_finds_speech(real_manager):
    from src.analysis import detect_speech

    speech = np.concatenate([np.zeros(16_000, dtype=np.float32), _synth_fi(real_manager, "Hyvää huomenta.")])
    key, cfg = real_manager.resolve("vad")
    result = real_manager.run_sync(key, cfg, lambda vad, c: detect_speech(speech, 16_000, vad, c.engine))
    assert result["segments"], result
    assert result["segments"][0]["start"] >= 0.8  # first second is silence
    assert 0.1 < result["speech_ratio"] < 0.9


def test_piper_speed(real_manager):
    key, cfg = real_manager.resolve("tts", "fi")
    lengths = {}
    for speed in (0.75, 1.5):
        inp = SynthesisInput(text=FI_TEXT, voice=None, speed=speed, language="fi")
        lengths[speed] = len(real_manager.run_sync(key, cfg, lambda tts, c, inp=inp: synthesize_wav(tts, c, inp)))
    assert lengths[0.75] > lengths[1.5] * 1.5


def test_english_voice_variant(real_manager):
    key, cfg = real_manager.resolve("tts", "en")
    assert cfg.model == "en_US-lessac-medium"
    inp = SynthesisInput(text="Good morning.", voice=None, speed=1.0, language="en")
    assert len(real_manager.run_sync(key, cfg, lambda tts, c: synthesize_wav(tts, c, inp))) > 10_000

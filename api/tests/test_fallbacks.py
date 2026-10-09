"""Graceful degradation: gated pyannote → Resemblyzer, and loudness stats that ignore digital silence."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import extract_audio_features
from src.config import ModelConfig
from src.models import ModelManager

SR = 16_000


def _fake_diarization_loader(cfg: ModelConfig) -> str:
    if cfg.engine == "pyannote":
        raise RuntimeError("403 Client Error: Cannot access gated repo pyannote/speaker-diarization-community-1")
    return f"{cfg.engine}-encoder"


def test_gated_pyannote_falls_back_to_resemblyzer(manager: ModelManager, monkeypatch: pytest.MonkeyPatch):
    pyannote = ModelConfig(engine="pyannote", model="pyannote/speaker-diarization-community-1", device="cpu")
    manager.config.diarization = pyannote
    monkeypatch.setattr(manager, "_load_diarization", _fake_diarization_loader)

    model, engine = manager.run_sync("diarization", pyannote, lambda m, c: (m, c.engine))

    assert (model, engine) == ("resemblyzer-encoder", "resemblyzer")
    # The switch sticks, so /api/system/model-options reports what is really used
    assert manager.config.diarization is not None
    assert manager.config.diarization.engine == "resemblyzer"


def test_other_load_failures_still_raise(manager: ModelManager, monkeypatch: pytest.MonkeyPatch):
    cfg = ModelConfig(engine="emotion2vec", model="iic/emotion2vec_base_finetuned", device="cpu")

    def broken(_cfg: ModelConfig) -> None:
        raise RuntimeError("download failed")

    monkeypatch.setattr(manager, "_load_emotion", broken)
    with pytest.raises(RuntimeError, match="download failed"):
        manager.run_sync("emotion", cfg, lambda m, c: m)


class _PitchEncoder:
    """Stand-in for resemblyzer's VoiceEncoder: the 'speaker' is the dominant frequency band."""

    def embed_utterance(self, wav: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(wav))
        freqs = np.fft.rfftfreq(len(wav), 1 / SR)
        low = spectrum[(freqs > 100) & (freqs < 250)].sum()
        high = spectrum[(freqs > 350) & (freqs < 500)].sum()
        vec = np.array([low, high, 1e-3], dtype=np.float32)
        return vec / np.linalg.norm(vec)


def test_resemblyzer_diarization_keeps_original_timeline():
    pytest.importorskip("resemblyzer")
    pytest.importorskip("sklearn")
    from src.analysis import run_diarization

    def tone(hz: float, seconds: float) -> np.ndarray:
        t = np.arange(int(seconds * SR)) / SR
        return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)

    def silence(seconds: float) -> np.ndarray:
        return np.zeros(int(seconds * SR), np.float32)

    # speaker A 2–6 s, speaker B 9–13 s; the silences used to be trimmed away, shifting B to ~6 s
    audio = np.concatenate([silence(2), tone(180, 4), silence(3), tone(420, 4), silence(2)])

    segments, summary = run_diarization(audio, SR, _PitchEncoder(), engine="resemblyzer")

    assert summary["num_speakers"] == 2
    second = [s for s in segments if s["speaker"] != segments[0]["speaker"]]
    assert second, segments
    assert second[0]["start"] >= 7.5
    assert segments[-1]["end"] <= 15.0


def test_dynamic_range_ignores_digital_silence():
    t = np.arange(SR) / SR
    tone = (0.3 * np.sin(2 * np.pi * 220 * t) * np.linspace(0.2, 1.0, SR)).astype(np.float32)
    padded = np.concatenate([np.zeros(SR, np.float32), tone, np.zeros(SR, np.float32)])

    energy = extract_audio_features(padded, SR)["energy"]

    # A 5× amplitude ramp is ~14 dB; exact zeros used to push this past 150 dB
    assert 5 < energy["dynamic_range_db"] < 30

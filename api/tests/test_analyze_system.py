"""/api/analyze and /api/system/* with fake models."""

from __future__ import annotations

import numpy as np

from src.config import ComputeMode, ModelConfig


def test_analyze_features_only(client, wav_bytes):
    r = client.post("/api/analyze", files={"file": ("a.wav", wav_bytes, "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["features"]["duration"] == 1.0
    assert 200 < body["features"]["pitch"]["mean_hz"] < 240  # 220 Hz sine
    assert body["emotions"] == []
    assert body["emotion_available"] is False
    assert body["diarization_available"] is False
    assert body["speech"] is None


def test_analyze_with_models(client, app_config, manager, monkeypatch, wav_bytes):
    import src.routers.analyze as analyze_mod

    app_config.emotion = ModelConfig(engine="emotion2vec", model="fake")
    app_config.diarization = ModelConfig(engine="resemblyzer", model="dvector")
    app_config.vad = ModelConfig(engine="silero", model="v5")
    monkeypatch.setattr(manager, "_load_emotion", lambda cfg: "emo-model")
    monkeypatch.setattr(manager, "_load_diarization", lambda cfg: "dia-model")
    monkeypatch.setattr(manager, "_load_vad", lambda cfg: "vad-model")

    def fake_emotion(audio, sr, model):
        assert model == "emo-model" and sr == 16_000
        return [{"label": "happy", "score": 0.8}, {"label": "neutral", "score": 0.2}]

    def fake_diar(audio, sr, model, engine=None):
        assert engine == "resemblyzer"
        return [{"speaker": "SPEAKER_0", "start": 0.0, "end": 1.0, "duration": 1.0}], {
            "num_speakers": 1, "speakers": {"SPEAKER_0": 1.0},
        }

    def fake_vad(audio, sr, model, engine="silero"):
        return {"segments": [{"start": 0.1, "end": 0.9}], "speech_seconds": 0.8, "speech_ratio": 0.8}

    monkeypatch.setattr(analyze_mod, "predict_emotion", fake_emotion)
    monkeypatch.setattr(analyze_mod, "run_diarization", fake_diar)
    monkeypatch.setattr(analyze_mod, "detect_speech", fake_vad)

    body = client.post("/api/analyze", files={"file": ("a.wav", wav_bytes, "audio/wav")}).json()
    assert body["primary_emotion"] == "happy"
    assert body["diarization"]["summary"]["num_speakers"] == 1
    assert body["speech"]["speech_ratio"] == 0.8
    assert body["emotion_available"] and body["diarization_available"] and body["vad_available"]
    assert set(manager.loaded_models) == {"emotion", "diarization", "vad"}


def test_analyze_model_failure_is_reported_not_fatal(client, app_config, manager, monkeypatch, wav_bytes):
    app_config.emotion = ModelConfig(engine="emotion2vec", model="fake")

    def boom(cfg):
        raise ImportError("funasr missing")

    monkeypatch.setattr(manager, "_load_emotion", boom)
    body = client.post("/api/analyze", files={"file": ("a.wav", wav_bytes, "audio/wav")}).json()
    assert body["emotion_available"] is False
    assert "funasr missing" in body["emotion_error"]


def test_merge_diarization_segments():
    from src.analysis import _merge_diarization_segments

    segs, summary = _merge_diarization_segments(
        [(0.0, 1.5), (0.75, 2.25), (1.5, 3.0)], np.array([0, 0, 1]), 3.0
    )
    assert [s["speaker"] for s in segs] == ["SPEAKER_0", "SPEAKER_1"]
    assert segs[0]["end"] == 2.25
    assert summary["num_speakers"] == 2


def test_system_info(client):
    body = client.get("/api/system/info").json()
    assert body["hardware"]["tier"] == "cpu-8gb"
    assert body["models"]["configured"]["tts"]["model"] == "fi_FI-harri-medium"
    assert body["models"]["loaded"] == []


def test_model_options_mark_current(client):
    body = client.get("/api/system/model-options").json()
    assert body["current"]["tts"] == {"engine": "piper", "model": "fi_FI-harri-medium"}
    assert any(o["model"] == "fi_FI-harri-medium" for o in body["options"]["tts"])
    assert body["current"]["emotion"] == {"engine": "none", "model": "none"}


def test_switch_model_unloads_variants(client, manager, piper_voices, app_config):
    assert client.post("/api/synthesize", json={"text": "hi", "language": "en"}).status_code == 200
    assert client.post("/api/synthesize", json={"text": "moi"}).status_code == 200
    assert set(manager.loaded_models) == {"tts", "tts:piper:en_US-lessac-medium"}

    r = client.post("/api/system/switch-model", json={"category": "tts", "engine": "piper", "model": "fi_FI-harri-low"})
    assert r.status_code == 200
    assert manager.loaded_models == []
    assert app_config.tts.model == "fi_FI-harri-low"


def test_switch_model_respects_forced_cpu_mode(client, app_config):
    app_config.hardware.gpu_available = True
    app_config.hardware.mode = ComputeMode.CPU
    r = client.post("/api/system/switch-model", json={"category": "stt", "engine": "faster-whisper", "model": "small"})
    assert r.json()["device"] == "cpu"


def test_switch_model_disable_and_bad_category(client, app_config):
    r = client.post("/api/system/switch-model", json={"category": "stt", "engine": "none", "model": "none"})
    assert r.json()["status"] == "disabled"
    assert app_config.stt is None
    r = client.post("/api/system/switch-model", json={"category": "nope", "engine": "x", "model": "y"})
    assert r.status_code == 400

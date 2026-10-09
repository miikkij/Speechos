"""/api/synthesize: language routing, Piper speed, unsupported-language errors."""

from __future__ import annotations

import io
from dataclasses import replace

import pytest
import soundfile as sf

from src.config import ModelConfig


def synth(client, **body):
    return client.post("/api/synthesize", json={"text": "Hyvää huomenta", **body})


def test_finnish_default_voice(client, piper_voices):
    r = synth(client, language="fi")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers["x-tts-model"] == "fi_FI-harri-medium"
    info = sf.info(io.BytesIO(r.content))
    assert (info.channels, info.samplerate, info.subtype) == (1, 22050, "PCM_16")
    assert list(piper_voices) == ["fi_FI-harri-medium"]


def test_english_request_on_finnish_voice_switches_voice(client, piper_voices, manager):
    r = synth(client, text="Good morning", language="en")
    assert r.status_code == 200, r.text
    assert r.headers["x-tts-model"] == "en_US-lessac-medium"
    assert "tts:piper:en_US-lessac-medium" in manager.loaded_models


def test_finnish_request_on_english_voice_uses_harri(client, piper_voices, app_config):
    app_config.tts = ModelConfig(engine="piper", model="en_US-lessac-medium")
    r = synth(client, language="fi")
    assert r.status_code == 200, r.text
    assert r.headers["x-tts-model"] == "fi_FI-harri-medium"
    assert piper_voices["fi_FI-harri-medium"].calls[0]["text"] == "Hyvää huomenta"


def test_no_language_uses_configured_voice(client, piper_voices, app_config):
    app_config.tts = ModelConfig(engine="piper", model="en_US-lessac-medium")
    assert synth(client).headers["x-tts-model"] == "en_US-lessac-medium"


@pytest.mark.parametrize(("speed", "length_scale"), [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0)])
def test_piper_speed_maps_to_length_scale(client, piper_voices, speed, length_scale):
    assert synth(client, speed=speed).status_code == 200
    cfg = piper_voices["fi_FI-harri-medium"].calls[0]["syn_config"]
    assert cfg.length_scale == pytest.approx(length_scale)


@pytest.mark.parametrize("speed", [0, 0.1, 10])
def test_speed_out_of_range(client, piper_voices, speed):
    assert synth(client, speed=speed).status_code == 422


def test_unsupported_piper_language(client, piper_voices):
    r = synth(client, language="xh")
    assert r.status_code == 400
    assert "No Piper voice" in r.json()["detail"]


def test_kokoro_finnish_is_a_clear_error(client, app_config, manager, monkeypatch):
    app_config.tts = ModelConfig(engine="kokoro", model="default")
    monkeypatch.setattr(manager, "_load_tts", lambda cfg: pytest.fail("must not load"))
    r = synth(client, language="fi")
    assert r.status_code == 400
    assert "Kokoro does not support language 'fi'" in r.json()["detail"]
    assert "Piper" in r.json()["detail"]


def test_kokoro_language_variants(manager, app_config):
    app_config.tts = ModelConfig(engine="kokoro", model="default")
    assert manager.resolve("tts", None) == ("tts", replace(app_config.tts, language="a"))
    key, cfg = manager.resolve("tts", "fr")
    assert key == "tts:kokoro:f" and cfg.language == "f"


def test_kitten_rejects_finnish(client, app_config):
    app_config.tts = ModelConfig(engine="kitten", model="KittenML/kitten-tts-nano-0.8-fp32")
    r = synth(client, language="fi")
    assert r.status_code == 400
    assert "only supports English" in r.json()["detail"]


def test_chatterbox_multilingual_gets_language_id(client, app_config, manager, monkeypatch):
    calls = []

    class FakeTensor:
        def squeeze(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            import numpy as np

            return np.zeros(2400, dtype=np.float32)

    class FakeChatterbox:
        def generate(self, text, **kwargs):
            calls.append(kwargs)
            return FakeTensor()

    app_config.tts = ModelConfig(engine="chatterbox", model="multilingual")
    monkeypatch.setattr(
        manager, "_load_tts", lambda cfg: {"model": FakeChatterbox(), "multilingual": True, "sr": 24000}
    )
    assert synth(client, language="fi").status_code == 200
    assert calls == [{"language_id": "fi"}]


def test_slots(client, piper_voices, app_config):
    r = synth(client, slot="fast")
    assert r.status_code == 400  # tts_fast not configured
    app_config.tts_fast = ModelConfig(engine="piper", model="fi_FI-harri-low")
    r = synth(client, slot="fast")
    assert r.status_code == 200
    assert r.headers["x-tts-model"] == "fi_FI-harri-low"


def test_empty_text(client, piper_voices):
    assert synth(client, text="   ").status_code == 400


def test_docker_engine_is_proxied(client, app_config, monkeypatch):
    import src.routers.synthesize as synth_mod

    seen = {}

    async def fake_docker(engine, text, voice, language):
        seen.update(engine=engine, text=text, language=language)
        buf = io.BytesIO()
        sf.write(buf, [0.0] * 100, 24000, format="WAV")
        return buf.getvalue()

    monkeypatch.setattr(synth_mod, "docker_synthesize", fake_docker)
    app_config.tts = ModelConfig(engine="xtts", model="xtts_v2")
    r = synth(client, language="fi")
    assert r.status_code == 200
    assert seen == {"engine": "xtts", "text": "Hyvää huomenta", "language": "fi"}

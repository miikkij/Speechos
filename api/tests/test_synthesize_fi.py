"""Finnish TTS: text normalization in front of Piper and the Chatterbox-Finnish engine (no torch)."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from src.config import ModelConfig


def synth(client, text: str, **body):
    return client.post("/api/synthesize", json={"text": text, **body})


def test_piper_reads_numbers_as_finnish_words(client, piper_voices):
    assert synth(client, "Kokous alkaa klo 9.30 ja kestää 2 tuntia.", language="fi").status_code == 200
    spoken = piper_voices["fi_FI-harri-medium"].calls[0]["text"]
    assert "9.30" not in spoken and "2 tuntia" not in spoken
    assert "kello" in spoken and "kaksi tuntia" in spoken


def test_piper_english_text_is_not_normalized(client, piper_voices):
    assert synth(client, "Meet me at 9.30 for 2 hours.", language="en").status_code == 200
    assert piper_voices["en_US-lessac-medium"].calls[0]["text"] == "Meet me at 9.30 for 2 hours."


class FakeChatterboxFinnish:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.ref_wav: str | None = "sample-ref.wav"
        self.texts: list[str] = []
        self.voices: list[str] = []

    def set_voice(self, ref_wav: str) -> None:
        self.voices.append(ref_wav)
        self.ref_wav = ref_wav

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.texts.append(text)
        return np.zeros(2400, dtype=np.float32), self.sample_rate


def use_chatterbox_fi(app_config, manager, monkeypatch) -> FakeChatterboxFinnish:
    fake = FakeChatterboxFinnish()
    app_config.tts = ModelConfig(engine="chatterbox-fi", model="sample", device="cuda")
    monkeypatch.setattr(manager, "_load_tts", lambda cfg: fake)
    return fake


def test_chatterbox_finnish_speaks_finnish(client, app_config, manager, monkeypatch):
    fake = use_chatterbox_fi(app_config, manager, monkeypatch)

    r = synth(client, "Hyvää huomenta!", language="fi")

    assert r.status_code == 200, r.text
    info = sf.info(io.BytesIO(r.content))
    assert (info.channels, info.samplerate) == (1, 24_000)
    assert fake.texts == ["Hyvää huomenta!"]


def test_chatterbox_finnish_rejects_other_languages(client, app_config, manager, monkeypatch):
    use_chatterbox_fi(app_config, manager, monkeypatch)
    r = synth(client, "Good morning!", language="en")
    assert r.status_code == 400
    assert "only speaks Finnish" in r.text


def test_chatterbox_finnish_voice_cloning_switches_reference(client, app_config, manager, monkeypatch):
    fake = use_chatterbox_fi(app_config, manager, monkeypatch)
    assert synth(client, "Hei", language="fi", voice="my-voice.wav").status_code == 200
    assert synth(client, "Hei taas", language="fi", voice="my-voice.wav").status_code == 200
    assert fake.voices == ["my-voice.wav"]  # set once, reused after


def test_chatterbox_finnish_is_in_the_menu(client):
    options = client.get("/api/system/model-options").json()["options"]["tts"]
    assert ("chatterbox-fi", "sample") in [(o["engine"], o["model"]) for o in options]

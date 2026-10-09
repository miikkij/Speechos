"""/api/transcribe with a fake Whisper model; event loop responsiveness."""

from __future__ import annotations

import asyncio
import time

import pytest

from .conftest import FakeWhisper


def post_audio(client, data: bytes, **form):
    return client.post("/api/transcribe", files={"file": ("rec.wav", data, "audio/wav")}, data=form)


def test_transcribe_passes_language(client, fake_whisper, wav_bytes):
    r = post_audio(client, wav_bytes, language="fi")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "hei maailma"
    assert body["language"] == "fi"
    assert body["segments"] == [{"start": 0.0, "end": 1.0, "text": "hei maailma"}]
    assert body["duration"] == pytest.approx(1.0)
    assert fake_whisper.calls[0]["language"] == "fi"
    assert fake_whisper.calls[0]["samples"] == 16_000


@pytest.mark.parametrize("value", [None, "auto", ""])
def test_transcribe_auto_language_means_detect(client, fake_whisper, wav_bytes, value):
    form = {} if value is None else {"language": value}
    assert post_audio(client, wav_bytes, **form).status_code == 200
    assert fake_whisper.calls[0]["language"] is None


def test_transcribe_normalizes_language(client, fake_whisper, wav_bytes):
    assert post_audio(client, wav_bytes, language=" FI ").status_code == 200
    assert fake_whisper.calls[0]["language"] == "fi"


def test_transcribe_rejects_bad_language(client, fake_whisper, wav_bytes):
    r = post_audio(client, wav_bytes, language="fi; rm -rf")
    assert r.status_code == 400
    assert fake_whisper.calls == []


def test_transcribe_webm_upload(client, fake_whisper, webm_bytes):
    r = client.post("/api/transcribe", files={"file": ("rec.webm", webm_bytes, "audio/webm")})
    assert r.status_code == 200, r.text
    assert abs(fake_whisper.calls[0]["samples"] - 16_000) < 400  # resampled 48k → 16k


def test_transcribe_rejects_non_audio(client, fake_whisper):
    r = post_audio(client, b"definitely not audio")
    assert r.status_code == 400


def test_transcribe_upload_limit(client, fake_whisper, app_config, make_wav):
    app_config.max_upload_mb = 0
    r = post_audio(client, make_wav())
    assert r.status_code == 413


def test_transcribe_engine_unavailable_is_503(client, manager, monkeypatch, wav_bytes):
    def boom(cfg):
        raise ImportError("faster_whisper missing")

    monkeypatch.setattr(manager, "_load_stt", boom)
    r = post_audio(client, wav_bytes)
    assert r.status_code == 503
    assert "faster_whisper missing" in r.json()["detail"]


def test_transcribe_not_configured_is_503(client, app_config, wav_bytes):
    app_config.stt = None
    assert post_audio(client, wav_bytes).status_code == 503


async def test_health_answers_during_long_inference(aclient, manager, monkeypatch, wav_bytes):
    """Inference runs in a worker thread: /health must not wait for it."""
    slow = FakeWhisper(delay=1.5)
    monkeypatch.setattr(manager, "_load_stt", lambda cfg: slow)

    t0 = time.perf_counter()
    transcription = asyncio.create_task(
        aclient.post("/api/transcribe", files={"file": ("a.wav", wav_bytes, "audio/wav")})
    )
    # Wait until the request is inside the (blocking) model call
    while not slow.calls:
        await asyncio.sleep(0.01)
    health = await aclient.get("/health")
    health_latency = time.perf_counter() - t0

    assert health.status_code == 200
    assert not transcription.done()
    assert health_latency < 1.0
    assert (await transcription).status_code == 200


async def test_same_model_never_runs_concurrently(aclient, manager, monkeypatch, wav_bytes):
    active = 0
    peak = 0

    class Exclusive(FakeWhisper):
        def transcribe(self, audio, language=None, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                time.sleep(0.2)
                return super().transcribe(audio, language, **kwargs)
            finally:
                active -= 1

    model = Exclusive()
    monkeypatch.setattr(manager, "_load_stt", lambda cfg: model)

    responses = await asyncio.gather(*(
        aclient.post("/api/transcribe", files={"file": ("a.wav", wav_bytes, "audio/wav")}) for _ in range(4)
    ))
    assert [r.status_code for r in responses] == [200] * 4
    assert len(model.calls) == 4
    assert peak == 1

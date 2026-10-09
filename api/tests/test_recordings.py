"""/api/recordings: every stored file is 16 kHz mono PCM WAV, whatever was uploaded."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import soundfile as sf


def upload(client, data: bytes, name: str, mime: str):
    return client.post("/api/recordings", files={"file": (name, data, mime)})


def assert_16k_mono_pcm(path: Path) -> None:
    raw = path.read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    info = sf.info(str(path))
    assert (info.samplerate, info.channels, info.subtype) == (16_000, 1, "PCM_16")


def test_webm_upload_is_stored_as_real_wav(client, app_config, webm_bytes):
    r = upload(client, webm_bytes, "recording.wav", "audio/webm")  # browser lies about the name
    assert r.status_code == 200, r.text
    body = r.json()
    path = Path(app_config.recordings_dir) / body["filename"]
    assert_16k_mono_pcm(path)
    assert body["duration"] == pytest.approx(1.0, abs=0.1)
    assert body["size_bytes"] == path.stat().st_size


def test_44k_stereo_wav_is_converted(client, app_config, make_wav):
    r = upload(client, make_wav(duration=0.5, sr=44_100, channels=2), "a.wav", "audio/wav")
    assert r.status_code == 200
    path = Path(app_config.recordings_dir) / r.json()["filename"]
    assert_16k_mono_pcm(path)
    assert sf.info(str(path)).duration == pytest.approx(0.5, abs=0.01)


def test_list_get_delete(client, make_wav):
    rec_id = upload(client, make_wav(), "a.wav", "audio/wav").json()["id"]

    listing = client.get("/api/recordings").json()["recordings"]
    assert [r["id"] for r in listing] == [rec_id]
    assert listing[0]["duration"] == pytest.approx(1.0)

    audio = client.get(f"/api/recordings/{rec_id}/audio")
    assert audio.status_code == 200
    assert sf.info(io.BytesIO(audio.content)).samplerate == 16_000

    assert client.delete(f"/api/recordings/{rec_id}").status_code == 200
    assert client.get("/api/recordings").json()["recordings"] == []
    assert client.delete(f"/api/recordings/{rec_id}").status_code == 404


@pytest.mark.parametrize("bad_id", ["..%2Fsecret", "a.b", "x y"])
def test_invalid_ids(client, bad_id):
    assert client.get(f"/api/recordings/{bad_id}/audio").status_code in (400, 404)
    assert client.delete(f"/api/recordings/{bad_id}").status_code in (400, 404)


def test_rejects_garbage(client):
    assert upload(client, b"garbage" * 10, "a.wav", "audio/wav").status_code == 400


def test_rejects_undecodable_audio(client):
    # Valid RIFF/WAVE magic but no usable content
    assert upload(client, b"RIFF\x00\x00\x00\x00WAVEjunkjunk", "a.wav", "audio/wav").status_code == 400

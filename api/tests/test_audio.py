"""Audio decoding / conversion utilities."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from src.audio import (
    audio_duration,
    float32_to_wav_bytes,
    prepare_for_model,
    read_audio,
    to_wav_16k_mono,
    validate_audio_file,
)

from .conftest import encode_wav, sine


def test_read_wav_stereo_to_mono(make_wav):
    audio, sr = read_audio(make_wav(duration=0.5, sr=44_100, channels=2))
    assert sr == 44_100
    assert audio.ndim == 1
    assert len(audio) == 22_050


def test_prepare_resamples_to_16k(make_wav):
    audio = prepare_for_model(make_wav(duration=1.0, sr=48_000))
    assert audio.dtype == np.float32
    assert abs(len(audio) - 16_000) <= 1


def test_wav_roundtrip_is_pcm16():
    data = float32_to_wav_bytes(sine(0.2))
    info = sf.info(io.BytesIO(data))
    assert (info.samplerate, info.channels, info.subtype) == (16_000, 1, "PCM_16")


def test_float32_out_of_range_is_clipped():
    data = float32_to_wav_bytes(np.array([2.0, -2.0, 0.0], dtype=np.float32))
    audio, _ = sf.read(io.BytesIO(data), dtype="float32")
    assert audio.max() <= 1.0 and audio.min() >= -1.0


def test_webm_opus_decodes(webm_bytes):
    assert webm_bytes[:4] == b"\x1a\x45\xdf\xa3"
    assert validate_audio_file(webm_bytes)
    audio, sr = read_audio(webm_bytes)
    assert sr == 48_000
    assert audio.ndim == 1
    assert 0.9 < len(audio) / sr < 1.1
    assert 0.1 < np.abs(audio).max() <= 1.0


def test_to_wav_16k_mono_from_webm(webm_bytes):
    wav, duration = to_wav_16k_mono(webm_bytes)
    info = sf.info(io.BytesIO(wav))
    assert (info.samplerate, info.channels, info.subtype, info.format) == (16_000, 1, "PCM_16", "WAV")
    assert duration == pytest.approx(1.0, abs=0.1)


def test_to_wav_16k_mono_from_float_stereo_wav():
    stereo = np.stack([sine(0.5, 44_100)] * 2, axis=1)
    wav, duration = to_wav_16k_mono(encode_wav(stereo, 44_100, subtype="FLOAT"))
    info = sf.info(io.BytesIO(wav))
    assert (info.samplerate, info.channels, info.subtype) == (16_000, 1, "PCM_16")
    assert duration == pytest.approx(0.5, abs=0.01)


def test_audio_duration(make_wav, tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(make_wav(duration=0.75))
    assert audio_duration(path) == pytest.approx(0.75)
    assert audio_duration(path.read_bytes()) == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("data", "ok"),
    [
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", True),
        (b"fLaC" + b"\x00" * 10, True),
        (b"OggS" + b"\x00" * 10, True),
        (b"ID3" + b"\x00" * 10, True),
        (b"\xff\xfb" + b"\x00" * 10, True),
        (b"not audio at all", False),
        (b"RIFF", False),
    ],
)
def test_validate_audio_file(data, ok):
    assert validate_audio_file(data) is ok

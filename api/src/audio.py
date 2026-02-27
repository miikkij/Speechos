"""Speechos audio utilities — format conversion, validation, resampling."""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


def read_audio(source: bytes | str | Path) -> tuple[np.ndarray, int]:
    """Read audio from bytes or file path, return (samples_float32, sample_rate)."""
    if isinstance(source, (str, Path)):
        data, sr = sf.read(str(source), dtype="float32")
    else:
        data, sr = sf.read(io.BytesIO(source), dtype="float32")

    # Convert stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    return data, sr


def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz mono for model input."""
    if sr == TARGET_SAMPLE_RATE:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)


def prepare_for_model(source: bytes | str | Path) -> np.ndarray:
    """Read and prepare audio for model consumption: 16 kHz mono float32."""
    audio, sr = read_audio(source)
    return resample_to_16k(audio, sr)


def audio_duration(source: bytes | str | Path) -> float:
    """Get audio duration in seconds."""
    audio, sr = read_audio(source)
    return len(audio) / sr


def float32_to_wav_bytes(audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> bytes:
    """Convert float32 audio array to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def validate_audio_file(data: bytes) -> bool:
    """Check if bytes contain a valid audio file (WAV, FLAC, OGG, MP3)."""
    if len(data) < 12:
        return False
    # WAV: RIFF....WAVE
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return True
    # FLAC
    if data[:4] == b"fLaC":
        return True
    # OGG
    if data[:4] == b"OggS":
        return True
    # MP3 (frame sync)
    if data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return True
    # MP3 with ID3 tag
    if data[:3] == b"ID3":
        return True
    return False

"""Speechos audio utilities: format conversion, validation, resampling."""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


def _decode_with_av(source: bytes | str | Path) -> tuple[np.ndarray, int]:
    """Decode audio using PyAV (handles WebM, MP4, and other container formats)."""
    import av

    if isinstance(source, (str, Path)):
        container = av.open(str(source))
    else:
        container = av.open(io.BytesIO(source))

    stream = container.streams.audio[0]
    # Float formats (fltp, flt, dblp, dbl) are already in [-1, 1] range.
    # Integer formats (s16, s16p, s32, etc.) need normalization.
    fmt_name = stream.codec_context.format.name
    is_float_fmt = fmt_name in ("fltp", "flt", "dblp", "dbl")
    frames = []
    for frame in container.decode(stream):
        array = frame.to_ndarray()
        # av returns shape (channels, samples): average to mono
        if array.ndim > 1:
            array = array.mean(axis=0)
        frames.append(array)
    container.close()

    audio = np.concatenate(frames).astype(np.float32)
    if is_float_fmt:
        # Float audio may slightly exceed [-1, 1]; clip instead of dividing
        np.clip(audio, -1.0, 1.0, out=audio)
    elif audio.max() > 1.0 or audio.min() < -1.0:
        # Integer-format audio: normalize to [-1, 1]
        peak = max(abs(float(audio.max())), abs(float(audio.min())))
        audio = audio / peak
    return audio, stream.rate


def read_audio(source: bytes | str | Path) -> tuple[np.ndarray, int]:
    """Read audio from bytes or file path, return (samples_float32, sample_rate)."""
    raw = source
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()

    # Use PyAV for WebM/Matroska (soundfile can't handle it)
    if isinstance(raw, bytes) and len(raw) >= 4 and raw[:4] == b"\x1a\x45\xdf\xa3":
        return _decode_with_av(source)

    try:
        if isinstance(source, (str, Path)):
            data, sr = sf.read(str(source), dtype="float32")
        else:
            data, sr = sf.read(io.BytesIO(source), dtype="float32")
    except Exception:
        # Fallback to PyAV for any format soundfile can't handle
        return _decode_with_av(source)

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
    """Check if bytes contain a valid audio file (WAV, FLAC, OGG, MP3, WebM)."""
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
    # WebM / Matroska (EBML header)
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return True
    # MP3 (frame sync)
    if data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return True
    # MP3 with ID3 tag
    if data[:3] == b"ID3":
        return True
    return False

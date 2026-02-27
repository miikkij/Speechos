"""Speechos audio analysis — emotion detection and acoustic feature extraction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def extract_audio_features(audio: np.ndarray, sr: int = 16_000) -> dict[str, Any]:
    """Extract acoustic features from audio using librosa.

    Returns pitch, energy, tempo, spectral features, and speaking rate metrics.
    """
    import librosa

    duration = len(audio) / sr

    # Pitch (F0) via pyin
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    # RMS energy
    rms = librosa.feature.rms(y=audio)[0]

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    # Spectral centroid — brightness
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

    # Zero crossing rate — noisiness / consonants
    zcr = librosa.feature.zero_crossing_rate(audio)[0]

    # MFCCs — timbral features (first 13)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1).tolist()

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]

    # Speaking rate estimate via onset detection
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, units="time")
    syllable_rate = len(onsets) / duration if duration > 0 else 0

    return {
        "duration": round(duration, 3),
        "pitch": {
            "mean_hz": round(float(f0_valid.mean()), 1) if len(f0_valid) > 0 else None,
            "min_hz": round(float(f0_valid.min()), 1) if len(f0_valid) > 0 else None,
            "max_hz": round(float(f0_valid.max()), 1) if len(f0_valid) > 0 else None,
            "std_hz": round(float(f0_valid.std()), 1) if len(f0_valid) > 0 else None,
        },
        "energy": {
            "mean_db": round(float(20 * np.log10(rms.mean() + 1e-10)), 1),
            "max_db": round(float(20 * np.log10(rms.max() + 1e-10)), 1),
            "dynamic_range_db": round(
                float(20 * np.log10((rms.max() + 1e-10) / (rms.min() + 1e-10))), 1
            ),
        },
        "tempo_bpm": round(tempo_val, 1),
        "spectral": {
            "centroid_mean_hz": round(float(spectral_centroid.mean()), 1),
            "rolloff_mean_hz": round(float(rolloff.mean()), 1),
            "zero_crossing_rate": round(float(zcr.mean()), 4),
        },
        "speaking_rate": {
            "syllables_per_sec": round(syllable_rate, 2),
            "estimated_wpm": round(syllable_rate * 60 / 1.5, 0),  # rough: ~1.5 syllables/word
        },
        "mfcc_means": [round(v, 2) for v in mfcc_means],
    }


def predict_emotion(
    audio: np.ndarray, sr: int, emotion_model: Any
) -> list[dict[str, Any]]:
    """Run emotion2vec+ inference on audio.

    Returns list of {label, score} sorted by score descending.
    """
    import tempfile
    import soundfile as sf

    # emotion2vec via FunASR expects a file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        tmp_path = tmp.name

    try:
        results = emotion_model.generate(tmp_path, granularity="utterance", extract_embedding=False)
    finally:
        import os
        os.unlink(tmp_path)

    if not results or len(results) == 0:
        return []

    result = results[0]
    scores = result.get("scores", [])
    labels = result.get("labels", [])

    emotions = []
    for label, score in zip(labels, scores):
        emotions.append({"label": label, "score": round(float(score), 4)})
    emotions.sort(key=lambda x: x["score"], reverse=True)

    return emotions


def run_diarization(
    audio: np.ndarray, sr: int, diarization_model: Any
) -> list[dict[str, Any]]:
    """Run pyannote speaker diarization on audio.

    Returns list of speaker segments: {speaker, start, end, duration}.
    """
    import tempfile
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        tmp_path = tmp.name

    try:
        diarization = diarization_model(tmp_path)
    finally:
        import os
        os.unlink(tmp_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3),
        })

    # Build speaker summary
    speaker_times: dict[str, float] = {}
    for seg in segments:
        speaker_times[seg["speaker"]] = speaker_times.get(seg["speaker"], 0.0) + seg["duration"]

    return segments, {
        "num_speakers": len(speaker_times),
        "speakers": {
            spk: round(total, 3)
            for spk, total in sorted(speaker_times.items())
        },
    }


def analyze_audio(
    audio: np.ndarray,
    sr: int = 16_000,
    emotion_model: Any | None = None,
    diarization_model: Any | None = None,
) -> dict[str, Any]:
    """Full audio analysis: acoustic features + optional emotion + optional diarization."""
    result: dict[str, Any] = {}

    # Always extract acoustic features (librosa, no GPU needed)
    result["features"] = extract_audio_features(audio, sr)

    # Emotion detection (requires emotion2vec model)
    if emotion_model is not None:
        try:
            result["emotions"] = predict_emotion(audio, sr, emotion_model)
            if result["emotions"]:
                result["primary_emotion"] = result["emotions"][0]["label"]
        except Exception as e:
            logger.warning("Emotion prediction failed: %s", e)
            result["emotions"] = []
            result["emotion_error"] = str(e)
    else:
        result["emotions"] = []

    # Speaker diarization (requires pyannote model)
    if diarization_model is not None:
        try:
            segments, summary = run_diarization(audio, sr, diarization_model)
            result["diarization"] = {
                "segments": segments,
                "summary": summary,
            }
        except Exception as e:
            logger.warning("Diarization failed: %s", e)
            result["diarization"] = {"segments": [], "summary": {"num_speakers": 0, "speakers": {}}, "error": str(e)}
    else:
        result["diarization"] = {"segments": [], "summary": {"num_speakers": 0, "speakers": {}}}

    return result

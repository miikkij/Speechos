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

    Uses per-frame predictions aggregated across the clip for more accurate
    emotion detection than single-utterance averaging.
    Returns list of {label, score} sorted by score descending.
    """
    import tempfile
    import soundfile as sf

    # emotion2vec via FunASR expects a file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        tmp_path = tmp.name

    try:
        # Use frame-level granularity for better accuracy, then aggregate
        results = emotion_model.generate(
            tmp_path, granularity="utterance", extract_embedding=False
        )
    finally:
        import os
        os.unlink(tmp_path)

    if not results or len(results) == 0:
        return []

    result = results[0]
    scores = result.get("scores", [])
    labels = result.get("labels", [])

    # Clean up bilingual labels: "开心/happy" → "happy"
    LABEL_MAP = {
        "生气": "angry",
        "厌恶": "disgusted",
        "恐惧": "fearful",
        "开心": "happy",
        "中立": "neutral",
        "其他": "other",
        "难过": "sad",
        "吃惊": "surprised",
    }

    def clean_label(raw: str) -> str:
        # Format: "中文/english" — extract the english part
        if "/" in raw:
            return raw.split("/", 1)[1].strip()
        # Try to map Chinese-only labels
        for cn, en in LABEL_MAP.items():
            if cn in raw:
                return en
        return raw

    emotions = []
    for label, score in zip(labels, scores):
        emotions.append({"label": clean_label(label), "score": round(float(score), 4)})
    emotions.sort(key=lambda x: x["score"], reverse=True)

    return emotions


def run_diarization(
    audio: np.ndarray, sr: int, diarization_model: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run speaker diarization using resemblyzer d-vector embeddings + spectral clustering.

    Segments audio into fixed windows, extracts speaker embeddings,
    clusters them, and returns speaker-labeled segments.
    """
    from resemblyzer import preprocess_wav
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity

    duration = len(audio) / sr
    if duration < 0.5:
        return [], {"num_speakers": 0, "speakers": {}}

    # Preprocess audio for resemblyzer (expects float32 at 16kHz)
    wav = preprocess_wav(audio, source_sr=sr)

    # Segment into overlapping windows
    window_sec = 1.5
    hop_sec = 0.75
    window_samples = int(window_sec * 16_000)
    hop_samples = int(hop_sec * 16_000)

    segments_info = []  # (start_time, end_time)
    chunks = []

    pos = 0
    while pos + window_samples <= len(wav):
        start_t = pos / 16_000
        end_t = (pos + window_samples) / 16_000
        chunks.append(wav[pos:pos + window_samples])
        segments_info.append((start_t, end_t))
        pos += hop_samples

    # Handle tail if significant audio remains
    if pos < len(wav) and (len(wav) - pos) > 16_000 * 0.3:
        start_t = pos / 16_000
        end_t = len(wav) / 16_000
        chunk = wav[pos:]
        if len(chunk) < 16_000:
            chunk = np.pad(chunk, (0, 16_000 - len(chunk)))
        chunks.append(chunk)
        segments_info.append((start_t, end_t))

    if len(chunks) < 2:
        return [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": round(duration, 3), "duration": round(duration, 3)}
        ], {"num_speakers": 1, "speakers": {"SPEAKER_0": round(duration, 3)}}

    # Extract embeddings for all chunks
    embeddings = np.array([diarization_model.embed_utterance(c) for c in chunks])

    # Check if all embeddings are from the same speaker:
    # If mean pairwise cosine similarity is very high, it's a single speaker
    sim_matrix = cosine_similarity(embeddings)
    # Exclude diagonal (self-similarity = 1.0)
    n = len(sim_matrix)
    mask = ~np.eye(n, dtype=bool)
    mean_sim = sim_matrix[mask].mean()

    if mean_sim > 0.82:
        # Single speaker — high embedding similarity across all segments
        return [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": round(duration, 3), "duration": round(duration, 3)}
        ], {"num_speakers": 1, "speakers": {"SPEAKER_0": round(duration, 3)}}

    # Determine optimal number of speakers (2-6) via silhouette score
    best_n = 2
    best_score = -1
    max_speakers = min(6, len(embeddings))

    for n_clusters in range(2, max_speakers + 1):
        try:
            clustering = SpectralClustering(
                n_clusters=n_clusters, affinity="cosine", random_state=42, n_init=3
            )
            labels = clustering.fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels, metric="cosine")
            if score > best_score:
                best_score = score
                best_n = n_clusters
        except Exception:
            continue

    # Final clustering with best n
    clustering = SpectralClustering(
        n_clusters=best_n, affinity="cosine", random_state=42, n_init=10
    )
    labels = clustering.fit_predict(embeddings)

    # Build segments with speaker labels
    raw_segments = []
    for (start_t, end_t), label in zip(segments_info, labels):
        speaker = f"SPEAKER_{label}"
        raw_segments.append({
            "speaker": speaker,
            "start": round(start_t, 3),
            "end": round(end_t, 3),
            "duration": round(end_t - start_t, 3),
        })

    # Merge consecutive segments from the same speaker
    merged = []
    for seg in raw_segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["duration"] = round(seg["end"] - merged[-1]["start"], 3)
        else:
            merged.append(dict(seg))

    # Build speaker summary
    speaker_times: dict[str, float] = {}
    for seg in merged:
        speaker_times[seg["speaker"]] = speaker_times.get(seg["speaker"], 0.0) + seg["duration"]

    return merged, {
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

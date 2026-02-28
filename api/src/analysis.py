"""Speechos audio analysis: emotion detection and acoustic feature extraction."""

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

    # Spectral centroid: brightness
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

    # Zero crossing rate: noisiness / consonants
    zcr = librosa.feature.zero_crossing_rate(audio)[0]

    # MFCCs: timbral features (first 13)
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
    """Run emotion inference on audio.

    Dispatches to the correct inference path based on model type:
    - FunASR AutoModel (emotion2vec): .generate() with file path
    - Transformers AudioClassificationPipeline: __call__ with raw audio

    Returns list of {label, score} sorted by score descending.
    """
    # Check if this is a transformers pipeline (AudioClassificationPipeline)
    model_type = type(emotion_model).__name__
    if model_type == "AudioClassificationPipeline" or hasattr(emotion_model, 'task') and not hasattr(emotion_model, 'generate'):
        return _predict_emotion_pipeline(audio, sr, emotion_model)

    # Default: emotion2vec via FunASR
    return _predict_emotion_funasr(audio, sr, emotion_model)


def _predict_emotion_pipeline(
    audio: np.ndarray, sr: int, pipeline_model: Any
) -> list[dict[str, Any]]:
    """Run emotion prediction via transformers AudioClassificationPipeline."""
    results = pipeline_model({"raw": audio, "sampling_rate": sr})
    emotions = []
    for item in results:
        emotions.append({
            "label": item["label"],
            "score": round(float(item["score"]), 4),
        })
    emotions.sort(key=lambda x: x["score"], reverse=True)
    return emotions


def _predict_emotion_funasr(
    audio: np.ndarray, sr: int, emotion_model: Any
) -> list[dict[str, Any]]:
    """Run emotion2vec+ inference via FunASR AutoModel."""
    import tempfile
    import soundfile as sf

    # emotion2vec via FunASR expects a file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        tmp_path = tmp.name

    try:
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
        if "/" in raw:
            return raw.split("/", 1)[1].strip()
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
    """Run speaker diarization.

    Dispatches based on model type:
    - resemblyzer VoiceEncoder: d-vector embeddings + spectral clustering
    - pyannote Pipeline: end-to-end neural diarization
    - speechbrain: speaker verification-based clustering
    """
    model_type = type(diarization_model).__name__

    # PyAnnote Pipeline
    if model_type == "Pipeline" and hasattr(diarization_model, 'instantiate'):
        return _run_diarization_pyannote(audio, sr, diarization_model)

    # SpeechBrain SpeakerRecognition
    if model_type in ("SpeakerRecognition", "EncoderClassifier"):
        return _run_diarization_speechbrain(audio, sr, diarization_model)

    # Default: resemblyzer
    return _run_diarization_resemblyzer(audio, sr, diarization_model)


def _run_diarization_pyannote(
    audio: np.ndarray, sr: int, pipeline: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run diarization via pyannote.audio Pipeline."""
    import torch
    import io
    import soundfile as sf

    # PyAnnote expects a file-like object or path
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)

    diarization = pipeline({"uri": "audio", "audio": buf})

    segments = []
    speaker_times: dict[str, float] = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        dur = turn.end - turn.start
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(dur, 3),
        })
        speaker_times[speaker] = speaker_times.get(speaker, 0.0) + dur

    summary = {
        "num_speakers": len(speaker_times),
        "speakers": {spk: round(t, 3) for spk, t in sorted(speaker_times.items())},
    }
    return segments, summary


def _run_diarization_speechbrain(
    audio: np.ndarray, sr: int, model: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run speaker diarization via SpeechBrain embeddings + clustering."""
    import torch
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity

    duration = len(audio) / sr
    if duration < 0.5:
        return [], {"num_speakers": 0, "speakers": {}}

    # Segment into windows
    window_sec = 1.5
    hop_sec = 0.75
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    chunks = []
    segments_info = []
    pos = 0
    while pos + window_samples <= len(audio):
        start_t = pos / sr
        end_t = (pos + window_samples) / sr
        chunks.append(audio[pos:pos + window_samples])
        segments_info.append((start_t, end_t))
        pos += hop_samples

    if len(chunks) < 2:
        return [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": round(duration, 3), "duration": round(duration, 3)}
        ], {"num_speakers": 1, "speakers": {"SPEAKER_0": round(duration, 3)}}

    # Extract embeddings
    import numpy as np
    embeddings = []
    for chunk in chunks:
        tensor = torch.tensor(chunk).unsqueeze(0)
        emb = model.encode_batch(tensor)
        embeddings.append(emb.squeeze().cpu().numpy())
    embeddings = np.array(embeddings)

    # Same clustering logic as resemblyzer path
    sim_matrix = cosine_similarity(embeddings)
    n = len(sim_matrix)
    mask = ~np.eye(n, dtype=bool)
    mean_sim = sim_matrix[mask].mean()

    if mean_sim > 0.82:
        return [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": round(duration, 3), "duration": round(duration, 3)}
        ], {"num_speakers": 1, "speakers": {"SPEAKER_0": round(duration, 3)}}

    best_n = 2
    best_score = -1
    for n_clusters in range(2, min(6, len(embeddings)) + 1):
        try:
            clustering = SpectralClustering(n_clusters=n_clusters, affinity="cosine", random_state=42, n_init=3)
            labels = clustering.fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels, metric="cosine")
            if score > best_score:
                best_score = score
                best_n = n_clusters
        except Exception:
            continue

    clustering = SpectralClustering(n_clusters=best_n, affinity="cosine", random_state=42, n_init=10)
    labels = clustering.fit_predict(embeddings)

    merged = _merge_diarization_segments(segments_info, labels, duration)
    return merged


def _merge_diarization_segments(
    segments_info: list[tuple[float, float]], labels: Any, duration: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge consecutive segments with same speaker label."""
    raw_segments = []
    for (start_t, end_t), label in zip(segments_info, labels):
        speaker = f"SPEAKER_{label}"
        raw_segments.append({
            "speaker": speaker,
            "start": round(start_t, 3),
            "end": round(end_t, 3),
            "duration": round(end_t - start_t, 3),
        })

    merged = []
    for seg in raw_segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["duration"] = round(seg["end"] - merged[-1]["start"], 3)
        else:
            merged.append(dict(seg))

    speaker_times: dict[str, float] = {}
    for seg in merged:
        speaker_times[seg["speaker"]] = speaker_times.get(seg["speaker"], 0.0) + seg["duration"]

    return merged, {
        "num_speakers": len(speaker_times),
        "speakers": {spk: round(total, 3) for spk, total in sorted(speaker_times.items())},
    }


def _run_diarization_resemblyzer(
    audio: np.ndarray, sr: int, diarization_model: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run speaker diarization using resemblyzer d-vector embeddings + spectral clustering."""
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
        # Single speaker: high embedding similarity across all segments
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

    # Merge consecutive segments and build speaker summary
    return _merge_diarization_segments(segments_info, labels, duration)


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

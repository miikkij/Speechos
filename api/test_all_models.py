"""
Comprehensive test of all Speech Analysis & SER models against test WAV files.
Tests: librosa, openSMILE, Silero VAD, emotion2vec+, wav2vec2 SER models,
       HuBERT SER, WavLM SER, wav2small, resemblyzer diarization.
"""
import json, time, os, sys, traceback
import numpy as np
import librosa
import torch

SAMPLES_DIR = r"e:\dev\GitHub\Speechos\samples"
wav_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith(".wav")])

print(f"Test WAV files: {wav_files}")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
print(f"Device: cuda={torch.cuda.is_available()}, GPU={gpu_name}")
print("=" * 80)

results = {}
all_errors = []


def test_section(name, func):
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"{'=' * 80}")
    try:
        result = func()
        results[name] = result
        print(f"  STATUS: OK")
        return True
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        results[name] = {"error": error_msg}
        all_errors.append((name, error_msg))
        print(f"  STATUS: FAILED — {error_msg}")
        traceback.print_exc()
        return False


# ============================================================
# 1. LIBROSA Feature Extraction
# ============================================================
def test_librosa():
    librosa_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        y, sr = librosa.load(path, sr=16000)
        duration = len(y) / sr

        pitch_f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        elapsed = time.time() - t0
        res = {
            "duration_s": round(duration, 2),
            "pitch_mean_hz": round(float(np.nanmean(pitch_f0)), 1),
            "pitch_std_hz": round(float(np.nanstd(pitch_f0)), 1),
            "rms_energy": round(float(np.mean(rms)), 4),
            "zcr": round(float(np.mean(zcr)), 4),
            "mfcc1_mean": round(float(np.mean(mfccs[0])), 2),
            "spectral_centroid_hz": round(float(np.mean(spectral_centroid)), 1),
            "spectral_rolloff_hz": round(float(np.mean(spectral_rolloff)), 1),
            "time_ms": round(elapsed * 1000, 1),
        }
        librosa_results[wf] = res
        print(f"    {wf}: pitch={res['pitch_mean_hz']}Hz rms={res['rms_energy']} zcr={res['zcr']} [{res['time_ms']}ms]")
    return librosa_results


# ============================================================
# 2. openSMILE Feature Extraction
# ============================================================
def test_opensmile():
    import opensmile

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    opensmile_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        features = smile.process_file(path)
        elapsed = time.time() - t0

        n_features = features.shape[1]
        # Pick key features
        cols = features.columns.tolist()
        key_features = {}
        for col in ["F0semitoneFrom27.5Hz_sma3nz_amean", "jitterLocal_sma3nz_amean",
                     "shimmerLocaldB_sma3nz_amean", "HNRdBACF_sma3nz_amean",
                     "logRelF0-H1-H2_sma3nz_amean", "loudness_sma3_amean"]:
            if col in cols:
                key_features[col] = round(float(features[col].values[0]), 4)

        res = {
            "total_features": n_features,
            "key_features": key_features,
            "time_ms": round(elapsed * 1000, 1),
        }
        opensmile_results[wf] = res
        loudness = key_features.get("loudness_sma3_amean", "N/A")
        f0 = key_features.get("F0semitoneFrom27.5Hz_sma3nz_amean", "N/A")
        print(f"    {wf}: {n_features} features, F0={f0}, loudness={loudness} [{res['time_ms']}ms]")
    return opensmile_results


# ============================================================
# 3. Silero VAD
# ============================================================
def test_silero_vad():
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

    model = load_silero_vad()
    vad_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        wav = read_audio(path)
        timestamps = get_speech_timestamps(wav, model, return_seconds=True)
        elapsed = time.time() - t0

        res = {
            "speech_segments": len(timestamps),
            "segments": [{"start": round(s["start"], 2), "end": round(s["end"], 2)} for s in timestamps],
            "has_speech": len(timestamps) > 0,
            "time_ms": round(elapsed * 1000, 1),
        }
        vad_results[wf] = res
        print(f"    {wf}: {res['speech_segments']} speech segments, has_speech={res['has_speech']} [{res['time_ms']}ms]")
    return vad_results


# ============================================================
# 4. emotion2vec+ large (via FunASR)
# ============================================================
def test_emotion2vec():
    from funasr import AutoModel

    print("    Loading emotion2vec+ large model...")
    model = AutoModel(model="iic/emotion2vec_plus_large")

    LABEL_MAP = {
        0: "angry", 1: "disgusted", 2: "fearful", 3: "happy",
        4: "neutral", 5: "other", 6: "sad", 7: "surprised", 8: "unknown"
    }

    emo_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        res_raw = model.generate(path, granularity="utterance", extract_embedding=False)
        elapsed = time.time() - t0

        if res_raw and len(res_raw) > 0:
            entry = res_raw[0]
            labels = entry.get("labels", [])
            scores = entry.get("scores", [])
            # Clean labels
            clean_labels = []
            for lbl in labels:
                if "/" in lbl:
                    lbl = lbl.split("/")[-1]
                clean_labels.append(lbl)
            top_idx = int(np.argmax(scores)) if scores else -1
            top_label = clean_labels[top_idx] if top_idx >= 0 else "unknown"
            top_score = round(float(scores[top_idx]), 4) if top_idx >= 0 else 0

            all_scores = {clean_labels[i]: round(float(scores[i]), 4) for i in range(len(scores))}
            res = {
                "top_emotion": top_label,
                "top_score": top_score,
                "all_scores": all_scores,
                "time_ms": round(elapsed * 1000, 1),
            }
        else:
            res = {"error": "no output", "time_ms": round(elapsed * 1000, 1)}

        emo_results[wf] = res
        print(f"    {wf}: {res.get('top_emotion', 'ERROR')} ({res.get('top_score', 0):.2%}) [{res['time_ms']}ms]")
    return emo_results


# ============================================================
# 5. wav2vec2 SER — ehcalabres (XLSR)
# ============================================================
def test_wav2vec2_ehcalabres():
    from transformers import pipeline

    print("    Loading ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition...")
    classifier = pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device=0 if torch.cuda.is_available() else -1,
    )

    w2v_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        y, sr = librosa.load(path, sr=16000)
        preds = classifier({"raw": y, "sampling_rate": 16000}, top_k=None)
        elapsed = time.time() - t0

        top = preds[0]
        all_scores = {p["label"]: round(p["score"], 4) for p in preds}
        res = {
            "top_emotion": top["label"],
            "top_score": round(top["score"], 4),
            "all_scores": all_scores,
            "time_ms": round(elapsed * 1000, 1),
        }
        w2v_results[wf] = res
        print(f"    {wf}: {res['top_emotion']} ({res['top_score']:.2%}) [{res['time_ms']}ms]")
    return w2v_results


# ============================================================
# 6. wav2vec2 SER — r-f (97.5% accuracy claim)
# ============================================================
def test_wav2vec2_rf():
    from transformers import pipeline

    print("    Loading r-f/wav2vec-english-speech-emotion-recognition...")
    classifier = pipeline(
        "audio-classification",
        model="r-f/wav2vec-english-speech-emotion-recognition",
        device=0 if torch.cuda.is_available() else -1,
    )

    w2v_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        y, sr = librosa.load(path, sr=16000)
        preds = classifier({"raw": y, "sampling_rate": 16000}, top_k=None)
        elapsed = time.time() - t0

        top = preds[0]
        all_scores = {p["label"]: round(p["score"], 4) for p in preds}
        res = {
            "top_emotion": top["label"],
            "top_score": round(top["score"], 4),
            "all_scores": all_scores,
            "time_ms": round(elapsed * 1000, 1),
        }
        w2v_results[wf] = res
        print(f"    {wf}: {res['top_emotion']} ({res['top_score']:.2%}) [{res['time_ms']}ms]")
    return w2v_results


# ============================================================
# 7. wav2vec2 SER — Dpngtm
# ============================================================
def test_wav2vec2_dpngtm():
    from transformers import pipeline

    print("    Loading Dpngtm/wav2vec2-emotion-recognition...")
    classifier = pipeline(
        "audio-classification",
        model="Dpngtm/emotion-recognition-wav2vec2-IEMOCAP",
        device=0 if torch.cuda.is_available() else -1,
    )

    w2v_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        y, sr = librosa.load(path, sr=16000)
        preds = classifier({"raw": y, "sampling_rate": 16000}, top_k=None)
        elapsed = time.time() - t0

        top = preds[0]
        all_scores = {p["label"]: round(p["score"], 4) for p in preds}
        res = {
            "top_emotion": top["label"],
            "top_score": round(top["score"], 4),
            "all_scores": all_scores,
            "time_ms": round(elapsed * 1000, 1),
        }
        w2v_results[wf] = res
        print(f"    {wf}: {res['top_emotion']} ({res['top_score']:.2%}) [{res['time_ms']}ms]")
    return w2v_results


# ============================================================
# 8. WavLM SER — audeering (A/D/V dimensional)
# ============================================================
def test_wavlm_audeering():
    from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

    print("    Loading audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim...")
    model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    wavlm_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        y, sr = librosa.load(path, sr=16000)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
        elapsed = time.time() - t0

        # audeering model outputs: [arousal, dominance, valence]
        labels = ["arousal", "dominance", "valence"]
        adv = {labels[i]: round(float(logits[i]), 4) for i in range(min(len(labels), len(logits)))}
        res = {
            "arousal": adv.get("arousal", 0),
            "dominance": adv.get("dominance", 0),
            "valence": adv.get("valence", 0),
            "time_ms": round(elapsed * 1000, 1),
        }
        wavlm_results[wf] = res
        print(f"    {wf}: A={res['arousal']:.3f} D={res['dominance']:.3f} V={res['valence']:.3f} [{res['time_ms']}ms]")
    return wavlm_results


# ============================================================
# 9. wav2small (audeering) — tiny A/D/V model
# ============================================================
def test_wav2small():
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

    print("    Loading audeering/wav2small...")
    model_name = "audeering/wav2small"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    w2s_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        y, sr = librosa.load(path, sr=16000)
        inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
        elapsed = time.time() - t0

        labels = ["arousal", "dominance", "valence"]
        adv = {labels[i]: round(float(logits[i]), 4) for i in range(min(len(labels), len(logits)))}
        res = {
            "arousal": adv.get("arousal", 0),
            "dominance": adv.get("dominance", 0),
            "valence": adv.get("valence", 0),
            "time_ms": round(elapsed * 1000, 1),
        }
        w2s_results[wf] = res
        print(f"    {wf}: A={res['arousal']:.3f} D={res['dominance']:.3f} V={res['valence']:.3f} [{res['time_ms']}ms]")
    return w2s_results


# ============================================================
# 10. Resemblyzer Diarization
# ============================================================
def test_resemblyzer():
    from resemblyzer import VoiceEncoder, preprocess_wav

    print("    Loading resemblyzer VoiceEncoder...")
    encoder = VoiceEncoder(device="cuda" if torch.cuda.is_available() else "cpu")

    diar_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        wav = preprocess_wav(path)
        # Generate embeddings for segments
        segment_len = 1.0  # 1 second segments
        sr = 16000
        n_samples = int(segment_len * sr)
        segments = []
        embeddings = []
        for i in range(0, len(wav) - n_samples, n_samples):
            seg = wav[i:i + n_samples]
            if np.abs(seg).mean() > 0.001:  # skip silence
                emb = encoder.embed_utterance(seg)
                embeddings.append(emb)
                segments.append({"start": round(i / sr, 2), "end": round((i + n_samples) / sr, 2)})

        elapsed = time.time() - t0

        if len(embeddings) > 1:
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(embeddings)
            mean_sim = float(np.mean(sim_matrix[np.triu_indices(len(embeddings), k=1)]))
            n_speakers = 1 if mean_sim > 0.82 else "2+"
        elif len(embeddings) == 1:
            mean_sim = 1.0
            n_speakers = 1
        else:
            mean_sim = 0.0
            n_speakers = 0

        res = {
            "n_segments_analyzed": len(segments),
            "estimated_speakers": n_speakers,
            "mean_similarity": round(mean_sim, 4),
            "time_ms": round(elapsed * 1000, 1),
        }
        diar_results[wf] = res
        print(f"    {wf}: {res['n_segments_analyzed']} segments, speakers={res['estimated_speakers']}, sim={res['mean_similarity']:.3f} [{res['time_ms']}ms]")
    return diar_results


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    test_section("1. librosa", test_librosa)
    test_section("2. openSMILE (eGeMAPSv02)", test_opensmile)
    test_section("3. Silero VAD", test_silero_vad)
    test_section("4. emotion2vec+ large", test_emotion2vec)
    test_section("5. wav2vec2 SER (ehcalabres XLSR)", test_wav2vec2_ehcalabres)
    test_section("6. wav2vec2 SER (r-f 97.5%)", test_wav2vec2_rf)
    test_section("7. wav2vec2 SER (Dpngtm IEMOCAP)", test_wav2vec2_dpngtm)
    test_section("8. WavLM SER (audeering A/D/V)", test_wavlm_audeering)
    test_section("9. wav2small (audeering tiny A/D/V)", test_wav2small)
    test_section("10. Resemblyzer Diarization", test_resemblyzer)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = [k for k, v in results.items() if "error" not in v]
    failed = [k for k, v in results.items() if "error" in v]
    print(f"  Passed: {len(passed)}/{len(results)}")
    print(f"  Failed: {len(failed)}/{len(results)}")
    if failed:
        print(f"  Failed tests:")
        for name, err in all_errors:
            print(f"    - {name}: {err}")

    # Write JSON results
    output_path = os.path.join(SAMPLES_DIR, "..", "docs", "research", "speech-analysis", "TEST-RESULTS.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results written to: {output_path}")

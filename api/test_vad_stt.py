"""Test Silero VAD and faster-whisper STT."""
import torch, time, os, json, librosa

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
RESULTS_PATH = os.path.join(SAMPLES_DIR, "..", "docs", "research", "speech-analysis", "TEST-RESULTS.json")
wav_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith(".wav")])

# Load existing results
with open(RESULTS_PATH) as f:
    all_results = json.load(f)

# === Silero VAD ===
print("=== Silero VAD ===")
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = utils

vad_results = {}
for wf in wav_files:
    path = os.path.join(SAMPLES_DIR, wf)
    t0 = time.time()
    # Load with librosa instead of torchaudio (torchcodec not installed)
    y, sr = librosa.load(path, sr=16000)
    wav = torch.from_numpy(y)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
    elapsed = time.time() - t0
    n_segments = len(speech_timestamps)
    total_speech_s = sum((ts["end"] - ts["start"]) / 16000 for ts in speech_timestamps)
    res = {
        "n_speech_segments": n_segments,
        "total_speech_seconds": round(total_speech_s, 3),
        "segments": [
            {"start_s": round(ts["start"] / 16000, 3), "end_s": round(ts["end"] / 16000, 3)}
            for ts in speech_timestamps
        ],
        "time_ms": round(elapsed * 1000, 1),
    }
    vad_results[wf] = res
    print(f"  {wf}: {n_segments} segments, {total_speech_s:.2f}s speech [{elapsed*1000:.0f}ms]")

all_results["3. Silero VAD"] = vad_results

# === faster-whisper STT ===
print("\n=== faster-whisper STT ===")
from faster_whisper import WhisperModel

stt_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
stt_results = {}

for wf in wav_files:
    path = os.path.join(SAMPLES_DIR, wf)
    t0 = time.time()
    segments, info = stt_model.transcribe(path, beam_size=5, language="en")
    segment_list = list(segments)
    elapsed = time.time() - t0
    full_text = " ".join(s.text.strip() for s in segment_list)
    res = {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_s": round(info.duration, 2),
        "n_segments": len(segment_list),
        "transcription": full_text if full_text else "(empty)",
        "segments": [
            {
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip(),
                "avg_logprob": round(s.avg_logprob, 4),
                "no_speech_prob": round(s.no_speech_prob, 4),
            }
            for s in segment_list
        ],
        "time_ms": round(elapsed * 1000, 1),
    }
    stt_results[wf] = res
    print(f"  {wf}: lang={info.language}({info.language_probability:.2f}) text='{full_text[:80]}' [{elapsed*1000:.0f}ms]")

all_results["0. faster-whisper STT (large-v3)"] = stt_results

# Save
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print("\nAll results saved to TEST-RESULTS.json")
print("=== ALL PASSED ===")

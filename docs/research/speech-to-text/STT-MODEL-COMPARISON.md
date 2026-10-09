# Speech-to-Text (STT) Model Comparison Matrix

> Last updated: 2026-06-19

## Overview

This document compares open-source speech-to-text models that can be downloaded and run locally, with and without GPU acceleration.

> **2026-06 refresh (what changed since Feb 2026):** No new Whisper — `large-v3-turbo` (Oct 2024) is still the newest OpenAI model; faster-whisper/distil-whisper remain solid but are now beaten on accuracy by LLM-decoder ASR and on speed by Parakeet. New/updated open-weight models to track: **NVIDIA Parakeet TDT 0.6B v3** (25 EU languages, top throughput), **Canary-Qwen-2.5B** / **Canary-1B-v2** and **IBM Granite-Speech-4.1-2B** (accuracy leaders), **Mistral Voxtral-Mini-4B-Realtime** (open streaming ASR), and **Moonshine v2** (now multilingual + streaming, repo moved to `moonshine-ai/moonshine`). Older entries are retained and marked `⚠ superseded` where a newer drop-in exists.

---

## Model Size & Hardware Requirements

| Model | Parameters | Model Size (Disk) | VRAM (GPU) | RAM (CPU-only) | CPU Support | GPU Required | Real-time Capable |
|---|---|---|---|---|---|---|---|
| **Whisper Tiny** | 39M | ~150 MB | ~1 GB | ~2 GB | ✅ | ❌ | ✅ |
| **Whisper Base** | 74M | ~290 MB | ~1 GB | ~2 GB | ✅ | ❌ | ✅ |
| **Whisper Small** | 244M | ~960 MB | ~2 GB | ~4 GB | ✅ | ❌ | ✅ |
| **Whisper Medium** | 769M | ~3 GB | ~5 GB | ~8 GB | ✅ (slow) | ❌ | ⚠️ |
| **Whisper Large-v3** | 1.55B | ~6 GB | ~10 GB | ~16 GB | ⚠️ (very slow) | Recommended | ❌ |
| **Whisper Large-v3 Turbo** | 809M | ~3.1 GB | ~6 GB | ~10 GB | ⚠️ | Recommended | ⚠️ |
| **Distil-Whisper (large-v3)** | 756M | ~3 GB | ~4 GB | ~8 GB | ✅ | ❌ | ✅ |
| **Faster-Whisper (large-v3)** | 1.55B (CTranslate2) | ~3 GB (INT8) | ~4 GB (INT8) | ~6 GB (INT8) | ✅ | ❌ | ✅ |
| **WhisperX** | Uses Whisper weights | Same as Whisper | Same + alignment model | Same + alignment model | ✅ | ❌ | ⚠️ |
| **Vosk (small-en)** | N/A (Kaldi) | ~40 MB | N/A | ~512 MB | ✅ | ❌ | ✅ |
| **Vosk (large-en)** | N/A (Kaldi) | ~1.8 GB | N/A | ~4 GB | ✅ | ❌ | ✅ |
| **Moonshine v2** | 27–60M | ~100 MB | ~512 MB | ~1 GB | ✅ | ❌ | ✅ |
| **NVIDIA Parakeet TDT 1.1B** `⚠ superseded` | 1.1B | ~4 GB | ~6 GB | N/A | ❌ | ✅ | ✅ |
| **NVIDIA Parakeet TDT 0.6B v3** *(new)* | 0.6B | ~2.5 GB | ~4 GB | N/A | ❌ | ✅ | ✅ |
| **Canary Qwen 2.5B** | 2.5B | ~10 GB | ~12 GB | N/A | ❌ | ✅ | ❌ |
| **Canary 1B v2** *(new, 25 langs)* | 1B | ~4 GB | ~6 GB | N/A | ❌ | ✅ | ⚠️ |
| **IBM Granite-Speech 4.1 2B** *(new)* | 2B | ~8 GB | ~10 GB | N/A | ❌ | ✅ | ❌ |
| **Voxtral-Mini-4B-Realtime** *(new, streaming)* | 4B | ~8 GB | ~10 GB | N/A | ❌ | ✅ | ✅ (streaming) |
| **Wav2Vec 2.0 (base)** | 95M | ~360 MB | ~2 GB | ~4 GB | ✅ | ❌ | ✅ |
| **Wav2Vec 2.0 (large)** | 317M | ~1.2 GB | ~4 GB | ~8 GB | ✅ | ❌ | ⚠️ |

---

## Feature Comparison

| Model | Languages | Streaming | Timestamps | Diarization | Noise Robust | Voice Activity | Punctuation | WER (LibriSpeech Clean) |
|---|---|---|---|---|---|---|---|---|
| **Whisper Tiny** | 99 | ❌ | ✅ (segment) | ❌ | ⚠️ | ❌ | ✅ | ~7.6% |
| **Whisper Base** | 99 | ❌ | ✅ (segment) | ❌ | ⚠️ | ❌ | ✅ | ~5.0% |
| **Whisper Small** | 99 | ❌ | ✅ (segment) | ❌ | ✅ | ❌ | ✅ | ~3.4% |
| **Whisper Medium** | 99 | ❌ | ✅ (segment) | ❌ | ✅ | ❌ | ✅ | ~2.9% |
| **Whisper Large-v3** | 99 | ❌ | ✅ (segment) | ❌ | ✅ | ❌ | ✅ | ~2.7% |
| **Whisper Large-v3 Turbo** | 99 | ❌ | ✅ (segment) | ❌ | ✅ | ❌ | ✅ | ~2.9% |
| **Distil-Whisper** | EN only | ❌ | ✅ (segment) | ❌ | ✅ | ❌ | ✅ | ~3.0% |
| **Faster-Whisper** | 99 | ❌ | ✅ (word) | ❌ | ✅ | ✅ (VAD) | ✅ | Same as Whisper |
| **WhisperX** | 99 | ❌ | ✅ (word) | ✅ | ✅ | ✅ | ✅ | Same as Whisper |
| **Vosk** | 20+ | ✅ | ✅ (word) | ❌ | ⚠️ | ❌ | ❌ | ~9.8% |
| **Moonshine v2** | 8 langs | ✅ | ✅ | ❌ | ⚠️ | ❌ | ✅ | ~5–6% |
| **NVIDIA Parakeet TDT 1.1B** | EN | ✅ | ✅ (word) | ❌ | ✅ | ✅ | ✅ | ~3.0% |
| **NVIDIA Parakeet TDT 0.6B v3** | 25 EU langs | ✅ | ✅ (word) | ❌ | ✅ | ✅ | ✅ | ~6.3% (multiling.) |
| **Canary Qwen 2.5B** | EN (+summarize) | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ~5.6% |
| **Canary 1B v2** | 25 langs | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ~5–6% |
| **IBM Granite-Speech 4.1 2B** | en/fr/de/es/pt/ja | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ~5.3% |
| **Voxtral-Mini-4B-Realtime** | Multi | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ~5–6% |
| **Wav2Vec 2.0** | Fine-tunable | ❌ | ✅ (frame) | ❌ | ⚠️ | ❌ | ❌ | ~3–5% |

---

## Speed Comparison (Relative to Real-Time)

| Model | GPU Speed (RTFx) | CPU Speed (RTFx) | Notes |
|---|---|---|---|
| **Whisper Large-v3** | ~16x | ~0.5x | Baseline accuracy leader |
| **Whisper Large-v3 Turbo** | ~86x | ~2x | 6x faster than Large-v3, similar accuracy |
| **Faster-Whisper (large-v3)** | ~40x | ~4x | CTranslate2 INT8 quantization |
| **Faster-Whisper (turbo)** | ~100x | ~8x | Best balance of speed/accuracy |
| **Distil-Whisper** | ~96x | ~6x | 6x faster, English only |
| **WhisperX** | ~30x (batched) | ~3x | Adds alignment + diarization overhead |
| **Vosk (small)** | N/A | ~20x | Lightweight, streaming |
| **Vosk (large)** | N/A | ~5x | Better accuracy, still fast |
| **Moonshine** | N/A | ~50x+ | Designed for edge devices |
| **Parakeet TDT** | ~2,728x | N/A | GPU only, fastest available |

> RTFx = Real-Time Factor multiplier. 10x means it processes 10 seconds of audio in 1 second.

---

## Licensing

| Model | License | Commercial Use |
|---|---|---|
| Whisper (all sizes) | MIT | ✅ |
| Faster-Whisper | MIT | ✅ |
| Distil-Whisper | MIT | ✅ |
| WhisperX | BSD-4-Clause | ✅ (with attribution) |
| Vosk | Apache 2.0 | ✅ |
| Moonshine v2 | MIT | ✅ |
| Parakeet TDT (1.1B / 0.6B v3) | CC-BY-4.0 (NeMo) | ✅ |
| Canary (Qwen 2.5B / 1B v2) | CC-BY-4.0 | ✅ |
| IBM Granite-Speech 4.1 | Apache 2.0 | ✅ |
| Voxtral-Mini-4B-Realtime | Apache 2.0 | ✅ |
| Wav2Vec 2.0 | MIT | ✅ |

---

## Recommendations for This Project

### Primary: Faster-Whisper (GPU + CPU)
- **Why**: Best balance of accuracy, speed, and flexibility
- **GPU mode**: Use `large-v3` or `turbo` with `float16` for maximum accuracy
- **CPU mode**: Use `base` or `small` with `int8` for real-time on CPU
- **Install**: `pip install faster-whisper`

### Secondary: Vosk (CPU-only, lightweight)
- **Why**: Tiny footprint (40MB), real-time streaming, works on any hardware
- **Use case**: Quick previews, low-latency feedback in web UI
- **Install**: `pip install vosk`

### For Diarization: WhisperX
- **Why**: Only option with built-in speaker diarization + word-level timestamps
- **Use case**: Multi-speaker analysis
- **Install**: `pip install whisperx`

---

## Finnish (fi) — measured benchmark 2026-10-09

Local measurement with `api/bench/stt_fi.py` (RTX 4090, faster-whisper float16 beam 5 + VAD, Qwen bf16,
NeMo fp32 batch 1). Tables with revisions: `api/bench/results/stt_fi_results_{n100,lock300}.{md,json}`;
decision write-up: `docs/plans/2026-10-public/report-b.md`. WER/CER on normalized text (lowercase,
punctuation removed, digits kept).

**Clean comparison — FLEURS-fi test, n = 300** (`google/fleurs` parquet `168de34`):

| Model (revision) | WER | CER | RTF | latency median | License |
|---|---:|---:|---:|---:|---|
| **Finnish-NLP/whisper-large-finnish-v3** (`b23deb0`, CT2 float16) | **6.2 %** | 1.43 % | 0.044 | 0.54 s | Apache-2.0 |
| faster-whisper large-v3 (`Systran@edaa852`, current default) | 7.5 % | 1.53 % | 0.044 | 0.53 s | MIT |
| faster-whisper large-v3-turbo (`mobiuslabsgmbh`) | 8.1 % | 1.65 % | 0.018 | 0.21 s | MIT |

Paired bootstrap (5000×): Finnish-Whisper − large-v3 WER −1.3 pp (95 % CI −2.1…−0.5), CER −0.1 pp
(CI −0.3…+0.2, n.s.) → the gain is at word level (compounds, inflection boundaries), not phonetic.

**All candidates, n = 100** (FLEURS / VoxPopuli-fi test / VoxPopuli 1–5 s clips):

| Model | FLEURS WER | VoxPopuli WER | short 1–5 s WER | RTF (FLEURS) | Notes |
|---|---:|---:|---:|---:|---|
| Finnish-Whisper v3 | 6.4 % | ⚠ 6.1 % | ⚠ 3.5 % | 0.048 | ⚠ trained on all VoxPopuli-fi splits incl. test — invalid |
| large-v3 | 7.9 % | 11.5 % | 13.5 % | 0.052 | reproduces the 2026-06 lock (7.9 %) |
| large-v3-turbo | 8.0 % | 11.6 % | 16.3 % | 0.018 | 3× faster; weaker on short clips |
| nvidia/parakeet-tdt-0.6b-v3 (`541d1f9`) | 12.6 % | 14.5 % | 14.8 % | 0.008 | NeMo 3.0; ~0.09 s/clip; CC-BY-4.0 |
| Qwen/Qwen3-ASR-1.7B (`7278e1e`) | 27.3 % | 29.1 % | 31.7 % | 0.22 | transformers backend; not competitive in Finnish |

**Caveats.**
- Finnish-NLP's training mix (`start_train.sh` in the model repo) includes CV11 fi train+dev, FLEURS fi train,
  and `Finnish-NLP/voxpopuli_fi_audio_processed` = 9074 rows = VoxPopuli-fi train 7878 + validation 718 +
  test 478. FLEURS test is clean for it; VoxPopuli is leaked. A clean short-utterance test for it is still
  missing (Common Voice is distributed only via Mozilla Data Collective since 10/2025).
- VoxPopuli-fi has misaligned segments (audio ≠ transcript), which put a CER floor of ~5 % under every model.
- The Finnish fine-tune is Finnish-only; use it for `language="fi"` requests, not as a multilingual default.
- Packaging: no `nemo-toolkit` release fits the Speechos core lock (2.4–2.6 pin transformers~=4.53, 2.7 needs
  setuptools≥79, 3.0 pins transformers 5.x) → the bench runs Parakeet in an isolated uv project.

---

## Python Package Summary

```bash
# Core STT
pip install faster-whisper    # Primary engine
pip install vosk              # Lightweight CPU fallback
pip install whisperx          # Diarization support

# For Whisper directly
pip install openai-whisper    # Original OpenAI implementation

# For wav2vec2 models
pip install transformers torch torchaudio

# New in 2026 (open weights)
pip install -U nemo_toolkit[asr]   # NVIDIA Parakeet TDT 0.6B v3 + Canary 1B v2 / Qwen 2.5B
pip install transformers           # IBM Granite-Speech 4.1 (ibm-granite/granite-speech-4.1-2b)
pip install vllm                   # Voxtral-Mini-4B-Realtime streaming ASR (Apache-2.0)
pip install useful-moonshine       # Moonshine v2 (repo moonshine-ai/moonshine)
```

> **Model IDs (2026):** Parakeet `nvidia/parakeet-tdt-0.6b-v3` · Canary `nvidia/canary-qwen-2.5b`, `nvidia/canary-1b-v2` · Granite `ibm-granite/granite-speech-4.1-2b` · Voxtral `mistralai/Voxtral-Mini-4B-Realtime-2602`. **Open ASR Leaderboard** top open-weight tier is now Canary-Qwen-2.5B / Granite-Speech; Parakeet is the throughput leader. Whisper remains the multilingual workhorse (no new version).

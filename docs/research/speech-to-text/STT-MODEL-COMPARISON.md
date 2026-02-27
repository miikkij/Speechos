# Speech-to-Text (STT) Model Comparison Matrix

> Last updated: 2026-02-27

## Overview

This document compares open-source speech-to-text models that can be downloaded and run locally, with and without GPU acceleration.

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
| **Moonshine** | 27M | ~100 MB | ~512 MB | ~1 GB | ✅ | ❌ | ✅ |
| **NVIDIA Parakeet TDT** | 1.1B | ~4 GB | ~6 GB | N/A | ❌ | ✅ | ✅ |
| **Canary Qwen 2.5B** | 2.5B | ~10 GB | ~12 GB | N/A | ❌ | ✅ | ❌ |
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
| **Moonshine** | EN | ✅ | ✅ | ❌ | ⚠️ | ❌ | ✅ | ~5–6% |
| **NVIDIA Parakeet TDT** | EN | ✅ | ✅ (word) | ❌ | ✅ | ✅ | ✅ | ~3.0% |
| **Canary Qwen 2.5B** | Multi | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ~5.6% |
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
| Moonshine | MIT | ✅ |
| Parakeet TDT | Apache 2.0 (NeMo) | ✅ |
| Canary Qwen | Apache 2.0 | ✅ |
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
```

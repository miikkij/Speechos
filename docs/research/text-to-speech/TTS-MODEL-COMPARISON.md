# Text-to-Speech (TTS) Model Comparison Matrix

> Last updated: 2026-02-27

## Overview

This document compares open-source text-to-speech models that can be downloaded and run locally. Models range from ultra-lightweight CPU-only engines to GPU-powered systems with voice cloning and emotional control.

---

## Model Size & Hardware Requirements

| Model | Parameters | Model Size (Disk) | VRAM (GPU) | RAM (CPU-only) | CPU Support | GPU Required | Real-time Capable |
|---|---|---|---|---|---|---|---|
| **Piper** | ~15-65M (VITS) | 15–65 MB (ONNX) | N/A | ~2 GB | ✅ | ❌ | ✅ (fast) |
| **Kokoro** | 82M | ~200 MB | ~1 GB | ~2 GB | ✅ | ❌ | ✅ |
| **MeloTTS** | ~100M | ~300 MB | ~1 GB | ~2 GB | ✅ | ❌ | ✅ |
| **ChatTTS** | ~300M | ~800 MB | ~2 GB | ~4 GB | ✅ (slow) | Optional | ✅ |
| **Bark** | 1B+ | ~5 GB | ~8 GB | ~12 GB | ⚠️ (very slow) | Recommended | ❌ (slow) |
| **XTTS-v2** (Coqui) | 467M | ~1.8 GB | ~6 GB | ~8 GB | ⚠️ (slow) | Recommended | ⚠️ |
| **Parler TTS** | 880M | ~3.5 GB | ~6 GB | N/A | ❌ | ✅ | ✅ |
| **Orpheus TTS 3B** | 3B (Llama) | ~6 GB | ~8 GB | ~12 GB (llama.cpp) | ✅ (llama.cpp) | Recommended | ✅ |
| **Chatterbox** | ~400M | ~1.5 GB | ~4 GB | ~8 GB | ⚠️ | Recommended | ✅ |
| **Fish Speech 1.5** | ~500M | ~2 GB | ~4 GB | N/A | ❌ | ✅ | ✅ |
| **CosyVoice2 0.5B** | 500M | ~2 GB | ~4 GB | N/A | ❌ | ✅ | ✅ (150ms) |
| **Qwen3-TTS 0.6B** | 600M | ~2.5 GB | ~4 GB | N/A | ❌ | ✅ | ✅ (97ms) |
| **eSpeak-NG** | N/A (rule-based) | ~5 MB | N/A | ~64 MB | ✅ | ❌ | ✅ |
| **Festival** | N/A (rule-based) | ~50 MB | N/A | ~256 MB | ✅ | ❌ | ✅ |

---

## Feature Comparison

| Model | Voice Quality | Voice Cloning | Emotion Control | Multilingual | Streaming | Custom Voices | License |
|---|---|---|---|---|---|---|---|
| **Piper** | ★★★☆☆ Good | ❌ | ❌ | ✅ 30+ langs | ❌ | ✅ (train) | MIT |
| **Kokoro** | ★★★★★ Excellent | ❌ | ⚠️ (limited) | ✅ EN, JA, ZH, KO, FR, DE, IT, ES, PT, HI | ❌ | ✅ (fine-tune) | Apache 2.0 |
| **MeloTTS** | ★★★★☆ Very Good | ❌ | ❌ | ✅ EN, ZH, JA, KO, FR, ES | ❌ | ❌ | MIT |
| **ChatTTS** | ★★★★☆ Very Good | ❌ | ✅ (laugh, pause) | ✅ EN, ZH | ✅ | ✅ (random seeds) | AGPLv3 |
| **Bark** | ★★★★☆ Very Good | ⚠️ (voice prompts) | ✅ (music, effects) | ✅ 16+ langs | ❌ | ⚠️ (speaker prompts) | MIT |
| **XTTS-v2** | ★★★★★ Excellent | ✅ (6s audio) | ❌ | ✅ 17 langs | ✅ | ✅ (clone) | CPML (Non-commercial) |
| **Parler TTS** | ★★★★☆ Very Good | ❌ | ✅ (text prompt) | ✅ EN, FR, DE, ES, + | ❌ | ✅ (describe voice) | Apache 2.0 |
| **Orpheus TTS 3B** | ★★★★★ Excellent | ❌ | ✅ (emotion tags) | ✅ 7+ langs | ✅ | ✅ (fine-tune) | Apache 2.0 |
| **Chatterbox** | ★★★★★ Excellent | ✅ (best in class) | ⚠️ | ✅ EN (primary) | ❌ | ✅ (clone) | MIT |
| **Fish Speech 1.5** | ★★★★★ Excellent | ✅ | ✅ | ✅ Multi | ✅ | ✅ | Apache 2.0 |
| **CosyVoice2 0.5B** | ★★★★★ Excellent | ✅ | ✅ | ✅ ZH, EN, JA, KO | ✅ (150ms) | ✅ | Apache 2.0 |
| **Qwen3-TTS 0.6B** | ★★★★★ Excellent | ✅ (3s audio) | ✅ | ✅ 10 langs | ✅ (97ms) | ✅ | Apache 2.0 |
| **eSpeak-NG** | ★★☆☆☆ Robotic | ❌ | ❌ | ✅ 100+ langs | ❌ | ❌ | GPL v3 |
| **Festival** | ★★☆☆☆ Robotic | ❌ | ❌ | ✅ Limited | ❌ | ❌ | BSD-like |

---

## Speed Comparison

| Model | GPU RTFx | CPU RTFx | Latency (first chunk) | Notes |
|---|---|---|---|---|
| **Piper** | N/A | ~50-100x | <50ms | Fastest CPU TTS available |
| **Kokoro** | ~96x | ~10x | ~100ms | Best quality per parameter |
| **MeloTTS** | ~40x | ~15x | ~100ms | Easy to set up |
| **ChatTTS** | ~20x | ~3x | ~200ms | Good with GPU |
| **Bark** | ~2x | ~0.3x | ~2-5s | Very slow, not for production |
| **XTTS-v2** | ~8x | ~1x | ~500ms | Voice cloning adds overhead |
| **Parler TTS** | ~15x | N/A | ~300ms | GPU required |
| **Orpheus TTS 3B** | ~20x | ~2x (llama.cpp) | ~200ms | Can use llama.cpp for CPU |
| **Chatterbox** | ~12x | ~1.5x | ~300ms | Voice cloning focused |
| **Fish Speech 1.5** | ~15x | N/A | ~200ms | GPU required |
| **CosyVoice2 0.5B** | ~20x | N/A | ~150ms | Lowest streaming latency |
| **Qwen3-TTS 0.6B** | ~25x | N/A | ~97ms | Newest, fastest streaming |
| **eSpeak-NG** | N/A | ~500x+ | <10ms | Rule-based, instant |

> RTFx = How many seconds of audio generated per second of processing.

---

## TTS Arena ELO Rankings (as of early 2026)

| Rank | Model | ELO Score | Notes |
|---|---|---|---|
| 1 | Kokoro 82M | ~1580 | #1 despite tiny size |
| 2 | Chatterbox | ~1502 | Best voice cloning (MIT) |
| 3 | Orpheus TTS 3B | ~1490 | Best emotional control |
| 4 | Fish Speech 1.5 | ~1339 | Strong multilingual |
| 5 | XTTS-v2 | ~1300 | Pioneer in voice cloning |
| 6 | Parler TTS | ~1280 | Descriptive voice control |
| 7 | Bark | ~1200 | Versatile but slow |
| 8 | MeloTTS | ~1150 | Lightweight, easy setup |
| 9 | Piper | ~1100 | Speed champion |

---

## Licensing

| Model | License | Commercial Use |
|---|---|---|
| Piper | MIT | ✅ |
| Kokoro | Apache 2.0 | ✅ |
| MeloTTS | MIT | ✅ |
| ChatTTS | AGPLv3 | ⚠️ (copyleft) |
| Bark | MIT | ✅ |
| XTTS-v2 | CPML | ❌ Non-commercial only |
| Parler TTS | Apache 2.0 | ✅ |
| Orpheus TTS 3B | Apache 2.0 | ✅ |
| Chatterbox | MIT | ✅ |
| Fish Speech 1.5 | Apache 2.0 | ✅ |
| CosyVoice2 | Apache 2.0 | ✅ |
| Qwen3-TTS | Apache 2.0 | ✅ |
| eSpeak-NG | GPL v3 | ⚠️ (copyleft) |

---

## Recommendations for This Project

### Primary: Kokoro (Quality + Efficiency)
- **Why**: #1 TTS Arena ranking, only 82M params, runs on CPU and GPU
- **Use case**: High-quality speech output for testing and demos
- **Install**: `pip install kokoro`

### Secondary: Piper (Speed + Lightweight)
- **Why**: Ultra-fast, ONNX-based, runs on anything, 30+ languages
- **Use case**: Real-time TTS preview, low-latency feedback
- **Install**: `pip install piper-tts`

### For Voice Cloning: Chatterbox or XTTS-v2
- **Chatterbox**: MIT license, best cloning quality
- **XTTS-v2**: More languages, but non-commercial license
- **Install**: `pip install chatterbox-tts` / `pip install TTS`

### For Emotional Speech: Orpheus TTS 3B
- **Why**: Emotion tags (happy, sad, angry), Llama-based, can run via llama.cpp on CPU
- **Use case**: Generating expressive speech with emotion control
- **Install**: Clone from GitHub, use with vLLM or llama.cpp

---

## Python Package Summary

```bash
# Core TTS
pip install kokoro           # Primary - best quality/size ratio
pip install piper-tts        # Ultra-fast CPU fallback

# Voice Cloning
pip install chatterbox-tts   # Best cloning (MIT)
pip install TTS              # Coqui TTS (includes XTTS-v2)

# Additional
pip install bark             # Versatile (music/effects)
pip install melotts          # Lightweight multilingual
```

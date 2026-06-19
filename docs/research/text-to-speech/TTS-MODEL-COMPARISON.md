# Text-to-Speech (TTS) Model Comparison Matrix

> Last updated: 2026-06-19

## Overview

This document compares open-source text-to-speech models that can be downloaded and run locally. Models range from ultra-lightweight CPU-only engines to GPU-powered systems with voice cloning and emotional control.

> **2026-06 refresh (what changed since Feb 2026):** CosyVoice **3** open weights shipped (supersede CosyVoice2); Chatterbox gained a **Multilingual V3** (23+ langs) + a turbo variant; Fish Speech 1.5 was superseded by **OpenAudio S1** (NC license); **Kitten TTS** arrived as a 15M ultra-tiny CPU model alongside Kokoro; **Qwen3-TTS** is confirmed **open-weights (Apache-2.0)**, not API-only; and Microsoft **VibeVoice** (long-form, multi-speaker podcast TTS) is now tracked. Older entries are retained below and marked `⚠ superseded` where a newer drop-in exists. Kokoro 82M v1.0 remains the recommended tiny default (no v2).

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
| **CosyVoice2 0.5B** `⚠ superseded` | 500M | ~2 GB | ~4 GB | N/A | ❌ | ✅ | ✅ (150ms) |
| **CosyVoice3 0.5B** *(2025-12)* | 500M | ~2 GB | ~4 GB | N/A | ❌ | ✅ | ✅ |
| **Qwen3-TTS 0.6B/1.7B** | 600M / 1.7B | ~2.5 / ~4 GB | ~4 / ~6 GB | N/A | ❌ | ✅ | ✅ (97ms) |
| **Kitten TTS nano** *(new)* | 15M | <25 MB | N/A | ~0.5 GB | ✅ | ❌ | ✅ (CPU) |
| **OpenAudio S1-mini** *(succeeds Fish 1.5)* | ~500M | ~2 GB | ~4 GB | N/A | ❌ | ✅ | ✅ |
| **VibeVoice 1.5B** *(long-form, multi-speaker)* | 1.5B | ~5 GB | ~7–8 GB | N/A | ❌ | ✅ | ⚠ (long gen) |
| **VibeVoice 7B / Large** | 7B | ~17 GB | ~24 GB (8 GB w/ 4-bit) | N/A | ❌ | ✅ | ❌ |
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
| **CosyVoice3 0.5B** | ★★★★★ Excellent+ | ✅ | ✅ | ✅ 9+ langs | ✅ | ✅ | Apache 2.0 |
| **Qwen3-TTS 0.6B/1.7B** | ★★★★★ Excellent | ✅ (3s audio) | ✅ | ✅ 10 langs | ✅ (97ms) | ✅ | Apache 2.0 |
| **Kitten TTS nano** | ★★★☆☆ Good | ❌ | ❌ | ✅ EN | ❌ | ❌ | Apache 2.0 |
| **OpenAudio S1-mini** | ★★★★★ Excellent | ✅ | ✅ | ✅ Multi | ✅ | ✅ | CC-BY-NC-SA (NC) |
| **VibeVoice 1.5B/7B** | ★★★★★ Excellent | ✅ (zero-shot ref) | ✅ (emergent) | ✅ EN, ZH (others exp.) | ⚠ (0.5B only) | ✅ (4 speakers) | MIT (community fork) |
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
| CosyVoice2 / CosyVoice3 | Apache 2.0 | ✅ |
| Qwen3-TTS | Apache 2.0 | ✅ (open weights, confirmed) |
| Kitten TTS | Apache 2.0 | ✅ |
| OpenAudio S1-mini | CC-BY-NC-SA 4.0 | ❌ Non-commercial only |
| VibeVoice | MIT (code+weights, via community fork) | ✅ legally; model card asks "research only" — treat as advisory compliance flag |
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

### For Long-form / Multi-speaker (Podcast): VibeVoice 1.5B
- **Why**: Generates a single continuous take with up to 4 distinct speakers, stable voice identity and natural turn-taking over very long spans (~90 min at 1.5B). Uses ultra-low-frame-rate (7.5 Hz) tokenizers + LLM + diffusion head.
- **Use case**: Synthetic podcasts, dialogues, audiobooks — not single-sentence latency-sensitive playback.
- **Variants**: `VibeVoice-1.5B` (~7–8 GB VRAM), `VibeVoice-7B`/`-Large` (~24 GB, or ~8 GB 4-bit), `VibeVoice-Realtime-0.5B` (streaming, single-speaker).
- **License/status**: MIT. Microsoft pulled the original code (~Sep 2025); the canonical source is now the community fork `vibevoice-community/VibeVoice`; weights remain on HF. English + Chinese reliable, other languages experimental. **GPU-only** (assumes CUDA + flash-attn). Output is 24 kHz.
- **Integration in Speechos**: runs as a Docker TTS engine (`vibevoice`) via the community `vibevoice-api` OpenAI-compatible server — see `docker/tts-engines.yml`.

### Alternatives in the long-form space (consider)
- **Nari Labs Dia 1.6B** — Apache-2.0, dialogue + nonverbals + zero-shot cloning, single-GPU realtime (cleanest license substitute for VibeVoice).
- **Boson AI Higgs Audio v3 (4B)** — ~100 languages, 21 emotion tokens.
- **Fish Audio OpenAudio S1** — strong quality but NC license.

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

# New in 2026
pip install kittentts        # 15M ultra-tiny CPU TTS (Apache-2.0)
# CosyVoice3 0.5B  → FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (HF, Apache-2.0)
# Chatterbox ML V3 → ResembleAI/chatterbox via pip install chatterbox-tts (23+ langs, MIT)
# OpenAudio S1-mini → fishaudio/openaudio-s1-mini (HF, CC-BY-NC-SA, non-commercial)
# Qwen3-TTS open weights → Qwen/Qwen3-TTS-12Hz-1.7B-Base (+0.6B) (HF, Apache-2.0)
# VibeVoice (long-form/multi-speaker) → vibevoice-community/VibeVoice; server: pip install vibevoice-api
```

> **Model IDs (2026):** CosyVoice3 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` · OpenAudio `fishaudio/openaudio-s1-mini` · Kitten `KittenML/kitten-tts-nano-0.8-fp32` · Qwen3-TTS `Qwen/Qwen3-TTS-12Hz-1.7B-Base` · VibeVoice `vibevoice/VibeVoice-1.5B` (mirror of `microsoft/VibeVoice-1.5B`).

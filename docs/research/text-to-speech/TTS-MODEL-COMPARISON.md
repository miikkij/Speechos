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

## Finnish (fi) — measured benchmark 2026-10-09

Local measurement with `api/bench/tts_fi.py` (RTX 4090, Windows 11). Full tables, per-sentence
transcripts and revisions: `api/bench/results/tts_fi_results_v2.{md,json}`; decision write-up:
`docs/plans/2026-10-public/report-b.md`.

**Method.** 38 Finnish sentences (30 varied + 8 short conversational turns), every engine run with
3 seeds. Round-trip intelligibility: TTS → faster-whisper large-v3 (`Systran/faster-whisper-large-v3@edaa852`)
→ CER/WER on normalized text; mean ± sd across runs = stability. "catastr." = sentences with CER > 30 %.
TTFA = time to first audio via the engine API (= full synthesis for non-streaming engines).
UTMOS22-strong / UTMOSv2 are English/Japanese-trained naturalness predictors — relative only.

| Engine (revision) | CER mean ± sd | WER | short-turn CER | catastr./run | RTF | TTFA | UTMOS22 | License |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| edge-tts fi-FI-NooraNeural (cloud, reference) | 1.95 % ± 0.04 | 3.4 % | 0.0 % | 0 | 0.20 | 0.50 s | 3.88 | MS online service |
| edge-tts fi-FI-HarriNeural (cloud, reference) | 2.17 % ± 0.04 | 4.1 % | 0.0 % | 0 | 0.28 | 0.63 s | 4.00 | MS online service |
| **Chatterbox-Finnish cp986 + fi reference voice** (`Finnish-NLP/Chatterbox-Finnish@d15775e`) | **3.29 % ± 0.26** | 8.2 % | 4.3 % | 1.0 | 1.28 | 4.6 s | 3.72 | MIT |
| ↳ same, T3 decode as CUDA graph (`api/src/engines_fi/chatterbox_graph.py`) | 3.29 % ± 0.26 (bit-identical) | 8.2 % | 4.3 % | 1.0 | **0.43** | **1.5 s** | 3.72 | MIT |
| ↳ same, CUDA graph + bf16 backbone | 2.73 % ± 0.40 | 7.9 % | 2.8 % | 0.3 | 0.32 | 1.2 s | 3.71 | MIT |
| MMS-TTS fin (`facebook/mms-tts-fin@0b8a1a2`) | 3.41 % ± 0.78 | 13.9 % | 2.8 % | 0 | 0.02 | 0.07 s | 3.31 | CC-BY-NC-4.0 |
| Chatterbox Multilingual + fi reference (`ResembleAI/chatterbox@5bb1f6e`, t3_mtl23ls_v2) | 3.95 % ± 1.21 | 6.7 % | 0.9 % | 1.0 | 1.00 | 4.4 s | 3.46 | MIT |
| **Piper fi_FI-harri-medium** (current default) | 4.65 % ± 1.33 | 11.5 % | 15.0 % | 1.7 | 0.04 | 0.11 s | 3.38 | GPL-3.0 engine; voice CC0 data (from lessac) |
| Chatterbox-Finnish cp1060 "naturalness" + fi ref | 5.14 % ± 2.49 | 8.9 % | 24.5 % | 1.3 | 0.83 | 3.0 s | 3.65 | MIT |
| Chatterbox-Finnish cp986, default (English) voice | 5.34 % ± 1.33 | 13.6 % | 9.5 % | 2.0 | 1.01 | 2.3 s | 3.71 | MIT |
| eSpeak-NG 1.52 | 9.94 % ± 0.00 | 31.4 % | 3.7 % | 3.0 | 0.02 | 0.04 s | 1.77 | GPL-3.0 |
| Chatterbox Multilingual, default voice | 16.89 % ± 8.74 | 23.5 % | 59.9 % | 4.3 | 1.09 | 5.4 s | 3.06 | MIT |
| CSM-1B FIN parliament fine-tune (1 run, 10 sentences) | 55.7 % | 65.4 % | 281 % | 5/10 | 4.45 | 44.7 s | 2.52 | weights Apache-2.0, data CC-BY-NC-ND |
| ZipVoice-Suomi-12k | not run — needs k2 (no Windows build) | | | | | | | CC-BY-ND-4.0 |

**Findings.**
- **Chatterbox-Finnish** is the best *local* engine on intelligibility and the most stable autoregressive
  one (sd 0.26 pp vs 1.2 pp for the base multilingual model), and UTMOS ranks it closest to the cloud voices.
  It needs a Finnish reference voice: with the built-in (English) voice CER rises to 5.3 % ± 1.3.
  Its remaining errors are mostly digits ("12 A", "24."), not babble. **But it is slower than real time on
  an RTX 4090 (RTF ≈ 1.3, ~4.6 s to first audio)** with the card configuration (eager attention, CFG,
  alignment analyzer). Profile: T3 token generation is 84 % of the time (30.9 tokens/s; 25 tokens = 1 s audio),
  bound by Python/kernel-launch overhead. **Capturing the T3 decode step as a CUDA graph** (static KV cache,
  static 4D mask, eager attention only in the analyzer's layers 9/12/13) gives bit-identical audio at RTF 0.43
  and ~1.5 s to first audio (78 tokens/s); a bf16 backbone reaches RTF 0.32. Module:
  `api/src/engines_fi/chatterbox_fi.py`.
- **Piper-fi is not as stable as the single June run suggested** (3.5 %): CER 3.2–5.9 % across runs, and
  it fails on very short turns — "Joo, ymmärrän." comes out as ~6 s of babble in every run.
- The base Chatterbox Multilingual default voice remains unusable for Finnish (catastrophic in 4.3
  sentences/run); with a Finnish reference voice it becomes reasonable.
- MMS-TTS is intelligible and fast but non-commercial — evaluation only.
- Number/abbreviation reading is the common weak spot of all local engines. A Finnish text normalizer
  (`api/src/engines_fi/text_norm_fi.py`, num2words with case heuristics) in front of TTS fixes it: on 10
  number/abbreviation sentences × 3 seeds the share of key items read correctly rises from 40 % to 99 % for
  Chatterbox-Finnish and from 80 % to 91 % for Piper (`api/bench/results/tts_fi_number_items.md`).
- Packaging: PyPI `chatterbox-tts` 0.1.7 pins `torch==2.6` / `transformers==5.2` and cannot share the
  Speechos lock (torch 2.10, transformers 4.57). The bench runs both Chatterbox variants on the code
  bundled in the Chatterbox-Finnish repo (MIT) with eager attention (needed by its alignment analyzer
  on transformers ≥ 4.47).

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

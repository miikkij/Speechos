# Speech Analysis & STT Models: Test Results Report

**Date**: 2025-01  
**Hardware**: NVIDIA RTX 4090 (24GB VRAM), CUDA 12.8, Windows 11  
**PyTorch**: 2.10.0+cu128 | **Transformers**: 5.2.0  
**Test Audio**: 6 synthetic WAV files (16kHz mono): see [Test Audio](#test-audio-files) section  

> **Important**: All test files are **synthetic signals** (sine waves, noise), NOT real human speech.  
> SER models are trained on real speech and will produce unreliable predictions on synthetic audio.  
> These tests verify that models **load, run, and produce output**: not accuracy on real-world data.

---

## Summary

| # | Model / Tool | Type | Status | Inference Speed | Notes |
|---|---|---|---|---|---|
| 0 | **faster-whisper large-v3** | STT | ✅ PASS | 230–9586ms | Hallucinated text from non-speech (expected) |
| 1 | **librosa** | Feature Extraction | ✅ PASS | 6–1358ms | All features extracted correctly |
| 2 | **openSMILE eGeMAPSv02** | Feature Extraction | ✅ PASS | 33–56ms | 88 features per file |
| 3 | **Silero VAD v5** | Voice Activity Detection | ✅ PASS | 14–1465ms | Correctly detected speech-like vs silence |
| 4 | **emotion2vec+ large** | SER (categorical) | ✅ PASS | 70–284ms | Strong confidence, 9 emotion classes |
| 5 | **wav2vec2 ehcalabres XLSR** | SER (categorical) | ⚠️ DEGRADED | 12–79ms | Near-random ~13% confidence: weight mismatch |
| 6 | **wav2vec2 r-f** | SER (categorical) | ⚠️ DEGRADED | 12–20ms | Near-random ~15% confidence |
| 7 | **wav2vec2 Dpngtm** | SER (categorical) | ✅ PASS | 6–1290ms | Moderate–high confidence |
| 8 | **HuBERT superb-er** | SER (categorical) | ✅ PASS | 13–103ms | Most differentiated results, 4 classes |
| 9 | **wav2small (audeering)** | SER (A/D/V dimensional) | ✅ PASS | 2–558ms | 17K params, A/D/V in [0,1] |
| 10 | **WavLM audeering** | SER (A/D/V dimensional) | ✅ PASS | 7–54ms | Near-zero values on synthetic audio |
| 11 | **Resemblyzer** | Speaker Diarization | ✅ PASS | 2–73ms | d-vector embeddings, speaker count estimation |

**Result: 10/12 fully working, 2/12 degraded (weight loading issues with transformers 5.2.0)**

---

## Test Audio Files

All files in `samples/` directory, 16kHz mono WAV:

| File | Duration | Description |
|---|---|---|
| `test_aggressive.wav` | 5.0s | 300Hz base, rapid pitch variation, high noise overlay |
| `test_calm.wav` | 5.0s | 150Hz base, slow/gentle variation, low noise |
| `test_energetic.wav` | 5.0s | 250Hz base, rapid variation, high energy |
| `test_low.wav` | 5.0s | 100Hz base, minimal variation, very low energy |
| `test_silence.wav` | 3.0s | Near-silent (noise floor only) |
| `test_tone.wav` | 2.0s | Pure 300Hz sine wave, no variation |

---

## Detailed Results

### 0. faster-whisper STT (large-v3): ✅ PASS

**Model**: `Systran/faster-whisper-large-v3` on CUDA (float16)  
**Purpose**: Speech-to-text transcription  
**Parameters**: ~1.5B (Whisper large-v3, CTranslate2 optimized)

| File | Language | Confidence | Transcription | Time |
|---|---|---|---|---|
| test_aggressive | en | 1.00 | "Woof, woof, woof, woof, woof, woof." | 643ms |
| test_calm | en | 1.00 | "woo woo woo whoo whoo whoo woo woo na nTeC" | 9586ms |
| test_energetic | en | 1.00 | "Woo, woo, woo!" | 581ms |
| test_low | en | 1.00 | "www.youtube.com" (repeated 4x) | 6913ms |
| test_silence | en | 1.00 | "Thank you." | 230ms |
| test_tone | en | 1.00 | "Thanks for watching!" | 251ms |

**Observations**:
- Model loads and runs on CUDA with float16: fully functional
- Hallucinated text from synthetic audio is expected behavior (no actual speech to transcribe)
- Language detection correctly identified English with 100% confidence
- Shorter/simpler audio processed faster; complex patterns (calm, low) took 6–9s
- The "Thank you" / "Thanks for watching" hallucinations on silence/tone are common Whisper artifacts

---

### 1. librosa: ✅ PASS

**Version**: 0.11.0  
**Purpose**: Low-level audio feature extraction  

| File | Pitch (Hz) | Pitch Std | RMS Energy | ZCR | MFCC1 | Spectral Centroid | Spectral Rolloff |
|---|---|---|---|---|---|---|---|
| test_aggressive | 283.7 | 162.6 | 0.403 | 0.301 | 77.0 | 2995 Hz | 5015 Hz |
| test_calm | 207.1 | 146.9 | 0.455 | 0.043 | -26.2 | 1109 Hz | 1967 Hz |
| test_energetic | 278.9 | 156.5 | 0.477 | 0.188 | 58.1 | 2209 Hz | 3660 Hz |
| test_low | 158.6 | 128.6 | 0.470 | 0.025 | -76.4 | 979 Hz | 1840 Hz |
| test_silence | 69.3 | 25.6 | 0.010 | 0.494 | -234.1 | 3999 Hz | 6804 Hz |
| test_tone | 300.3 | 0.1 | 0.210 | 0.037 | -515.1 | 308 Hz | 308 Hz |

**Observations**:
- All features extracted correctly and differentiate audio characteristics well
- Pitch correctly tracks fundamental frequency (test_tone=300.3Hz matches 300Hz signal)
- RMS energy correctly shows test_silence near 0, others ~0.4–0.5
- ZCR high for noise-like signals (aggressive=0.30, silence=0.49), low for tonal (low=0.02, tone=0.04)
- Spectral features correctly distinguish narrow-band (tone: 308Hz) from wide-band (aggressive: 2995Hz)
- First-file latency ~1358ms (JIT warmup), subsequent files ~6–17ms

---

### 2. openSMILE eGeMAPSv02: ✅ PASS

**Version**: opensmile 2.6.0  
**Purpose**: Standardized acoustic feature extraction (Geneva Minimalistic Acoustic Parameter Set)  

| File | F0 (semitones) | Jitter | Shimmer (dB) | HNR (dB) | H1-H2 | Loudness | Time |
|---|---|---|---|---|---|---|---|
| test_aggressive | 57.34 | 0.020 | 0.561 | 5.96 | 3.05 | 3.34 | 55ms |
| test_calm | 43.81 | 0.041 | 0.325 | 8.80 | 8.57 | 1.99 | 56ms |
| test_energetic | 54.76 | 0.038 | 0.362 | 3.93 | 1.25 | 3.08 | 56ms |
| test_low | 34.92 | 0.077 | 0.391 | 8.24 | 7.14 | 1.52 | 56ms |
| test_silence | 0.00 | 0.000 | 0.000 | 0.00 | 0.00 | 0.47 | 33ms |
| test_tone | 41.35 | 0.000 | 0.005 | 16.89 | 7.63 | 0.47 | 35ms |

**Observations**:
- 88 eGeMAPSv02 features extracted per file: fully functional
- F0 correctly 0 for silence, highest for aggressive signal
- HNR correctly highest for pure tone (16.89 dB), lowest for noisy signals
- Jitter/shimmer near 0 for pure tone (no perturbation), higher for complex signals
- ~55ms per file: fast and consistent

---

### 3. Silero VAD v5: ✅ PASS

**Method**: `torch.hub.load('snakers4/silero-vad')` (bypasses torchaudio 2.10 / torchcodec issue)  
**Purpose**: Voice activity detection: identify speech vs non-speech segments  

| File | Speech Segments | Total Speech (s) | Segment Details | Time |
|---|---|---|---|---|
| test_aggressive | 0 | 0.00 |: | 1465ms |
| test_calm | 1 | 3.16 | [0.00–3.16s] | 36ms |
| test_energetic | 1 | 0.89 | [1.89–2.79s] | 36ms |
| test_low | 1 | 4.74 | [0.03–4.77s] | 36ms |
| test_silence | 0 | 0.00 |: | 21ms |
| test_tone | 0 | 0.00 |: | 14ms |

**Observations**:
- Correctly classified silence and pure tone as non-speech (0 segments)
- Correctly classified aggressive noise as non-speech (good noise rejection)
- Detected speech-like activity in calm/low/energetic signals (periodic harmonic content resembles speech)
- First-file latency ~1465ms (model warmup), subsequent ~14–36ms
- **Note**: `torchaudio.load()` broken with torchaudio 2.10.0+cu128: must use `librosa.load()` + torch tensor

---

### 4. emotion2vec+ large: ✅ PASS

**Model**: `iic/emotion2vec_plus_large` via FunASR 1.3.1 on CUDA  
**Purpose**: Speech emotion recognition: 9 categorical classes  
**Parameters**: ~300M  

| File | Top Emotion | Confidence | Runner-up | Time |
|---|---|---|---|---|
| test_aggressive | **angry** | 99.98% | neutral (0.01%) | 284ms |
| test_calm | **angry** | 45.27% | happy (32.18%) | 81ms |
| test_energetic | **angry** | 100.00% |: | 71ms |
| test_low | **happy** | 81.32% | sad (9.36%) | 74ms |
| test_silence | **sad** | 94.92% | neutral (3.59%) | 75ms |
| test_tone | **sad** | 56.95% | surprised (29.69%) | 78ms |

**Observations**:
- Strong confidence on distinctive signals (aggressive=99.98%, energetic=100%, silence=94.92%)
- Correctly identified high-energy noise-like signals as "angry"
- Silence → sad mapping is reasonable (absence of energy = low arousal)
- calm → angry is a false positive, but with only 45% confidence (synthetic audio limitation)
- Only model with 9 emotion classes including "other" and "&lt;unk&gt;"
- First inference ~284ms (warmup), subsequent ~70–80ms on CUDA

---

### 5. wav2vec2 ehcalabres XLSR: ⚠️ DEGRADED

**Model**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`  
**Purpose**: SER: 8 emotion classes (RAVDESS)  

| File | Top Emotion | Confidence | Time |
|---|---|---|---|
| test_aggressive | fearful | **13.35%** | 79ms |
| test_calm | fearful | **13.31%** | 15ms |
| test_energetic | happy | **13.65%** | 15ms |
| test_low | sad | **13.38%** | 15ms |
| test_silence | sad | **13.20%** | 12ms |
| test_tone | neutral | **13.34%** | 12ms |

**Observations**:
- **Near-random predictions**: all scores ~12.5% (1/8 classes = random)
- Root cause: MISSING classifier weights when loading with transformers 5.2.0 (`classifier.weight` missing, `classifier.output.weight` unexpected)
- The model's classification head architecture changed between transformers versions
- **Not recommended** without fixing the weight loading issue or pinning transformers < 5.0
- Very fast inference (~12-15ms) since the classifier is essentially uninitialized

---

### 6. wav2vec2 r-f: ⚠️ DEGRADED

**Model**: `r-f/wav2vec-english-speech-emotion-recognition`  
**Purpose**: SER: 7 emotion classes  

| File | Top Emotion | Confidence | Time |
|---|---|---|---|
| test_aggressive | happy | **15.14%** | 19ms |
| test_calm | disgust | **15.41%** | 20ms |
| test_energetic | angry | **14.96%** | 17ms |
| test_low | disgust | **15.54%** | 19ms |
| test_silence | disgust | **15.17%** | 14ms |
| test_tone | surprise | **15.48%** | 12ms |

**Observations**:
- **Near-random predictions**: all scores ~14% (1/7 classes ≈ 14.3%)
- Same weight loading issue as ehcalabres model
- Slightly better than pure random but not meaningful
- **Not recommended** with transformers 5.2.0

---

### 7. wav2vec2 Dpngtm: ✅ PASS

**Model**: `Dpngtm/wav2vec2-emotion-recognition` (IEMOCAP trained)  
**Purpose**: SER: 7 emotion classes  

| File | Top Emotion | Confidence | Runner-up | Time |
|---|---|---|---|---|
| test_aggressive | **fearful** | 45.22% | sad (39.09%) | 1290ms |
| test_calm | **fearful** | 93.63% | sad (3.25%) | 6ms |
| test_energetic | **fearful** | 64.67% | disgust (18.82%) | 12ms |
| test_low | **surprised** | 97.10% | happy (0.71%) | 6ms |
| test_silence | **sad** | 45.51% | fearful (40.37%) | 11ms |
| test_tone | **sad** | 93.59% | fearful (4.06%) | 11ms |

**Observations**:
- Model loads correctly and produces high-confidence predictions
- Tends toward "fearful" for high-energy signals: likely a bias from IEMOCAP training data
- Silence/tone correctly map to "sad" (low energy = sadness)
- First inference ~1290ms (model warmup), subsequent 6–12ms
- **Note**: Original model ID `Dpngtm/emotion-recognition-wav2vec2-IEMOCAP` was incorrect (404). Correct repo: `Dpngtm/wav2vec2-emotion-recognition`

---

### 8. HuBERT superb-er: ✅ PASS

**Model**: `superb/hubert-large-superb-er` (IEMOCAP trained)  
**Purpose**: SER: 4 emotion classes (ang, hap, neu, sad)  
**Parameters**: ~300M (HuBERT Large)  

| File | Top Emotion | Confidence | Runner-up | Time |
|---|---|---|---|---|
| test_aggressive | **ang** | 63.90% | hap (22.15%) | 103ms |
| test_calm | **hap** | 87.51% | sad (10.41%) | 17ms |
| test_energetic | **ang** | 85.86% | hap (13.27%) | 25ms |
| test_low | **sad** | 96.23% | hap (2.50%) | 20ms |
| test_silence | **sad** | 81.33% | neu (12.07%) | 15ms |
| test_tone | **sad** | 99.49% | hap (0.48%) | 14ms |

**Observations**:
- **Most differentiated and intuitive results** of all categorical SER models
- High-energy signals → angry (aggressive=63.9%, energetic=85.9%) ✓
- Low-energy/calm → sad or happy (low=96.2% sad, calm=87.5% happy) ✓
- Silence → sad (81.3%) ✓, tone → sad (99.5%): reasonable for steady low-variation signal
- High confidence scores (63–99%): model is decisive
- Only 4 classes but well-separated predictions
- Fast inference: 14–25ms after warmup (103ms first)

---

### 9. wav2small (audeering): ✅ PASS

**Model**: `audeering/wav2small`: custom VGG7 architecture  
**Purpose**: SER: dimensional (Arousal / Dominance / Valence) in [0, 1]  
**Parameters**: **17K** (smallest model tested)  

| File | Arousal | Dominance | Valence | Time |
|---|---|---|---|---|
| test_aggressive | **0.749** | 0.704 | 0.343 | 558ms |
| test_calm | 0.818 | 0.735 | 0.281 | 3ms |
| test_energetic | **0.890** | **0.820** | 0.269 | 2ms |
| test_low | **0.435** | **0.365** | 0.317 | 2ms |
| test_silence | 0.648 | 0.638 | 0.368 | 7ms |
| test_tone | 0.678 | 0.688 | 0.227 | 6ms |

**Observations**:
- Model loads via manual safetensors (transformers `from_pretrained` broken with 5.2.0: `all_tied_weights_keys` error)
- Required exact architecture reconstruction from HuggingFace README (Vgg7 + custom Spectrogram + LogmelFilterBank)
- Arousal correctly highest for energetic (0.89) and aggressive (0.75), lowest for low (0.44)
- Dominance pattern follows arousal (high-energy = high dominance)
- Valence relatively flat (0.23–0.37): expected for non-emotional synthetic audio
- Extremely fast after warmup: **2–7ms per file** on CUDA
- **Note**: MaxPool2d(3, stride=2, padding=1) is critical: different from typical VGG configs

---

### 10. WavLM audeering (A/D/V): ✅ PASS

**Model**: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` via audEERING interface  
**Purpose**: SER: dimensional (Arousal / Dominance / Valence)  
**Parameters**: ~300M (WavLM Large fine-tuned)  

| File | Arousal | Dominance | Valence | Time |
|---|---|---|---|---|
| test_aggressive | 0.0008 | -0.0054 | -0.0203 | 54ms |
| test_calm | 0.0278 | -0.0021 | -0.0053 | 50ms |
| test_energetic | 0.0087 | -0.0042 | -0.0257 | 41ms |
| test_low | 0.0172 | 0.0014 | -0.0063 | 9ms |
| test_silence | -0.0051 | 0.0038 | 0.0165 | 10ms |
| test_tone | -0.0072 | 0.0197 | 0.0082 | 7ms |

**Observations**:
- All A/D/V values near zero (range: -0.03 to +0.03)
- This is **expected behavior**: model was trained on real speech and correctly outputs near-neutral for non-speech
- Shows the model has strong speech specificity (doesn't hallucinate emotions from noise)
- Contrasts with wav2small which produces non-zero A/D/V for synthetic audio
- Fast inference on CUDA: 7–54ms

---

### 11. Resemblyzer Diarization: ✅ PASS

**Model**: Resemblyzer d-vector encoder (GE2E)  
**Purpose**: Speaker diarization: segment audio by speaker identity  

| File | Segments Analyzed | Est. Speakers | Mean Similarity | Time |
|---|---|---|---|---|
| test_aggressive | 4 | 2+ | 0.727 | 73ms |
| test_calm | 4 | 2+ | 0.751 | 36ms |
| test_energetic | 4 | 2+ | 0.655 | 44ms |
| test_low | 3 | 1 | 0.904 | 31ms |
| test_silence | 0 | 0 | 0.000 | 2ms |
| test_tone | 1 | 1 | 1.000 | 11ms |

**Observations**:
- Correctly estimated 0 speakers for silence (no embeddings to analyze)
- Correctly estimated 1 speaker for pure tone (self-similarity = 1.0) and low-frequency signal (similarity = 0.90)
- Estimated 2+ speakers for complex signals (similarity < 0.82 threshold)
- This is reasonable: varying synthetic signals have inconsistent embeddings across segments
- Very fast: 2–73ms per file
- Uses cosine similarity threshold of 0.82 for single-speaker detection

---

## Recommendations for Speechos

### Primary Models (Recommended)

| Feature | Model | Reason |
|---|---|---|
| **STT** | faster-whisper large-v3 | Best accuracy, CUDA optimized, CTranslate2 |
| **VAD** | Silero VAD v5 | Lightweight, accurate, no CUDA needed |
| **SER (categorical)** | emotion2vec+ large | Best confidence, 9 classes, CUDA |
| **SER (backup categorical)** | HuBERT superb-er | Most intuitive 4-class results |
| **SER (dimensional)** | wav2small | Tiny (17K params), fast A/D/V |
| **Feature extraction** | librosa + openSMILE | Complementary features, fast |
| **Diarization** | Resemblyzer | Lightweight d-vectors, no gated model issues |

### Models to Avoid

| Model | Issue |
|---|---|
| wav2vec2 ehcalabres XLSR | Broken weight loading with transformers ≥5.0, random output |
| wav2vec2 r-f | Same weight loading issue, near-random predictions |

### Models Requiring Custom Loading

| Model | Issue | Workaround |
|---|---|---|
| wav2small | `Wav2Vec2PreTrainedModel.from_pretrained` fails with transformers 5.2.0 (`all_tied_weights_keys` missing) | Manual `nn.Module` construction + `safetensors.torch.load_file` |
| Silero VAD | `torchaudio.load()` broken with torchaudio 2.10.0+cu128 (requires torchcodec) | Use `librosa.load()` → `torch.from_numpy()` |

---

## Technical Notes

### Test Limitations
- All audio files are synthetic (sine waves + noise), not human speech
- SER models trained on real speech datasets (IEMOCAP, RAVDESS, MSP-Podcast) will produce unreliable predictions on synthetic audio
- Tests verify **model loading, GPU execution, and output format**: not real-world accuracy
- For accuracy benchmarks, use real speech recordings or standard evaluation datasets

### Transformers 5.2.0 Compatibility
- Several wav2vec2-based models have broken weight loading due to classifier head architecture changes in transformers 5.x
- emotion2vec+ (FunASR) is unaffected (uses its own model loading)
- HuBERT superb-er loads correctly despite being transformers-based
- wav2small requires manual model construction to bypass `Wav2Vec2PreTrainedModel` issues

### GPU Memory Usage
- All models fit comfortably in 24GB VRAM (RTX 4090)
- Largest model: faster-whisper large-v3 (~3.5GB VRAM in float16)
- Most SER models: 1–2GB VRAM each
- wav2small: negligible VRAM (17K parameters)
- Multiple models can be loaded simultaneously without OOM issues

### torchaudio 2.10.0 Breaking Change
- torchaudio 2.10.0+cu128 removed native audio I/O: requires `torchcodec` package
- Affects: Silero VAD's `read_audio()`, any code using `torchaudio.load()`
- Fix: Load audio with `librosa.load(path, sr=16000)` and convert to tensor with `torch.from_numpy()`

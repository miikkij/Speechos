# Speech Analysis & Emotion Recognition Model Comparison

> Last updated: 2026-02-27

## Overview

Beyond transcription, speech audio can be analyzed for emotion, sentiment, speaker characteristics, and more. This document covers models for analyzing `.wav` files to extract non-textual insights.

---

## Speech Emotion Recognition (SER) Models

| Model | Parameters | VRAM/RAM | Emotions Detected | Accuracy | Languages | CPU Support | License |
|---|---|---|---|---|---|---|---|
| **emotion2vec+ large** | ~300M | ~2 GB VRAM / ~4 GB RAM | 9 (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown) | State-of-the-art | Multilingual | ✅ (slow) | Custom (research) |
| **emotion2vec+ base** | ~90M | ~1 GB VRAM / ~2 GB RAM | 9 emotions | Very good | Multilingual | ✅ | Custom (research) |
| **emotion2vec+ seed** | ~90M | ~1 GB VRAM / ~2 GB RAM | 9 emotions | Good | Multilingual | ✅ | Custom (research) |
| **wav2vec2-lg-xlsr SER** (ehcalabres) | ~317M | ~2 GB VRAM / ~4 GB RAM | 8 (angry, calm, disgust, fearful, happy, neutral, sad, surprised) | 82.2% | English | ✅ | Apache 2.0 |
| **wav2vec2 emotion** (Dpngtm) | ~95M | ~1 GB VRAM / ~2 GB RAM | 7 (angry, disgust, fear, happy, neutral, sad, surprise) | 79.9% | English | ✅ | MIT |
| **wav2vec2 SER** (r-f) | ~317M | ~2 GB VRAM / ~4 GB RAM | 7 emotions | 97.5% | English | ✅ | Apache 2.0 |
| **HuBERT SER** (various) | ~95-317M | ~1-2 GB VRAM | 7-8 emotions | 80-85% | Various | ✅ | MIT/Apache |
| **WavLM SER** (3loi) | ~317M | ~2 GB VRAM | Multi-attribute (arousal, valence, dominance) | Good | English | ✅ | MIT |
| **wav2small** (audeering) | ~10M | <1 GB | Arousal, Dominance, Valence | Good | Multilingual | ✅ | CC BY-NC-SA 4.0 |

### Key Notes:
- **emotion2vec+** is the "Whisper of SER": best general-purpose model, works across languages
- **wav2vec2-based models** are fine-tuned on RAVDESS, TESS, CREMA-D, SAVEE datasets
- All models require **16kHz mono WAV** input
- Models classify per-utterance; chunk long files into 3-10 second segments

---

## Speaker Analysis Models

| Model/Tool | Task | Parameters | CPU Support | Notes |
|---|---|---|---|---|
| **PyAnnote** (pyannote/speaker-diarization-3.1) | Speaker diarization | ~50M | ✅ | "Who spoke when": segments audio by speaker |
| **PyAnnote VAD** | Voice Activity Detection | ~5M | ✅ | Detects speech vs silence |
| **SpeechBrain** (spkrec-ecapa-voxceleb) | Speaker verification/ID | ~15M | ✅ | Identify/verify speaker identity |
| **Silero VAD** | Voice Activity Detection | ~2M | ✅ | Lightweight VAD, 100+ languages |
| **Wav2Vec 2.0 XLSR** | Gender classification | ~317M | ✅ | 100% accuracy on TIMIT (gender) |
| **Wav2Vec 2.0** | Age estimation | ~317M | ✅ | ~7.1 years MAE |

---

## Audio Feature Extraction Libraries

| Library | Features | Use Case |
|---|---|---|
| **librosa** | MFCCs, spectrograms, tempo, pitch, chroma | General audio analysis |
| **pyAudioAnalysis** | ZCR, energy, spectral features, emotion, speaker diarization | Full analysis pipeline |
| **torchaudio** | Spectrograms, MFCCs, resampling, augmentation | PyTorch integration |
| **openSMILE** | 6,373 acoustic features (eGeMAPS, ComParE) | Emotion/paralinguistic analysis |
| **VANPY** | Comprehensive: VAD, diarization, emotion, features | All-in-one framework |

---

## Combined Analysis Pipeline (Recommended)

For the Speechos project, a **wav analysis pipeline** should include:

```
.wav file
├── 1. Preprocessing
│   ├── Resample to 16kHz mono
│   ├── Silero VAD → detect speech segments
│   └── Normalize audio levels
│
├── 2. Transcription (STT)
│   └── Faster-Whisper → text + timestamps
│
├── 3. Emotion Recognition
│   └── emotion2vec+ base → per-segment emotions
│
├── 4. Speaker Analysis
│   ├── PyAnnote → speaker diarization
│   └── SpeechBrain → speaker identification
│
├── 5. Audio Features
│   ├── librosa → pitch, tempo, energy
│   └── openSMILE → detailed acoustic features
│
└── 6. Text Analysis (on transcription)
    ├── Sentiment analysis (positive/negative/neutral)
    ├── Key phrase extraction
    └── Topic detection
```

---

## Recommendations for This Project

### Emotion Recognition: emotion2vec+ base
- **Why**: Best accuracy, multilingual, ~90M params, CPU capable
- **Install**: `pip install funasr modelscope`

### Speaker Diarization: PyAnnote
- **Why**: State-of-the-art, easy API, well-maintained
- **Install**: `pip install pyannote.audio`

### Voice Activity Detection: Silero VAD
- **Why**: Tiny (2M params), fast, 100+ languages
- **Install**: `pip install silero-vad` or via `torch.hub`

### Audio Features: librosa + torchaudio
- **Why**: Standard tools, well-documented, cover all basics
- **Install**: `pip install librosa torchaudio`

---

## Python Package Summary

```bash
# Emotion Recognition
pip install funasr modelscope        # emotion2vec+
pip install transformers             # wav2vec2 SER models

# Speaker Analysis
pip install pyannote.audio           # Diarization
pip install speechbrain              # Speaker ID

# VAD
pip install silero-vad               # Lightweight VAD

# Audio Features
pip install librosa                  # Audio analysis
pip install opensmile                # Acoustic features
pip install torchaudio               # PyTorch audio
```

---

## Real-World Reliability & Community Feedback

> Research compiled: 2026-02-27. Based on published papers, HuggingFace community discussions, GitHub issues, and independent benchmarks.

### Speech Emotion Recognition (SER) Models

#### emotion2vec+ (large / base / seed)

**Reliability: HIGH (best available open-source SER)**

- **Strengths**:
  - Trained on 40k+ hours of pseudo-labeled speech data: by far the largest SER training corpus
  - Consistently outperforms all other open-source SER models on EmoBox benchmarks (4-class primary emotions)
  - Truly multilingual: works across languages without fine-tuning, unlike wav2vec2-based models that are English-only
  - Self-supervised pre-training (emotion2vec base) + iterative data engineering pipeline (seed → base → large)
  - Used as the de facto evaluation metric by other TTS/SER researchers (e.g., EmoVoice, EmoSphere-TTS papers all use emotion2vec for scoring)

- **Known Problems & Pitfalls**:
  - **Bilingual label output**: Returns labels like `难过/sad` (Chinese/English): requires post-processing to extract English labels. This catches many users off guard
  - **Data bias from pseudo-labels**: The large-scale training data uses pseudo-labels (model-generated, not human-annotated), which can propagate systematic biases. If the seed model misjudged certain emotions, those errors are amplified in base/large
  - **Limited context understanding**: Classifies based purely on acoustic features: cannot understand sarcasm, irony, or conversational context. "I'm so angry!" said in a joking tone will still register as angry
  - **Coarse emotion granularity**: Only 9 discrete categories. Cannot capture nuanced emotions (e.g., "frustrated" vs "angry", "anxious" vs "fearful", "bored" vs "neutral")
  - **"Other" and "Unknown" categories are noisy**: These two classes are catch-alls that often absorb ambiguous samples, leading to inconsistent classification
  - **Reliability gap for fine-grained evaluation**: Recent EmoVoice (2025) research found that emotion2vec's emotion similarity scores have **low sentence-level Spearman's ρ (below 50%)**: meaning per-sample predictions are often unreliable despite good aggregate/system-level metrics
  - **CPU inference is slow**: The large model (~300M params) processes very slowly on CPU; base (~90M) is more practical for non-GPU deployments
  - **FunASR dependency**: Requires the FunASR framework, which has its own dependency chain (modelscope, torch, etc.) and can conflict with other packages
  - **Bug history**: A bug was fixed in June 2024: users with older code/checkpoints may get wrong results

- **Community Verdict**: Best option available, but don't trust individual predictions blindly: aggregate/majority-vote across segments for better accuracy.

---

#### wav2vec2 SER models (ehcalabres, Dpngtm, r-f)

**Reliability: LOW-MEDIUM (dataset-specific, poor generalization)**

- **Strengths**:
  - Leverages wav2vec2's powerful self-supervised speech representations
  - Easy to use via HuggingFace Transformers pipeline
  - Apache 2.0 / MIT licenses: fully permissive, no restrictions
  - The ehcalabres XLSR model supports cross-lingual features

- **Known Problems & Pitfalls**:
  - **The 97.5% accuracy claim (r-f model) is misleading**: This accuracy is on RAVDESS only: a tiny acted dataset of 1,440 recordings by 24 actors. This number does NOT generalize to real-world speech. One HuggingFace user fine-tuned for RAVDESS and explicitly confirmed: "Overfitting is correct in this case because [it's] built only to solve this problem, predicting emotions on the RAVDESS dataset"
  - **Acted vs. natural speech gap**: All wav2vec2 SER models are fine-tuned on acted emotion datasets (RAVDESS, TESS, CREMA-D, SAVEE). Acted emotions have exaggerated prosody that doesn't match real conversational speech. Real-world accuracy drops to **60-75%** for most of these models
  - **English-only**: Unlike emotion2vec+ which is multilingual, these models only work reliably on English speech. Non-English audio gets unpredictable results
  - **Audio-only modality ceiling**: Research (WavFusion, 2024) showed that audio-only SER using wav2vec2 achieved a maximum accuracy of ~66%: the audio modality alone has inherent limitations for emotion classification
  - **No "calm" in some models**: The r-f and Dpngtm models don't distinguish "calm" from "neutral", while ehcalabres does. This inconsistency makes model comparison difficult
  - **Large model size for marginal gains**: At ~317M parameters, the XLSR-based models are large but don't outperform emotion2vec+ base (~90M) in cross-corpus evaluation
  - **Dataset contamination risk**: Small fine-tuning datasets mean speaker identity can leak into emotion predictions: models may learn to recognize specific actors rather than emotions

- **Community Verdict**: Useful as baselines or for RAVDESS-like acted speech. Do NOT rely on for production real-world SER. emotion2vec+ is strictly better.

---

#### HuBERT SER (various fine-tuned models)

**Reliability: MEDIUM**

- **Strengths**:
  - HuBERT's masked prediction pre-training learns robust speech representations
  - Research (INTERSPEECH 2022) showed HuBERT embeddings transfer well across emotion corpora (cross-corpus SER)
  - Multiple fine-tuned variants available on HuggingFace
  - Better intermediate representations than wav2vec2 for some emotion tasks (Chakhtouna, 2024)

- **Known Problems & Pitfalls**:
  - **Same acted-dataset problem**: Most HuBERT SER models are still fine-tuned on RAVDESS/IEMOCAP: same generalization issues as wav2vec2
  - **No single "best" HuBERT SER model**: The "various" tag means there's no standard checkpoint: users must evaluate multiple community models, quality varies wildly
  - **Large compute footprint**: The large variant (~317M params) has similar resource requirements to wav2vec2 XLSR with no significant accuracy advantage for SER
  - **Emotion-specific embeddings less studied**: Most HuBERT research focuses on ASR, not SER: fewer papers validate SER-specific performance compared to emotion2vec+

- **Community Verdict**: Good embeddings for research. For production SER, emotion2vec+ is more battle-tested.

---

#### WavLM SER (3loi / audeering)

**Reliability: MEDIUM-HIGH (dimensional A/D/V only)**

- **Strengths**:
  - WavLM's denoising pre-training gives superior noise robustness: better in real-world noisy conditions than wav2vec2/HuBERT
  - audeering's wav2vec2-large-robust-12-ft-emotion-msp-dim model achieves SOTA on MSP-Podcast for dimensional emotion (valence CCC=0.638)
  - Dimensional output (Arousal/Dominance/Valence) provides more nuanced emotion representation than discrete categories
  - Used by audeering commercially in their devAIce product: production-proven

- **Known Problems & Pitfalls**:
  - **Not categorical**: Outputs continuous A/D/V values, not discrete emotion labels. Requires post-processing to map to categories (angry, sad, etc.): this mapping is lossy and subjective
  - **Valence is inherently hard from audio alone**: The "valence gap" is a known problem: predicting whether speech is positive/negative from acoustics alone (without text) is very difficult. CCC=0.638 sounds good but still misses many samples
  - **Annotation disagreement**: A/D/V annotations have high inter-annotator disagreement. The model is trained to match average ratings, but individuals may perceive the same audio very differently
  - **Large model**: ~317M parameters with significant VRAM/RAM needs: overkill for simple emotion classification
  - **English-centric**: Primary MSP-Podcast training data is English: multilingual generalization is less validated than emotion2vec+

- **Community Verdict**: Best choice for dimensional emotion analysis (A/D/V). Complementary to emotion2vec+ (categorical). Not a replacement for discrete emotion classification.

---

#### wav2small (audeering)

**Reliability: MEDIUM (edge/resource-constrained use case)**

- **Strengths**:
  - Incredibly tiny: only 72K parameters, 120KB ONNX: can run on microcontrollers and edge devices
  - Knowledge-distilled from a large WavLM teacher model: inherits some of its quality
  - A/D/V output like its teacher
  - CC BY-NC-SA 4.0: available for non-commercial research

- **Known Problems & Pitfalls**:
  - **Distillation quality loss**: Significantly less accurate than the teacher model. The paper acknowledges this is a trade-off for extreme size reduction
  - **Non-commercial license**: CC BY-NC-SA 4.0 prohibits commercial use: not suitable for production applications
  - **Trained on teacher outputs, not human labels**: Inherits all biases and errors from the teacher model, plus adds its own distillation noise
  - **A/D/V only**: No discrete emotion labels: same mapping challenge as WavLM above
  - **New and less validated**: Published August 2024, limited community testing beyond the original paper
  - **Multilingual claim is weak**: The "multilingual" label comes from the pre-trained backbone, not from explicit multilingual SER evaluation

- **Community Verdict**: Impressive engineering for extreme edge deployment. Not recommended for accuracy-critical applications.

---

### Speaker Analysis Models

#### PyAnnote (speaker-diarization-3.1)

**Reliability: HIGH (but major access friction)**

- **Strengths**:
  - State-of-the-art speaker diarization: consistently top results on DIHARD and AMI benchmarks
  - Version 3.1 removed onnxruntime dependency: pure PyTorch for easier deployment
  - Well-maintained by Hervé Bredin, active development
  - Used in 100+ HuggingFace Spaces (whisper-webui, SoniTranslate, etc.)
  - MIT license: fully permissive

- **Known Problems & Pitfalls**:
  - **Gated model access is the #1 complaint**: Requires accepting terms on BOTH `pyannote/speaker-diarization-3.1` AND `pyannote/segmentation-3.0` on HuggingFace. HuggingFace forums have **8,600+ views** on 401 error threads alone
  - **Token permission confusion**: Need a fine-grained token with "public gated repos" read enabled. Many users create tokens without this checkbox. Some report needing to "toggle" permissions or clear `~/.cache/huggingface` to fix
  - **Windows symlink errors**: `[WinError 1314] A required privilege is not held by the client`: a known Windows issue (7,500+ views on HF forums) requiring developer mode or admin privileges
  - **Cascading gated dependencies**: The diarization pipeline internally loads segmentation-3.0, which is separately gated. Users who only accept diarization-3.1 still get 403 errors
  - **Email collection requirement**: PyAnnote gates access specifically to collect user emails for marketing their premium/paid services: not a pure technical restriction
  - **Speaker count estimation**: Default clustering can over-estimate speaker count, especially for short audio. Requires tuning `min_speakers`/`max_speakers` parameters
  - **High latency**: Not suitable for real-time: designed for offline whole-file processing
  - **"Pyannote gives wrong results"** is a recurring HF forum thread (1,000+ views): sensitivity to audio quality, background noise, and short segments

- **Community Verdict**: Best accuracy when it works, but the gated access and Windows issues make deployment painful. Consider resemblyzer or NeMo for simpler alternatives.

---

#### SpeechBrain (spkrec-ecapa-voxceleb)

**Reliability: MEDIUM-HIGH (speaker verification/ID)**

- **Strengths**:
  - ECAPA-TDNN architecture is robust: maintains stable EER even when scaling to large user bases (unlike weaker models)
  - Clean API: `SpeakerRecognition.from_hparams()` → `verify_files()`: simple verification
  - Part of the larger SpeechBrain ecosystem (200+ recipes, 40+ datasets)
  - Used in scalability research (Lviv Polytechnic, 2025): confirmed most robust architecture for voice biometrics

- **Known Problems & Pitfalls**:
  - **torchaudio API breaking changes**: SpeechBrain relies on torchaudio, which has had breaking changes (e.g., `list_audio_backends()` removed in torchaudio 2.10). This causes crashes with newer PyTorch versions
  - **Heavy dependency chain**: SpeechBrain installs many transitive dependencies that can conflict with other packages (particularly along PyTorch version lines)
  - **No performance warranty**: The model card explicitly states "The SpeechBrain team does not provide any warranty on the performance achieved by this model when used on other datasets"
  - **Speaker verification ≠ speaker identification**: ECAPA-TDNN does pairwise comparison ("are these the same speaker?"), not "who is speaking?": building full speaker ID requires additional enrollment/matching infrastructure
  - **VoxCeleb training domain**: Trained on celebrity speech from YouTube: may not generalize well to telephony, far-field, or noisy environments

- **Community Verdict**: Reliable for speaker verification tasks. Be cautious with torchaudio/PyTorch version compatibility.

---

#### Silero VAD

**Reliability: HIGH (for its class)**

- **Strengths**:
  - Trained on 6,000+ languages: extremely robust across diverse audio
  - Tiny (~2M params): one audio chunk takes <1ms on a single CPU thread
  - Multiple runtimes: PyTorch, ONNX, ExecuTorch, C++: runs practically anywhere
  - MIT license, no telemetry, no registration, no expiration
  - 551+ commits, actively maintained: mature project

- **Known Problems & Pitfalls**:
  - **Not the most accurate**: Independent benchmarks (Picovoice, 2025-2026) show Cobra VAD achieves 98.9% TPR at 5% FPR vs Silero's 87.7% TPR: Silero misses 1 in 8 speech frames while Cobra misses 1 in 100
  - **At 1% FPR, drops to 80.4% TPR**: For strict false-positive requirements, Silero misses 1 in 5 speech frames
  - **Heavy for true edge**: Despite being "lightweight", requires PyTorch or ONNX runtime: too heavy for microcontrollers or Raspberry Pi Zero (uses ~43% CPU vs Cobra's 5%)
  - **Architecture is opaque**: Training data details, architecture spec, loss function, and augmentation strategy are NOT publicly disclosed: this limits reproducibility and trust for academic use
  - **Threshold sensitivity**: Performance varies dramatically with threshold selection. Default threshold may not be optimal for all use cases (telephony vs. meetings vs. broadcast)
  - **Not real-time on constrained devices**: While fast on desktop CPUs, the PyTorch/ONNX overhead makes it impractical for truly resource-constrained IoT devices
  - **AUC is lower than Cobra**: In overall AUC comparison (vendor-neutral metric), Cobra > Silero > WebRTC consistently

- **Community Verdict**: Excellent default choice for desktop/server VAD. For edge/IoT, consider Cobra. For simplest possible implementation, WebRTC VAD is lighter (but much less accurate).

---

#### Wav2Vec 2.0 for Gender/Age

**Reliability: MEDIUM**

- **Strengths**:
  - Gender classification: 100% accuracy on TIMIT is genuine: gender is acoustically very distinguishable
  - Age estimation: ~7.1 years MAE is reasonable given the inherent difficulty
  - audeering models (wav2vec2-large-robust-24-ft-age-gender) have 432K+ downloads: well-tested

- **Known Problems & Pitfalls**:
  - **TIMIT's 100% accuracy is misleading**: TIMIT is a clean, studio-recorded, balanced dataset of 630 speakers. Real-world noisy audio with diverse accents/ages will see accuracy drops
  - **Binary gender classification is problematic**: Only classifies male/female: no support for non-binary, and accuracy drops for children, elderly, and trans speakers
  - **Age estimation has high variance**: 7.1 years MAE means a 30-year-old could be estimated as 23-37: not precise enough for many applications
  - **Large models for simple tasks**: ~317M parameters for gender/age is overkill: simpler signal processing features (F0, formants) can achieve 95%+ gender accuracy with far less compute

- **Community Verdict**: Works well for rough demographic estimation. Don't rely on for individual-level accuracy.

---

### Audio Feature Extraction Libraries

#### librosa

**Reliability: VERY HIGH (industry standard)**

- **Strengths**:
  - The standard Python audio analysis library: cited in thousands of papers
  - Comprehensive feature set: MFCCs, spectrograms, chromagrams, tempograms, pitch, onset detection
  - Excellent documentation and tutorials
  - BSD license: fully permissive
  - Stable API, few breaking changes

- **Known Problems & Pitfalls**:
  - **Slower than torchaudio for batch processing**: No GPU acceleration: CPU-only. For large-scale feature extraction, torchaudio is faster
  - **Feature values differ from OpenSMILE/Praat**: A 2025 study (Cornell/UPenn, INTERSPEECH 2025) comparing OpenSMILE, Praat, and librosa found **significant discrepancies** in extracted features (F0, HNR, MFCCs) despite analyzing the same audio. OpenSMILE showed **higher discrimination potential** (AUC >0.75) for clinical speech analysis
  - **No streaming/real-time mode**: Designed for offline analysis of complete files: not suitable for real-time audio processing
  - **Limited speaker/emotion features**: Only extracts acoustic features: doesn't provide higher-level features like emotion or speaker identity

- **Community Verdict**: Essential library for audio analysis. Use alongside OpenSMILE when fine-grained or clinical-grade features are needed.

---

#### openSMILE

**Reliability: HIGH (research & clinical standard)**

- **Strengths**:
  - 6,373 acoustic features via ComParE 2016 feature set: the most comprehensive extraction tool
  - eGeMAPS feature set (88 features) is the standard for speech emotion research
  - Used as the baseline in nearly all INTERSPEECH Computational Paralinguistic challenges
  - Proven in clinical applications (depression, Parkinson's, schizophrenia detection)
  - INTERSPEECH 2025 study found OpenSMILE features had highest AUC for discriminating SSD vs HC groups

- **Known Problems & Pitfalls**:
  - **Complex configuration**: Feature extraction requires understanding SMILE config files: steeper learning curve than librosa
  - **6,373 features can be overwhelming**: ComParE feature set produces very high-dimensional output, requires careful feature selection to avoid curse of dimensionality
  - **Python wrapper (opensmile) has limitations**: Not all C++ openSMILE features are exposed through the Python bindings
  - **Academic/research license**: The open-source version (audEERING) has restrictions for commercial use: check license carefully
  - **Overkill for simple tasks**: If you just need MFCCs or a spectrogram, librosa is simpler

- **Community Verdict**: Gold standard for paralinguistic feature extraction. Use eGeMAPS (88 features) for emotion-related tasks, ComParE for exhaustive analysis.

---

#### torchaudio

**Reliability: HIGH (PyTorch ecosystem)**

- **Strengths**:
  - GPU-accelerated: fastest option for batch feature extraction
  - Native PyTorch integration: features flow directly into neural network training
  - Active development by Meta/PyTorch team
  - Supports streaming audio processing

- **Known Problems & Pitfalls**:
  - **Breaking API changes**: History of breaking changes between major versions (e.g., `list_audio_backends()` removed in v2.10, backend system restructured multiple times)
  - **Audio backend complexity**: Requires separate audio I/O backends (sox, soundfile, or ffmpeg). Which backend is available depends on OS and installation method: a common source of issues
  - **Not a standalone analysis library**: Provides building blocks (transforms, I/O) but not high-level analysis functions like librosa's `beat_track()` or `piptrack()`
  - **Heavy dependency**: Pulls in full PyTorch: overkill if you only need feature extraction without training

- **Community Verdict**: Use when you're already in the PyTorch ecosystem and need GPU-accelerated processing. Pair with librosa for analysis functions.

---

#### pyAudioAnalysis

**Reliability: LOW (outdated)**

- **Strengths**:
  - All-in-one: feature extraction, classification, segmentation, regression, visualization in a single library
  - Good for learning: covers the full audio analysis pipeline conceptually
  - Published as a PLoS ONE paper (2015) with clear methodology

- **Known Problems & Pitfalls**:
  - **Effectively unmaintained**: Last significant update was 2022. Core methodology is from 2015: uses classical ML (SVM, kNN) rather than modern deep learning
  - **Outdated dependencies**: Built around older numpy/scipy patterns, can conflict with modern package versions
  - **Inferior accuracy**: Classical feature extraction + SVM classification is vastly outperformed by modern self-supervised models (wav2vec2, HuBERT, emotion2vec+)
  - **No GPU acceleration**: All processing is CPU-based
  - **Limited community**: Very few active users compared to librosa or torchaudio
  - **Author's own recommendation shifted**: The author now points to "deep-audio-features" (CNN + PyTorch) as the preferred approach, implicitly acknowledging pyAudioAnalysis is outdated

- **Community Verdict**: Historical significance only. Use librosa for features, emotion2vec+ for SER, PyAnnote for diarization instead.

---

#### VANPY

**Reliability: LOW (very new, unproven)**

- **Strengths**:
  - Comprehensive: 15+ voice analysis components in one framework (music/speech separation, VAD, diarization, emotion, speaker characterization)
  - Published as arXiv paper (February 2025): actively developed
  - Open-source end-to-end framework for speaker characterization

- **Known Problems & Pitfalls**:
  - **Brand new**: Published February 2025: minimal community adoption or independent validation
  - **Wraps existing models**: VANPY is largely a convenience wrapper around existing tools (Silero VAD, PyAnnote, emotion2vec+, etc.): inherits their limitations
  - **No significant accuracy advantages**: Doesn't introduce novel models: just orchestrates existing ones
  - **Unknown maintenance future**: Single-paper publication with no established maintenance track record
  - **Complex dependency chain**: Wrapping 15+ analysis components means 15+ dependency trees: high risk of conflicts

- **Community Verdict**: Interesting for rapid prototyping. Too early to recommend for production. Build your own pipeline using individual proven components instead.

---

## Summary: Reliability Tiers

| Tier | Models/Tools | Recommendation |
|---|---|---|
| **Tier 1: Production Ready** | emotion2vec+ (large/base), Silero VAD, librosa, PyAnnote (if access works) | Use with confidence. Known limitations are manageable. |
| **Tier 2: Solid with Caveats** | WavLM SER, SpeechBrain ECAPA, openSMILE, torchaudio | Good for specific use cases. Understand the caveats before deploying. |
| **Tier 3: Research/Baseline Only** | wav2vec2 SER models (all), HuBERT SER, wav2small | Fine for benchmarks. High accuracy claims don't generalize. |
| **Tier 4: Avoid for New Projects** | pyAudioAnalysis, VANPY (too new) | Use alternatives instead. |

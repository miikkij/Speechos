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
- **emotion2vec+** is the "Whisper of SER" — best general-purpose model, works across languages
- **wav2vec2-based models** are fine-tuned on RAVDESS, TESS, CREMA-D, SAVEE datasets
- All models require **16kHz mono WAV** input
- Models classify per-utterance; chunk long files into 3-10 second segments

---

## Speaker Analysis Models

| Model/Tool | Task | Parameters | CPU Support | Notes |
|---|---|---|---|---|
| **PyAnnote** (pyannote/speaker-diarization-3.1) | Speaker diarization | ~50M | ✅ | "Who spoke when" — segments audio by speaker |
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

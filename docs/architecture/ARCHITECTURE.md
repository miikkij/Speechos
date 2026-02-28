# Speechos: Architecture & Project Plan

> Last updated: 2026-02-27

## Project Vision

Speechos is a local-first speech analysis and synthesis platform. Record speech via microphone, store as WAV, then analyze with AI models for transcription, emotion recognition, speaker analysis, and more. Also supports text-to-speech synthesis.

**Core principle**: All processing runs locally: no cloud APIs, no data leaving the machine.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Web Interface (pnpm/Node.js)          │
│                                                         │
│  ┌──────────┐  ┌───────────┐  ┌─────────────────────┐  │
│  │ Recorder  │  │ Playback  │  │ Analysis Dashboard  │  │
│  │ (mic→wav) │  │ (TTS out) │  │ (results, charts)   │  │
│  └──────────┘  └───────────┘  └─────────────────────┘  │
│                                                         │
│  WebSocket / REST API ←──────────────────────────────── │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ HTTP / WebSocket
                       │
┌──────────────────────▼──────────────────────────────────┐
│               Python Backend (uv)                        │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │                  API Server (FastAPI)                ││
│  │                                                     ││
│  │  POST /api/upload        → receive .wav             ││
│  │  POST /api/transcribe    → STT pipeline             ││
│  │  POST /api/analyze       → emotion + features       ││
│  │  POST /api/synthesize    → TTS pipeline              ││
│  │  GET  /api/models        → list available models     ││
│  │  WS   /ws/stream         → real-time streaming      ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Processing Pipeline                    ││
│  │                                                     ││
│  │  ┌─────────┐  ┌──────────┐  ┌───────────────────┐  ││
│  │  │   STT   │  │ Analysis │  │       TTS         │  ││
│  │  │         │  │          │  │                   │  ││
│  │  │ Faster- │  │emotion2v │  │ Kokoro / Piper /  │  ││
│  │  │ Whisper │  │ PyAnnote │  │ Orpheus / XTTS    │  ││
│  │  │ Vosk    │  │ librosa  │  │                   │  ││
│  │  └─────────┘  └──────────┘  └───────────────────┘  ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │           Model Manager                             ││
│  │  - Download models on demand                        ││
│  │  - Auto-detect GPU/CPU                              ││
│  │  - Select optimal model per hardware                ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
Speechos/
├── CLAUDE.md                  # AI coding instructions
├── .gitignore
├── README.md
├── config.example.yaml        # Configuration template
├── start.sh                   # Linux/Mac launcher
├── start.ps1                  # Windows launcher
│
├── docker/                    # Docker deployment
│   ├── api.Dockerfile         # GPU-enabled API image
│   ├── api-cpu.Dockerfile     # CPU-only API image
│   ├── web.Dockerfile         # Frontend image
│   ├── docker-compose.gpu.yml
│   ├── docker-compose.cpu.yml
│   └── nginx.conf
│
├── web/                       # Node.js frontend (pnpm)
│   ├── package.json
│   ├── pnpm-lock.yaml
│   ├── src/
│   │   ├── app/               # Main app pages
│   │   ├── components/        # UI components
│   │   │   ├── Recorder.tsx   # Microphone recording → WAV
│   │   │   ├── Player.tsx     # Audio playback
│   │   │   ├── Analyzer.tsx   # Analysis results display
│   │   │   └── Synthesizer.tsx # TTS input/output
│   │   ├── lib/               # API client, utils
│   │   └── hooks/             # React hooks for audio
│   └── public/
│
├── api/                       # Python backend (uv)
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── src/
│   │   ├── server.py          # FastAPI main
│   │   ├── routers/
│   │   │   ├── transcribe.py  # STT endpoints
│   │   │   ├── analyze.py     # Analysis endpoints
│   │   │   └── synthesize.py  # TTS endpoints
│   │   ├── pipelines/
│   │   │   ├── stt.py         # STT pipeline (faster-whisper, vosk)
│   │   │   ├── tts.py         # TTS pipeline (kokoro, piper, orpheus)
│   │   │   ├── emotion.py     # Emotion recognition (emotion2vec)
│   │   │   ├── diarization.py # Speaker diarization (pyannote)
│   │   │   └── features.py    # Audio features (librosa)
│   │   ├── models/
│   │   │   └── manager.py     # Model download & management
│   │   └── utils/
│   │       ├── audio.py       # WAV processing utilities
│   │       └── gpu.py         # GPU detection & allocation
│   └── tests/
│
├── models/                    # Downloaded model files (gitignored)
│
├── recordings/                # Recorded WAV files (gitignored)
│
├── docs/
│   ├── research/
│   │   ├── speech-to-text/
│   │   │   └── STT-MODEL-COMPARISON.md
│   │   ├── text-to-speech/
│   │   │   └── TTS-MODEL-COMPARISON.md
│   │   └── speech-analysis/
│   │       └── SPEECH-ANALYSIS-MODELS.md
│   ├── architecture/
│   │   └── ARCHITECTURE.md    # This file
│   └── api/
│       └── API-REFERENCE.md   # API documentation
│
└── samples/                   # Sample WAV files for testing
```

---

## Technology Stack

### Frontend (web/)
| Component | Technology | Why |
|---|---|---|
| Runtime | Node.js | Standard web runtime |
| Package Manager | pnpm | Fast, disk-efficient |
| Framework | Next.js 15 or Vite + React | Modern, fast dev experience |
| Audio Recording | MediaRecorder API | Native browser mic access |
| Audio Playback | Web Audio API | Low-latency playback |
| Charts | Recharts or Chart.js | Visualize analysis results |
| Communication | REST + WebSocket | Real-time streaming support |
| Styling | Tailwind CSS | Utility-first, fast prototyping |

### Backend (api/)
| Component | Technology | Why |
|---|---|---|
| Runtime | Python 3.11+ | ML ecosystem, GPU support |
| Package Manager | uv | Fast, reliable Python pkg manager |
| Framework | FastAPI | Async, WebSocket support, auto-docs |
| STT Primary | faster-whisper | Best speed/accuracy, CPU+GPU |
| STT Lightweight | Vosk | Tiny footprint, streaming |
| Emotion | emotion2vec+ | Best SER model available |
| Diarization | pyannote.audio | State-of-the-art speaker detection |
| TTS Primary | Kokoro | Best quality at 82M params |
| TTS Fast | Piper | Ultra-fast, CPU-only |
| TTS Expressive | Orpheus TTS 3B | Emotion control tags |
| Audio Processing | librosa, torchaudio | Feature extraction |
| VAD | Silero VAD | Lightweight voice detection |
| GPU Framework | PyTorch + CUDA | GPU acceleration |

---

## Development Phases

### Phase 1: Foundation
- [ ] Initialize pnpm project (web/)
- [ ] Initialize uv project (api/)
- [ ] Set up FastAPI server with health check
- [ ] Set up Next.js/Vite frontend with basic layout
- [ ] Implement microphone recording → WAV in browser
- [ ] Implement WAV file upload to backend
- [ ] Implement WAV playback in browser

### Phase 2: Speech-to-Text
- [ ] Integrate faster-whisper (GPU + CPU modes)
- [ ] Integrate Vosk as lightweight fallback
- [ ] Create transcription API endpoint
- [ ] Display transcription results in frontend
- [ ] Add model selection UI (size/speed tradeoff)
- [ ] Add word-level timestamps display

### Phase 3: Speech Analysis
- [ ] Integrate emotion2vec+ for emotion recognition
- [ ] Integrate Silero VAD for speech detection
- [ ] Extract audio features with librosa
- [ ] Create analysis dashboard in frontend
- [ ] Visualize emotions over time
- [ ] Add pitch/energy/tempo charts

### Phase 4: Text-to-Speech
- [ ] Integrate Kokoro for high-quality TTS
- [ ] Integrate Piper for fast CPU TTS
- [ ] Create synthesis API endpoint
- [ ] Add TTS playground in frontend
- [ ] Support voice selection
- [ ] Add audio download for generated speech

### Phase 5: Advanced Features
- [ ] Speaker diarization with PyAnnote
- [ ] Voice cloning with Chatterbox/XTTS
- [ ] Orpheus TTS with emotion tags
- [ ] Real-time streaming (WebSocket)
- [ ] Model manager (download/switch models in UI)
- [ ] Batch file processing

---

## API Endpoints (Planned)

```
# Health
GET  /api/health                   → { status: "ok", gpu: true/false }

# Models
GET  /api/models                   → list available models
POST /api/models/download          → download a model

# Speech-to-Text
POST /api/transcribe               → { file: .wav } → { text, segments[], language }
POST /api/transcribe/stream        → WebSocket streaming transcription

# Analysis
POST /api/analyze                  → { file: .wav } → { emotions[], features, speakers[] }
POST /api/analyze/emotion          → { file: .wav } → { emotions[] }
POST /api/analyze/features         → { file: .wav } → { pitch, energy, tempo, ... }
POST /api/analyze/diarize          → { file: .wav } → { speakers[] }

# Text-to-Speech
POST /api/synthesize               → { text, model, voice } → .wav file
GET  /api/voices                   → list available voices
POST /api/synthesize/stream        → WebSocket streaming TTS

# Files
POST /api/upload                   → upload .wav file
GET  /api/recordings               → list recordings
GET  /api/recordings/:id           → get recording + analysis
```

---

## Hardware Detection & Tier System

The system auto-detects CPU/GPU/RAM/VRAM at startup and selects optimal models. Users can override via `config.yaml` or environment variables. See **[HARDWARE-TIERS.md](HARDWARE-TIERS.md)** for the full tier matrix.

**Supported configurations:**
- CPU-only: 2GB, 4GB, 8GB, 16GB, 32GB RAM
- GPU-only: 4GB, 8GB, 12GB, 24GB, 32GB VRAM
- Hybrid CPU+GPU: combinations of the above

**VRAM management:** Lazy-load models, LRU eviction when VRAM is tight, CPU fallback always available.

---

## Docker Deployment

Three Dockerfiles, two compose configs, launcher scripts for Linux/Windows:

```
docker/
├── api.Dockerfile           # GPU-enabled (NVIDIA CUDA base)
├── api-cpu.Dockerfile       # CPU-only (slim Python)
├── web.Dockerfile           # Node.js frontend
├── docker-compose.gpu.yml   # Full stack with GPU passthrough
├── docker-compose.cpu.yml   # CPU-only stack
└── nginx.conf               # Reverse proxy config
```

**Quick start:**
```bash
# Auto-detect GPU/CPU and launch
./start.sh                # Linux/Mac
.\start.ps1               # Windows

# Force specific mode
./start.sh --gpu
./start.sh --cpu
./start.sh --tier gpu-32gb
./start.sh --dev           # Dev mode without Docker
```

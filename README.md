# Speechos

Local-first speech analysis and synthesis platform. Record speech from your microphone, transcribe it, analyze emotion, and synthesize speech — all running locally on your machine.

## Quick Start (Development)

### Prerequisites

- **Node.js** 22+ and **pnpm** (`npm install -g pnpm`)  
- **Python** 3.11+ and **uv** (`pip install uv` or [install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **FFmpeg** (optional, for non-WAV audio files)

### 1. Install dependencies

```bash
# Install web dependencies
cd web && pnpm install && cd ..

# Install Python API dependencies
cd api && uv sync && cd ..
```

### 2. Start both servers

```powershell
# Windows (recommended — handles process cleanup properly)
.\dev.ps1
```

This starts both:
- **API** (Python/FastAPI) → `http://localhost:36300`
- **Web** (Next.js) → `http://localhost:36301`

Press `Ctrl+C` to stop — both processes are killed cleanly by PID.

You can also start them individually:

```bash
pnpm dev:api   # only the API
pnpm dev:web   # only the web frontend

# Or with flags:
.\dev.ps1 -ApiOnly
.\dev.ps1 -WebOnly
```

### 3. Open in browser

Go to [http://localhost:36301](http://localhost:36301). The status bar at the top shows whether the API is connected.

## What You Can Do

| Feature | Tab | Description |
|---|---|---|
| **Record & Transcribe** | Record | Click the mic button, speak, click again → automatic transcription |
| **Upload & Transcribe** | Record | Upload a WAV/MP3/FLAC/OGG file for transcription |
| **Text to Speech** | TTS | Type text → synthesize speech and play it back |

## Project Structure

```
speechos/
├── api/               Python backend (FastAPI + uv)
│   └── src/
│       ├── server.py        FastAPI app, CORS, router mounts
│       ├── config.py        Hardware detection, tier system, settings
│       ├── models.py        Lazy model loading with LRU cache
│       ├── audio.py         Audio I/O, resampling, validation
│       └── routers/
│           ├── health.py        GET /health
│           ├── transcribe.py    POST /api/transcribe
│           ├── synthesize.py    POST /api/synthesize
│           ├── recordings.py    GET/POST/DELETE /api/recordings
│           └── system.py        GET /api/system/info
├── web/               Node.js frontend (Next.js + pnpm)
│   └── src/
│       ├── app/             Next.js App Router pages
│       ├── components/      React UI components
│       ├── hooks/           Custom hooks (useRecorder)
│       └── lib/             API client
├── docker/            Docker deployment files
├── docs/              Research & architecture docs
├── models/            Downloaded AI models (auto-created, gitignored)
├── recordings/        Saved recordings (auto-created, gitignored)
├── samples/           Sample WAV files for testing
└── package.json       Root dev scripts (pnpm dev)
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/transcribe` | Upload audio → get transcription |
| `POST` | `/api/synthesize` | Send text → get WAV audio |
| `GET` | `/api/recordings` | List saved recordings |
| `POST` | `/api/recordings` | Upload and save a recording |
| `DELETE` | `/api/recordings/:id` | Delete a recording |
| `GET` | `/api/system/info` | Hardware profile, tier, loaded models |

## Hardware Auto-Detection

The API auto-detects your GPU/CPU/RAM and selects optimal models. Override with:

```bash
# Environment variable
SPEECHOS_TIER=gpu-32gb pnpm dev:api

# Or create config.yaml from the template
cp config.example.yaml config.yaml
```

Available tiers: `cpu-{2,4,8,16,32}gb`, `gpu-{4,8,12,24,32}gb`, `hybrid-{4,8,12,24,32}gb-gpu`

## Docker (Production)

Docker is **not required** for development. Use it for deployment or if you prefer containers:

```bash
# GPU (requires NVIDIA Container Toolkit)
docker compose -f docker/docker-compose.gpu.yml up --build

# CPU-only
docker compose -f docker/docker-compose.cpu.yml up --build

# Auto-detect (Linux/Mac)
./start.sh

# Auto-detect (Windows)
.\start.ps1
```

## Models

Models are downloaded automatically on first use to the `models/` directory. First run will be slower.

| Feature | Engine | Model | Size |
|---|---|---|---|
| STT (GPU) | faster-whisper | large-v3 | ~3 GB |
| STT (CPU) | faster-whisper | tiny/base | 75–140 MB |
| STT (low RAM) | Vosk | small-en | ~40 MB |
| TTS | Piper | lessac-medium | ~60 MB |
| VAD | Silero | v5 | ~2 MB |

## Tech Stack

- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS 4
- **Backend**: Python 3.11+, FastAPI, uvicorn
- **STT**: faster-whisper, Vosk
- **TTS**: Piper
- **VAD**: Silero
- **Package managers**: pnpm (JS), uv (Python)

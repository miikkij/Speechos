# Speechos

Local-first speech analysis and synthesis platform. Record speech from your microphone, transcribe it, analyze emotion and speakers, and synthesize speech — all running locally on your machine. Finnish is the default language (English and auto-detect are one click away).

## Quick Start (Development)

### Prerequisites

- **Node.js** 22+ and **pnpm** (`npm install -g pnpm`)
- **Python** 3.11–3.13 and **uv** ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **FFmpeg** is not required: non-WAV input (WebM/Opus from the browser, MP3, …) is decoded with PyAV.

### 1. Install dependencies

```bash
cd web && pnpm install && cd ..
cd api && uv sync && cd ..          # add --extra dev for tests, --extra gpu for pyannote
```

### 2. Start both servers

```powershell
# Windows (recommended — handles process cleanup properly)
.\dev.ps1
```

This starts both:
- **API** (Python/FastAPI) → `http://localhost:36300` (Swagger: `/docs`)
- **Web** (Next.js) → `http://localhost:36301`

Press `Ctrl+C` to stop — both process trees are killed by PID.

You can also start them individually:

```bash
pnpm dev:api   # only the API  (cd api && uv run python -m src)
pnpm dev:web   # only the web frontend

.\dev.ps1 -ApiOnly
.\dev.ps1 -WebOnly
```

### 3. Open in browser

Go to [http://localhost:36301](http://localhost:36301). The status bar shows whether the API is connected. Pick the language (Suomi / English / Auto) in the header; the choice is remembered.

## What You Can Do

| Feature | Tab | Description |
|---|---|---|
| **Record & Transcribe** | Record | Click the mic button, speak, click again → transcription + analysis |
| **Upload & Transcribe** | Record | Upload any audio file (WAV, MP3, FLAC, OGG, WebM) |
| **Recordings library** | Record | Recordings are stored as 16 kHz mono WAV and can be re-analyzed |
| **Analysis** | Record | Pitch, energy, tempo, speaking rate, speech activity; emotion and speakers on GPU tiers |
| **Text to Speech** | TTS | Type text → synthesize in the selected language, adjustable speed |

## Tests

```bash
cd api && uv sync --extra dev
uv run pytest                   # unit tests — no GPU, no model downloads
uv run pytest -m models         # real-model smoke tests on CPU (downloads weights)
uv run ruff check src tests

cd web && pnpm lint && pnpm typecheck && pnpm build
```

CI (`.github/workflows/ci.yml`) runs the same checks on every push and pull request.

## Finnish STT/TTS benchmark

```bash
cd api && uv sync --extra bench
uv run python bench/stt_fi.py --models large-v3 --datasets fleurs --n 100
uv run python bench/tts_fi.py --engines piper-fi espeak-fi
```

See [api/bench/README.md](api/bench/README.md).

## Project Structure

```
speechos/
├── api/               Python backend (FastAPI + uv)
│   ├── src/
│   │   ├── __main__.py      Entry point (python -m src)
│   │   ├── server.py        App factory, CORS, router mounts
│   │   ├── config.py        Hardware detection, tiers, config/env overrides
│   │   ├── lang.py          Language codes and per-engine voice maps
│   │   ├── models.py        Lazy model loading, cache, per-model locks
│   │   ├── audio.py         Decoding, resampling, 16 kHz WAV encoding
│   │   ├── analysis.py      Features, emotion, diarization, VAD
│   │   ├── docker_tts.py    Docker-backed TTS engines (async)
│   │   └── routers/         health, transcribe, synthesize, analyze, recordings, system
│   ├── tests/         pytest suite
│   └── bench/         Finnish STT/TTS benchmark
├── web/               Next.js frontend (pnpm)
├── docker/            Dockerfiles, compose files, nginx
├── docs/              Architecture & research docs
├── samples/           Sample WAV files (incl. a Finnish sample)
└── package.json       Root dev scripts
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/transcribe` | Audio (+ optional `language`, e.g. `fi`) → transcription |
| `POST` | `/api/synthesize` | `{text, language?, speed?, voice?}` → WAV audio |
| `POST` | `/api/analyze` | Audio → features, emotion, diarization, speech activity |
| `GET` | `/api/recordings` | List saved recordings |
| `POST` | `/api/recordings` | Upload and save a recording (converted to 16 kHz mono WAV) |
| `GET` | `/api/recordings/:id/audio` | Download a recording |
| `DELETE` | `/api/recordings/:id` | Delete a recording |
| `GET` | `/api/system/info` | Hardware profile, tier, loaded models |
| `GET` | `/api/system/model-options` | Selectable models |
| `POST` | `/api/system/switch-model` | Switch a model at runtime |

Full reference: [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md).

## Hardware Auto-Detection

The API detects your GPU/CPU/RAM and selects models per tier. Override with:

```bash
SPEECHOS_TIER=gpu-24gb pnpm dev:api          # force a tier
SPEECHOS_COMPUTE_MODE=cpu pnpm dev:api       # ignore the GPU
SPEECHOS_STT_MODEL=large-v3 pnpm dev:api     # override one model
cp config.example.yaml config.yaml           # or use a config file
```

Tiers: `cpu-{2,4,8,16,32}gb`, `gpu-{4,8,12,24,32}gb`, `hybrid-{4,8,12,24,32}gb-gpu`. Details and the full list of overrides: [docs/architecture/HARDWARE-TIERS.md](docs/architecture/HARDWARE-TIERS.md).

## Docker

Docker is **not required** for development. Use it for deployment:

```bash
docker compose -f docker/docker-compose.gpu.yml up --build   # NVIDIA Container Toolkit
docker compose -f docker/docker-compose.cpu.yml up --build   # CPU-only
./start.sh      # auto-detect (Linux/Mac)
.\start.ps1     # auto-detect (Windows)
```

Set `HF_TOKEN` to enable pyannote community-1 diarization on GPU tiers. The web image bakes in the API URL the browser uses (`SPEECHOS_PUBLIC_API_URL`, default `http://localhost:36300`).

## Models

Models download automatically on first use (into `models/` under the API's working directory, or `SPEECHOS_MODEL_DIR`).

| Feature | Engine | Default model | Size |
|---|---|---|---|
| STT (GPU) | faster-whisper | large-v3 / turbo | ~3 GB / ~1.6 GB |
| STT (CPU) | faster-whisper | tiny … medium (int8) | 75 MB – 1.5 GB |
| TTS | Piper | fi_FI-harri-medium (+ en_US-lessac-medium for English) | ~60 MB each |
| Emotion (GPU) | emotion2vec+ | base / large | |
| Diarization (GPU ≥ 12 GB) | pyannote community-1 (HF_TOKEN) or Resemblyzer | | |
| VAD | Silero | v5 (bundled with silero-vad) | ~2 MB |

## Tech Stack

- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS 4
- **Backend**: Python 3.11+, FastAPI, uvicorn
- **STT**: faster-whisper (+ Vosk, WhisperX, NeMo, … selectable)
- **TTS**: Piper (+ Kokoro, Chatterbox, Kitten, eSpeak-NG, Docker engines)
- **Analysis**: librosa, emotion2vec+, pyannote / Resemblyzer, Silero VAD
- **Package managers**: pnpm (JS), uv (Python)

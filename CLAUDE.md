# CLAUDE.md: Speechos Project

## Project Overview

Speechos is a local-first speech analysis and synthesis platform. It records speech from a microphone, stores it as WAV, and uses AI models to analyze the audio for transcription, emotion, speaker identification, and more. It also supports text-to-speech synthesis.

## Repository Structure

- `web/`: Node.js frontend (pnpm, Next.js/Vite + React)
- `api/`: Python backend (uv, FastAPI)
- `docs/`: Research, architecture, and API documentation
- `models/`: Downloaded AI models (gitignored)
- `recordings/`: Recorded WAV files (gitignored)
- `samples/`: Sample WAV files for testing

## Tech Stack

### Frontend (web/)
- **Package manager**: pnpm
- **Framework**: Next.js or Vite + React + TypeScript
- **Audio**: MediaRecorder API (recording), Web Audio API (playback)
- **Styling**: Tailwind CSS
- **Communication**: REST + WebSocket to Python backend

### Backend (api/)
- **Package manager**: uv
- **Framework**: FastAPI (async, WebSocket)
- **STT**: faster-whisper (primary), Vosk (lightweight CPU)
- **TTS**: Kokoro (quality), Piper (speed), Orpheus 3B (emotion)
- **Emotion**: emotion2vec+ (speech emotion recognition)
- **Analysis**: PyAnnote (diarization), Silero VAD, librosa
- **GPU**: PyTorch + CUDA when available, CPU fallback always provided

## Key Commands

```bash
# Quick start (auto-detects GPU/CPU)
./start.sh              # Linux/Mac
.\start.ps1             # Windows

# Docker (GPU)
docker compose -f docker/docker-compose.gpu.yml up --build

# Docker (CPU-only)
docker compose -f docker/docker-compose.cpu.yml up --build

# Development (no Docker)
cd web && pnpm install && pnpm dev
cd api && uv sync && uv run python -m src.server

# Development (Linux/WSL2, auto-cleanup)
./dev.sh

# Development (Windows, auto-cleanup)
.\dev.ps1

# Run tests
cd api && uv run pytest
```

## Hardware Configuration

- Auto-detects GPU/CPU/RAM/VRAM and selects optimal models
- Override with `SPEECHOS_TIER=gpu-32gb` env var or `config.yaml`
- Tiers: cpu-{2,4,8,16,32}gb, gpu-{4,8,12,24,32}gb, hybrid-{4,8,12,24,32}gb-gpu
- See `docs/architecture/HARDWARE-TIERS.md` for full tier matrix

## Coding Conventions

- Python: Use type hints, async/await for API handlers
- TypeScript: Strict mode, no `any`
- Recording defaults to 48kHz (configurable: 16k/44.1k/48k); downsampled to 16kHz mono only for model input
- Models auto-detect GPU/CPU and select optimal configuration
- REST API returns JSON; binary audio uses streaming responses
- File uploads use multipart/form-data

## Model Management

- Models are downloaded on first use to `models/` directory
- The system auto-detects GPU VRAM and selects appropriate model sizes
- CPU fallbacks exist for all features (may be slower)
- Model paths are relative to project root

## Docker

### Core Stack
- `docker/api.Dockerfile`: GPU-enabled API (NVIDIA CUDA base)
- `docker/api-cpu.Dockerfile`: CPU-only API (slim Python)
- `docker/web.Dockerfile`: Node.js frontend
- `docker/docker-compose.gpu.yml`: Full stack with GPU passthrough
- `docker/docker-compose.cpu.yml`: CPU-only stack
- Nginx reverse proxy: web on /, API on /api/, WebSocket on /ws/

### Engine Compose Files (run one engine at a time to share GPU)
- `docker/tts-engines.yml`: 8 TTS engines (ports 36310-36317)
- `docker/stt-engines.yml`: 7 STT engines (ports 36320-36326)
- `docker/analysis-engines.yml`: 2 analysis engines (ports 36330-36331)

```bash
# Start/stop individual engines
docker compose -f docker/tts-engines.yml up -d xtts
docker compose -f docker/tts-engines.yml stop xtts
docker compose -f docker/stt-engines.yml up -d speaches
docker compose -f docker/stt-engines.yml stop speaches
```

### Port Allocation
```
Core Services:    36300 (API), 36301 (Web)
TTS Engines:      36310-36319
STT Engines:      36320-36329
Analysis Engines: 36330-36339
```

## Architecture

See `docs/architecture/ARCHITECTURE.md` for full architecture diagram and API reference.
See `docs/architecture/HARDWARE-TIERS.md` for hardware tier system.

## Research & Plans

- STT models: `docs/research/speech-to-text/STT-MODEL-COMPARISON.md`
- TTS models: `docs/research/text-to-speech/TTS-MODEL-COMPARISON.md`
- Analysis models: `docs/research/speech-analysis/SPEECH-ANALYSIS-MODELS.md`
- Engine catalog & benchmarking: `docs/plans/PLAN-002-ENGINE-CATALOG-AND-BENCHMARKING.md`
- WSL2 fixes plan: `docs/plans/PLAN-001-WSL2-FIXES-AND-TESTING.md`

## Process Tracking

- `docs/work.log`: Session-by-session work log for continuity across interruptions
- `docs/learned_lessons.md`: Mistakes made and how they were fixed (knowledge base)
- `docs/conditionanalysis/`: Project condition analysis snapshots

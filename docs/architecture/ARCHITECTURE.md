# Speechos — Architecture

> Last updated: 2026-10-09. Describes what is in the code; planned work is in [Roadmap](#roadmap).

## Project Vision

Speechos is a local-first speech analysis and synthesis platform. Record speech via microphone, store it as WAV, then analyze it with AI models for transcription, emotion recognition, speaker analysis, and more. It also synthesizes speech from text. Finnish is a first-class language: Finnish is the UI default and every tier ships a Finnish TTS voice.

**Core principle**: All processing runs locally — no cloud APIs, no data leaving the machine.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│ Web UI — Next.js 15 + React 19 + Tailwind (web/, pnpm, :36301)│
│                                                              │
│  Recorder (MediaRecorder) · Upload · Recordings library      │
│  Transcription view · Analysis view · TTS playground         │
│  Language selector (Suomi / English / Auto, localStorage)    │
└──────────────────────────────┬───────────────────────────────┘
                               │ REST (JSON, multipart, audio/wav)
┌──────────────────────────────▼───────────────────────────────┐
│ API — FastAPI (api/, uv, :36300)            src/server.py     │
│                                                              │
│  routers/  transcribe · synthesize · analyze · recordings ·  │
│            system · health                                   │
│     │  async handlers: decode audio + run models in worker   │
│     │  threads (asyncio.to_thread) — the event loop stays free│
│     ▼                                                        │
│  ModelManager (src/models.py)                                │
│   - lazy load + cache per key ("stt", "tts",                 │
│     "tts:piper:en_US-lessac-medium", ...)                    │
│   - one threading.Lock per cached model: load + inference    │
│     of the same model never run concurrently                 │
│   - per-language TTS variants (src/lang.py)                  │
│     │                                                        │
│     ├─ STT: faster-whisper (default), Vosk, WhisperX,        │
│     │       Moonshine, NeMo, wav2vec2, Granite-Speech        │
│     ├─ TTS: Piper (default, fi_FI-harri-medium), Kokoro,     │
│     │       Kitten, Chatterbox, Bark, eSpeak-NG,             │
│     │       Docker engines → src/docker_tts.py               │
│     ├─ Emotion: emotion2vec+, transformers SER pipelines     │
│     ├─ Diarization: pyannote community-1, Resemblyzer,       │
│     │       SpeechBrain, Sortformer                          │
│     └─ VAD: Silero (silero-vad package)                      │
│                                                              │
│  src/config.py  tier detection + config.yaml + SPEECHOS_* env │
│  src/audio.py   decode (soundfile / PyAV) → 16 kHz mono      │
│  src/analysis.py librosa features, emotion, diarization, VAD │
└──────────────────────────────────────────────────────────────┘
        │ optional: docker compose (WSL2) for heavy TTS engines
        ▼
  docker/tts-engines.yml — XTTS, ChatTTS, MeloTTS, Orpheus, Fish Speech,
  CosyVoice, Qwen3-TTS, Parler, VibeVoice (one GPU container at a time)
```

### Request flow

1. The handler validates the upload (size limit, magic bytes) and parses `language`
   (`fi`, `en`, …; empty / `auto` → `None`).
2. Audio is decoded and resampled to **16 kHz mono float32** in a worker thread
   (`src/audio.py`; WebM/Opus and other containers via PyAV, everything else via soundfile).
3. `ModelManager.run(category, fn, language=...)` resolves the cache key and config,
   then — in a worker thread, under that model's lock — loads the model if needed and
   calls `fn(model, cfg)`.
4. The result is returned as JSON, or as `audio/wav` for synthesis (headers
   `X-Processing-Time`, `X-TTS-Engine`, `X-TTS-Model`).

`/api/analyze` runs acoustic features, emotion, diarization and VAD concurrently
(`asyncio.gather`), each under its own model lock; a failing optional model is reported
in the response (`emotion_error`, `diarization.error`, `vad_error`) instead of failing
the request.

### Audio storage

Every uploaded or recorded file is **re-encoded to 16 kHz mono 16-bit PCM WAV before it
is stored** (`to_wav_16k_mono`). The browser's container (Chrome/Firefox produce
WebM/Opus, Safari MP4) is never trusted, so `recordings/*.wav` are always real WAV files.

### Languages

| Engine | `language` handling |
|---|---|
| faster-whisper / WhisperX | passed through; `None` = auto-detect (response has detected `language`) |
| Piper | if the configured voice speaks another language, the request language's voice is loaded as a separate cached model (`fi` → `fi_FI-harri-medium`, `en` → `en_US-lessac-medium`, also `sv`, `de`) |
| Kokoro | one `KPipeline` per lang code (en, en-gb, es, fr, hi, it, pt, ja, zh); Finnish → **400** with a pointer to Piper |
| Kitten, Parler | English only; other languages → 400 |
| Chatterbox | Multilingual: `language_id`; English model: non-English → 400 |
| eSpeak-NG | language selects the eSpeak voice |
| XTTS (Docker) | `language_id` |

The UI sends the selected language with every transcribe and synthesize call (default `fi`).

---

## Folder Structure

```
Speechos/
├── CLAUDE.md, README.md, config.example.yaml
├── start.sh / start.ps1        # Docker launcher (auto GPU/CPU), --dev for local
├── dev.ps1, package.json       # Local dev (API + web) with process-tree cleanup
├── .github/workflows/ci.yml    # API pytest + ruff, web lint/typecheck/build
├── docker/
│   ├── api.Dockerfile          # CUDA 12.8 runtime base (matches torch cu128)
│   ├── api-cpu.Dockerfile      # python:3.12-slim
│   ├── web.Dockerfile          # node:22-alpine, serves :36301
│   ├── *.Dockerfile.dockerignore
│   ├── docker-compose.gpu.yml / docker-compose.cpu.yml
│   ├── tts-engines.yml         # Docker-backed TTS engines (WSL2)
│   └── nginx.conf              # / → web, /api/ → api
├── api/
│   ├── pyproject.toml, uv.lock, requirements-ci.txt
│   ├── src/
│   │   ├── __main__.py         # python -m src (uvicorn factory)
│   │   ├── server.py           # create_app(), CORS from config
│   │   ├── config.py           # hardware detection, tiers, overrides
│   │   ├── lang.py             # language codes, Piper voices, Kokoro codes
│   │   ├── models.py           # ModelManager (loaders, cache, locks)
│   │   ├── audio.py            # decode / resample / WAV encode
│   │   ├── analysis.py         # features, emotion, diarization, VAD
│   │   ├── docker_tts.py       # async docker compose control + HTTP proxy
│   │   └── routers/            # HTTP endpoints
│   ├── tests/                  # pytest (no GPU, no model downloads)
│   └── bench/                  # Finnish STT/TTS benchmark (separate, `bench` extra)
├── web/src/{app,components,hooks,lib}
├── docs/{architecture,research,plans}
└── samples/                    # Test WAVs (incl. fi_harri_sample.wav)
```

`models/` and `recordings/` are created relative to the API's working directory
(`api/` when run with `uv run python -m src`) unless `SPEECHOS_MODEL_DIR` /
`SPEECHOS_RECORDINGS_DIR` say otherwise.

---

## API Reference

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok"}` |
| POST | `/api/transcribe` | multipart `file`, optional `language` (`fi`, `en`, `auto`) | `{text, segments[{start,end,text}], language, language_probability, duration, processing_time}` |
| POST | `/api/synthesize` | JSON `{text, language?, speed? (0.25–4), voice?, slot? (default\|fast\|expressive\|clone)}` | `audio/wav` (mono 16-bit) |
| POST | `/api/analyze` | multipart `file` | `{features, emotions[], primary_emotion?, diarization{segments,summary}, speech{segments,speech_seconds,speech_ratio}?, *_available, processing_time}` |
| GET | `/api/recordings` | — | `{recordings[{id, filename, size_bytes, duration, created_at}]}` |
| POST | `/api/recordings` | multipart `file` (any supported format) | stored 16 kHz mono WAV metadata |
| GET | `/api/recordings/{id}/audio` | — | `audio/wav` |
| DELETE | `/api/recordings/{id}` | — | `{deleted}` |
| GET | `/api/system/info` | — | hardware profile, configured + loaded models |
| GET | `/api/system/model-options` | — | selectable models per category with `installed` flag |
| POST | `/api/system/switch-model` | JSON `{category, engine, model}` (`engine: "none"` disables) | new selection; unloads the old model and its variants |
| POST | `/api/models/unload/{key}` | — | unload status |
| GET | `/api/system/docker-tts` | — | Docker TTS container status |

Errors: 400 invalid input / unsupported language, 413 upload too large, 422 schema
validation, 503 engine not installed or not configured. Interactive docs: `/docs`.

---

## Configuration

Precedence: **tier defaults < `config.yaml` < `SPEECHOS_*` env vars**. See
[HARDWARE-TIERS.md](HARDWARE-TIERS.md) for the tier matrix and the full override list,
and `config.example.yaml` for the file format. CORS origins come from
`server.cors_origins` / `SPEECHOS_CORS_ORIGINS`.

`model_loading: eager` preloads STT, TTS and VAD in the background at startup
(`/health` answers immediately); the default `lazy` loads on first request.

---

## Testing and CI

```bash
cd api && uv run pytest              # unit tests: no GPU, no model downloads
cd api && uv run pytest -m models    # real-model smoke tests (CPU, downloads weights)
cd api && uv run ruff check src tests
cd web && pnpm lint && pnpm typecheck && pnpm build
```

Unit tests mock the models (fake Whisper/Piper/Chatterbox objects injected into
`ModelManager`) and torch (`sys.modules`), so CI installs only `api/requirements-ci.txt`
(no torch). Tests cover tier detection, config overrides, audio conversion (incl.
WebM/Opus), every router, language routing, CORS, `/health` latency during a long
inference, and per-model locking.

---

## Docker Deployment

```bash
./start.sh            # Linux/Mac — auto-detect GPU/CPU
.\start.ps1           # Windows
docker compose -f docker/docker-compose.gpu.yml up --build
docker compose -f docker/docker-compose.cpu.yml up --build
```

- API image: `uv sync --frozen` from `api/uv.lock`, `python -m src`, port 36300,
  healthcheck `GET /health`; volumes `/models`, `/recordings`.
- Web image: built with `NEXT_PUBLIC_API_URL` (build arg, default
  `http://localhost:36300` — the browser calls the API directly), serves port 36301.
- nginx (port 80): `/` → web, `/api/` → api.
- Pass `HF_TOKEN` (GPU compose) to enable pyannote community-1 diarization.

---

## Roadmap

Not implemented yet (previous versions of this document listed some of these as existing):

- **Streaming**: WebSocket endpoints (`/ws/stream`, streaming STT/TTS). nginx already
  proxies `/ws/`.
- **VRAM management**: budget-based LRU eviction, `vram.max_usage_gb` / `pinned`.
  Today models stay loaded until switched or unloaded (`/api/models/unload/{key}`).
- **Fine-grained analysis endpoints** (`/api/analyze/emotion`, `/features`, `/diarize`)
  and `/api/voices`; model download endpoint.
- **Expressive / cloning slots in tier defaults**: `tts_expressive` / `tts_clone` are
  selectable via config and the `slot` request field, but no tier enables them by default.
- **CPU-only torch build**: the lockfile pins CUDA 12.8 torch wheels for every platform;
  the CPU image works but carries the CUDA libraries.
- **Finnish models beyond Whisper + Piper** — see `docs/research/` and the FI benchmark
  (`api/bench/`).
- Batch processing, word-level timestamps in the UI, analysis charts over time.

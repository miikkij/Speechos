# Speechos Project: Condition Analysis

**Date**: 2026-02-27
**Branch**: main (clean, no uncommitted changes)
**Total Commits**: 8
**Development Span**: ~6.5 hours (single session, 15:42–22:02)

---

## Executive Summary

Speechos is in **early development / functional MVP stage**. The core recording, transcription, TTS, and analysis pipeline works end-to-end. The project was built rapidly in a single session and has solid architecture, documentation, and Docker infrastructure: but has not yet been battle-tested in production or multi-session development.

---

## CLAUDE.md Accuracy Audit

| Claim | Status | Notes |
|-------|--------|-------|
| Frontend: Next.js + React + TypeScript + Tailwind | **Correct** | Next.js 15.3.3, React 19, TS 5.8, Tailwind 4.1 |
| Backend: Python + FastAPI + uv | **Correct** | Python 3.11+, FastAPI 0.115.0+, uv verified |
| STT: faster-whisper, Vosk | **Correct** | Both in models.py and pyproject.toml |
| TTS: Kokoro, Piper, Orpheus 3B | **Correct** | In config and docker/tts-engines.yml |
| Emotion: emotion2vec+ | **Correct** | In analysis.py and config |
| Analysis: PyAnnote, Silero VAD, librosa | **Correct** | In dependencies and code |
| Hardware tier auto-detection | **Correct** | config.py implements full tier system |
| Audio: 16kHz mono WAV | **Correct** | audio.py: TARGET_SAMPLE_RATE=16000, TARGET_CHANNELS=1 |
| Docker GPU/CPU variants | **Correct** | Both compose files + Dockerfiles present |
| start.sh / start.ps1 launchers | **Correct** | Both scripts present and functional |
| Models dir gitignored | **Correct** | .gitignore has `models/` |
| Recordings dir gitignored | **Correct** | .gitignore has `recordings/` |

**Verdict**: CLAUDE.md is **fully accurate**: no discrepancies found.

---

## Codebase Statistics

| Metric | Count |
|--------|-------|
| Frontend Components | 8 (~1,033 lines) |
| Frontend Hooks | 2 |
| API Core Modules | 8 (~2,500 lines) |
| API Routers | 6 (~915 lines) |
| Test Files | 5 (~1,031 lines) |
| Docker Compose Files | 3 |
| Dockerfiles | 3 (GPU, CPU, Web) |
| Documentation Files | 7 |
| Sample Audio Files | 6 |
| API Dependencies | 65+ |
| Web Dependencies | 7 core + 9 dev |

---

## Component Inventory

### Frontend (`web/src/`)

| Component | Lines | Status | Purpose |
|-----------|-------|--------|---------|
| `Recorder.tsx` | 191 | Functional | Mic recording + file upload |
| `AudioPlayer.tsx` | 86 | Functional | WAV playback + duration |
| `TranscriptionView.tsx` | 63 | Functional | STT results display |
| `AnalysisView.tsx` | 259 | Functional | Emotion + acoustic features |
| `TtsPlayground.tsx` | 126 | Functional | TTS input + model select |
| `ModelSelectors.tsx` | 103 | Functional | STT/TTS model switching |
| `StatusBar.tsx` | 78 | Functional | API connection status |
| `RecordingsLibrary.tsx` | 127 | Functional | Recording management |

### Backend (`api/src/`)

| Module | Lines | Status | Purpose |
|--------|-------|--------|---------|
| `server.py` | 74 | Functional | FastAPI app + CORS + lifespan |
| `config.py` | 334 | Functional | Hardware detection + tier system |
| `models.py` | 367 | Functional | Model manager + lazy loading + LRU |
| `audio.py` | 120 | Functional | Audio conversion + validation |
| `analysis.py` | 471 | Functional | Feature extraction + emotion |
| `docker_tts.py` | 232 | Functional | Docker container management |

### API Endpoints

| Router | Endpoints | Status |
|--------|-----------|--------|
| `health.py` | `GET /health` | Functional |
| `transcribe.py` | `POST /api/transcribe` | Functional |
| `synthesize.py` | `POST /api/synthesize` | Functional |
| `analyze.py` | `POST /api/analyze` | Functional |
| `system.py` | `GET /api/system/info` + model switching | Functional |
| `recordings.py` | `GET/POST/DELETE /api/recordings` | Functional |

---

## Development Timeline

| Commit | Time | Phase | Description |
|--------|------|-------|-------------|
| `72f2298` | 15:42 | Scaffolding | Docker, docs, model research, launchers |
| `28d33ed` | 16:14 | Core MVP | Full app: backend + frontend + deps |
| `99c7ef7` | 16:14 | Cleanup | Formatting/consistency refactor |
| `d44be72` | 17:03 | Analysis | AnalysisView + analysis backend |
| `3934907` | 21:07 | Model UI | ModelSelectors, RecordingsLibrary, tests |
| `1f302e0` | 21:07 | Fix | Whitespace in catch blocks |
| `e28c21b` | 21:26 | Docker TTS | Docker engine management + HTTPX |
| `110a40b` | 22:02 | Robustness | Enhanced model mgmt + integration test |

---

## Strengths

1. **Well-organized architecture**: Clear separation of frontend, backend, Docker, docs
2. **Hardware abstraction**: Sophisticated 15-tier system with auto-detection (CPU/GPU/hybrid)
3. **Model flexibility**: Multiple STT/TTS/emotion engines with CPU fallbacks
4. **Production infrastructure**: Multi-stage Dockerfiles, nginx reverse proxy, health checks
5. **Comprehensive documentation**: Architecture docs, model comparison research, hardware tier matrix
6. **Test coverage**: 5 test files with integration, model, and fix verification tests
7. **Cross-platform**: Bash (Linux/Mac) and PowerShell (Windows) launchers
8. **Type safety**: TypeScript strict mode, Python type hints throughout
9. **Modern patterns**: Async/await, lazy model loading, LRU cache eviction

---

## Risks and Concerns

### High Priority
1. **No CI/CD pipeline**: No GitHub Actions, no automated testing on push
2. **Single-session development**: Entire codebase built in ~6.5 hours; no multi-session iteration yet
3. **No `.env` template**: Missing `.env.example` for environment variables
4. **`tmp_options.json` in api/**: Development artifact left in codebase

### Medium Priority
5. **No error boundary in React**: Frontend crashes could be unhandled
6. **No rate limiting**: API endpoints have no request throttling
7. **WebSocket not fully implemented**: Infrastructure exists but real-time streaming not wired
8. **No authentication**: API is open; fine for local-first but risky if exposed

### Low Priority
9. **No changelog**: Hard to track changes between versions
10. **No linting CI**: ESLint config exists but not enforced in pipeline
11. **`dev.ps1` not mentioned in CLAUDE.md**: Minor documentation gap

---

## Recommended Next Steps

1. **Validate end-to-end locally**: Run `api` and `web` together, test recording → transcription → analysis flow
2. **Set up CI/CD**: GitHub Actions for linting, type checking, and pytest
3. **Add `.env.example`**: Document required/optional environment variables
4. **Clean up `tmp_options.json`**: Remove development artifact
5. **Add error boundaries**: React error boundary for graceful frontend failures
6. **Test Docker builds**: Verify both GPU and CPU Docker stacks build and run
7. **Add WebSocket streaming**: Wire up real-time transcription for live recording
8. **Create `learned_lessons.md`**: Track mistakes and corrections going forward
9. **Create `work.log`**: Track session progress for continuity

---

## Conclusion

The project is in a **solid MVP state** with good architecture and documentation. The main risk is that it was built in a single rapid session and needs multi-session validation, testing, and hardening. The CLAUDE.md is accurate and comprehensive. The next phase should focus on validation, CI/CD, and iterative improvement.

# Plan 001: WSL2 Fixes, GPU Model Verification & Testing

**Created**: 2026-02-27
**Status**: Ready for execution
**Environment**: WSL2, Ubuntu, RTX 4090 (24GB VRAM, ~20GB free), Docker 29.1.2

---

## Current Environment Audit

| Component | Status | Issue |
|-----------|--------|-------|
| GPU | RTX 4090 24GB, ~20GB free | None |
| Docker | v29.1.2, Compose v5.0.0 | None |
| NVIDIA Docker | GPU passthrough works | None |
| uv | v0.5.11 | None |
| Python | **3.10.12 (system)** | Needs 3.11+ (pyproject.toml requires >=3.11) |
| Node.js | v24.13.0 (nvm) | None |
| pnpm | 10.22.0 (**Windows binary** at /mnt/c/) | Needs native Linux pnpm |
| .env | HF_TOKEN set | None |
| Recordings | 11 WAV files in api/recordings/ | None |
| Samples | 6 WAV files in samples/ | None |

---

## Bugs Found

### Bug 1: Root `pnpm dev` calls PowerShell (CRITICAL)

**File**: `package.json` (root)
**Line**: `"dev": "powershell -ExecutionPolicy Bypass -File dev.ps1"`
**Problem**: WSL2 has no PowerShell. This is a Windows-only script.
**Fix**: Add cross-platform dev script using bash.

```json
{
  "scripts": {
    "dev": "bash dev.sh",
    "dev:win": "powershell -ExecutionPolicy Bypass -File dev.ps1",
    "dev:api": "cd api && uv run python -m src",
    "dev:web": "cd web && pnpm dev",
    "install:all": "cd web && pnpm install",
    "build": "cd web && pnpm build"
  }
}
```

Create `dev.sh`: a bash equivalent of `dev.ps1` that starts API + web concurrently with cleanup on Ctrl+C.

### Bug 2: Python 3.10 vs 3.11+ requirement

**File**: `api/pyproject.toml`
**Line**: `requires-python = ">=3.11,<3.14"`
**Problem**: WSL2 system Python is 3.10.12. uv needs Python 3.11+ to install deps.
**Fix**: Use `uv python install 3.12` to get a managed Python. Then `uv sync` will use it.

### Bug 3: pnpm is the Windows binary

**Path**: `/mnt/c/Program Files/nodejs/pnpm`
**Problem**: This is the Windows pnpm via PATH leak. It may work for some things but is unreliable in WSL2.
**Fix**: Install native pnpm via `corepack enable pnpm` (Node.js already has corepack) or `npm install -g pnpm`.

### Bug 4: Docker Healthcheck port mismatch (api.Dockerfile)

**File**: `docker/api.Dockerfile`
**Lines**: HEALTHCHECK uses port 8000, EXPOSE uses 8000
**Problem**: The API server starts on port 36300 (from config.py). The Dockerfile EXPOSE and HEALTHCHECK reference port 8000.
**Fix**:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:36300/health || exit 1

EXPOSE 36300
```

### Bug 5: Docker Healthcheck port mismatch (web.Dockerfile)

**File**: `docker/web.Dockerfile`
**Lines**: HEALTHCHECK on :3000, EXPOSE 3000
**Problem**: Next.js default is 3000 but compose/nginx expects 36301. The `pnpm start` command doesn't specify `-p 36301`.
**Fix**:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD wget -q --spider http://localhost:36301/ || exit 1

EXPOSE 36301

CMD ["pnpm", "start", "-p", "36301"]
```

### Bug 6: api.Dockerfile health check endpoint wrong

**File**: `docker/api.Dockerfile`
**Problem**: Health check calls `/api/health` but the route is mounted as `/health` (not under /api prefix).
**Fix**: Use `http://localhost:36300/health`

---

## Phase 1: Fix WSL2 Development Environment

### Step 1.1: Install Python 3.12 via uv
```bash
uv python install 3.12
```

### Step 1.2: Install native pnpm
```bash
corepack enable pnpm
# Or if corepack isn't available:
npm install -g pnpm
```

### Step 1.3: Create dev.sh (Linux equivalent of dev.ps1)

Create a bash script that:
- Starts API server (`cd api && uv run python -m src`)
- Starts web frontend (`cd web && pnpm dev`)
- Traps SIGINT/SIGTERM and kills both
- Shows both outputs with colored prefixes

### Step 1.4: Update root package.json

Change `"dev"` to use `dev.sh` on Linux, keep `dev:win` for PowerShell.

### Step 1.5: Install frontend dependencies
```bash
cd web && pnpm install
```

### Step 1.6: Sync API dependencies
```bash
cd api && uv sync
```

### Step 1.7: Verify both services start
```bash
# Terminal 1
cd api && uv run python -m src

# Terminal 2
cd web && pnpm dev
```

---

## Phase 2: Fix Docker Configuration

### Step 2.1: Fix api.Dockerfile
- Change EXPOSE from 8000 to 36300
- Fix HEALTHCHECK URL to `http://localhost:36300/health`

### Step 2.2: Fix web.Dockerfile
- Change EXPOSE from 3000 to 36301
- Fix CMD to include `-p 36301`
- Fix HEALTHCHECK to port 36301

### Step 2.3: Test Docker GPU stack
```bash
docker compose -f docker/docker-compose.gpu.yml up --build
```

### Step 2.4: Test Docker CPU stack
```bash
docker compose -f docker/docker-compose.cpu.yml up --build
```

---

## Phase 3: GPU Model Verification

With RTX 4090 (24GB), the system will auto-detect as `hybrid-24gb-gpu` tier.

### Models that run natively (inside API server with GPU):

| Model | Type | VRAM Needed | Available | Docker Image Needed |
|-------|------|-------------|-----------|-------------------|
| **faster-whisper large-v3** | STT | ~4 GB (float16) | Via pip (faster-whisper) | No |
| **emotion2vec+ large** | Emotion | ~2 GB | Via pip (funasr) | No |
| **Silero VAD v5** | VAD | <1 GB | Via pip (silero-vad) | No |
| **resemblyzer** | Diarization | <1 GB | Via pip (resemblyzer) | No |
| **Piper** | TTS | CPU only | Via pip (piper-tts) | No |
| **Kokoro** | TTS | ~1 GB | Via pip (kokoro) | No |

**Total native VRAM**: ~7-8 GB (plenty of room in 16 GB budget)

### TTS Docker engines (optional, GPU):

| Engine | Docker Image | Port | VRAM | Image Verified |
|--------|-------------|------|------|----------------|
| **XTTS-v2** | `ghcr.io/coqui-ai/tts` | 36310 | ~6 GB | Likely works (official image) |
| **ChatTTS** | `yikchunnnn/chattts-dockerized:latest` | 36311 | ~2 GB | Community image: needs testing |
| **MeloTTS** | `sensejworld/melotts:v0.0.4` | 36312 | ~1 GB | Community image: needs testing |
| **Orpheus 3B** | `neosun/orpheus-tts:v1.5.0-allinone` | 36313 | ~8 GB | Community image: needs testing |
| **Fish Speech** | `fishaudio/fish-speech:latest-webui-cuda` | 36314 | ~4 GB | Official image: likely works |
| **CosyVoice** | `catcto/cosyvoice:latest` | 36315 | ~4 GB | Community image: needs testing |
| **Qwen3-TTS** | Needs build from source | 36316 | ~4 GB | Requires Dockerfile creation |
| **Parler-TTS** | `fedirz/parler-tts-server:latest` | 36317 | ~6 GB | Community image: needs testing |

**Important**: Only one Docker TTS engine should run at a time (they share GPU). Total VRAM with native models (~8GB) + one TTS engine (~6GB max) = ~14GB, well within 16GB budget.

### Verification strategy (one engine at a time):
```bash
# Test each Docker TTS engine
docker compose -f docker/tts-engines.yml up -d xtts
curl http://localhost:36310/api/tts?text=Hello
docker compose -f docker/tts-engines.yml stop xtts

# Repeat for each engine
```

---

## Phase 4: End-to-End Testing

### Step 4.1: Test with existing recordings
Use the 11 WAV files in `api/recordings/` to test:
1. **Transcription** (STT): POST /api/transcribe with each WAV
2. **Analysis** (emotion + features): POST /api/analyze with each WAV
3. **System info**: GET /api/system/info: verify tier detection

### Step 4.2: Test TTS pipeline
1. **Piper TTS**: POST /api/synthesize with text
2. **Kokoro TTS**: Switch model then synthesize
3. **Docker TTS**: Start one engine, test via API

### Step 4.3: Run existing tests
```bash
cd api && uv run pytest -v
```

### Step 4.4: Test from web UI
1. Open http://localhost:36301
2. Record audio from mic (test 48kHz default)
3. Transcribe recording
4. Run analysis
5. Test TTS playback
6. Switch models from UI

---

## Phase 5: Documentation Updates

### Step 5.1: Update CLAUDE.md
- Add WSL2/Linux-specific commands
- Reference dev.sh

### Step 5.2: Update docs/work.log
- Record all changes made

### Step 5.3: Update docs/learned_lessons.md
- Document any issues found during testing

---

## Execution Order

```
Phase 1 (WSL2 dev fixes): must be first, everything depends on this
    ↓
Phase 2 (Docker fixes): can happen in parallel with Phase 1 testing
    ↓
Phase 3 (GPU model verification): requires Phase 1 working
    ↓
Phase 4 (E2E testing): requires Phase 1 + Phase 3
    ↓
Phase 5 (Documentation): after all fixes are verified
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Python 3.12 install fails in WSL2 | Low | High | Use deadsnakes PPA as fallback |
| uv sync fails (dep conflicts) | Medium | High | Check error logs, pin conflicting deps |
| Docker TTS images don't exist/broken | Medium | Medium | Test one at a time, document which work |
| CUDA OOM with multiple models loaded | Low | Medium | Lazy loading + LRU eviction already implemented |
| Qwen3-TTS needs custom Dockerfile | High | Low | Skip for now, test other engines first |
| piper-tts won't install on 3.12 | Medium | Medium | Check wheel availability, use 3.11 if needed |

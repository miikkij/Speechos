# Plan 002: Comprehensive Engine Catalog & Benchmarking Strategy

**Created**: 2026-02-27
**Environment**: WSL2, RTX 4090 24GB (~16GB usable), Docker 29.1.2

---

## TTS Engines (Text-to-Speech)

### Already in tts-engines.yml

| # | Engine | Docker Image | Port | VRAM | API Type | Status |
|---|--------|-------------|------|------|----------|--------|
| 1 | XTTS-v2 | `ghcr.io/coqui-ai/tts` | 36310 | ~6 GB | REST | Configured |
| 2 | ChatTTS | `yikchunnnn/chattts-dockerized` | 36311 | ~2 GB | REST | Configured |
| 3 | MeloTTS | `sensejworld/melotts:v0.0.4` | 36312 | ~1 GB | REST | Configured |
| 4 | Orpheus 3B | `neosun/orpheus-tts:v1.5.0-allinone` | 36313 | ~8 GB | REST | Configured |
| 5 | Fish Speech | `fishaudio/fish-speech:latest-webui-cuda` | 36314 | ~4 GB | WebUI | Configured |
| 6 | CosyVoice | `catcto/cosyvoice:latest` | 36315 | ~4 GB | REST | Configured |
| 7 | Qwen3-TTS | Needs update (see below) | 36316 | ~6 GB | REST | **Needs fix** |
| 8 | Parler-TTS | `fedirz/parler-tts-server:latest` | 36317 | ~6 GB | REST | Configured |

### NEW: Qwen3-TTS (Updated Config)

**Best option: groxaxo/Qwen3-TTS-Openai-Fastapi**: OpenAI-compatible `/v1/audio/speech` endpoint.

| Property | Value |
|----------|-------|
| Repo | https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi |
| API | OpenAI-compatible: `POST /v1/audio/speech` |
| Voices | 9 preset (Vivian, Ryan, Serena, Dylan, Eric, Aiden, etc.) + clone:ProfileName |
| Languages | 10 (EN, ZH, JA, KO, DE, FR, RU, PT, ES, IT) |
| Formats | mp3, opus, aac, flac, wav, pcm |
| Streaming | Yes (PCM) |
| Port | 8880 |
| 0.6B VRAM | ~3-4 GB |
| 1.7B VRAM | ~5-6 GB |
| RTX 4090 RTF (1.7B) | 0.65-0.85 (faster than real-time) |
| First token latency | 97ms |

**Alternative pre-built image:** `ghcr.io/louis-pujol/qwen3-tts:latest` (Gradio UI, port 8000, less API-friendly)

### Native TTS (pip-installed in API server)

| Engine | Package | VRAM | CPU? |
|--------|---------|------|------|
| Piper | `piper-tts` | N/A (CPU only) | Yes |
| Kokoro | `kokoro` | ~1 GB | Yes |

---

## STT Engines (Speech-to-Text)

### Tier 1: Ready to benchmark (pre-built Docker + HTTP API + GPU)

| # | Engine | Docker Image | Port | VRAM | API | Key Feature |
|---|--------|-------------|------|------|-----|-------------|
| 1 | **Speaches** (faster-whisper) | `ghcr.io/speaches-ai/speaches:latest-cuda` | 8000 | 4 GB (large-v3 INT8) | OpenAI-compat | Dynamic model loading, streaming |
| 2 | **Whisper ASR** (3 engines) | `onerahmet/openai-whisper-asr-webservice:latest-gpu` | 9000 | Varies | REST+Swagger | Supports openai_whisper, faster_whisper, whisperx from ONE image |
| 3 | **Whisper.cpp** | `ghcr.io/ggml-org/whisper.cpp:main-cuda` | 8080 | 5 GB (large-v3 GGML) | REST /inference | Low memory, C++ speed |
| 4 | **Parakeet TDT** (NIM) | `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us` | 9000 | 4-6 GB | OpenAI-compat + gRPC | SOTA accuracy, requires NGC key |

### Tier 2: Usable with minor config

| # | Engine | Docker Image | Port | VRAM | API | Notes |
|---|--------|-------------|------|------|-----|-------|
| 5 | **LinTO NeMo** (Parakeet) | `lintoai/linto-stt-nemo` | 80 | ~2 GB (0.6B) | REST | No NGC key needed |
| 6 | **LinTO Whisper** | `lintoai/linto-stt-whisper-gpu` | 80 | Varies | REST | Supports HF model IDs |
| 7 | **WhisperX API** | `pluja/whisperx-api` | 8000 | ~10 GB | REST | With diarization |

### Tier 3: Requires custom Docker build

| # | Engine | Install | VRAM | Notes |
|---|--------|---------|------|-------|
| 8 | **Vosk** | `alphacep/kaldi-en` | ~512 MB | WebSocket only (port 2700), no REST |
| 9 | **Canary Qwen 2.5B** | `nvcr.io/nvidia/nemo:25.11.01` | ~8 GB | Needs custom server wrapper |
| 10 | **Wav2Vec 2.0** | pip + Triton | 2-4 GB | Needs custom FastAPI wrapper |
| 11 | **Moonshine** | pip only | CPU only | Edge-focused, needs custom server |
| 12 | **Distil-Whisper** | Via Speaches/LinTO | ~3 GB | Specify model ID in request |

### Native STT (pip-installed in API server)

| Engine | Package | VRAM | Notes |
|--------|---------|------|-------|
| faster-whisper | `faster-whisper` | 4 GB (large-v3 INT8) | Primary |
| Vosk | `vosk` | CPU only | Lightweight fallback |

---

## Speech Analysis Engines (Emotion, Diarization, VAD, Features)

### Pre-built Docker available

| # | Engine | Docker Image | Port | VRAM | API | Task |
|---|--------|-------------|------|------|-----|------|
| 1 | **FunASR** (emotion2vec+) | `funasr/funasr` or Alibaba CR | 10095 | 1-2 GB | WebSocket | ASR+VAD+Emotion+Diarization all-in-one |
| 2 | **PyAnnote** (LinTO) | `lintoai/linto-diarization-pyannote` | 8080 | 2-6 GB | REST POST /diarization | Speaker diarization |
| 3 | **NeMo MSDD** | `nvcr.io/nvidia/nemo:25.11.01` | N/A | 4-64 GB | Framework (needs wrapper) | Multi-scale diarization |

### Requires custom Docker build

| # | Engine | Install | VRAM | Task | Notes |
|---|--------|---------|------|------|-------|
| 4 | **emotion2vec+ large** | pip (funasr) | ~2 GB | Emotion (9 categories) | Best SER model |
| 5 | **emotion2vec+ base** | pip (funasr) | ~1 GB | Emotion (9 categories) | Lighter version |
| 6 | **SpeechBrain ECAPA** | pip (speechbrain) | ~1 GB | Speaker verification | Pairwise comparison |
| 7 | **wav2vec2 SER** | pip (transformers) | 1-2 GB | Emotion (English) | Fine-tuned on RAVDESS |
| 8 | **WavLM SER** | pip (audonnx) | ~2 GB | Arousal/Valence/Dominance | Dimensional emotion |
| 9 | **Resemblyzer** | pip (resemblyzer) | <0.5 GB | Speaker embeddings | Very lightweight |

### CPU-only (no GPU needed)

| # | Engine | Install | RAM | Task |
|---|--------|---------|-----|------|
| 10 | **Silero VAD** | pip (silero-vad) | ~50 MB | Voice activity detection |
| 11 | **openSMILE** | pip (opensmile) | <1 GB | 6,373 acoustic features |
| 12 | **librosa** | pip (librosa) | <1 GB | MFCCs, pitch, tempo |

### Native Analysis (pip-installed in API server)

| Engine | Package | GPU? | Current Status |
|--------|---------|------|----------------|
| emotion2vec+ | funasr | Yes | Configured for gpu-24gb tier |
| Silero VAD | silero-vad | No | Configured all tiers |
| Resemblyzer | resemblyzer | Optional | Configured for diarization |
| librosa | librosa | No | Configured for features |
| openSMILE | opensmile | No | Installed |

---

## Benchmarking Architecture

### Docker Port Allocation

```
TTS Engines:     36310-36319
STT Engines:     36320-36329
Analysis Engines: 36330-36339
Core Services:   36300 (API), 36301 (Web)
```

### Proposed STT Docker Config (new file: `docker/stt-engines.yml`)

| Service | Image | Port | Env |
|---------|-------|------|-----|
| speaches | `ghcr.io/speaches-ai/speaches:latest-cuda` | 36320 |: |
| whisper-asr | `onerahmet/openai-whisper-asr-webservice:latest-gpu` | 36321 | ASR_ENGINE=faster_whisper |
| whisper-cpp | `ghcr.io/ggml-org/whisper.cpp:main-cuda` | 36322 |: |
| linto-nemo | `lintoai/linto-stt-nemo` | 36323 | MODEL=nvidia/parakeet-tdt-0.6b-v2 |
| linto-whisper | `lintoai/linto-stt-whisper-gpu` | 36324 | MODEL=large-v3 |
| whisperx-api | `pluja/whisperx-api` | 36325 | HF_TOKEN from .env |
| vosk | `alphacep/kaldi-en` | 36326 | WebSocket |

### Proposed Analysis Docker Config (new file: `docker/analysis-engines.yml`)

| Service | Image | Port | Task |
|---------|-------|------|------|
| pyannote | `lintoai/linto-diarization-pyannote` | 36330 | Speaker diarization |
| funasr | `funasr/funasr` | 36331 | ASR+emotion (WebSocket) |

---

## VRAM Budget (RTX 4090 24GB, 16GB usable)

### Scenario A: Benchmarking one engine at a time
Run one Docker engine + native API models (~8GB) = ~14GB max. All engines fit.

### Scenario B: Native API with max models loaded
| Model | VRAM |
|-------|------|
| faster-whisper large-v3 (float16) | ~4 GB |
| emotion2vec+ large | ~2 GB |
| Kokoro TTS | ~1 GB |
| Silero VAD | CPU |
| **Total** | **~7 GB** |

Leaves ~9 GB for one Docker TTS/STT engine.

### Scenario C: Full benchmark run
Run engines sequentially (one at a time), collecting results per engine.

---

## Execution Plan

### Phase 1: Fix Qwen3-TTS in tts-engines.yml (replace custom build with groxaxo)
### Phase 2: Create `docker/stt-engines.yml`
### Phase 3: Create `docker/analysis-engines.yml`
### Phase 4: Test each engine (one at a time):

```
For each engine:
  1. docker compose -f docker/<category>-engines.yml up -d <engine>
  2. Run test with sample recordings
  3. Collect: accuracy, latency, VRAM usage, error rate
  4. docker compose -f docker/<category>-engines.yml stop <engine>
  5. Log results to docs/benchmarks/
```

### Phase 5: Build benchmarking UI in web frontend
### Phase 6: Write results to `docs/benchmarks/`

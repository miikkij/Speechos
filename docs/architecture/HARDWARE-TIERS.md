# Hardware Tier System & Model Selection

> Last updated: 2026-10-09. Source of truth: `TIER_CONFIGS` in `api/src/config.py`.

## Overview

At startup Speechos detects CPU cores, RAM, and (via PyTorch) CUDA GPU + VRAM, picks a
**compute mode** and a **tier**, and loads that tier's model defaults. `config.yaml` and
`SPEECHOS_*` environment variables override any of it.

Defaults are Finnish-first: every tier synthesizes with **Piper `fi_FI-harri-medium`**
(FI benchmark CER 3.5 %) and transcribes with multilingual **faster-whisper**
(large-v3 FLEURS-fi WER 7.9 % on GPU tiers). A request in another language switches to
that language's Piper voice automatically (see ARCHITECTURE.md → Languages).

---

## Mode and tier detection

```
mode = SPEECHOS_COMPUTE_MODE / config.yaml compute_mode     (cpu | gpu | hybrid | auto)
       else hybrid  if a CUDA GPU is usable and RAM >= 8 GB
       else gpu     if a CUDA GPU is usable
       else cpu

tier = SPEECHOS_TIER / config.yaml tier  (validated against the tier list)
       else by mode:
         cpu:    RAM  <= 2 / 4 / 8 / 16 GB  → cpu-2gb / cpu-4gb / cpu-8gb / cpu-16gb, else cpu-32gb
         gpu:    VRAM <= 4 / 8 / 12 / 24 GB → gpu-4gb / gpu-8gb / gpu-12gb / gpu-24gb, else gpu-32gb
         hybrid: VRAM <= 4 / 8 / 12 / 24 GB → hybrid-4gb-gpu … hybrid-24gb-gpu, else hybrid-32gb-gpu
```

"Usable" means `torch.cuda.is_available()`, at least one device, and the device can be
queried; anything else (no torch, broken driver, `CUDA_VISIBLE_DEVICES=""`) is CPU.
Forcing `SPEECHOS_COMPUTE_MODE=cpu` on a GPU machine selects a CPU tier and makes
`/api/system/switch-model` place new models on the CPU.

---

## Tier matrix

| Tier | STT (faster-whisper) | TTS | Emotion | Diarization | VAD |
|---|---|---|---|---|---|
| **cpu-2gb** | tiny int8 (cpu) | Piper fi_FI-harri-medium | — | — | Silero |
| **cpu-4gb** | tiny int8 (cpu) | Piper fi_FI-harri-medium | — | — | Silero |
| **cpu-8gb** | base int8 (cpu) | Piper fi_FI-harri-medium | — | — | Silero |
| **cpu-16gb** | small int8 (cpu) | Piper fi_FI-harri-medium (+ `tts_fast`) | — | — | Silero |
| **cpu-32gb** | medium int8 (cpu) | Piper fi_FI-harri-medium | — | — | Silero |
| **gpu-4gb** | small float16 (cuda) | Piper fi_FI-harri-medium | — | — | Silero |
| **gpu-8gb** | turbo float16 (cuda) | Piper fi_FI-harri-medium | emotion2vec base (cuda) | — | Silero |
| **gpu-12gb** | large-v3 int8_float16 (cuda) | Piper fi_FI-harri-medium | emotion2vec base (cuda) | pyannote† | Silero |
| **gpu-24gb** | large-v3 float16 (cuda) | Piper fi_FI-harri-medium | emotion2vec+ large (cuda) | pyannote† | Silero |
| **gpu-32gb** | large-v3 float16 (cuda) | Piper fi_FI-harri-medium | emotion2vec+ large (cuda) | pyannote† | Silero |
| **hybrid-4gb-gpu** | small float16 (cuda) | Piper fi_FI-harri-medium | — | — | Silero |
| **hybrid-8gb-gpu** | turbo float16 (cuda) | Piper fi_FI-harri-medium | emotion2vec base (cuda) | — | Silero |
| **hybrid-12gb-gpu** | large-v3 int8_float16 (cuda) | Piper fi_FI-harri-medium | emotion2vec base (cuda) | pyannote† | Silero |
| **hybrid-24gb-gpu** | large-v3 float16 (cuda) | Piper fi_FI-harri-medium | emotion2vec+ large (cuda) | pyannote† | Silero |
| **hybrid-32gb-gpu** | large-v3 float16 (cuda) | Piper fi_FI-harri-medium | emotion2vec+ large (cuda) | pyannote† | Silero |

Piper and Silero always run on the CPU. emotion2vec models are
`iic/emotion2vec_base_finetuned` and `iic/emotion2vec_plus_large` (FunASR).

† **pyannote `pyannote/speaker-diarization-community-1`** (cuda) when `HF_TOKEN` is set
*and* `pyannote.audio` is installed (`uv sync --extra gpu`; accept the model's terms on
HuggingFace first). Otherwise **Resemblyzer** d-vector clustering (cuda if available).

cpu-2gb used Vosk before; the bundled Vosk model is English-only, so it now uses Whisper
tiny. Vosk remains selectable (`SPEECHOS_STT_ENGINE=vosk`).

Models download on first use: Whisper → `<model_dir>/whisper`, Piper →
`<model_dir>/piper`, Vosk → `<model_dir>/vosk`; HuggingFace models use the HF cache.

---

## Overrides

### config.yaml

Searched in this order: `SPEECHOS_CONFIG`, `./config.yaml`, `./config.yml`,
`../config.yaml` (relative to the API's working directory). See `config.example.yaml`.

```yaml
compute_mode: auto          # cpu | gpu | hybrid | auto
tier: auto                  # any tier name, or auto
stt: {engine: faster-whisper, model: large-v3, device: cuda, compute_type: float16}
tts: {engine: piper, model: fi_FI-harri-medium}
emotion: {engine: none}     # "none" disables a slot
server: {host: 0.0.0.0, port: 36300, workers: 1, cors_origins: ["http://localhost:36301"]}
storage: {model_dir: ./models, recordings_dir: ./recordings, samples_dir: ./samples, max_upload_mb: 100}
model_loading: lazy         # lazy | eager
```

Model slots: `stt`, `tts`, `tts_fast`, `tts_expressive`, `tts_clone`, `emotion`,
`diarization`, `vad`. A slot given in YAML replaces the tier default completely.

### Environment variables

```bash
SPEECHOS_TIER=gpu-24gb                # force tier
SPEECHOS_COMPUTE_MODE=cpu             # cpu | gpu | hybrid

# Per slot: SPEECHOS_<SLOT>_ENGINE / _MODEL / _DEVICE / _COMPUTE_TYPE
# SLOT = STT, TTS, TTS_FAST, TTS_EXPRESSIVE, TTS_CLONE, EMOTION, DIARIZATION, VAD
SPEECHOS_STT_MODEL=large-v3
SPEECHOS_STT_COMPUTE_TYPE=int8_float16
SPEECHOS_TTS_ENGINE=kokoro            # a new engine does not inherit the old model name
SPEECHOS_EMOTION_ENGINE=none          # disable

SPEECHOS_HOST=0.0.0.0
SPEECHOS_PORT=36300
SPEECHOS_WORKERS=1                    # keep 1 with GPU models (each worker loads its own)
SPEECHOS_MODEL_DIR=/models
SPEECHOS_RECORDINGS_DIR=/recordings
SPEECHOS_CORS_ORIGINS=http://localhost:36301,http://localhost
SPEECHOS_MODEL_LOADING=eager
SPEECHOS_CONFIG=/path/to/config.yaml
HF_TOKEN=hf_...                       # enables pyannote community-1 on GPU >= 12 GB tiers
```

Env vars win over `config.yaml`, which wins over tier defaults. Runtime changes through
`POST /api/system/switch-model` last until restart.

---

## Memory management

- Models load lazily on first request (or in the background with `model_loading: eager`)
  and stay cached until switched or unloaded (`POST /api/models/unload/{key}`).
- Each cached model has its own lock: concurrent requests to the same model queue up,
  different models run in parallel.
- Docker TTS engines run one GPU container at a time (starting one stops the others).

Not implemented (see ARCHITECTURE.md → Roadmap): VRAM budgets and LRU eviction, pinned
models, automatic CPU fallback when VRAM runs out.

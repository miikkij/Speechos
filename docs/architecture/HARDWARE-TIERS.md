# Hardware Tier System & Model Selection

> Last updated: 2026-02-27

## Overview

Speechos auto-detects available hardware (CPU cores, RAM, GPU presence, VRAM) at startup and selects the optimal model combination. Users can also override with explicit configuration via environment variables or `config.yaml`.

---

## Hardware Tiers

### CPU-Only Tiers

| Tier | RAM Available | STT Model | TTS Model | Emotion Model | Diarization | VAD | Total RAM Used |
|---|---|---|---|---|---|---|---|
| **cpu-2gb** | ≤ 2 GB | Vosk small (40MB) | Piper (15MB) | wav2small (10M) | ❌ | Silero (2M) | ~1.5 GB |
| **cpu-4gb** | ≤ 4 GB | Faster-Whisper tiny INT8 | Piper medium | emotion2vec+ seed | ❌ | Silero | ~3 GB |
| **cpu-8gb** | ≤ 8 GB | Faster-Whisper base INT8 | Kokoro | emotion2vec+ base | PyAnnote | Silero | ~6 GB |
| **cpu-16gb** | ≤ 16 GB | Faster-Whisper small INT8 | Kokoro + Piper | emotion2vec+ base | PyAnnote | Silero | ~10 GB |
| **cpu-32gb** | ≤ 32 GB | Faster-Whisper medium INT8 | Kokoro + Orpheus (llama.cpp Q4) | emotion2vec+ large | PyAnnote | Silero | ~20 GB |

### GPU-Only Tiers

| Tier | VRAM Available | STT Model | TTS Model | Emotion Model | Diarization | VAD | Total VRAM Used |
|---|---|---|---|---|---|---|---|
| **gpu-4gb** | ≤ 4 GB | Faster-Whisper small FP16 | Kokoro | emotion2vec+ seed | ❌ | Silero | ~3.5 GB |
| **gpu-8gb** | ≤ 8 GB | Faster-Whisper turbo FP16 | Kokoro | emotion2vec+ base | PyAnnote | Silero | ~7 GB |
| **gpu-12gb** | ≤ 12 GB | Faster-Whisper large-v3 INT8 | Kokoro + Chatterbox | emotion2vec+ base | PyAnnote | Silero | ~10 GB |
| **gpu-24gb** | ≤ 24 GB | Faster-Whisper large-v3 FP16 | Kokoro + Orpheus 3B FP16 | emotion2vec+ large | PyAnnote | Silero | ~22 GB |
| **gpu-32gb** (4090 Ti) | ≤ 32 GB | Faster-Whisper large-v3 FP16 | Orpheus 3B FP16 + Kokoro + Chatterbox | emotion2vec+ large | PyAnnote | Silero | ~28 GB |

### CPU + GPU Combination Tiers

In hybrid mode, GPU handles inference-heavy models (STT, TTS) while CPU handles lightweight tasks (VAD, feature extraction, diarization). This allows loading more models by spreading across both.

| Tier | RAM / VRAM | GPU Models | CPU Models | Total VRAM | Total RAM |
|---|---|---|---|---|---|
| **hybrid-4gb-gpu** | 8GB RAM / 4GB VRAM | STT: FW small FP16 | TTS: Piper, Emotion: e2v seed, VAD: Silero | ~2.5 GB | ~3 GB |
| **hybrid-8gb-gpu** | 16GB RAM / 8GB VRAM | STT: FW turbo FP16, TTS: Kokoro | Emotion: e2v base, Diarize: PyAnnote, VAD: Silero | ~6 GB | ~4 GB |
| **hybrid-12gb-gpu** | 16GB RAM / 12GB VRAM | STT: FW large-v3 INT8, TTS: Kokoro | Emotion: e2v base, Diarize: PyAnnote, VAD: Silero, TTS2: Piper | ~8 GB | ~4 GB |
| **hybrid-24gb-gpu** | 32GB RAM / 24GB VRAM | STT: FW large-v3 FP16, TTS: Orpheus FP16 | Emotion: e2v large CPU, Diarize: PyAnnote, TTS2: Piper | ~20 GB | ~6 GB |
| **hybrid-32gb-gpu** (4090 Ti) | 64GB RAM / 32GB VRAM | STT: FW large-v3 FP16, TTS: Orpheus FP16 + Kokoro GPU | Emotion: e2v large CPU, Diarize: PyAnnote, TTS3: Piper CPU | ~26 GB | ~6 GB |

---

## Your 4090 Ti Setup (32GB VRAM)

With a 4090 Ti, you can run **everything at maximum quality simultaneously**:

```
GPU (32 GB VRAM):
├── STT:  Faster-Whisper large-v3 FP16              ~10 GB
├── TTS:  Orpheus 3B FP16 (emotional, highest quality) ~8 GB
├── TTS:  Kokoro (fast fallback, on GPU)               ~1 GB
├── SER:  emotion2vec+ large                           ~2 GB
├── Diarize: PyAnnote 3.1                              ~1 GB
├── Reserved/overhead                                  ~10 GB
└── Total: ~22 GB used, ~10 GB headroom

CPU (system RAM):
├── VAD:  Silero VAD                                   ~50 MB
├── TTS:  Piper (instant CPU fallback)                 ~100 MB
├── Features: librosa, torchaudio                      ~200 MB
└── Total: ~350 MB
```

**Capabilities unlocked:**
- ✅ Maximum accuracy transcription (2.7% WER)
- ✅ Emotional TTS with tags (`<happy>`, `<sad>`, etc.)
- ✅ Voice cloning (Chatterbox can be loaded on demand)
- ✅ 9-class emotion recognition (multilingual)
- ✅ Speaker diarization ("who spoke when")
- ✅ Real-time streaming
- ✅ All features running concurrently

---

## Auto-Detection Logic

```python
# api/src/config/hardware.py

import os
from dataclasses import dataclass
from enum import Enum

class ComputeMode(Enum):
    CPU_ONLY = "cpu"
    GPU_ONLY = "gpu"
    HYBRID = "hybrid"

@dataclass
class HardwareProfile:
    mode: ComputeMode
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    gpu_name: str | None
    vram_gb: float
    tier: str

def detect_hardware() -> HardwareProfile:
    """Auto-detect available hardware and determine tier."""
    import psutil
    
    cpu_cores = psutil.cpu_count(logical=False)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    gpu_available = False
    gpu_name = None
    vram_gb = 0.0
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        pass
    
    # Determine mode from env or auto-detect
    mode_override = os.environ.get("SPEECHOS_COMPUTE_MODE")
    if mode_override:
        mode = ComputeMode(mode_override)
    elif gpu_available and ram_gb >= 8:
        mode = ComputeMode.HYBRID
    elif gpu_available:
        mode = ComputeMode.GPU_ONLY
    else:
        mode = ComputeMode.CPU_ONLY
    
    # Determine tier
    tier = _resolve_tier(mode, ram_gb, vram_gb)
    
    return HardwareProfile(
        mode=mode,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        tier=tier,
    )

def _resolve_tier(mode: ComputeMode, ram_gb: float, vram_gb: float) -> str:
    """Map hardware specs to a named tier."""
    tier_override = os.environ.get("SPEECHOS_TIER")
    if tier_override:
        return tier_override
    
    if mode == ComputeMode.CPU_ONLY:
        if ram_gb <= 2: return "cpu-2gb"
        if ram_gb <= 4: return "cpu-4gb"
        if ram_gb <= 8: return "cpu-8gb"
        if ram_gb <= 16: return "cpu-16gb"
        return "cpu-32gb"
    
    if mode == ComputeMode.GPU_ONLY:
        if vram_gb <= 4: return "gpu-4gb"
        if vram_gb <= 8: return "gpu-8gb"
        if vram_gb <= 12: return "gpu-12gb"
        if vram_gb <= 24: return "gpu-24gb"
        return "gpu-32gb"
    
    # Hybrid
    if vram_gb <= 4: return "hybrid-4gb-gpu"
    if vram_gb <= 8: return "hybrid-8gb-gpu"
    if vram_gb <= 12: return "hybrid-12gb-gpu"
    if vram_gb <= 24: return "hybrid-24gb-gpu"
    return "hybrid-32gb-gpu"
```

---

## Model Selection per Tier

```python
# api/src/config/tiers.py

TIER_CONFIGS = {
    # ── CPU-Only ──────────────────────────────────────────
    "cpu-2gb": {
        "stt": {"engine": "vosk", "model": "vosk-model-small-en-us-0.15", "device": "cpu"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "wav2small", "model": "audeering/wav2small", "device": "cpu"},
        "diarization": None,
        "vad": {"engine": "silero", "device": "cpu"},
    },
    "cpu-4gb": {
        "stt": {"engine": "faster-whisper", "model": "tiny", "device": "cpu", "compute_type": "int8"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+seed", "device": "cpu"},
        "diarization": None,
        "vad": {"engine": "silero", "device": "cpu"},
    },
    "cpu-8gb": {
        "stt": {"engine": "faster-whisper", "model": "base", "device": "cpu", "compute_type": "int8"},
        "tts": {"engine": "kokoro", "model": "kokoro-v0.19", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+base", "device": "cpu"},
        "diarization": {"engine": "pyannote", "device": "cpu"},
        "vad": {"engine": "silero", "device": "cpu"},
    },
    "cpu-16gb": {
        "stt": {"engine": "faster-whisper", "model": "small", "device": "cpu", "compute_type": "int8"},
        "tts": {"engine": "kokoro", "model": "kokoro-v0.19", "device": "cpu"},
        "tts_fast": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+base", "device": "cpu"},
        "diarization": {"engine": "pyannote", "device": "cpu"},
        "vad": {"engine": "silero", "device": "cpu"},
    },
    "cpu-32gb": {
        "stt": {"engine": "faster-whisper", "model": "medium", "device": "cpu", "compute_type": "int8"},
        "tts": {"engine": "kokoro", "model": "kokoro-v0.19", "device": "cpu"},
        "tts_expressive": {"engine": "orpheus", "model": "orpheus-3b-Q4_K_M", "device": "cpu", "runtime": "llama.cpp"},
        "tts_fast": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+large", "device": "cpu"},
        "diarization": {"engine": "pyannote", "device": "cpu"},
        "vad": {"engine": "silero", "device": "cpu"},
    },

    # ── GPU-Only ──────────────────────────────────────────
    "gpu-4gb": {
        "stt": {"engine": "faster-whisper", "model": "small", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "kokoro", "model": "kokoro-v0.19", "device": "cuda"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+seed", "device": "cuda"},
        "diarization": None,
        "vad": {"engine": "silero", "device": "cuda"},
    },
    "gpu-8gb": {
        "stt": {"engine": "faster-whisper", "model": "turbo", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "kokoro", "model": "kokoro-v0.19", "device": "cuda"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+base", "device": "cuda"},
        "diarization": {"engine": "pyannote", "device": "cuda"},
        "vad": {"engine": "silero", "device": "cuda"},
    },
    "gpu-12gb": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "int8_float16"},
        "tts": {"engine": "kokoro", "model": "kokoro-v0.19", "device": "cuda"},
        "tts_clone": {"engine": "chatterbox", "model": "chatterbox-tts", "device": "cuda"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+base", "device": "cuda"},
        "diarization": {"engine": "pyannote", "device": "cuda"},
        "vad": {"engine": "silero", "device": "cuda"},
    },
    "gpu-24gb": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "kokoro", "model": "kokoro-v0.19", "device": "cuda"},
        "tts_expressive": {"engine": "orpheus", "model": "orpheus-3b", "device": "cuda", "compute_type": "float16"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+large", "device": "cuda"},
        "diarization": {"engine": "pyannote", "device": "cuda"},
        "vad": {"engine": "silero", "device": "cuda"},
    },
    "gpu-32gb": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "kokoro", "model": "kokoro-v0.19", "device": "cuda"},
        "tts_expressive": {"engine": "orpheus", "model": "orpheus-3b", "device": "cuda", "compute_type": "float16"},
        "tts_clone": {"engine": "chatterbox", "model": "chatterbox-tts", "device": "cuda"},
        "tts_fast": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+large", "device": "cuda"},
        "diarization": {"engine": "pyannote", "device": "cuda"},
        "vad": {"engine": "silero", "device": "cuda"},
    },

    # ── Hybrid (CPU + GPU) ────────────────────────────────
    "hybrid-4gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "small", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+seed", "device": "cpu"},
        "diarization": None,
        "vad": {"engine": "silero", "device": "cpu"},
    },
    "hybrid-8gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "turbo", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "kokoro", "model": "kokoro-v0.19", "device": "cuda"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+base", "device": "cpu"},
        "diarization": {"engine": "pyannote", "device": "cpu"},
        "vad": {"engine": "silero", "device": "cpu"},
    },
    "hybrid-12gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "int8_float16"},
        "tts": {"engine": "kokoro", "model": "kokoro-v0.19", "device": "cuda"},
        "tts_fast": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+base", "device": "cpu"},
        "diarization": {"engine": "pyannote", "device": "cpu"},
        "vad": {"engine": "silero", "device": "cpu"},
    },
    "hybrid-24gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "tts_expressive": {"engine": "orpheus", "model": "orpheus-3b", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "kokoro", "model": "kokoro-v0.19", "device": "cuda"},
        "tts_fast": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+large", "device": "cpu"},
        "diarization": {"engine": "pyannote", "device": "cpu"},
        "vad": {"engine": "silero", "device": "cpu"},
    },
    "hybrid-32gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "tts_expressive": {"engine": "orpheus", "model": "orpheus-3b", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "kokoro", "model": "kokoro-v0.19", "device": "cuda"},
        "tts_clone": {"engine": "chatterbox", "model": "chatterbox-tts", "device": "cuda"},
        "tts_fast": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "emotion2vec+large", "device": "cpu"},
        "diarization": {"engine": "pyannote", "device": "cpu"},
        "vad": {"engine": "silero", "device": "cpu"},
    },
}
```

---

## Environment Variable Overrides

```bash
# Force a specific tier (skip auto-detection)
SPEECHOS_TIER=gpu-32gb

# Force compute mode
SPEECHOS_COMPUTE_MODE=hybrid   # cpu | gpu | hybrid

# Override individual models
SPEECHOS_STT_ENGINE=faster-whisper
SPEECHOS_STT_MODEL=large-v3
SPEECHOS_STT_DEVICE=cuda
SPEECHOS_STT_COMPUTE_TYPE=float16

SPEECHOS_TTS_ENGINE=kokoro
SPEECHOS_TTS_DEVICE=cuda

SPEECHOS_EMOTION_ENGINE=emotion2vec
SPEECHOS_EMOTION_MODEL=emotion2vec+large

# Model cache directory
SPEECHOS_MODEL_DIR=/models

# Server config
SPEECHOS_HOST=0.0.0.0
SPEECHOS_PORT=36300
SPEECHOS_WORKERS=1
```

---

## VRAM Budget Calculator

When running multiple models concurrently on GPU, VRAM is the bottleneck. Models are loaded/unloaded dynamically, but the system reserves VRAM budgets:

```
Priority order (what stays loaded in VRAM):
1. STT model    : always loaded (most used)
2. Primary TTS  : always loaded
3. Emotion model: loaded on demand, kept warm
4. Diarization  : loaded on demand
5. Voice cloning: loaded on demand (can evict #4)
6. Expressive TTS: loaded on demand (can evict #5)

Strategy: "Lazy load, LRU evict"
- Models are loaded into VRAM on first request
- Least-recently-used models are evicted when VRAM is tight
- CPU fallback is always available for evicted models
```

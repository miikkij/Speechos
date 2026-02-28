"""System info and model management endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import ModelConfig
from src.server import get_config, get_models

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])

# Available model options the user can choose from
MODEL_OPTIONS = {
    "stt": [
        # Faster-Whisper (CTranslate2): primary engine
        {"engine": "faster-whisper", "model": "tiny", "label": "Whisper Tiny (fastest, least accurate)"},
        {"engine": "faster-whisper", "model": "base", "label": "Whisper Base"},
        {"engine": "faster-whisper", "model": "small", "label": "Whisper Small"},
        {"engine": "faster-whisper", "model": "medium", "label": "Whisper Medium"},
        {"engine": "faster-whisper", "model": "turbo", "label": "Whisper Turbo (fast + accurate)"},
        {"engine": "faster-whisper", "model": "large-v3", "label": "Whisper Large v3 (best accuracy)"},
        # Distil-Whisper via faster-whisper
        {"engine": "faster-whisper", "model": "Systran/faster-distil-whisper-large-v3", "label": "Distil-Whisper Large v3 (fast + accurate)"},
        # WhisperX: word-level timestamps + alignment
        {"engine": "whisperx", "model": "large-v3", "label": "WhisperX Large v3 (word timestamps + alignment)"},
        # Vosk: lightweight CPU-only Kaldi
        {"engine": "vosk", "model": "vosk-model-small-en-us-0.15", "label": "Vosk Small EN (40MB, CPU, real-time)"},
        {"engine": "vosk", "model": "vosk-model-en-us-0.22", "label": "Vosk Large EN (1.8GB, better accuracy)"},
        # Moonshine: ultra-lightweight
        {"engine": "moonshine", "model": "moonshine/tiny", "label": "Moonshine Tiny (27M, ultra-fast)"},
        {"engine": "moonshine", "model": "moonshine/base", "label": "Moonshine Base"},
        # NVIDIA NeMo: GPU-only
        {"engine": "nemo", "model": "nvidia/parakeet-tdt-1.1b", "label": "NVIDIA Parakeet TDT 1.1B (GPU only)"},
        {"engine": "nemo", "model": "nvidia/canary-qwen-2.5b", "label": "Canary Qwen 2.5B (GPU only, multilingual)"},
        # Wav2Vec 2.0: via transformers
        {"engine": "wav2vec2-stt", "model": "facebook/wav2vec2-base-960h", "label": "Wav2Vec 2.0 Base (95M)"},
        {"engine": "wav2vec2-stt", "model": "facebook/wav2vec2-large-960h", "label": "Wav2Vec 2.0 Large (317M)"},
        # ── Docker engines (run in WSL2 containers, tested 2026-02-28) ──
        {"engine": "speaches", "model": "Systran/faster-whisper-small", "label": "[Docker] Speaches (faster-whisper, OpenAI-compatible)"},
        {"engine": "whisper-asr", "model": "large-v3", "label": "[Docker] Whisper ASR Webservice (3 engines)"},
        {"engine": "linto-nemo", "model": "parakeet-tdt-0.6b-v2", "label": "[Docker] Parakeet TDT 0.6B v2 (best balanced)"},
        {"engine": "linto-whisper", "model": "large-v3", "label": "[Docker] LinTO Whisper (GPU)"},
        {"engine": "linto-nemo-1.1b", "model": "parakeet-tdt-1.1b", "label": "[Docker] Parakeet TDT 1.1B (best accuracy, lowercase)"},
    ],
    "tts": [
        # ── Native engines (run in-process) ─────────────────
        # Piper: ultra-fast CPU ONNX
        {"engine": "piper", "model": "en_US-lessac-medium", "label": "Piper: Lessac Medium (default, fastest)"},
        {"engine": "piper", "model": "en_US-lessac-high", "label": "Piper: Lessac High Quality"},
        {"engine": "piper", "model": "en_US-amy-medium", "label": "Piper: Amy Medium"},
        {"engine": "piper", "model": "en_US-ryan-medium", "label": "Piper: Ryan Medium"},
        {"engine": "piper", "model": "en_US-arctic-medium", "label": "Piper: Arctic Medium"},
        {"engine": "piper", "model": "en_US-libritts-high", "label": "Piper: LibriTTS High"},
        {"engine": "piper", "model": "en_GB-alan-medium", "label": "Piper: Alan British Medium"},
        # Kokoro: #1 TTS Arena, 82M params
        {"engine": "kokoro", "model": "default", "label": "Kokoro 82M (best quality/size, CPU+GPU)"},
        # Chatterbox: best voice cloning, MIT
        {"engine": "chatterbox", "model": "default", "label": "Chatterbox (best voice cloning, MIT)"},
        # Bark: versatile with effects
        {"engine": "bark", "model": "suno/bark", "label": "Bark (music/effects, slow, GPU rec.)"},
        {"engine": "bark", "model": "suno/bark-small", "label": "Bark Small (faster, less quality)"},
        # eSpeak-NG: rule-based, instant
        {"engine": "espeak", "model": "en", "label": "eSpeak-NG (robotic, instant, 100+ langs)"},
        # ── Docker engines (run in WSL2 containers) ─────────
        # Qwen3-TTS: SOTA quality, emotion control
        {"engine": "qwen3-tts", "model": "Qwen3-TTS-0.6B", "label": "[Docker] Qwen3-TTS 0.6B (SOTA, emotion control)"},
        # XTTS-v2: Coqui TTS, 17 languages, voice cloning
        {"engine": "xtts", "model": "xtts_v2", "label": "[Docker] XTTS-v2 (17 langs, voice cloning)"},
        # ChatTTS: dialog-optimized, prosody control
        {"engine": "chattts", "model": "default", "label": "[Docker] ChatTTS (dialog speech, prosody)"},
        # MeloTTS: multilingual, fast
        {"engine": "melotts", "model": "EN", "label": "[Docker] MeloTTS (multilingual, fast)"},
        # Orpheus 3B: emotional speech
        {"engine": "orpheus", "model": "orpheus-3b", "label": "[Docker] Orpheus 3B (emotional speech)"},
        # Fish Speech 1.5: real-time, voice cloning
        {"engine": "fish-speech", "model": "fish-speech-1.5", "label": "[Docker] Fish Speech 1.5 (voice cloning)"},
        # CosyVoice 2: DISABLED: Docker image catcto/cosyvoice doesn't exist on Docker Hub
        # Parler-TTS: description-driven voice control
        {"engine": "parler", "model": "parler-tts-mini-v1", "label": "[Docker] Parler-TTS Mini (description-driven)"},
    ],
    "emotion": [
        # emotion2vec+: best general-purpose SER
        {"engine": "emotion2vec", "model": "iic/emotion2vec_base_finetuned", "label": "emotion2vec+ Base (recommended)"},
        {"engine": "emotion2vec", "model": "iic/emotion2vec_plus_large", "label": "emotion2vec+ Large (best accuracy)"},
        # HuBERT SER: via transformers audio-classification pipeline
        {"engine": "hubert-ser", "model": "superb/hubert-large-superb-er", "label": "HuBERT SER Large (4 emotions)"},
        {"engine": "none", "model": "none", "label": "Disabled"},
        # REMOVED: wav2vec2-ser (ehcalabres, Dpngtm, r-f): untrained classifier heads, random scores
        # REMOVED: WavLM SER (3loi): model not found on HuggingFace
        # REMOVED: wav2small: regression model, incompatible with pipeline
    ],
    "diarization": [
        # PyAnnote: state-of-the-art (requires HuggingFace gated access)
        {"engine": "pyannote", "model": "pyannote/speaker-diarization-3.1", "label": "PyAnnote 3.1 (best accuracy, gated model)"},
        # SpeechBrain: speaker verification/ID
        {"engine": "speechbrain", "model": "spkrec-ecapa-voxceleb", "label": "SpeechBrain ECAPA (speaker verification)"},
        # Resemblyzer: lightweight d-vector embeddings
        {"engine": "resemblyzer", "model": "dvector", "label": "Resemblyzer (lightweight speaker ID)"},
        {"engine": "none", "model": "none", "label": "Disabled"},
    ],
    "vad": [
        {"engine": "silero", "model": "v5", "label": "Silero VAD v5 (default, lightweight)"},
        {"engine": "pyannote-vad", "model": "pyannote/segmentation-3.0", "label": "PyAnnote VAD (more accurate, gated)"},
    ],
}


@router.get("/system/info")
async def system_info():
    """Return hardware profile, tier, and loaded models."""
    config = get_config()
    hw = config.hardware

    model_configs = {}
    for key in ("stt", "tts", "emotion", "diarization", "vad"):
        mc = getattr(config, key, None)
        if mc:
            model_configs[key] = {
                "engine": mc.engine,
                "model": mc.model,
                "device": mc.device,
            }

    return {
        "hardware": {
            "mode": hw.mode.value,
            "tier": hw.tier,
            "cpu_cores": hw.cpu_cores,
            "ram_gb": hw.ram_gb,
            "gpu_available": hw.gpu_available,
            "gpu_name": hw.gpu_name,
            "vram_gb": hw.vram_gb,
        },
        "models": {
            "configured": model_configs,
            "loaded": get_models().loaded_models,
        },
    }


def _check_engine_installed(engine: str) -> bool:
    """Check if the Python package for an engine is actually installed."""
    try:
        if engine == "faster-whisper":
            import faster_whisper  # noqa: F401
        elif engine == "vosk":
            import vosk  # noqa: F401
        elif engine == "nemo":
            import nemo.collections.asr  # noqa: F401
        elif engine == "whisperx":
            import whisperx  # noqa: F401
        elif engine == "moonshine":
            import moonshine  # noqa: F401
        elif engine in ("wav2vec2-stt", "wav2vec2-ser", "hubert-ser", "wavlm-ser", "wav2small"):
            import transformers  # noqa: F401
        elif engine == "emotion2vec":
            import funasr  # noqa: F401
        elif engine == "pyannote":
            import pyannote.audio  # noqa: F401
        elif engine == "pyannote-vad":
            import pyannote.audio  # noqa: F401
        elif engine == "speechbrain":
            import speechbrain  # noqa: F401
        elif engine == "resemblyzer":
            from resemblyzer import VoiceEncoder  # noqa: F401
        elif engine == "silero":
            import torch  # noqa: F401
        elif engine == "piper":
            import piper  # noqa: F401
        elif engine == "kokoro":
            import kokoro  # noqa: F401
        elif engine == "chatterbox":
            import chatterbox  # noqa: F401
        elif engine == "bark":
            import transformers  # noqa: F401
        elif engine == "espeak":
            pass  # system binary, assume available
        elif engine == "none":
            pass
        elif engine in ("xtts", "chattts", "melotts", "orpheus", "fish-speech", "parler", "qwen3-tts"):
            pass  # Docker TTS engines, always "available" as options
        elif engine in ("speaches", "whisper-asr", "linto-nemo", "linto-whisper", "linto-nemo-1.1b"):
            pass  # Docker STT engines, always "available" as options
        else:
            return False
        return True
    except ImportError:
        return False

# Cache availability at import time so we don't re-check every request
_engine_availability: dict[str, bool] = {}

def _is_engine_installed(engine: str) -> bool:
    if engine not in _engine_availability:
        _engine_availability[engine] = _check_engine_installed(engine)
    return _engine_availability[engine]


@router.get("/system/model-options")
async def model_options():
    """Return available model options for each category."""
    config = get_config()
    hw = config.hardware

    # Return current selection alongside options
    current = {}
    for key in ("stt", "tts", "emotion", "diarization", "vad"):
        mc = getattr(config, key, None)
        if mc:
            current[key] = {"engine": mc.engine, "model": mc.model}
        else:
            current[key] = {"engine": "none", "model": "none"}

    # Add installed flag to each option
    options_with_status: dict[str, list[dict]] = {}
    for category, opts in MODEL_OPTIONS.items():
        options_with_status[category] = [
            {**opt, "installed": _is_engine_installed(opt["engine"])}
            for opt in opts
        ]

    return {
        "options": options_with_status,
        "current": current,
        "gpu_available": hw.gpu_available,
    }


class SwitchModelRequest(BaseModel):
    category: str
    engine: str
    model: str


@router.post("/system/switch-model")
async def switch_model(req: SwitchModelRequest):
    """Switch a model at runtime. Unloads the old model and configures the new one."""
    config = get_config()
    models = get_models()

    if req.category not in ("stt", "tts", "emotion", "diarization", "vad"):
        raise HTTPException(400, f"Invalid category: {req.category}")

    # Handle disabling a model
    if req.engine == "none":
        models.unload(req.category)
        setattr(config, req.category, None)
        return {"status": "disabled", "category": req.category}

    # Determine device
    device = "cpu"
    compute_type = "int8"
    if config.hardware.gpu_available:
        if req.category in ("stt", "emotion", "diarization"):
            device = "cuda"
            compute_type = "float16"

    new_cfg = ModelConfig(
        engine=req.engine,
        model=req.model,
        device=device,
        compute_type=compute_type if req.category == "stt" else None,
    )

    # Unload old model if loaded
    models.unload(req.category)
    setattr(config, req.category, new_cfg)

    # Pre-warm Docker containers so they're ready when needed
    docker_status = None
    if req.category == "stt":
        from src.docker_stt import is_docker_stt_engine, ensure_stt_ready
        if is_docker_stt_engine(req.engine):
            ready = ensure_stt_ready(req.engine)
            docker_status = "ready" if ready else "failed"
    elif req.category == "tts":
        from src.docker_tts import is_docker_engine, is_container_running, start_container
        if is_docker_engine(req.engine):
            if not is_container_running(req.engine):
                ready = start_container(req.engine)
                docker_status = "ready" if ready else "failed"
            else:
                docker_status = "ready"

    result = {
        "status": "switched",
        "category": req.category,
        "engine": req.engine,
        "model": req.model,
        "device": device,
    }
    if docker_status:
        result["docker_status"] = docker_status
    return result


@router.post("/models/unload/{model_key}")
async def unload_model(model_key: str):
    """Unload a model from memory."""
    models = get_models()
    if model_key not in models.loaded_models:
        return {"status": "not_loaded", "model": model_key}
    models.unload(model_key)
    return {"status": "unloaded", "model": model_key}


@router.get("/system/docker-tts")
async def docker_tts_status():
    """Return status of Docker TTS engine containers."""
    from src.docker_tts import DOCKER_TTS_ENGINES, is_container_running

    statuses = {}
    for engine, cfg in DOCKER_TTS_ENGINES.items():
        statuses[engine] = {
            "service": cfg["service"],
            "port": cfg["port"],
            "running": is_container_running(engine),
        }
    return {"engines": statuses}


@router.get("/system/docker-stt")
async def docker_stt_status():
    """Return status of Docker STT engine containers."""
    from src.docker_stt import DOCKER_STT_ENGINES, is_stt_container_running

    statuses = {}
    for engine, cfg in DOCKER_STT_ENGINES.items():
        statuses[engine] = {
            "service": cfg["service"],
            "port": cfg["port"],
            "running": is_stt_container_running(engine),
        }
    return {"engines": statuses}

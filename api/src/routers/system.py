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
        # Faster-Whisper (CTranslate2) — primary engine
        {"engine": "faster-whisper", "model": "tiny", "label": "Whisper Tiny (fastest, least accurate)"},
        {"engine": "faster-whisper", "model": "base", "label": "Whisper Base"},
        {"engine": "faster-whisper", "model": "small", "label": "Whisper Small"},
        {"engine": "faster-whisper", "model": "medium", "label": "Whisper Medium"},
        {"engine": "faster-whisper", "model": "turbo", "label": "Whisper Turbo (fast + accurate)"},
        {"engine": "faster-whisper", "model": "large-v3", "label": "Whisper Large v3 (best accuracy)"},
        # Distil-Whisper via faster-whisper
        {"engine": "faster-whisper", "model": "Systran/faster-distil-whisper-large-v3", "label": "Distil-Whisper Large v3 (fast + accurate)"},
        # WhisperX — word-level timestamps + alignment
        {"engine": "whisperx", "model": "large-v3", "label": "WhisperX Large v3 (word timestamps + alignment)"},
        # Vosk — lightweight CPU-only Kaldi
        {"engine": "vosk", "model": "vosk-model-small-en-us-0.15", "label": "Vosk Small EN (40MB, CPU, real-time)"},
        {"engine": "vosk", "model": "vosk-model-en-us-0.22", "label": "Vosk Large EN (1.8GB, better accuracy)"},
        # Moonshine — ultra-lightweight
        {"engine": "moonshine", "model": "moonshine/tiny", "label": "Moonshine Tiny (27M, ultra-fast)"},
        {"engine": "moonshine", "model": "moonshine/base", "label": "Moonshine Base"},
        # NVIDIA NeMo — GPU-only
        {"engine": "nemo", "model": "nvidia/parakeet-tdt-1.1b", "label": "NVIDIA Parakeet TDT 1.1B (GPU only)"},
        {"engine": "nemo", "model": "nvidia/canary-qwen-2.5b", "label": "Canary Qwen 2.5B (GPU only, multilingual)"},
        # Wav2Vec 2.0 — via transformers
        {"engine": "wav2vec2-stt", "model": "facebook/wav2vec2-base-960h", "label": "Wav2Vec 2.0 Base (95M)"},
        {"engine": "wav2vec2-stt", "model": "facebook/wav2vec2-large-960h", "label": "Wav2Vec 2.0 Large (317M)"},
    ],
    "tts": [
        # ── Native engines (run in-process) ─────────────────
        # Piper — ultra-fast CPU ONNX
        {"engine": "piper", "model": "en_US-lessac-medium", "label": "Piper: Lessac Medium (default, fastest)"},
        {"engine": "piper", "model": "en_US-lessac-high", "label": "Piper: Lessac High Quality"},
        {"engine": "piper", "model": "en_US-amy-medium", "label": "Piper: Amy Medium"},
        {"engine": "piper", "model": "en_US-ryan-medium", "label": "Piper: Ryan Medium"},
        {"engine": "piper", "model": "en_US-arctic-medium", "label": "Piper: Arctic Medium"},
        {"engine": "piper", "model": "en_US-libritts-high", "label": "Piper: LibriTTS High"},
        {"engine": "piper", "model": "en_GB-alan-medium", "label": "Piper: Alan British Medium"},
        # Kokoro — #1 TTS Arena, 82M params
        {"engine": "kokoro", "model": "default", "label": "Kokoro 82M (best quality/size, CPU+GPU)"},
        # Chatterbox — best voice cloning, MIT
        {"engine": "chatterbox", "model": "default", "label": "Chatterbox (best voice cloning, MIT)"},
        # Bark — versatile with effects
        {"engine": "bark", "model": "suno/bark", "label": "Bark (music/effects, slow, GPU rec.)"},
        {"engine": "bark", "model": "suno/bark-small", "label": "Bark Small (faster, less quality)"},
        # eSpeak-NG — rule-based, instant
        {"engine": "espeak", "model": "en", "label": "eSpeak-NG (robotic, instant, 100+ langs)"},
        # ── Docker engines (run in WSL2 containers) ─────────
        # Qwen3-TTS — SOTA quality, emotion control
        {"engine": "qwen3-tts", "model": "Qwen3-TTS-0.6B", "label": "[Docker] Qwen3-TTS 0.6B (SOTA, emotion control)"},
        # XTTS-v2 — Coqui TTS, 17 languages, voice cloning
        {"engine": "xtts", "model": "xtts_v2", "label": "[Docker] XTTS-v2 (17 langs, voice cloning)"},
        # ChatTTS — dialog-optimized, prosody control
        {"engine": "chattts", "model": "default", "label": "[Docker] ChatTTS (dialog speech, prosody)"},
        # MeloTTS — multilingual, fast
        {"engine": "melotts", "model": "EN", "label": "[Docker] MeloTTS (multilingual, fast)"},
        # Orpheus 3B — emotional speech
        {"engine": "orpheus", "model": "orpheus-3b", "label": "[Docker] Orpheus 3B (emotional speech)"},
        # Fish Speech 1.5 — real-time, voice cloning
        {"engine": "fish-speech", "model": "fish-speech-1.5", "label": "[Docker] Fish Speech 1.5 (voice cloning)"},
        # CosyVoice 2 — zero-shot cloning, streaming
        {"engine": "cosyvoice", "model": "cosyvoice-300m", "label": "[Docker] CosyVoice 2 (zero-shot cloning)"},
        # Parler-TTS — description-driven voice control
        {"engine": "parler", "model": "parler-tts-mini-v1", "label": "[Docker] Parler-TTS Mini (description-driven)"},
    ],
    "emotion": [
        # emotion2vec+ — best general-purpose SER
        {"engine": "emotion2vec", "model": "iic/emotion2vec_base_finetuned", "label": "emotion2vec+ Base (recommended)"},
        {"engine": "emotion2vec", "model": "iic/emotion2vec_plus_large", "label": "emotion2vec+ Large (best accuracy)"},
        # wav2vec2-based SER — via transformers
        {"engine": "wav2vec2-ser", "model": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", "label": "wav2vec2-XLSR SER (8 emotions, 82.2%)"},
        {"engine": "wav2vec2-ser", "model": "Dpngtm/wav2vec2-emotion", "label": "wav2vec2 Emotion (7 emotions, 79.9%)"},
        {"engine": "wav2vec2-ser", "model": "r-f/wav2vec-english-speech-emotion-recognition", "label": "wav2vec2 SER r-f (7 emotions, 97.5% RAVDESS)"},
        # HuBERT SER
        {"engine": "hubert-ser", "model": "superb/hubert-large-superb-er", "label": "HuBERT SER Large (7-8 emotions)"},
        # WavLM SER — dimensional A/D/V
        {"engine": "wavlm-ser", "model": "3loi/SER-Odyssey-Baseline-WavLM-Multi-Fused", "label": "WavLM SER (arousal/valence/dominance)"},
        # wav2small — tiny dimensional model
        {"engine": "wav2small", "model": "audeering/wav2small", "label": "wav2small (10M, A/D/V, multilingual)"},
        {"engine": "none", "model": "none", "label": "Disabled"},
    ],
    "diarization": [
        # PyAnnote — state-of-the-art (requires HuggingFace gated access)
        {"engine": "pyannote", "model": "pyannote/speaker-diarization-3.1", "label": "PyAnnote 3.1 (best accuracy, gated model)"},
        # SpeechBrain — speaker verification/ID
        {"engine": "speechbrain", "model": "spkrec-ecapa-voxceleb", "label": "SpeechBrain ECAPA (speaker verification)"},
        # Resemblyzer — lightweight d-vector embeddings
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

    return {
        "options": MODEL_OPTIONS,
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

    return {
        "status": "switched",
        "category": req.category,
        "engine": req.engine,
        "model": req.model,
        "device": device,
    }


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

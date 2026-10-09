"""System info and model management endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import MODEL_SLOTS, ComputeMode, ModelConfig
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
        # Moonshine v2 — ultra-lightweight, now multilingual + streaming
        {"engine": "moonshine", "model": "moonshine/tiny", "label": "Moonshine v2 Tiny (27M, ultra-fast)"},
        {"engine": "moonshine", "model": "moonshine/base", "label": "Moonshine v2 Base (multilingual)"},
        # NVIDIA NeMo — GPU-only
        {"engine": "nemo", "model": "nvidia/parakeet-tdt-0.6b-v3", "label": "NVIDIA Parakeet TDT 0.6B v3 (GPU, 25 EU langs, fastest)"},
        {"engine": "nemo", "model": "nvidia/parakeet-tdt-1.1b", "label": "NVIDIA Parakeet TDT 1.1B (GPU only)"},
        {"engine": "nemo", "model": "nvidia/canary-qwen-2.5b", "label": "Canary Qwen 2.5B (GPU only, top accuracy)"},
        {"engine": "nemo", "model": "nvidia/canary-1b-v2", "label": "Canary 1B v2 (GPU, 25 languages)"},
        # IBM Granite Speech — accuracy leader via transformers
        {"engine": "granite-speech", "model": "ibm-granite/granite-speech-4.1-2b", "label": "IBM Granite-Speech 4.1 2B (GPU, high accuracy)"},
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
        # Piper — Finnish voices (rhasspy/piper-voices)
        {"engine": "piper", "model": "fi_FI-harri-medium", "label": "Piper: Harri Finnish Medium (fi, fast, CPU)"},
        {"engine": "piper", "model": "fi_FI-harri-low", "label": "Piper: Harri Finnish Low (fi, fastest)"},
        # Kokoro — #1 TTS Arena, 82M params
        {"engine": "kokoro", "model": "default", "label": "Kokoro 82M (best quality/size, CPU+GPU)"},
        # Kitten TTS — 15M ultra-tiny CPU model (new 2026)
        {"engine": "kitten", "model": "KittenML/kitten-tts-nano-0.8-fp32", "label": "Kitten TTS nano (15M, CPU, tiny)"},
        # Chatterbox — best voice cloning, MIT (Multilingual V3 in 2026)
        {"engine": "chatterbox", "model": "multilingual", "label": "Chatterbox Multilingual V3 (fi + 22 langs, voice cloning, MIT)"},
        {"engine": "chatterbox", "model": "english", "label": "Chatterbox English (voice cloning, MIT)"},
        # Bark — versatile with effects
        {"engine": "bark", "model": "suno/bark", "label": "Bark (music/effects, slow, GPU rec.)"},
        {"engine": "bark", "model": "suno/bark-small", "label": "Bark Small (faster, less quality)"},
        # eSpeak-NG — rule-based, instant
        {"engine": "espeak", "model": "en", "label": "eSpeak-NG (robotic, instant, 100+ langs)"},
        {"engine": "espeak", "model": "fi", "label": "eSpeak-NG Finnish (robotic, instant, fi)"},
        # ── Docker engines (run in WSL2 containers) ─────────
        # VibeVoice — long-form, multi-speaker podcast TTS (new 2026)
        {"engine": "vibevoice", "model": "vibevoice/VibeVoice-1.5B", "label": "[Docker] VibeVoice 1.5B (long-form, multi-speaker)"},
        # Qwen3-TTS — SOTA quality, emotion control (confirmed open weights, Apache-2.0)
        {"engine": "qwen3-tts", "model": "Qwen3-TTS-0.6B", "label": "[Docker] Qwen3-TTS 0.6B (open weights, emotion control)"},
        # XTTS-v2 — Coqui TTS, 17 languages, voice cloning
        {"engine": "xtts", "model": "xtts_v2", "label": "[Docker] XTTS-v2 (17 langs, voice cloning)"},
        # ChatTTS — dialog-optimized, prosody control
        {"engine": "chattts", "model": "default", "label": "[Docker] ChatTTS (dialog speech, prosody)"},
        # MeloTTS — multilingual, fast
        {"engine": "melotts", "model": "EN", "label": "[Docker] MeloTTS (multilingual, fast)"},
        # Orpheus 3B — emotional speech
        {"engine": "orpheus", "model": "orpheus-3b", "label": "[Docker] Orpheus 3B (emotional speech)"},
        # OpenAudio S1-mini — succeeds Fish Speech 1.5 (NON-COMMERCIAL license)
        {"engine": "fish-speech", "model": "fishaudio/openaudio-s1-mini", "label": "[Docker] OpenAudio S1-mini (voice cloning, NC license)"},
        # Fish Speech 1.5 — real-time, voice cloning
        {"engine": "fish-speech", "model": "fish-speech-1.5", "label": "[Docker] Fish Speech 1.5 (voice cloning)"},
        # CosyVoice 3 — zero-shot cloning, streaming (2025-12, supersedes CosyVoice2)
        {"engine": "cosyvoice", "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "label": "[Docker] CosyVoice 3 (zero-shot cloning, 2026)"},
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
        # PyAnnote 4.0 community-1 — supersedes 3.1, free CC-BY-4.0 (new 2026)
        {"engine": "pyannote", "model": "pyannote/speaker-diarization-community-1", "label": "PyAnnote community-1 (4.0, best accuracy, free)"},
        # PyAnnote 3.1 — previous SOTA (requires HuggingFace gated access)
        {"engine": "pyannote", "model": "pyannote/speaker-diarization-3.1", "label": "PyAnnote 3.1 (gated model)"},
        # NVIDIA Sortformer — end-to-end streaming diarization (new 2026, GPU)
        {"engine": "sortformer", "model": "nvidia/diar_streaming_sortformer_4spk-v2", "label": "NVIDIA Streaming Sortformer v2 (GPU, real-time, ≤4 spk)"},
        {"engine": "sortformer", "model": "nvidia/diar_sortformer_4spk-v1", "label": "NVIDIA Sortformer v1 (GPU, offline, ≤4 spk)"},
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
    for key in MODEL_SLOTS:
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
        elif engine in ("wav2vec2-stt", "wav2vec2-ser", "hubert-ser", "wavlm-ser", "wav2small", "granite-speech"):
            import transformers  # noqa: F401
        elif engine == "kitten":
            import kittentts  # noqa: F401
        elif engine == "sortformer":
            import nemo.collections.asr  # noqa: F401
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
            import silero_vad  # noqa: F401
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
        elif engine in ("xtts", "chattts", "melotts", "orpheus", "fish-speech", "cosyvoice", "parler", "qwen3-tts", "vibevoice"):
            pass  # Docker engines, always "available" as options
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
    for key in ("stt", "stt_fi", "tts", "emotion", "diarization", "vad"):
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

    if req.category not in MODEL_SLOTS:
        raise HTTPException(400, f"Invalid category: {req.category}")

    # Handle disabling a model
    if req.engine == "none":
        models.unload(req.category)
        setattr(config, req.category, None)
        return {"status": "disabled", "category": req.category}

    # Determine device
    device = "cpu"
    compute_type = "int8"
    # Native TTS engines that benefit from / require a GPU (Docker engines manage
    # their own device, and piper/kitten/espeak stay on CPU).
    GPU_CAPABLE_TTS = {"chatterbox", "kokoro", "bark", "orpheus"}
    if config.hardware.gpu_available and config.hardware.mode != ComputeMode.CPU:
        if req.category in ("stt", "emotion", "diarization"):
            device = "cuda"
            compute_type = "float16"
        elif req.category.startswith("tts") and req.engine in GPU_CAPABLE_TTS:
            device = "cuda"

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
    import asyncio

    from src.docker_tts import DOCKER_TTS_ENGINES, is_container_running

    engines = list(DOCKER_TTS_ENGINES)
    running = await asyncio.gather(*(is_container_running(e) for e in engines))
    return {
        "engines": {
            engine: {
                "service": DOCKER_TTS_ENGINES[engine]["service"],
                "port": DOCKER_TTS_ENGINES[engine]["port"],
                "running": up,
            }
            for engine, up in zip(engines, running)
        }
    }

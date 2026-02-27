"""System info and model management endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from src.server import get_config, get_models

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


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


@router.post("/models/unload/{model_key}")
async def unload_model(model_key: str):
    """Unload a model from memory."""
    models = get_models()
    if model_key not in models.loaded_models:
        return {"status": "not_loaded", "model": model_key}
    models.unload(model_key)
    return {"status": "unloaded", "model": model_key}

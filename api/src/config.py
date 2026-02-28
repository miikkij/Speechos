"""Speechos API: Hardware detection and tier configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import psutil
import yaml


class ComputeMode(Enum):
    CPU = "cpu"
    GPU = "gpu"
    HYBRID = "hybrid"


@dataclass
class ModelConfig:
    engine: str
    model: str
    device: str = "cpu"
    compute_type: str | None = None
    runtime: str | None = None


@dataclass
class HardwareProfile:
    mode: ComputeMode
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    gpu_name: str | None
    vram_gb: float
    tier: str


@dataclass
class AppConfig:
    hardware: HardwareProfile
    stt: ModelConfig | None = None
    tts: ModelConfig | None = None
    tts_fast: ModelConfig | None = None
    tts_expressive: ModelConfig | None = None
    tts_clone: ModelConfig | None = None
    emotion: ModelConfig | None = None
    diarization: ModelConfig | None = None
    vad: ModelConfig | None = None
    host: str = "0.0.0.0"
    port: int = 36300
    workers: int = 1
    model_dir: str = "./models"
    recordings_dir: str = "./recordings"
    samples_dir: str = "./samples"
    max_upload_mb: int = 100
    model_loading: str = "lazy"
    cors_origins: list[str] = field(default_factory=lambda: [
        "http://localhost:36301",
        "http://localhost:80",
        "http://localhost",
    ])


# ── Tier definitions ──────────────────────────────────────

TIER_CONFIGS: dict[str, dict[str, Any]] = {
    # CPU-only
    "cpu-2gb": {
        "stt": {"engine": "vosk", "model": "vosk-model-small-en-us-0.15", "device": "cpu"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "cpu-4gb": {
        "stt": {"engine": "faster-whisper", "model": "tiny", "device": "cpu", "compute_type": "int8"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "cpu-8gb": {
        "stt": {"engine": "faster-whisper", "model": "base", "device": "cpu", "compute_type": "int8"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "cpu-16gb": {
        "stt": {"engine": "faster-whisper", "model": "small", "device": "cpu", "compute_type": "int8"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "tts_fast": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "cpu-32gb": {
        "stt": {"engine": "faster-whisper", "model": "medium", "device": "cpu", "compute_type": "int8"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    # GPU-only
    "gpu-4gb": {
        "stt": {"engine": "faster-whisper", "model": "small", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "gpu-8gb": {
        "stt": {"engine": "faster-whisper", "model": "turbo", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_base_finetuned", "device": "cuda"},
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "gpu-12gb": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "int8_float16"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_base_finetuned", "device": "cuda"},
        "diarization": {"engine": "resemblyzer", "model": "dvector", "device": "cuda"},
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "gpu-24gb": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_plus_large", "device": "cuda"},
        "diarization": {"engine": "resemblyzer", "model": "dvector", "device": "cuda"},
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "gpu-32gb": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_plus_large", "device": "cuda"},
        "diarization": {"engine": "resemblyzer", "model": "dvector", "device": "cuda"},
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    # Hybrid
    "hybrid-4gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "small", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "hybrid-8gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "turbo", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_base_finetuned", "device": "cuda"},
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "hybrid-12gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "int8_float16"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_base_finetuned", "device": "cuda"},
        "diarization": {"engine": "resemblyzer", "model": "dvector", "device": "cuda"},
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "hybrid-24gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_plus_large", "device": "cuda"},
        "diarization": {"engine": "resemblyzer", "model": "dvector", "device": "cuda"},
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "hybrid-32gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "tts": {"engine": "piper", "model": "en_US-lessac-medium", "device": "cpu"},
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_plus_large", "device": "cuda"},
        "diarization": {"engine": "resemblyzer", "model": "dvector", "device": "cuda"},
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
}


def detect_hardware() -> HardwareProfile:
    """Auto-detect available hardware and determine optimal tier."""
    cpu_cores = psutil.cpu_count(logical=False) or 4
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

    mode_override = os.environ.get("SPEECHOS_COMPUTE_MODE")
    if mode_override:
        mode = ComputeMode(mode_override)
    elif gpu_available and ram_gb >= 8:
        mode = ComputeMode.HYBRID
    elif gpu_available:
        mode = ComputeMode.GPU
    else:
        mode = ComputeMode.CPU

    tier = _resolve_tier(mode, ram_gb, vram_gb)

    return HardwareProfile(
        mode=mode,
        cpu_cores=cpu_cores,
        ram_gb=round(ram_gb, 1),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        vram_gb=round(vram_gb, 1),
        tier=tier,
    )


def _resolve_tier(mode: ComputeMode, ram_gb: float, vram_gb: float) -> str:
    tier_override = os.environ.get("SPEECHOS_TIER")
    if tier_override:
        return tier_override

    if mode == ComputeMode.CPU:
        if ram_gb <= 2:
            return "cpu-2gb"
        if ram_gb <= 4:
            return "cpu-4gb"
        if ram_gb <= 8:
            return "cpu-8gb"
        if ram_gb <= 16:
            return "cpu-16gb"
        return "cpu-32gb"

    if mode == ComputeMode.GPU:
        if vram_gb <= 4:
            return "gpu-4gb"
        if vram_gb <= 8:
            return "gpu-8gb"
        if vram_gb <= 12:
            return "gpu-12gb"
        if vram_gb <= 24:
            return "gpu-24gb"
        return "gpu-32gb"

    # Hybrid
    if vram_gb <= 4:
        return "hybrid-4gb-gpu"
    if vram_gb <= 8:
        return "hybrid-8gb-gpu"
    if vram_gb <= 12:
        return "hybrid-12gb-gpu"
    if vram_gb <= 24:
        return "hybrid-24gb-gpu"
    return "hybrid-32gb-gpu"


def _dict_to_model_config(d: dict[str, Any] | None) -> ModelConfig | None:
    if d is None:
        return None
    return ModelConfig(**d)


def load_config(config_path: str | None = None) -> AppConfig:
    """Load configuration from file, env vars, and auto-detection."""
    hw = detect_hardware()
    tier_defaults = TIER_CONFIGS.get(hw.tier, TIER_CONFIGS["cpu-8gb"])

    # Start with tier defaults
    cfg = AppConfig(
        hardware=hw,
        stt=_dict_to_model_config(tier_defaults.get("stt")),
        tts=_dict_to_model_config(tier_defaults.get("tts")),
        tts_fast=_dict_to_model_config(tier_defaults.get("tts_fast")),
        tts_expressive=_dict_to_model_config(tier_defaults.get("tts_expressive")),
        tts_clone=_dict_to_model_config(tier_defaults.get("tts_clone")),
        emotion=_dict_to_model_config(tier_defaults.get("emotion")),
        diarization=_dict_to_model_config(tier_defaults.get("diarization")),
        vad=_dict_to_model_config(tier_defaults.get("vad")),
    )

    # Override from YAML config file
    if config_path is None:
        for candidate in ["config.yaml", "config.yml", "../config.yaml"]:
            if Path(candidate).exists():
                config_path = candidate
                break

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        _apply_yaml_overrides(cfg, yaml_cfg)

    # Override from env vars
    _apply_env_overrides(cfg)

    # Ensure directories exist
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.recordings_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.samples_dir).mkdir(parents=True, exist_ok=True)

    return cfg


def _apply_yaml_overrides(cfg: AppConfig, yaml_cfg: dict[str, Any]) -> None:
    if "server" in yaml_cfg:
        srv = yaml_cfg["server"]
        cfg.host = srv.get("host", cfg.host)
        cfg.port = srv.get("port", cfg.port)
        cfg.workers = srv.get("workers", cfg.workers)
        cfg.cors_origins = srv.get("cors_origins", cfg.cors_origins)
    if "storage" in yaml_cfg:
        st = yaml_cfg["storage"]
        cfg.model_dir = st.get("model_dir", cfg.model_dir)
        cfg.recordings_dir = st.get("recordings_dir", cfg.recordings_dir)
        cfg.max_upload_mb = st.get("max_upload_mb", cfg.max_upload_mb)
    if "model_loading" in yaml_cfg:
        cfg.model_loading = yaml_cfg["model_loading"]
    for key in ("stt", "tts", "tts_fast", "tts_expressive", "emotion", "diarization", "vad"):
        if key in yaml_cfg and yaml_cfg[key] is not None:
            setattr(cfg, key, _dict_to_model_config(yaml_cfg[key]))


def _apply_env_overrides(cfg: AppConfig) -> None:
    if v := os.environ.get("SPEECHOS_HOST"):
        cfg.host = v
    if v := os.environ.get("SPEECHOS_PORT"):
        cfg.port = int(v)
    if v := os.environ.get("SPEECHOS_MODEL_DIR"):
        cfg.model_dir = v

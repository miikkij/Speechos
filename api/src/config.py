"""Speechos API — Hardware detection and tier configuration."""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import psutil
import yaml

logger = logging.getLogger(__name__)


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
    # Language a TTS model instance is built for (Kokoro lang pipeline). Set by
    # ModelManager when it resolves a per-language variant; not a user setting.
    language: str | None = None


@dataclass
class HardwareProfile:
    mode: ComputeMode
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    gpu_name: str | None
    vram_gb: float
    tier: str


MODEL_SLOTS = ("stt", "stt_fi", "tts", "tts_fast", "tts_expressive", "tts_clone", "emotion", "diarization", "vad")


@dataclass
class AppConfig:
    hardware: HardwareProfile
    stt: ModelConfig | None = None
    # Finnish-only STT used when a request says language=fi (see _auto_finnish_stt)
    stt_fi: ModelConfig | None = None
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

# Finnish-first default voice (FI benchmark winner, CER 3.5 %). Requests with
# another language switch to that language's Piper voice (see src/lang.py).
_PIPER_FI: dict[str, Any] = {"engine": "piper", "model": "fi_FI-harri-medium", "device": "cpu"}

# Placeholder resolved in load_config(): pyannote community-1 when HF_TOKEN is set
# and pyannote.audio is installed, Resemblyzer otherwise.
_PYANNOTE_OR_RESEMBLYZER = "pyannote-or-resemblyzer"
PYANNOTE_DIARIZATION: dict[str, Any] = {
    "engine": "pyannote", "model": "pyannote/speaker-diarization-community-1", "device": "cuda",
}
RESEMBLYZER_DIARIZATION: dict[str, Any] = {"engine": "resemblyzer", "model": "dvector", "device": "cuda"}

TIER_CONFIGS: dict[str, Any] = {
    # CPU-only
    "cpu-2gb": {
        # Whisper tiny instead of Vosk: the Vosk default model is English-only.
        "stt": {"engine": "faster-whisper", "model": "tiny", "device": "cpu", "compute_type": "int8"},
        "tts": _PIPER_FI,
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "cpu-4gb": {
        "stt": {"engine": "faster-whisper", "model": "tiny", "device": "cpu", "compute_type": "int8"},
        "tts": _PIPER_FI,
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "cpu-8gb": {
        "stt": {"engine": "faster-whisper", "model": "base", "device": "cpu", "compute_type": "int8"},
        "tts": _PIPER_FI,
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "cpu-16gb": {
        "stt": {"engine": "faster-whisper", "model": "small", "device": "cpu", "compute_type": "int8"},
        "tts": _PIPER_FI,
        "tts_fast": _PIPER_FI,
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "cpu-32gb": {
        "stt": {"engine": "faster-whisper", "model": "medium", "device": "cpu", "compute_type": "int8"},
        "tts": _PIPER_FI,
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    # GPU-only
    "gpu-4gb": {
        "stt": {"engine": "faster-whisper", "model": "small", "device": "cuda", "compute_type": "float16"},
        "tts": _PIPER_FI,
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "gpu-8gb": {
        "stt": {"engine": "faster-whisper", "model": "turbo", "device": "cuda", "compute_type": "float16"},
        "tts": _PIPER_FI,
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_base_finetuned", "device": "cuda"},
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "gpu-12gb": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "int8_float16"},
        "tts": _PIPER_FI,
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_base_finetuned", "device": "cuda"},
        "diarization": _PYANNOTE_OR_RESEMBLYZER,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "gpu-24gb": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "tts": _PIPER_FI,
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_plus_large", "device": "cuda"},
        "diarization": _PYANNOTE_OR_RESEMBLYZER,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "gpu-32gb": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "tts": _PIPER_FI,
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_plus_large", "device": "cuda"},
        "diarization": _PYANNOTE_OR_RESEMBLYZER,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    # Hybrid
    "hybrid-4gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "small", "device": "cuda", "compute_type": "float16"},
        "tts": _PIPER_FI,
        "emotion": None,
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "hybrid-8gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "turbo", "device": "cuda", "compute_type": "float16"},
        "tts": _PIPER_FI,
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_base_finetuned", "device": "cuda"},
        "diarization": None,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "hybrid-12gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "int8_float16"},
        "tts": _PIPER_FI,
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_base_finetuned", "device": "cuda"},
        "diarization": _PYANNOTE_OR_RESEMBLYZER,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "hybrid-24gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "tts": _PIPER_FI,
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_plus_large", "device": "cuda"},
        "diarization": _PYANNOTE_OR_RESEMBLYZER,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
    "hybrid-32gb-gpu": {
        "stt": {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "tts": _PIPER_FI,
        "emotion": {"engine": "emotion2vec", "model": "iic/emotion2vec_plus_large", "device": "cuda"},
        "diarization": _PYANNOTE_OR_RESEMBLYZER,
        "vad": {"engine": "silero", "model": "v5", "device": "cpu"},
    },
}


def detect_hardware(mode_override: str | None = None, tier_override: str | None = None) -> HardwareProfile:
    """Auto-detect available hardware and determine optimal tier.

    ``SPEECHOS_COMPUTE_MODE`` / ``SPEECHOS_TIER`` env vars win over the arguments,
    which carry the ``compute_mode`` / ``tier`` keys of config.yaml.
    """
    cpu_cores = psutil.cpu_count(logical=False) or 4
    ram_gb = psutil.virtual_memory().total / (1024**3)

    gpu_available = False
    gpu_name = None
    vram_gb = 0.0

    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_available = True
    except ImportError:
        pass
    except Exception as e:  # broken driver, CUDA_VISIBLE_DEVICES="" with a stale context, ...
        logger.warning("CUDA reported available but could not be queried (%s); using CPU", e)
        gpu_name, vram_gb = None, 0.0

    mode_override = os.environ.get("SPEECHOS_COMPUTE_MODE") or mode_override
    if mode_override and mode_override != "auto":
        mode = ComputeMode(mode_override)
    elif gpu_available and ram_gb >= 8:
        mode = ComputeMode.HYBRID
    elif gpu_available:
        mode = ComputeMode.GPU
    else:
        mode = ComputeMode.CPU

    tier_override = os.environ.get("SPEECHOS_TIER") or tier_override
    if tier_override and tier_override != "auto":
        if tier_override not in TIER_CONFIGS:
            raise ValueError(f"Unknown tier: {tier_override!r}. Valid: {', '.join(TIER_CONFIGS)}")
        tier = tier_override
    else:
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


def pyannote_available() -> bool:
    """pyannote community-1 is gated on HuggingFace: needs HF_TOKEN + the package."""
    if not os.environ.get("HF_TOKEN"):
        return False
    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except ModuleNotFoundError:
        return False


def _resolve_slot(value: Any) -> dict[str, Any] | None:
    if value == _PYANNOTE_OR_RESEMBLYZER:
        return dict(PYANNOTE_DIARIZATION if pyannote_available() else RESEMBLYZER_DIARIZATION)
    return dict(value) if value is not None else None


def _dict_to_model_config(d: dict[str, Any] | None) -> ModelConfig | None:
    if d is None or d.get("engine") in (None, "none"):
        return None
    d = dict(d)
    d.setdefault("model", "default")
    return ModelConfig(**d)


def _find_config_file() -> str | None:
    if v := os.environ.get("SPEECHOS_CONFIG"):
        return v
    for candidate in ["config.yaml", "config.yml", "../config.yaml"]:
        if Path(candidate).exists():
            return candidate
    return None


def load_config(config_path: str | None = None) -> AppConfig:
    """Load configuration: tier defaults < config.yaml < SPEECHOS_* env vars."""
    if config_path is None:
        config_path = _find_config_file()
    yaml_cfg: dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}

    hw = detect_hardware(
        mode_override=yaml_cfg.get("compute_mode"),
        tier_override=yaml_cfg.get("tier"),
    )
    tier_defaults = TIER_CONFIGS[hw.tier]

    cfg = AppConfig(hardware=hw)
    for slot in MODEL_SLOTS:
        setattr(cfg, slot, _dict_to_model_config(_resolve_slot(tier_defaults.get(slot))))

    _apply_yaml_overrides(cfg, yaml_cfg)
    _apply_env_overrides(cfg)
    explicit_fi = "stt_fi" in yaml_cfg or any(k.startswith("SPEECHOS_STT_FI_") for k in os.environ)
    if not explicit_fi:
        _auto_finnish_stt(cfg)

    # Ensure directories exist
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.recordings_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.samples_dir).mkdir(parents=True, exist_ok=True)

    return cfg


# Finnish-NLP/whisper-large-finnish-v3 converted to CTranslate2 float16 (see api/bench/README.md).
# FLEURS-fi n=300: WER 6.2 % vs 7.5 % for large-v3 at the same speed; Finnish only.
FINNISH_WHISPER_DIR = "whisper-large-finnish-v3-ct2-f16"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def finnish_whisper_candidates(model_dir: str) -> list[Path]:
    return [
        Path(model_dir) / "whisper-ct2" / FINNISH_WHISPER_DIR,
        _REPO_ROOT / "models" / "whisper-ct2" / FINNISH_WHISPER_DIR,
    ]


def _auto_finnish_stt(cfg: AppConfig) -> None:
    """Use the Finnish Whisper for language=fi on tiers that already run large-v3 on the GPU,
    when the converted model is on disk. Other languages keep using ``stt``."""
    stt = cfg.stt
    if stt is None or stt.engine != "faster-whisper" or stt.device != "cuda" or stt.model != "large-v3":
        return
    for path in finnish_whisper_candidates(cfg.model_dir):
        if (path / "model.bin").is_file():
            cfg.stt_fi = ModelConfig(
                engine="faster-whisper", model=str(path), device="cuda", compute_type=stt.compute_type
            )
            return


def _apply_yaml_overrides(cfg: AppConfig, yaml_cfg: dict[str, Any]) -> None:
    if "server" in yaml_cfg:
        srv = yaml_cfg["server"] or {}
        cfg.host = srv.get("host", cfg.host)
        cfg.port = int(srv.get("port", cfg.port))
        cfg.workers = int(srv.get("workers", cfg.workers))
        cfg.cors_origins = list(srv.get("cors_origins", cfg.cors_origins))
    if "storage" in yaml_cfg:
        st = yaml_cfg["storage"] or {}
        cfg.model_dir = st.get("model_dir", cfg.model_dir)
        cfg.recordings_dir = st.get("recordings_dir", cfg.recordings_dir)
        cfg.samples_dir = st.get("samples_dir", cfg.samples_dir)
        cfg.max_upload_mb = int(st.get("max_upload_mb", cfg.max_upload_mb))
    if "model_loading" in yaml_cfg:
        cfg.model_loading = yaml_cfg["model_loading"]
    for key in MODEL_SLOTS:
        if key in yaml_cfg and yaml_cfg[key] is not None:
            setattr(cfg, key, _dict_to_model_config(yaml_cfg[key]))


_ENV_MODEL_FIELDS = ("engine", "model", "device", "compute_type")


def _apply_env_overrides(cfg: AppConfig) -> None:
    if v := os.environ.get("SPEECHOS_HOST"):
        cfg.host = v
    if v := os.environ.get("SPEECHOS_PORT"):
        cfg.port = int(v)
    if v := os.environ.get("SPEECHOS_WORKERS"):
        cfg.workers = int(v)
    if v := os.environ.get("SPEECHOS_MODEL_DIR"):
        cfg.model_dir = v
    if v := os.environ.get("SPEECHOS_RECORDINGS_DIR"):
        cfg.recordings_dir = v
    if v := os.environ.get("SPEECHOS_CORS_ORIGINS"):
        cfg.cors_origins = [o.strip() for o in v.split(",") if o.strip()]
    if v := os.environ.get("SPEECHOS_MODEL_LOADING"):
        cfg.model_loading = v

    # SPEECHOS_<SLOT>_<FIELD>, e.g. SPEECHOS_STT_MODEL=large-v3, SPEECHOS_EMOTION_ENGINE=none
    for slot in MODEL_SLOTS:
        values = {
            f: os.environ[f"SPEECHOS_{slot.upper()}_{f.upper()}"]
            for f in _ENV_MODEL_FIELDS
            if os.environ.get(f"SPEECHOS_{slot.upper()}_{f.upper()}")
        }
        if not values:
            continue
        if values.get("engine") == "none":
            setattr(cfg, slot, None)
            continue
        current: ModelConfig | None = getattr(cfg, slot)
        if current is None or ("engine" in values and values["engine"] != current.engine):
            # New engine: don't inherit the old engine's model name
            setattr(cfg, slot, _dict_to_model_config({"device": "cpu", **values}))
        else:
            setattr(cfg, slot, replace(current, **values))

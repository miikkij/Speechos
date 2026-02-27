"""Speechos model manager — lazy loading, caching, VRAM management."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from src.config import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model lifecycle: download, load, cache, evict."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model_dir = Path(config.model_dir)
        self._loaded: OrderedDict[str, Any] = OrderedDict()

    def get_stt(self) -> Any:
        return self._get_or_load("stt", self.config.stt, self._load_stt)

    def get_tts(self) -> Any:
        return self._get_or_load("tts", self.config.tts, self._load_tts)

    def get_vad(self) -> Any:
        return self._get_or_load("vad", self.config.vad, self._load_vad)

    def get_emotion(self) -> Any | None:
        if self.config.emotion is None:
            return None
        return self._get_or_load("emotion", self.config.emotion, self._load_emotion)

    def get_diarization(self) -> Any | None:
        if self.config.diarization is None:
            return None
        return self._get_or_load("diarization", self.config.diarization, self._load_diarization)

    def _get_or_load(self, key: str, cfg: ModelConfig | None, loader: Any) -> Any:
        if cfg is None:
            raise ValueError(f"No configuration for model: {key}")
        if key in self._loaded:
            self._loaded.move_to_end(key)
            return self._loaded[key]
        logger.info("Loading model: %s (%s/%s on %s)", key, cfg.engine, cfg.model, cfg.device)
        model = loader(cfg)
        self._loaded[key] = model
        return model

    def _load_stt(self, cfg: ModelConfig) -> Any:
        if cfg.engine == "faster-whisper":
            from faster_whisper import WhisperModel
            return WhisperModel(
                cfg.model,
                device=cfg.device,
                compute_type=cfg.compute_type or "int8",
                download_root=str(self.model_dir / "whisper"),
            )
        if cfg.engine == "vosk":
            from vosk import Model as VoskModel
            model_path = self.model_dir / "vosk" / cfg.model
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Vosk model not found: {model_path}. "
                    f"Download from https://alphacephei.com/vosk/models"
                )
            return VoskModel(str(model_path))
        raise ValueError(f"Unknown STT engine: {cfg.engine}")

    def _load_tts(self, cfg: ModelConfig) -> Any:
        if cfg.engine == "piper":
            from piper import PiperVoice
            model_path = self.model_dir / "piper" / f"{cfg.model}.onnx"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Piper model not found: {model_path}. "
                    f"Download from https://github.com/rhasspy/piper/releases"
                )
            return PiperVoice.load(str(model_path))
        raise ValueError(f"Unknown TTS engine: {cfg.engine}")

    def _load_vad(self, cfg: ModelConfig) -> Any:
        if cfg.engine == "silero":
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            return model, utils
        raise ValueError(f"Unknown VAD engine: {cfg.engine}")

    def _load_emotion(self, cfg: ModelConfig) -> Any:
        raise NotImplementedError("Emotion model loading not yet implemented")

    def _load_diarization(self, cfg: ModelConfig) -> Any:
        raise NotImplementedError("Diarization model loading not yet implemented")

    def unload(self, key: str) -> None:
        if key in self._loaded:
            del self._loaded[key]
            logger.info("Unloaded model: %s", key)

    def unload_all(self) -> None:
        keys = list(self._loaded.keys())
        for key in keys:
            self.unload(key)

    @property
    def loaded_models(self) -> list[str]:
        return list(self._loaded.keys())

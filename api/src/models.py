"""Speechos model manager: lazy loading, caching, VRAM management."""

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
        from src.docker_stt import is_docker_stt_engine
        if is_docker_stt_engine(cfg.engine):
            return {"type": "docker", "engine": cfg.engine}
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
                self._download_vosk_model(cfg.model, model_path)
            return VoskModel(str(model_path))
        if cfg.engine == "whisperx":
            try:
                import whisperx
            except ImportError:
                raise ImportError("WhisperX not installed. Run: pip install whisperx")
            device = cfg.device if cfg.device != "cpu" else "cpu"
            return whisperx.load_model(
                cfg.model, device=device, compute_type=cfg.compute_type or "int8",
            )
        if cfg.engine == "moonshine":
            try:
                from moonshine import Moonshine
            except ImportError:
                raise ImportError("Moonshine not installed. Run: pip install moonshine")
            return Moonshine(model_name=cfg.model)
        if cfg.engine == "nemo":
            try:
                import nemo.collections.asr as nemo_asr
            except ImportError:
                raise ImportError("NeMo not installed. Run: pip install nemo_toolkit[asr]")
            if cfg.device == "cpu":
                raise ValueError(f"NeMo model {cfg.model} requires GPU (CUDA)")
            return nemo_asr.models.ASRModel.from_pretrained(cfg.model)
        if cfg.engine == "wav2vec2-stt":
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError("transformers not installed. Run: pip install transformers")
            device = 0 if cfg.device == "cuda" else -1
            return pipeline("automatic-speech-recognition", model=cfg.model, device=device)
        raise ValueError(f"Unknown STT engine: {cfg.engine}")

    def _load_tts(self, cfg: ModelConfig) -> Any:
        # Docker-backed engines don't need native loading
        from src.docker_tts import is_docker_engine
        if is_docker_engine(cfg.engine):
            return {"type": "docker", "engine": cfg.engine}

        if cfg.engine == "piper":
            from piper import PiperVoice
            model_path = self.model_dir / "piper" / f"{cfg.model}.onnx"
            if not model_path.exists():
                self._download_piper_model(cfg.model, model_path.parent)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Piper model not found: {model_path}. "
                    f"Auto-download failed."
                )
            return PiperVoice.load(str(model_path))
        if cfg.engine == "kokoro":
            try:
                from kokoro import KPipeline
            except ImportError:
                raise ImportError("Kokoro not installed. Run: pip install kokoro")
            return KPipeline(lang_code="a")
        if cfg.engine == "chatterbox":
            try:
                from chatterbox.tts import ChatterboxTTS
            except ImportError:
                raise ImportError("Chatterbox not installed. Run: pip install chatterbox-tts")
            import torch
            device = "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
            return ChatterboxTTS.from_pretrained(device=device)
        if cfg.engine == "orpheus":
            try:
                from orpheus_tts import OrpheusModel
            except ImportError:
                raise ImportError("Orpheus TTS not installed. Run: pip install orpheus-tts")
            return OrpheusModel(model_name=cfg.model)
        if cfg.engine == "xtts":
            try:
                from TTS.api import TTS
            except ImportError:
                raise ImportError("Coqui TTS not installed. Run: pip install TTS")
            import torch
            device = "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
            return TTS(cfg.model).to(device)
        if cfg.engine == "melotts":
            try:
                from melo.api import TTS as MeloTTS
            except ImportError:
                raise ImportError("MeloTTS not installed. Run: pip install melotts")
            import torch
            device = "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
            return MeloTTS(language=cfg.model, device=device)
        if cfg.engine == "bark":
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError("transformers not installed. Run: pip install transformers")
            device = 0 if cfg.device == "cuda" else -1
            return pipeline("text-to-audio", model=cfg.model, device=device)
        if cfg.engine == "parler":
            try:
                from parler_tts import ParlerTTSForConditionalGeneration
                from transformers import AutoTokenizer
            except ImportError:
                raise ImportError("Parler TTS not installed. Run: pip install parler-tts")
            import torch
            device = "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
            model = ParlerTTSForConditionalGeneration.from_pretrained(cfg.model).to(device)
            tokenizer = AutoTokenizer.from_pretrained(cfg.model)
            return {"model": model, "tokenizer": tokenizer, "device": device}
        if cfg.engine == "chattts":
            try:
                import ChatTTS
            except ImportError:
                raise ImportError("ChatTTS not installed. Run: pip install chattts")
            import torch
            chat = ChatTTS.Chat()
            device = "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
            chat.load(compile=False, device=device)
            return chat
        if cfg.engine == "fish-speech":
            try:
                from fish_speech.inference import TTSInference
            except ImportError:
                raise ImportError("Fish Speech not installed. See: https://github.com/fishaudio/fish-speech")
            return TTSInference()
        if cfg.engine == "cosyvoice":
            try:
                from cosyvoice import CosyVoice
            except ImportError:
                raise ImportError("CosyVoice not installed. See: https://github.com/FunAudioLLM/CosyVoice")
            return CosyVoice(cfg.model)
        if cfg.engine == "qwen3-tts":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                raise ImportError("transformers not installed. Run: pip install transformers")
            import torch
            device = "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(cfg.model, trust_remote_code=True).to(device)
            tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
            return {"model": model, "tokenizer": tokenizer, "device": device}
        if cfg.engine == "espeak":
            # eSpeak-NG via espeakng_loader DLL (ctypes) or system binary
            import ctypes
            import os
            try:
                import espeakng_loader
                dll = ctypes.CDLL(str(espeakng_loader.get_library_path()))
                data = os.path.dirname(espeakng_loader.get_data_path())
                # AUDIO_OUTPUT_SYNCHRONOUS = 0x0002
                sr = dll.espeak_Initialize(0x0002, 500, data.encode("utf-8"), 0)
                if sr <= 0:
                    raise RuntimeError(f"espeak_Initialize failed: {sr}")
                lang = (cfg.model or "en").encode("utf-8")
                dll.espeak_SetVoiceByName(lang)
                return {"engine": "espeak", "lang": cfg.model or "en", "dll": dll, "sample_rate": sr}
            except ImportError:
                pass
            import shutil
            if not (shutil.which("espeak-ng") or shutil.which("espeak")):
                raise FileNotFoundError(
                    "eSpeak-NG not found. Install espeak-ng system package "
                    "or pip install espeakng-loader"
                )
            return {"engine": "espeak", "lang": cfg.model or "en", "dll": None, "sample_rate": 22050}
        raise ValueError(f"Unknown TTS engine: {cfg.engine}")

    def _download_piper_model(self, model_name: str, dest_dir: Path) -> None:
        """Download Piper ONNX voice model from HuggingFace."""
        import urllib.request

        # Parse model name: en_US-lessac-medium → lang=en, region=en_US, name=lessac, quality=medium
        parts = model_name.split("-")
        if len(parts) < 3:
            raise ValueError(f"Cannot parse Piper model name: {model_name}")
        lang_region = parts[0]       # en_US
        voice_name = parts[1]        # lessac
        quality = parts[2]           # medium
        lang = lang_region.split("_")[0]  # en

        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
        files = [f"{model_name}.onnx", f"{model_name}.onnx.json"]

        dest_dir.mkdir(parents=True, exist_ok=True)

        for filename in files:
            url = f"{base_url}/{lang}/{lang_region}/{voice_name}/{quality}/{filename}"
            dest_path = dest_dir / filename
            logger.info("Downloading Piper model: %s", url)
            try:
                urllib.request.urlretrieve(url, str(dest_path))  # noqa: S310
                logger.info("Saved: %s", dest_path)
            except Exception as e:
                logger.error("Failed to download %s: %s", url, e)
                if dest_path.exists():
                    dest_path.unlink()
                raise

    def _load_vad(self, cfg: ModelConfig) -> Any:
        if cfg.engine == "silero":
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            return model, utils
        if cfg.engine == "pyannote-vad":
            try:
                from pyannote.audio import Pipeline
            except ImportError:
                raise ImportError(
                    "pyannote.audio not installed. Run: pip install pyannote.audio\n"
                    "Also requires accepting terms at:\n"
                    "  https://huggingface.co/pyannote/segmentation-3.0"
                )
            import torch
            device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
            pipe = Pipeline.from_pretrained(cfg.model)
            pipe.to(device)
            return pipe
        raise ValueError(f"Unknown VAD engine: {cfg.engine}")

    def _load_emotion(self, cfg: ModelConfig) -> Any:
        if cfg.engine == "emotion2vec":
            from funasr import AutoModel
            return AutoModel(
                model=cfg.model,
                device=cfg.device,
            )
        if cfg.engine in ("wav2vec2-ser", "hubert-ser", "wavlm-ser", "wav2small"):
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError("transformers not installed. Run: pip install transformers")
            device = 0 if cfg.device == "cuda" else -1
            return pipeline("audio-classification", model=cfg.model, device=device)
        raise ValueError(f"Unknown emotion engine: {cfg.engine}")

    def _load_diarization(self, cfg: ModelConfig) -> Any:
        if cfg.engine == "resemblyzer":
            from resemblyzer import VoiceEncoder
            import torch

            device = cfg.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            encoder = VoiceEncoder(device=device)
            return encoder
        if cfg.engine == "pyannote":
            try:
                from pyannote.audio import Pipeline
            except ImportError:
                raise ImportError(
                    "pyannote.audio not installed. Run: pip install pyannote.audio\n"
                    "Also requires accepting terms at:\n"
                    "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                    "  https://huggingface.co/pyannote/segmentation-3.0"
                )
            import torch
            device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
            pipe = Pipeline.from_pretrained(cfg.model)
            pipe.to(device)
            return pipe
        if cfg.engine == "speechbrain":
            try:
                from speechbrain.inference.speaker import SpeakerRecognition
            except ImportError:
                raise ImportError("SpeechBrain not installed. Run: pip install speechbrain")
            return SpeakerRecognition.from_hparams(
                source=f"speechbrain/{cfg.model}",
                savedir=str(self.model_dir / "speechbrain" / cfg.model),
            )
        raise ValueError(f"Unknown diarization engine: {cfg.engine}")

    def _download_vosk_model(self, model_name: str, dest: Path) -> None:
        """Download a Vosk model from alphacephei.com."""
        import io
        import urllib.request
        import zipfile

        url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
        logger.info("Downloading Vosk model: %s", url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as resp:  # noqa: S310
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(dest.parent)
        if not dest.exists():
            raise FileNotFoundError(f"Vosk model download failed: expected {dest}")
        logger.info("Vosk model downloaded: %s", dest)

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

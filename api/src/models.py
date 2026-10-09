"""Speechos model manager — lazy loading, caching, per-model locking.

Inference runs in worker threads (``ModelManager.run``) so the event loop stays
free. Every cached model instance has its own lock, so one model is never used
from two threads at once (most are not thread-safe: CTranslate2 Whisper, Silero's
stateful JIT model, Piper's ONNX session, ...).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

from src.config import AppConfig, ModelConfig
from src.lang import (
    UnsupportedLanguageError,
    base_language,
    kokoro_lang_code,
    piper_voice_for,
    piper_voice_language,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

TTS_SLOTS = ("tts", "tts_fast", "tts_expressive", "tts_clone")

# Engines whose models only speak English
_ENGLISH_ONLY_TTS = {"kitten", "parler"}


class ModelNotConfiguredError(LookupError):
    """No model configured for the requested category/slot."""


class ModelManager:
    """Manages model lifecycle: download, load, cache, unload."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model_dir = Path(config.model_dir)
        self._loaded: OrderedDict[str, Any] = OrderedDict()
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    # ── Resolution: cache key + effective config ─────────────

    def resolve(self, category: str, language: str | None = None) -> tuple[str, ModelConfig]:
        """Return (cache_key, config) for a category.

        For TTS slots this picks a per-language variant: a Finnish request on an
        English Piper voice loads the Finnish voice under its own cache key.
        For STT, a Finnish request uses ``stt_fi`` when one is configured.
        Raises ModelNotConfiguredError / UnsupportedLanguageError.
        """
        if category == "stt" and self.config.stt_fi is not None and base_language(language) == "fi":
            return "stt_fi", self.config.stt_fi
        cfg: ModelConfig | None = getattr(self.config, category, None)
        if cfg is None:
            raise ModelNotConfiguredError(f"No model configured for '{category}'")
        if category not in TTS_SLOTS:
            return category, cfg
        return self._resolve_tts(category, cfg, language)

    def _resolve_tts(self, slot: str, cfg: ModelConfig, language: str | None) -> tuple[str, ModelConfig]:
        lang = base_language(language)
        if cfg.engine == "piper":
            if lang is None or piper_voice_language(cfg.model) == lang:
                return slot, cfg
            voice = piper_voice_for(lang)
            return f"{slot}:piper:{voice}", replace(cfg, model=voice)
        if cfg.engine == "kokoro":
            code = kokoro_lang_code(language)
            key = slot if code == "a" else f"{slot}:kokoro:{code}"
            return key, replace(cfg, language=code)
        if cfg.engine == "chatterbox-fi" and lang not in (None, "fi"):
            raise UnsupportedLanguageError("Chatterbox-Finnish only speaks Finnish ('fi').")
        if cfg.engine in _ENGLISH_ONLY_TTS and lang not in (None, "en"):
            raise UnsupportedLanguageError(
                f"TTS engine '{cfg.engine}' only supports English, not '{language}'. "
                "For Finnish use Piper (fi_FI-harri-medium) or Chatterbox Multilingual."
            )
        return slot, cfg

    # ── Locked access ───────────────────────────────────────

    def lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = self._locks[key] = threading.Lock()
            return lock

    def run_sync(self, key: str, cfg: ModelConfig, fn: Callable[[Any, ModelConfig], T]) -> T:
        """Load (if needed) and call ``fn(model, cfg)`` while holding the model's lock."""
        category = key.split(":", 1)[0]
        with self.lock_for(key):
            try:
                model = self._get_or_load(key, cfg, self._loader_for(category))
            except Exception as e:
                fallback = self._diarization_fallback(category, cfg, e)
                if fallback is None:
                    raise
                cfg = fallback
                model = self._get_or_load(key, cfg, self._loader_for(category))
            return fn(model, cfg)

    def _diarization_fallback(self, category: str, cfg: ModelConfig, error: Exception) -> ModelConfig | None:
        """pyannote is gated on Hugging Face: when it cannot load (no token, terms not accepted,
        offline), switch diarization to Resemblyzer for this process instead of failing every analysis."""
        if category != "diarization" or cfg.engine != "pyannote":
            return None
        logger.warning(
            "pyannote diarization (%s) failed to load, falling back to Resemblyzer: %s",
            cfg.model,
            str(error).splitlines()[0][:200],
        )
        fallback = ModelConfig(engine="resemblyzer", model="dvector", device=cfg.device)
        self.config.diarization = fallback
        return fallback

    async def run(
        self, category: str, fn: Callable[[Any, ModelConfig], T], *, language: str | None = None
    ) -> T:
        """Run ``fn(model, cfg)`` for a category in a worker thread (non-blocking)."""
        key, cfg = self.resolve(category, language)
        return await asyncio.to_thread(self.run_sync, key, cfg, fn)

    def _loader_for(self, category: str) -> Callable[[ModelConfig], Any]:
        if category in ("stt", "stt_fi"):
            return self._load_stt
        if category in TTS_SLOTS:
            return self._load_tts
        if category == "vad":
            return self._load_vad
        if category == "emotion":
            return self._load_emotion
        if category == "diarization":
            return self._load_diarization
        raise ValueError(f"Unknown model category: {category}")

    def preload(self, categories: tuple[str, ...] = ("stt", "tts", "vad")) -> None:
        """Eager loading (``model_loading: eager``). Failures are logged, not raised."""
        for category in categories:
            try:
                key, cfg = self.resolve(category)
                with self.lock_for(key):
                    self._get_or_load(key, cfg, self._loader_for(category))
            except Exception as e:  # noqa: BLE001 — preload is best-effort
                logger.warning("Preload of %s failed: %s", category, e)

    # ── Direct (unlocked) getters — for scripts / integration tests ─

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
            raise ModelNotConfiguredError(f"No configuration for model: {key}")
        with self._guard:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return self._loaded[key]
        logger.info("Loading model: %s (%s/%s on %s)", key, cfg.engine, cfg.model, cfg.device)
        model = loader(cfg)
        with self._guard:
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
        if cfg.engine == "granite-speech":
            # IBM Granite-Speech 4.1 — speech LLM via transformers (chat template + audio)
            try:
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            except ImportError:
                raise ImportError("transformers not installed. Run: pip install -U transformers")
            import torch
            device = "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
            processor = AutoProcessor.from_pretrained(cfg.model, trust_remote_code=True)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                cfg.model, trust_remote_code=True
            ).to(device)
            return {"processor": processor, "model": model, "device": device}
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
        if cfg.engine == "chatterbox-fi":
            from src.engines_fi.chatterbox_fi import ChatterboxFinnish

            # cfg.model: "sample" (the model repo's Finnish reference voice), a path to a Finnish
            # reference WAV, or "" for the built-in voice
            return ChatterboxFinnish.load(self.model_dir, device=cfg.device, ref_wav=cfg.model or None)
        if cfg.engine == "kokoro":
            try:
                from kokoro import KPipeline
            except ImportError:
                raise ImportError("Kokoro not installed. Run: pip install kokoro")
            return KPipeline(lang_code=cfg.language or "a", repo_id="hexgrad/Kokoro-82M")
        if cfg.engine == "kitten":
            try:
                from kittentts import KittenTTS
            except ImportError:
                raise ImportError("Kitten TTS not installed. Run: pip install kittentts")
            return KittenTTS(cfg.model or "KittenML/kitten-tts-nano-0.8-fp32")
        if cfg.engine == "chatterbox":
            import torch
            device = "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
            # model="multilingual" → Chatterbox Multilingual (23+ langs incl. Finnish, needs
            # language_id at generate-time). Anything else → English-only model.
            multilingual = (cfg.model or "").lower() in ("multilingual", "mtl", "ml")
            if multilingual:
                try:
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                except ImportError:
                    raise ImportError("Chatterbox not installed. Run: pip install chatterbox-tts")
                model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            else:
                try:
                    from chatterbox.tts import ChatterboxTTS
                except ImportError:
                    raise ImportError("Chatterbox not installed. Run: pip install chatterbox-tts")
                model = ChatterboxTTS.from_pretrained(device=device)
            return {
                "model": model,
                "multilingual": multilingual,
                "sr": getattr(model, "sr", 24000),
            }
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
            # The pinned silero-vad pip package ships the model — no torch.hub download
            from silero_vad import load_silero_vad
            return load_silero_vad()
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
            import torch
            from resemblyzer import VoiceEncoder

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
                    "  https://huggingface.co/pyannote/speaker-diarization-community-1"
                )
            import torch
            device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
            # Gated model: huggingface_hub picks up HF_TOKEN from the environment
            pipe = Pipeline.from_pretrained(cfg.model)
            if pipe is None:
                raise RuntimeError(
                    f"Could not load {cfg.model}: set HF_TOKEN and accept the model's terms on HuggingFace"
                )
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
        if cfg.engine == "sortformer":
            # NVIDIA Sortformer — end-to-end diarization via NeMo (GPU)
            try:
                from nemo.collections.asr.models import SortformerEncLabelModel
            except ImportError:
                raise ImportError("NeMo not installed. Run: pip install nemo_toolkit[asr]")
            if cfg.device == "cpu":
                raise ValueError(f"Sortformer model {cfg.model} requires GPU (CUDA)")
            model = SortformerEncLabelModel.from_pretrained(cfg.model)
            model.eval()
            return model
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
            raise FileNotFoundError(f"Vosk model download failed — expected {dest}")
        logger.info("Vosk model downloaded: %s", dest)

    def unload(self, key: str) -> None:
        """Unload a category and its per-language variants (``tts:piper:...``)."""
        with self._guard:
            keys = [k for k in self._loaded if k == key or k.startswith(f"{key}:")]
            for k in keys:
                del self._loaded[k]
        for k in keys:
            logger.info("Unloaded model: %s", k)

    def unload_all(self) -> None:
        keys = list(self._loaded.keys())
        for key in keys:
            self.unload(key)

    @property
    def loaded_models(self) -> list[str]:
        with self._guard:
            return list(self._loaded.keys())

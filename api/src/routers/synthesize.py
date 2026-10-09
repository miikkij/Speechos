"""Text-to-speech synthesis endpoints."""

from __future__ import annotations

import ctypes
import io
import logging
import os
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import ModelConfig
from src.docker_tts import docker_synthesize, is_docker_engine
from src.lang import UnsupportedLanguageError, base_language, piper_voice_language
from src.models import ModelNotConfiguredError
from src.routers._common import parse_language
from src.server import get_models

logger = logging.getLogger(__name__)
router = APIRouter(tags=["synthesis"])

SLOT_KEYS = {"default": "tts", "fast": "tts_fast", "expressive": "tts_expressive", "clone": "tts_clone"}


class SynthesizeRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float = Field(1.0, ge=0.25, le=4.0)
    # ISO 639-1 code, e.g. "fi". Piper switches to that language's voice, Chatterbox
    # Multilingual uses it as language_id, English-only engines reject other languages.
    language: str | None = None
    # Which configured TTS slot to use (config keys tts / tts_fast / tts_expressive / tts_clone)
    slot: Literal["default", "fast", "expressive", "clone"] = "default"


@dataclass
class SynthesisInput:
    text: str
    voice: str | None
    speed: float
    language: str | None


@router.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Synthesize speech from text, returns WAV audio."""
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")
    if len(req.text) > 10_000:
        raise HTTPException(400, "Text exceeds 10,000 character limit")

    language = parse_language(req.language)
    slot = SLOT_KEYS[req.slot]
    models = get_models()
    try:
        _key, cfg = models.resolve(slot, language)
    except ModelNotConfiguredError as e:
        raise HTTPException(400 if slot != "tts" else 500, str(e)) from e
    except UnsupportedLanguageError as e:
        raise HTTPException(400, str(e)) from e

    inp = SynthesisInput(text=req.text, voice=req.voice, speed=req.speed, language=language)
    t0 = time.perf_counter()

    if is_docker_engine(cfg.engine):
        # Docker-backed engines (proxy to WSL2 container)
        try:
            wav_bytes = await docker_synthesize(cfg.engine, req.text, req.voice, language)
        except Exception as e:
            logger.error("Docker TTS (%s) failed: %s", cfg.engine, e)
            raise HTTPException(500, f"Docker TTS engine '{cfg.engine}' failed: {e}") from e
    else:
        try:
            wav_bytes = await models.run(slot, lambda tts, c: synthesize_wav(tts, c, inp), language=language)
        except UnsupportedLanguageError as e:
            raise HTTPException(400, str(e)) from e
        except (ImportError, FileNotFoundError) as e:
            raise HTTPException(503, f"TTS engine not available: {e}") from e

    elapsed = time.perf_counter() - t0
    logger.info("TTS (%s/%s) completed in %.2fs for %d chars", cfg.engine, cfg.model, elapsed, len(req.text))
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={
            "X-Processing-Time": str(round(elapsed, 3)),
            "X-TTS-Engine": cfg.engine,
            "X-TTS-Model": cfg.model,
            "Content-Disposition": "inline; filename=speech.wav",
        },
    )


def _to_int16(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    return (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)


def _wav_bytes(audio_int16: np.ndarray, sample_rate: int) -> bytes:
    if audio_int16.ndim > 1:
        audio_int16 = audio_int16.mean(axis=-1).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def synthesize_wav(tts: Any, cfg: ModelConfig, inp: SynthesisInput) -> bytes:
    """Run one native TTS engine and return mono 16-bit WAV bytes (blocking)."""
    engine = cfg.engine
    text = inp.text

    if engine == "piper":
        from piper import SynthesisConfig

        if base_language(inp.language) == "fi" or piper_voice_language(cfg.model) == "fi":
            from src.engines_fi.text_norm_fi import normalize_fi

            # Numbers, dates, money, clock times, abbreviations → words (benchmark: 80 % → 91 % read right)
            text = normalize_fi(text)

        # length_scale < 1 is faster, > 1 slower
        syn_config = SynthesisConfig(length_scale=1.0 / inp.speed)
        chunks, sample_rate = [], 22050
        for chunk in tts.synthesize(text, syn_config=syn_config):
            sample_rate = chunk.sample_rate
            chunks.append(chunk.audio_float_array)
        if not chunks:
            raise HTTPException(500, "TTS produced no audio")
        return _wav_bytes(_to_int16(np.concatenate(chunks)), sample_rate)

    if engine == "chatterbox-fi":
        # ChatterboxFinnish (src/engines_fi): normalizes Finnish text itself; no tempo control, so speed is ignored
        if inp.voice and inp.voice != tts.ref_wav:
            tts.set_voice(inp.voice)  # path to a Finnish reference WAV for voice cloning
        audio, sample_rate = tts.synthesize(text)
        return _wav_bytes(_to_int16(audio), sample_rate)

    if engine == "kokoro":
        # KPipeline yields (graphemes, phonemes, audio_tensor)
        voice = inp.voice or "af_heart"
        chunks = [audio.numpy() for _gs, _ps, audio in tts(text, voice=voice, speed=inp.speed)]
        if not chunks:
            raise HTTPException(500, "TTS produced no audio")
        return _wav_bytes(_to_int16(np.concatenate(chunks)), 24000)

    if engine == "kitten":
        # KittenTTS.generate returns a float32 numpy waveform at 24kHz
        audio = tts.generate(text, voice=inp.voice or "expr-voice-2-f")
        return _wav_bytes(_to_int16(audio), 24000)

    if engine == "chatterbox":
        # tts is a dict: {"model", "multilingual", "sr"} (see ModelManager._load_tts)
        gen_kwargs: dict[str, Any] = {}
        if tts.get("multilingual"):
            gen_kwargs["language_id"] = inp.language or "en"
        elif inp.language not in (None, "en"):
            raise UnsupportedLanguageError(
                f"Chatterbox English cannot speak '{inp.language}'. Use Chatterbox Multilingual."
            )
        if inp.voice:
            # For Chatterbox, `voice` is a path to a reference WAV for voice cloning
            gen_kwargs["audio_prompt_path"] = inp.voice
        wav = tts["model"].generate(text, **gen_kwargs)  # tensor [1, samples]
        return _wav_bytes(_to_int16(wav.squeeze().cpu().numpy()), tts.get("sr", 24000))

    if engine == "orpheus":
        audio_data = tts.generate_speech(text)
        return _wav_bytes(_to_int16(audio_data["audio"]), audio_data.get("sample_rate", 24000))

    if engine == "xtts":
        return _wav_bytes(_to_int16(tts.tts(text=text, language=inp.language or "en")), 22050)

    if engine == "melotts":
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            first_speaker = next(iter(tts.hps.data.spk2id.values()))
            tts.tts_to_file(text, first_speaker, tmp_path, speed=inp.speed)
            audio, sample_rate = sf.read(tmp_path, dtype="int16")
        finally:
            os.unlink(tmp_path)
        return _wav_bytes(np.asarray(audio, dtype=np.int16), sample_rate)

    if engine == "bark":
        result = tts(text)
        return _wav_bytes(_to_int16(result["audio"]), result["sampling_rate"])

    if engine == "parler":
        import torch

        model, tokenizer, device = tts["model"], tts["tokenizer"], tts["device"]
        description = inp.voice or "A female speaker delivers a clear, natural speech."
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
        return _wav_bytes(_to_int16(generation.cpu().float().numpy()), model.config.sampling_rate)

    if engine == "chattts":
        import ChatTTS

        params = ChatTTS.Chat.InferCodeParams(spk_emb=tts.sample_random_speaker())
        wavs = tts.infer([text], skip_refine_text=True, params_infer_code=params)
        return _wav_bytes(_to_int16(wavs[0]), 24000)

    if engine == "fish-speech":
        result = tts.synthesize(text)
        return _wav_bytes(_to_int16(result["audio"]), result.get("sample_rate", 44100))

    if engine == "cosyvoice":
        result = tts.inference_sft(text, "English")
        return _wav_bytes(_to_int16(result["tts_speech"].numpy()), 22050)

    if engine == "qwen3-tts":
        import torch

        model, tokenizer, device = tts["model"], tts["tokenizer"], tts["device"]
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=4096)
        return _wav_bytes(_to_int16(outputs.cpu().float().numpy()), 24000)

    if engine == "espeak":
        return _espeak(tts, inp)

    raise HTTPException(500, f"Unknown TTS engine: {engine}")


def _espeak(tts: dict[str, Any], inp: SynthesisInput) -> bytes:
    lang = inp.language or tts.get("lang", "en")
    # eSpeak rate is words per minute; 175 is its default
    rate = int(175 * inp.speed)
    dll = tts.get("dll")
    if not dll:
        # System binary fallback
        result = subprocess.run(
            ["espeak-ng", "--stdout", "-v", lang, "-s", str(rate), inp.text],
            capture_output=True, check=True,
        )
        return result.stdout

    # espeakng_loader DLL via ctypes
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)
    chunks: list[np.ndarray] = []

    @callback_type
    def _cb(wav, numsamples, events):
        if wav and numsamples > 0:
            arr = (ctypes.c_short * numsamples)()
            ctypes.memmove(arr, wav, numsamples * 2)
            chunks.append(np.frombuffer(arr, dtype=np.int16).copy())
        return 0

    dll.espeak_SetSynthCallback(_cb)
    dll.espeak_SetVoiceByName(lang.encode("utf-8"))
    dll.espeak_SetParameter(1, rate, 0)  # espeakRATE = 1
    text_bytes = inp.text.encode("utf-8")
    text_buf = ctypes.create_string_buffer(text_bytes)
    dll.espeak_Synth(ctypes.cast(text_buf, ctypes.c_void_p), len(text_bytes) + 1, 0, 0, 0, 0, None, None)
    if not chunks:
        raise HTTPException(500, "eSpeak produced no audio")
    return _wav_bytes(np.concatenate(chunks), tts["sample_rate"])

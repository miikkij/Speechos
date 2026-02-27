"""Text-to-speech synthesis endpoints."""

from __future__ import annotations

import io
import logging
import time
import wave

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.docker_tts import docker_synthesize, is_docker_engine
from src.server import get_config, get_models

logger = logging.getLogger(__name__)
router = APIRouter(tags=["synthesis"])


class SynthesizeRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float = 1.0


@router.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Synthesize speech from text, returns WAV audio."""
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")
    if len(req.text) > 10_000:
        raise HTTPException(400, "Text exceeds 10,000 character limit")

    config = get_config()
    models = get_models()
    tts = models.get_tts()

    t0 = time.perf_counter()

    tts_cfg = config.tts
    if not tts_cfg:
        raise HTTPException(500, "No TTS engine configured")

    engine = tts_cfg.engine
    audio_int16 = None
    sample_rate = 22050

    # ── Docker-backed engines (proxy to WSL2 container) ──
    if is_docker_engine(engine):
        try:
            wav_bytes = await docker_synthesize(engine, req.text, req.voice)
        except Exception as e:
            logger.error("Docker TTS (%s) failed: %s", engine, e)
            raise HTTPException(500, f"Docker TTS engine '{engine}' failed: {e}")

        elapsed = time.perf_counter() - t0
        logger.info("TTS (%s/docker) completed in %.2fs for %d chars", engine, elapsed, len(req.text))
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={
                "X-Processing-Time": str(round(elapsed, 3)),
                "Content-Disposition": "inline; filename=speech.wav",
            },
        )

    # ── Native engines (run in-process) ──────────────────
    if engine == "piper":
        audio_chunks = []
        for chunk in tts.synthesize(req.text):
            sample_rate = chunk.sample_rate
            audio_chunks.append(chunk.audio_float_array)
        if not audio_chunks:
            raise HTTPException(500, "TTS produced no audio")
        audio = np.concatenate(audio_chunks)
        audio_int16 = (audio * 32767).astype(np.int16)

    elif engine == "kokoro":
        # Kokoro KPipeline returns generator of (graphemes, phonemes, audio_tensor)
        voice = req.voice or "af_heart"
        audio_chunks = []
        for _gs, _ps, audio_tensor in tts(req.text, voice=voice):
            audio_chunks.append(audio_tensor.numpy())
        if not audio_chunks:
            raise HTTPException(500, "TTS produced no audio")
        audio = np.concatenate(audio_chunks)
        sample_rate = 24000
        audio_int16 = (audio * 32767).astype(np.int16)

    elif engine == "chatterbox":
        wav = tts.generate(req.text)
        # Returns tensor [1, samples] at 24kHz
        sample_rate = 24000
        audio = wav.squeeze().cpu().numpy()
        audio_int16 = (audio * 32767).astype(np.int16)

    elif engine == "orpheus":
        audio_data = tts.generate_speech(req.text)
        sample_rate = audio_data.get("sample_rate", 24000)
        audio = np.array(audio_data["audio"], dtype=np.float32)
        audio_int16 = (audio * 32767).astype(np.int16)

    elif engine == "xtts":
        # Coqui TTS .tts() returns numpy array
        audio = tts.tts(text=req.text)
        audio = np.array(audio, dtype=np.float32)
        sample_rate = 22050
        audio_int16 = (audio * 32767).astype(np.int16)

    elif engine == "melotts":
        # MeloTTS .tts_to_file returns audio, or we use synthesize
        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        speaker_ids = tts.hps.data.spk2id
        first_speaker = list(speaker_ids.values())[0]
        tts.tts_to_file(req.text, first_speaker, tmp_path, speed=req.speed)
        audio, sample_rate = sf.read(tmp_path, dtype="int16")
        import os
        os.unlink(tmp_path)
        audio_int16 = np.array(audio, dtype=np.int16)

    elif engine == "bark":
        # Transformers text-to-audio pipeline returns ndarray
        result = tts(req.text)
        audio = np.array(result["audio"], dtype=np.float32).squeeze()
        sample_rate = result["sampling_rate"]
        audio_int16 = (audio * 32767).astype(np.int16)

    elif engine == "parler":
        import torch
        model = tts["model"]
        tokenizer = tts["tokenizer"]
        device = tts["device"]
        description = "A female speaker delivers a clear, natural speech."
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(req.text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
        audio = generation.cpu().float().numpy().squeeze()
        sample_rate = model.config.sampling_rate
        audio_int16 = (audio * 32767).astype(np.int16)

    elif engine == "chattts":
        import ChatTTS
        params = ChatTTS.Chat.InferCodeParams(spk_emb=tts.sample_random_speaker())
        wavs = tts.infer([req.text], skip_refine_text=True, params_infer_code=params)
        audio = np.array(wavs[0], dtype=np.float32).squeeze()
        sample_rate = 24000
        audio_int16 = (audio * 32767).astype(np.int16)

    elif engine == "fish-speech":
        result = tts.synthesize(req.text)
        audio = np.array(result["audio"], dtype=np.float32)
        sample_rate = result.get("sample_rate", 44100)
        audio_int16 = (audio * 32767).astype(np.int16)

    elif engine == "cosyvoice":
        result = tts.inference_sft(req.text, "English")
        audio = result["tts_speech"].numpy().squeeze()
        sample_rate = 22050
        audio_int16 = (audio * 32767).astype(np.int16)

    elif engine == "qwen3-tts":
        import torch
        model = tts["model"]
        tokenizer = tts["tokenizer"]
        device = tts["device"]
        inputs = tokenizer(req.text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=4096)
        audio = outputs.cpu().float().numpy().squeeze()
        sample_rate = 24000
        audio_int16 = (audio * 32767).astype(np.int16)

    elif engine == "espeak":
        import ctypes
        dll = tts.get("dll")
        if dll:
            # Use espeakng_loader DLL via ctypes
            sr = tts["sample_rate"]
            sample_rate = sr
            CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)
            audio_chunks: list[np.ndarray] = []

            @CB
            def _cb(wav, numsamples, events):
                if wav and numsamples > 0:
                    arr = (ctypes.c_short * numsamples)()
                    ctypes.memmove(arr, wav, numsamples * 2)
                    audio_chunks.append(np.frombuffer(arr, dtype=np.int16).copy())
                return 0

            dll.espeak_SetSynthCallback(_cb)
            lang = (tts.get("lang", "en")).encode("utf-8")
            dll.espeak_SetVoiceByName(lang)
            text_bytes = req.text.encode("utf-8")
            text_buf = ctypes.create_string_buffer(text_bytes)
            dll.espeak_Synth(ctypes.cast(text_buf, ctypes.c_void_p), len(text_bytes) + 1, 0, 0, 0, 0, None, None)

            if not audio_chunks:
                raise HTTPException(500, "eSpeak produced no audio")
            audio_int16 = np.concatenate(audio_chunks)
        else:
            # Fallback to subprocess if system binary available
            import subprocess
            result = subprocess.run(
                ["espeak-ng", "--stdout", "-v", tts.get("lang", "en"), req.text],
                capture_output=True, check=True,
            )
            wav_bytes = result.stdout
            buf = io.BytesIO(wav_bytes)
            elapsed = time.perf_counter() - t0
            logger.info("TTS (espeak) completed in %.2fs", elapsed)
            return StreamingResponse(
                buf,
                media_type="audio/wav",
                headers={
                    "X-Processing-Time": str(round(elapsed, 3)),
                    "Content-Disposition": "inline; filename=speech.wav",
                },
            )

    else:
        raise HTTPException(500, f"Unknown TTS engine: {engine}")

    if audio_int16 is None:
        raise HTTPException(500, "TTS produced no audio")

    # Ensure mono
    if audio_int16.ndim > 1:
        audio_int16 = audio_int16.mean(axis=-1).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)

    elapsed = time.perf_counter() - t0
    logger.info("TTS (%s) completed in %.2fs for %d chars", engine, elapsed, len(req.text))

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "X-Processing-Time": str(round(elapsed, 3)),
            "Content-Disposition": "inline; filename=speech.wav",
        },
    )

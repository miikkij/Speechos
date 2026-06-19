# ============================================================
# VibeVoice TTS server (OpenAI-compatible) for Speechos
# ============================================================
# Long-form, multi-speaker podcast TTS. Serves the community fork
# (vibevoice-community/VibeVoice) behind the OpenAI-compatible
# `vibevoice-api` server on port 8000.
#
#   POST /v1/audio/speech
#   {"model": "...", "input": "...", "voice": "Frank", "response_format": "wav"}
#
# GPU-only (CUDA + PyTorch). Output is 24kHz WAV.
# ============================================================
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    VIBEVOICE_MODEL=vibevoice/VibeVoice-1.5B \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the community VibeVoice library + the OpenAI-compatible API server.
# flash-attn is optional; the server falls back to SDPA attention if absent.
RUN pip install --no-cache-dir \
        "git+https://github.com/vibevoice-community/VibeVoice.git" \
        vibevoice-api

EXPOSE 8000

# The model is downloaded from HuggingFace on first request and cached in the volume.
CMD ["sh", "-c", "python -m vibevoice_api.server --model_path ${VIBEVOICE_MODEL} --host 0.0.0.0 --port ${PORT}"]

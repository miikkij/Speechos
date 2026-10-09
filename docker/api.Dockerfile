# ============================================================
# Speechos API — GPU-enabled Python backend
# ============================================================
# Multi-stage build: base → dependencies → runtime
# CUDA 12.8 base to match the torch cu128 wheels in api/uv.lock.
# ============================================================

# --- Stage 1: Base with CUDA ---
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# System dependencies for audio processing (Python comes from uv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    espeak-ng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv

WORKDIR /app

# --- Stage 2: Dependencies ---
FROM base AS deps

# Compiler for sdist-only deps (webrtcvad via resemblyzer), deps stage only
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api/pyproject.toml api/uv.lock ./
RUN uv python install 3.12 \
    && uv sync --frozen --no-install-project

# --- Stage 3: Runtime ---
FROM base AS runtime

# Interpreter + venv from the deps stage (the venv links to /opt/python)
COPY --from=deps /opt/python /opt/python
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source
COPY api/src ./src

# Model cache / recordings volumes
VOLUME /models
VOLUME /recordings
ENV SPEECHOS_MODEL_DIR=/models \
    SPEECHOS_RECORDINGS_DIR=/recordings \
    SPEECHOS_PORT=36300

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:36300/health || exit 1

EXPOSE 36300

CMD ["python", "-m", "src"]

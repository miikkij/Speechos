# ============================================================
# Speechos API: GPU-enabled Python backend
# ============================================================
# Multi-stage build: base → dependencies → runtime
# Supports CUDA GPU acceleration with CPU fallback
# ============================================================

# --- Stage 1: Base with CUDA ---
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    espeak-ng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --- Stage 2: Dependencies ---
FROM base AS deps

COPY api/pyproject.toml api/uv.lock* ./
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# --- Stage 3: Runtime ---
FROM base AS runtime

# Copy installed packages from deps stage
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source
COPY api/src ./src

# Model cache volume mount point
VOLUME /models
ENV SPEECHOS_MODEL_DIR=/models

# Recordings volume mount point
VOLUME /recordings

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:36300/health || exit 1

EXPOSE 36300

CMD ["python", "-m", "src.server"]

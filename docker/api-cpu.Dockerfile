# ============================================================
# Speechos API — CPU-only Python backend
# ============================================================
# No CUDA base image (smaller, runs anywhere). Note: api/uv.lock pins the
# cu128 torch wheels, which also run on CPU; a CPU-only torch would need a
# separate lock/index (see docs/architecture/ARCHITECTURE.md, Roadmap).
# ============================================================

FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
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
RUN uv sync --frozen --no-install-project

# --- Stage 3: Runtime ---
FROM base AS runtime

COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY api/src ./src

VOLUME /models
VOLUME /recordings
ENV SPEECHOS_MODEL_DIR=/models \
    SPEECHOS_RECORDINGS_DIR=/recordings \
    SPEECHOS_PORT=36300 \
    SPEECHOS_COMPUTE_MODE=cpu

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:36300/health || exit 1

EXPOSE 36300

CMD ["python", "-m", "src"]

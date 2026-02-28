# ============================================================
# Speechos API: CPU-only Python backend
# ============================================================
# Lighter image without CUDA (smaller download, works anywhere)
# ============================================================

FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    espeak-ng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --- Stage 2: Dependencies ---
FROM base AS deps

COPY api/pyproject.toml api/uv.lock* ./
# Install CPU-only PyTorch variant
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# --- Stage 3: Runtime ---
FROM base AS runtime

COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY api/src ./src

VOLUME /models
ENV SPEECHOS_MODEL_DIR=/models
ENV SPEECHOS_COMPUTE_MODE=cpu

VOLUME /recordings

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "src.server"]

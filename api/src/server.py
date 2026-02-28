"""Speechos FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import AppConfig, load_config
from src.models import ModelManager

logger = logging.getLogger(__name__)

config: AppConfig | None = None
model_manager: ModelManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, model_manager
    config = load_config()
    model_manager = ModelManager(config)
    logger.info(
        "Speechos API starting: tier=%s mode=%s gpu=%s",
        config.hardware.tier,
        config.hardware.mode.value,
        config.hardware.gpu_name or "none",
    )
    yield
    if model_manager:
        model_manager.unload_all()
    logger.info("Speechos API shutdown complete")


app = FastAPI(
    title="Speechos API",
    description="Local-first speech analysis and synthesis platform",
    version="0.1.0",
    lifespan=lifespan,
)


def get_config() -> AppConfig:
    assert config is not None
    return config


def get_models() -> ModelManager:
    assert model_manager is not None
    return model_manager


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:36301", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────

from src.routers import health, recordings, system, transcribe, synthesize, analyze  # noqa: E402

app.include_router(health.router)
app.include_router(transcribe.router, prefix="/api")
app.include_router(synthesize.router, prefix="/api")
app.include_router(analyze.router, prefix="/api")
app.include_router(recordings.router, prefix="/api")
app.include_router(system.router, prefix="/api")

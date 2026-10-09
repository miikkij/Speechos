"""Speechos FastAPI application.

``create_app()`` builds the app from a config (loaded from tier defaults,
config.yaml and SPEECHOS_* env vars when not given). ``src.server:app`` is
created lazily on first access, so importing this module has no side effects.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import AppConfig, load_config
from src.models import ModelManager

logger = logging.getLogger(__name__)

config: AppConfig | None = None
model_manager: ModelManager | None = None


def get_config() -> AppConfig:
    assert config is not None, "Speechos app not initialised (call create_app())"
    return config


def get_models() -> ModelManager:
    assert model_manager is not None, "Speechos app not initialised (call create_app())"
    return model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg, models = get_config(), get_models()
    logger.info(
        "Speechos API starting — tier=%s mode=%s gpu=%s",
        cfg.hardware.tier,
        cfg.hardware.mode.value,
        cfg.hardware.gpu_name or "none",
    )
    preload: asyncio.Task[None] | None = None
    if cfg.model_loading == "eager":
        # Load in the background so /health answers while models load
        preload = asyncio.create_task(asyncio.to_thread(models.preload))
    yield
    if preload is not None and not preload.done():
        preload.cancel()
    models.unload_all()
    logger.info("Speechos API shutdown complete")


def create_app(app_config: AppConfig | None = None, models: ModelManager | None = None) -> FastAPI:
    global config, model_manager
    config = app_config or load_config()
    model_manager = models or ModelManager(config)

    app = FastAPI(
        title="Speechos API",
        description="Local-first speech analysis and synthesis platform",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Processing-Time", "X-TTS-Engine", "X-TTS-Model"],
    )

    from src.routers import analyze, health, recordings, synthesize, system, transcribe

    app.include_router(health.router)
    app.include_router(transcribe.router, prefix="/api")
    app.include_router(synthesize.router, prefix="/api")
    app.include_router(analyze.router, prefix="/api")
    app.include_router(recordings.router, prefix="/api")
    app.include_router(system.router, prefix="/api")
    return app


_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    # Lazy module attribute: `uvicorn src.server:app` keeps working
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)

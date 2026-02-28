"""Speechos API entry point: ``python -m src``."""

import uvicorn

from src.config import load_config

config = load_config()
uvicorn.run(
    "src.server:app",
    host=config.host,
    port=config.port,
    workers=config.workers,
    log_level="info",
)

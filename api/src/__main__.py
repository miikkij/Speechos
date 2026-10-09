"""Speechos API entry point — ``python -m src``."""

import logging

import uvicorn

from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

config = load_config()
uvicorn.run(
    "src.server:create_app",
    factory=True,
    host=config.host,
    port=config.port,
    workers=config.workers,
    log_level="info",
)

"""Lightweight per-site status HTTP server — polled by the FL server heartbeat."""
from __future__ import annotations

import threading
from typing import Any

import uvicorn
from fastapi import FastAPI

from client.config import get_client_settings
from client.engine.state import get_state

_app = FastAPI(docs_url=None, redoc_url=None)   # no docs UI needed


@_app.get("/site/status")
def site_status() -> dict[str, Any]:
    """Return site identity, run count, last run timestamp, and current phase."""
    settings = get_client_settings()
    s = get_state()
    return {
        "site_id":     settings.site_id,
        "run_count":   s.run_count,
        "last_run_at": s.last_run_at,
        "phase":       s.phase,
    }


def start_status_server(port: int) -> None:
    """Start the status HTTP server as a background daemon thread."""
    threading.Thread(
        target=uvicorn.run,
        kwargs={"app": _app, "host": "0.0.0.0", "port": port, "log_level": "warning"},
        daemon=True,
        name="fl-status-server",
    ).start()

"""Periodic heartbeat ping to the aggregation server."""
from __future__ import annotations

import time
import threading

import httpx

from client.config             import get_client_settings
from shared.utils.logging_config import get_logger

log = get_logger(__name__)
INTERVAL = 30   # seconds


def _beat() -> None:
    settings = get_client_settings()
    while True:
        try:
            r = httpx.get(f"{settings.server_url}/health/", timeout=5)
            log.debug("heartbeat", site=settings.site_id, status=r.status_code)
        except Exception as exc:
            log.warning("heartbeat_failed", site=settings.site_id, error=str(exc))
        time.sleep(INTERVAL)


def start_heartbeat() -> None:
    threading.Thread(target=_beat, daemon=True, name="fl-heartbeat").start()

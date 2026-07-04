"""
Round watcher — polls the server for active FL rounds and triggers
local training when a new COLLECTING round is detected.
"""
from __future__ import annotations

import time
import threading

import httpx

from client.comms.fl_client      import FLClient
from client.engine.local_trainer import LocalTrainer
from client.config               import get_client_settings
from shared.utils.logging_config import get_logger

log = get_logger(__name__)
POLL_SECONDS = 15


def _watch() -> None:
    settings = get_client_settings()
    fl       = FLClient()
    trainer  = LocalTrainer()

    try:
        fl.authenticate()
    except Exception as exc:
        log.error("auth_failed_on_start", error=str(exc))
        return

    last_seen_round = 0

    while True:
        try:
            resp = httpx.get(
                f"{settings.server_url}/federation/round/{last_seen_round + 1}",
                headers=fl.auth_headers,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                rid    = data.get("round_id", 0)
                status = data.get("status", "")
                if rid > last_seen_round and status == "collecting":
                    log.info("new_round", round_id=rid)
                    update = trainer.train_and_prepare_update(rid)
                    fl.upload_update(update)
                    last_seen_round = rid
        except Exception as exc:
            log.warning("scheduler_poll_error", error=str(exc))

        time.sleep(POLL_SECONDS)


def start_scheduler() -> None:
    threading.Thread(target=_watch, daemon=True, name="fl-scheduler").start()

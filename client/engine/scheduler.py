"""
FL round watcher (scheduler) — polls the server and triggers local training.

WHAT THIS MODULE DOES
----------------------
The scheduler is the "main loop" of the FL client. It runs as a background
thread, continuously asking the server: "Is there a new round for me to join?"

When a new round in "collecting" status is detected, the scheduler:
  1. Runs local training (LocalTrainer.train_and_prepare_update)
  2. Uploads the resulting model update (FLClient.upload_update)
  3. Records the round_id so it doesn't reprocess the same round

This polling approach means each site independently decides when to participate.
Sites don't need to be synchronised with each other — a slow site can submit
its update late (up to round_timeout_seconds after the round starts) and the
server will wait.

POLLING INTERVAL
-----------------
The scheduler polls every POLL_SECONDS = 15 seconds. This is a deliberate
trade-off:
  - Too frequent (e.g., 1s): unnecessary server load, round detection latency
    is already < 1s which adds no value
  - Too infrequent (e.g., 60s): slow round participation; sites take up to 60s
    to notice a round started, delaying convergence

KNOWN ISSUE
------------
The current implementation calls `httpx.get()` directly instead of using
`fl.authenticate()` credentials management via `FLClient._request()`.
This means:
  1. It bypasses the exponential backoff retry logic
  2. It does not handle token expiry (the auth header will stop working after 15 min)
A future fix: use `fl._request()` for the poll request too.

PYTHON CONCEPT: threading.Thread with daemon=True
  A daemon thread exits when the main thread (the Flet UI) exits.
  If we used a non-daemon thread, Python would wait for the scheduler to
  finish before the process could exit — which would never happen because
  the while-loop runs forever. Daemon threads allow clean shutdown.

PYTHON CONCEPT: infinite while loop with time.sleep()
  `while True:` runs forever. `time.sleep(POLL_SECONDS)` pauses execution
  for 15 seconds between iterations. This is appropriate for a background
  polling task that should run for the lifetime of the process.
"""
from __future__ import annotations

import time
import threading

import httpx

from client.comms.fl_client      import FLClient
from client.engine.local_trainer  import LocalTrainer
from client.config                import get_client_settings
from shared.utils.logging_config  import get_logger

log = get_logger(__name__)

# How often to check if the server has started a new round (seconds)
POLL_SECONDS = 15


def _watch() -> None:
    """
    Main loop for the round-watching scheduler thread.

    Runs until the process exits. On each iteration:
      1. Fetch the next expected round (last_seen_round + 1) from the server
      2. If a "collecting" round is found, train locally and upload the update
      3. Sleep for POLL_SECONDS, then repeat

    The `last_seen_round` counter prevents re-processing a round that was
    already handled in a previous iteration. When round 3 is handled,
    `last_seen_round` becomes 3, so we only watch for round 4 next time.
    """
    settings = get_client_settings()
    fl       = FLClient()      # HTTP client with retry, SSL, auth
    trainer  = LocalTrainer()  # local fitting engine

    # Authenticate on startup — obtain access + refresh tokens
    try:
        fl.authenticate()
    except Exception as exc:
        log.error("auth_failed_on_start", error=str(exc))
        return   # if we can't authenticate, there's no point polling

    last_seen_round = 0   # we have not participated in any round yet

    while True:
        try:
            # Poll for the next round.
            # NOTE: This uses httpx.get() directly — a known issue.
            # It should use fl._request() to get retry/backoff support.
            resp = httpx.get(
                f"{settings.server_url}/federation/round/{last_seen_round + 1}",
                headers=fl.auth_headers,
                timeout=10,
            )
            if resp.status_code == 200:
                data   = resp.json()
                rid    = data.get("round_id", 0)
                status = data.get("status", "")

                # Only train for rounds we haven't seen before that are in COLLECTING state
                if rid > last_seen_round and status == "collecting":
                    log.info("new_round", round_id=rid)

                    # Run local training — this is the computationally expensive part.
                    # It loads data, fits Hermia models, applies DP noise, and returns
                    # a ModelUpdate object.
                    update = trainer.train_and_prepare_update(rid)

                    # Upload the model update to the server.
                    # FLClient handles token refresh if the 15-min access token expired.
                    fl.upload_update(update)

                    last_seen_round = rid   # mark this round as handled

        except Exception as exc:
            # Catch all exceptions to prevent the polling loop from dying.
            # Common causes: server unavailable, network timeout, parse error.
            # We log a warning and continue — the next poll iteration will retry.
            log.warning("scheduler_poll_error", error=str(exc))

        # Wait before the next poll.
        # `time.sleep()` releases the GIL, allowing other threads to run.
        time.sleep(POLL_SECONDS)


def start_scheduler() -> None:
    """
    Start the round-watcher scheduler as a background daemon thread.

    Called once from client/main.py at startup. The thread runs for the
    entire lifetime of the client process and cannot be stopped externally
    (other than by the process exiting).

    Thread name "fl-scheduler" appears in log messages and OS process views,
    making it easy to identify in debugging.
    """
    threading.Thread(target=_watch, daemon=True, name="fl-scheduler").start()

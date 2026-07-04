"""
Heartbeat — periodic keep-alive ping to the aggregation server.

PURPOSE
--------
The heartbeat thread sends a lightweight GET /health/ ping to the server
every 30 seconds. This serves several purposes:

  1. LIVENESS CHECK: Confirms this site can reach the server. If the heartbeat
     fails repeatedly, it signals a network problem that the site operator
     should investigate.

  2. SERVER AWARENESS: The server can track which sites are actively connected
     vs. which appear to have gone offline (no heartbeat for > 60s). Currently,
     this data is only logged — a future enhancement would feed it into the
     `site_statuses` dictionary in RoundManager to show connectivity status
     on the dashboard.

  3. KEEPALIVE: Some firewalls and NAT devices drop idle TCP connections after
     60–300 seconds of inactivity. Regular pings prevent the connection pool
     from being killed by network infrastructure.

WHY A SEPARATE THREAD?
-----------------------
The heartbeat needs to run continuously regardless of whether a training round
is in progress. Using a separate thread ensures the ping is sent even when the
scheduler thread is busy running local training (which can take 10–30 seconds).

The heartbeat thread has its own httpx.Client (separate from FLClient's client)
because httpx.Client is not thread-safe by default — concurrent calls from
multiple threads can cause race conditions. Separate clients, separate pools.

CONNECTION POOLING NOTE
------------------------
The heartbeat client is created ONCE and reused throughout the thread's lifetime.
This means:
  - The TCP connection is established on the first ping and kept alive
  - Subsequent pings reuse the existing connection (much faster)
  - The connection is closed when the thread exits via the `finally` block

PYTHON CONCEPT: try/finally
  `try: ... finally: client.close()` guarantees that `client.close()` runs
  even if an exception is raised inside the try block. This prevents resource
  leaks (open file descriptors, TCP connections) even if the thread crashes.

PYTHON CONCEPT: logging.debug vs logging.warning
  `log.debug("heartbeat", ...)` writes to the log only when LOG_LEVEL=DEBUG.
  At the default INFO level, successful heartbeats are silent — they would
  otherwise flood the log with 2 entries per minute per site.
  `log.warning("heartbeat_failed", ...)` always appears because network failures
  are worth alerting on.
"""
from __future__ import annotations

import time
import threading

import httpx

from client.config import get_client_settings
from shared.utils.logging_config import get_logger

log = get_logger(__name__)

# How often to ping the server — every 30 seconds is a common keepalive interval
INTERVAL = 30  # seconds


def _beat() -> None:
    """
    Infinite loop that pings the server health endpoint every INTERVAL seconds.

    Runs as a daemon thread for the entire client process lifetime.
    Failures are logged as warnings but do not stop the loop — a failed
    heartbeat should not kill the FL client.
    """
    settings = get_client_settings()

    # Create a dedicated httpx.Client for the heartbeat.
    # verify=settings.verify_ssl respects the same SSL setting as FLClient.
    # timeout is the connect+read timeout; for a simple health ping, the
    # connect_timeout is appropriate (we don't need a long read timeout).
    client = httpx.Client(
        verify=settings.verify_ssl,
        timeout=float(settings.connect_timeout),
    )

    try:
        while True:
            try:
                # Simple GET /health/ — no auth required, very fast response
                r = client.get(f"{settings.server_url}/health/")
                # Log at DEBUG level — only visible when LOG_LEVEL=DEBUG
                # At INFO level (default), successful pings are silent
                log.debug("heartbeat", site=settings.site_id, status=r.status_code)

            except Exception as exc:
                # Any network error (timeout, connection refused, DNS failure)
                # is logged as a warning but the loop continues.
                # The scheduler thread will detect the failure when it tries
                # to upload an update.
                log.warning("heartbeat_failed", site=settings.site_id, error=str(exc))

            # Wait 30 seconds before the next ping.
            # time.sleep() releases the GIL — other threads (scheduler, UI)
            # can run freely during this wait.
            time.sleep(INTERVAL)

    finally:
        # Always close the HTTP client when this thread exits,
        # whether it exits normally (impossible for an infinite loop)
        # or due to an unexpected exception.
        client.close()


def start_heartbeat() -> None:
    """
    Start the heartbeat as a background daemon thread.

    Called once from client/main.py at startup. The thread name
    "fl-heartbeat" appears in OS-level thread listings and log entries,
    making it easy to identify when debugging.
    """
    threading.Thread(target=_beat, daemon=True, name="fl-heartbeat").start()

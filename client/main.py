"""
Client entry point — starts FL background threads then launches the Flet status UI.

HOW THE CLIENT STARTS
----------------------
When a manufacturing site's Docker container starts (or you run `python client/main.py`),
this file is the entry point. It does three things:

  1. Configures structured logging (call first, before any log messages)
  2. Starts background threads for the FL protocol:
       - heartbeat thread: sends keep-alive pings to the server every 30 seconds
       - scheduler thread: polls for new rounds and triggers local training
  3. Starts the Flet web UI on the configured port so site operators can
     see the site's status, training progress, and local results

THREADING MODEL
---------------
Python threads share memory within one process, but the GIL (Global Interpreter
Lock) limits true CPU parallelism for Python code. However, for I/O-bound tasks
like network calls (httpx requests), threads can run truly in parallel because
the GIL is released during I/O operations.

This client uses three concurrent execution contexts:
  - Main thread      → runs the Flet UI event loop
  - Heartbeat thread → runs an infinite loop sending pings
  - Scheduler thread → runs an infinite loop polling for rounds

Both background threads are started as `daemon=True`. A daemon thread is
automatically killed when the main thread exits. This ensures clean shutdown:
when the user closes the browser window (Flet exits), the entire process stops.

PYTHON CONCEPT: threading.Thread
  Creates a new OS-level thread that runs `target` concurrently with the main code.
  `daemon=True` means: kill this thread if the main thread exits.

PYTHON CONCEPT: ft.run()
  This is a blocking call — it starts the Flet web server and enters an event
  loop that handles browser interactions. The call returns only when the Flet
  app is shut down.

PYTHON CONCEPT: `if __name__ == "__main__":`
  Python sets the special variable __name__ to "__main__" only when a file
  is run directly (e.g., `python client/main.py`). When the file is imported
  as a module, __name__ is set to the module name instead. This idiom prevents
  the startup code from running on import.
"""
import os
import threading      # standard library: create concurrent threads
import flet as ft     # Flet: Python framework for building web/desktop UI

from client.ui.app          import main as flet_main   # Flet page builder function
from client.comms.heartbeat import start_heartbeat      # starts the ping thread
from client.comms.status_server import start_status_server  # per-site status HTTP server
from client.engine.data_source import DevDataSource, ProdDataSource
from client.engine.scheduler import start_scheduler     # starts the round-watcher thread
from client.config           import get_client_settings
from shared.utils.logging_config import configure_logging

# Configure structured JSON logging before any logger is used
configure_logging()
settings = get_client_settings()


def _background() -> None:
    """
    Start all background service threads.

    This function runs in its own thread (not the main thread). It:
      1. Starts the heartbeat thread (keep-alive pings to the FL server)
      2. Selects a DataSource based on dev_mode config (DevDataSource or ProdDataSource)
      3. Starts the per-site status HTTP server on client_status_port
      4. Starts the scheduler thread (polls for rounds and triggers local training)

    All spawned sub-threads are daemon threads that keep running independently
    after this function returns.
    """
    start_heartbeat()    # begins pinging the server every 30 seconds

    cfg = get_client_settings()

    if cfg.dev_mode:
        physics = {
            "J0":       cfg.dev_j0,
            "k1":       cfg.dev_k1,
            "k2":       cfg.dev_k2,
            "noise":    cfg.dev_noise,
            "tmp_base": cfg.dev_tmp_base,
        }
        data_source = DevDataSource(physics, jitter=cfg.dev_jitter_fraction)
    else:
        data_dir = os.path.dirname(cfg.local_data_path) or f"data/{cfg.site_id}"
        data_source = ProdDataSource(data_dir)

    start_status_server(cfg.client_status_port)
    start_scheduler(data_source=data_source)   # data_source wired through in Task 5


if __name__ == "__main__":
    # Start background services in a daemon thread.
    # We use a thread here rather than calling _background() directly
    # because start_scheduler() eventually blocks in a polling loop,
    # and we need ft.app() to run on the main thread (Flet requirement).
    threading.Thread(target=_background, daemon=True).start()

    # Launch the Flet web interface.
    # - flet_main: the function that builds the UI page
    # - port: the HTTP port the browser connects to (8551-8555 per site)
    # - view=WEB_BROWSER: serve as a web app (not a desktop window)
    # This call BLOCKS until the Flet server shuts down.
    ft.run(
        flet_main,
        port=settings.flet_client_port,
        view=ft.AppView.WEB_BROWSER,
    )

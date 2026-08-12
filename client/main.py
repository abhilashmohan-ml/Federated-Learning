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
import threading      # standard library: create concurrent threads
import flet as ft     # Flet: Python framework for building web/desktop UI

from client.ui.app          import main as flet_main   # Flet page builder function
from client.comms.heartbeat import start_heartbeat      # starts the ping thread
from client.engine.scheduler import start_scheduler     # starts the round-watcher thread
from client.config           import get_client_settings
from shared.utils.logging_config import configure_logging

# Configure structured JSON logging before any logger is used
configure_logging()
settings = get_client_settings()


def _background() -> None:
    """
    Start all background service threads.

    This function runs in its own thread (not the main thread). It calls
    `start_heartbeat()` and `start_scheduler()` which each spawn their
    own daemon threads internally. After starting them, this function returns —
    the spawned threads keep running independently.
    """
    start_heartbeat()    # begins pinging the server every 30 seconds
    start_scheduler()    # begins polling for new FL rounds


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

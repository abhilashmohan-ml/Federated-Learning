"""Client entry point — starts FL engine threads then launches Flet UI."""
import threading
import flet as ft

from client.ui.app         import main as flet_main
from client.comms.heartbeat import start_heartbeat
from client.engine.scheduler import start_scheduler
from client.config          import get_client_settings
from shared.utils.logging_config import configure_logging

configure_logging()
settings = get_client_settings()


def _background() -> None:
    start_heartbeat()
    start_scheduler()


if __name__ == "__main__":
    threading.Thread(target=_background, daemon=True).start()
    ft.app(
        target=flet_main,
        port=settings.flet_client_port,
        view=ft.AppView.WEB_BROWSER,
    )

"""Client Flet UI - simple operator dashboard (two tabs)."""
from __future__ import annotations

import asyncio

import flet as ft

from client.comms.fl_client import FLClient
from shared.utils.logging_config import get_logger

log = get_logger(__name__)
from client.config import get_client_settings
from client.engine.state import get_state
from client.ui.pages.local_results import LocalResultsPage
from client.ui.pages.status import StatusPage
from shared.utils.theme import LC


def main(page: ft.Page, fl_client: FLClient | None = None) -> None:
    settings = get_client_settings()
    page.title      = f"Viral FL Client - {settings.site_id}"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor    = LC.BG_PRIMARY
    page.padding    = 20

    if fl_client is None:
        log.warning("fl_client_not_injected", reason="FLClient created inside flet_main — browser reconnect will re-authenticate")
        fl_client = FLClient()
        fl_client.authenticate()
    fl = fl_client

    status_pg  = StatusPage(page, fl_client=fl)
    results_pg = LocalResultsPage(page)

    status_content  = status_pg.build()
    results_content = results_pg.build()

    tab_bar  = ft.TabBar(tabs=[ft.Tab(label="Status"), ft.Tab(label="Local Results")])
    tab_view = ft.TabBarView(controls=[status_content, results_content], expand=True)

    tabs = ft.Tabs(
        content=ft.Column([tab_bar, tab_view], expand=True),
        length=2,
        selected_index=0,
        expand=True,
    )

    page.add(
        ft.Text(f"Site: {settings.site_id}", size=20,
                weight=ft.FontWeight.BOLD),
        ft.Divider(),
        tabs,
    )

    running = [True]

    def on_disconnect(_: ft.ControlEvent) -> None:
        running[0] = False

    page.on_disconnect = on_disconnect

    async def poll_loop() -> None:
        while running[0]:
            await asyncio.sleep(5)
            try:
                state = get_state()
                status_pg.update_from_state(state)
                results_pg.update_from_state(state)
                page.update()
            except Exception as exc:
                log.warning("client_poll_error", error=str(exc))

    page.run_task(poll_loop)

"""
Flet server dashboard entry point.

Pages
-----
0  Dashboard      all sites status + round progress bar
1  Site Monitor   per-site J(t), LRV, Amin live charts
2  Global Model   global PINN parameters + performance history
3  Graphs         comparative charts across all sites
4  Settings       server config + site management
"""
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import flet as ft
import httpx

from server.config import get_settings
from shared.utils.logging_config import get_logger

log = get_logger(__name__)
from server.ui.pages.dashboard    import DashboardPage
from server.ui.pages.site_monitor import SiteMonitorPage
from server.ui.pages.global_model import GlobalModelPage
from server.ui.pages.graphs       import GraphsPage
from server.ui.pages.settings     import SettingsPage
from server.ui.components.nav_rail import build_nav_rail
from shared.utils.theme import LC

_ASSETS_DIR = str(Path(__file__).parent / "assets")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


def main(page: ft.Page) -> None:
    page.title      = "Viral FL - Server Dashboard"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor    = LC.BG_PRIMARY
    page.padding    = 0

    settings = get_settings()
    poll_url = f"http://localhost:{settings.port}/internal/status"

    pages = [
        DashboardPage(page),
        SiteMonitorPage(page),
        GlobalModelPage(page),
        GraphsPage(page),
        SettingsPage(page),
    ]
    body = ft.Container(expand=True, content=pages[0].build())

    def on_nav(e: ft.ControlEvent) -> None:
        body.content = pages[e.control.selected_index].build()
        page.update()

    utc_clock = ft.Text(
        _utc_now(),
        size=10,
        color=LC.TEXT_MUTED,
        font_family="monospace",
        text_align=ft.TextAlign.CENTER,
    )

    header = ft.Container(
        content=ft.Row(
            [
                ft.Container(expand=True),
                ft.Column(
                    [
                        ft.Image(
                            src="merck_logo.svg",
                            width=70,
                            height=33,
                            fit=ft.BoxFit.CONTAIN,
                        ),
                        utc_clock,
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=2,
                ),
                ft.Container(width=16),
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.Padding(left=0, top=6, right=0, bottom=6),
        bgcolor=LC.SURFACE,
        border=ft.Border(bottom=ft.BorderSide(1, LC.BORDER)),
        height=56,
    )

    page.add(
        ft.Column(
            [
                header,
                ft.Row(
                    [build_nav_rail(on_nav), ft.VerticalDivider(width=1), body],
                    expand=True,
                ),
            ],
            expand=True,
            spacing=0,
        )
    )

    running = [True]

    def on_disconnect(_: ft.ControlEvent) -> None:
        running[0] = False

    page.on_disconnect = on_disconnect

    async def poll_loop() -> None:
        dashboard: DashboardPage  = pages[0]
        gm_page:   GlobalModelPage = pages[2]
        graphs_pg: GraphsPage      = pages[3]
        last_model_version = -1

        async with httpx.AsyncClient() as client:
            while running[0]:
                await asyncio.sleep(5)
                try:
                    r = await client.get(poll_url, timeout=4.0)
                    if r.status_code == 200:
                        data             = r.json()
                        sites            = data.get("sites", {})
                        mv               = data.get("model_version", 0)
                        rid              = data.get("current_round_id", 0)
                        round_status     = data.get("round_status", "idle")
                        participating    = data.get("participating_sites", [])
                        min_sites        = data.get("min_sites", 1)
                        started_at       = data.get("round_started_at")
                        completed_at     = data.get("round_completed_at")
                        rounds_completed = data.get("rounds_completed", 0)
                        global_metrics   = data.get("global_metrics", {})

                        dashboard.update_sites(sites)
                        dashboard.timeline.update(
                            rid, sites,
                            round_status=round_status,
                            participating=participating,
                            min_sites=min_sites,
                            started_at=started_at,
                            completed_at=completed_at,
                        )

                        # Update run counts and last-run timestamps on all site cards
                        run_counts  = data.get("run_counts", {})
                        last_run_at = data.get("last_run_at", {})
                        for site_id, card in dashboard.cards.items():
                            card.set_run_info(
                                run_count=run_counts.get(site_id, 0),
                                last_run_at=last_run_at.get(site_id),
                            )

                        # Update Site Monitor and comparative charts with per-site metrics
                        site_metrics     = data.get("site_metrics", {})
                        site_best_models = data.get("site_best_models", {})
                        pages[1].update_data(site_metrics, site_best_models, rid)
                        if site_metrics:
                            graphs_pg.update(site_metrics)

                        if mv != last_model_version:
                            last_model_version   = mv
                            last_round_sites     = data.get("last_round_participating_sites", [])
                            sites_last_round     = len(last_round_sites) if mv > 0 else 0
                            gm_page.update_model_data(
                                mv, rounds_completed, sites_last_round, global_metrics
                            )

                        page.update()
                except Exception as exc:
                    log.warning("dashboard_poll_error", error=str(exc))
                    dashboard.timeline._chips_row.controls = [
                        ft.Chip(
                            label=ft.Text("Server unreachable", size=11, color=ft.Colors.WHITE),
                            bgcolor=ft.Colors.RED,
                        )
                    ]
                    page.update()

    async def clock_tick() -> None:
        while running[0]:
            utc_clock.value = _utc_now()
            page.update()
            await asyncio.sleep(1)

    page.run_task(poll_loop)
    page.run_task(clock_tick)


if __name__ == "__main__":
    s = get_settings()
    ft.run(main, port=s.flet_port, view=ft.AppView.WEB_BROWSER, assets_dir=_ASSETS_DIR)

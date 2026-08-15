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


def main(page: ft.Page) -> None:
    page.title      = "Viral FL - Server Dashboard"
    page.theme_mode = ft.ThemeMode.DARK
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

    page.add(
        ft.Row(
            [build_nav_rail(on_nav), ft.VerticalDivider(width=1), body],
            expand=True,
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
                        data   = r.json()
                        sites  = data.get("sites", {})
                        mv     = data.get("model_version", 0)
                        rid    = data.get("current_round_id", 0)
                        n_done = len(data.get("participating_sites", []))

                        dashboard.update_sites(sites)
                        dashboard.timeline.update(rid, sites)

                        # Update run counts and last-run timestamps on all site cards
                        run_counts  = data.get("run_counts", {})
                        last_run_at = data.get("last_run_at", {})
                        for site_id, card in dashboard.cards.items():
                            card.set_run_info(
                                run_count=run_counts.get(site_id, 0),
                                last_run_at=last_run_at.get(site_id),
                            )

                        # Update comparative charts with latest per-site metrics
                        site_metrics = data.get("site_metrics", {})
                        if site_metrics:
                            graphs_pg.update(site_metrics)

                        if mv != last_model_version:
                            last_model_version = mv
                            gm_page.update_tiles(mv, rid, n_done)

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

    page.run_task(poll_loop)


if __name__ == "__main__":
    s = get_settings()
    ft.run(main, port=s.flet_port, view=ft.AppView.WEB_BROWSER)

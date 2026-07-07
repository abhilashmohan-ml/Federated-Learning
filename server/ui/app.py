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
import flet as ft

from server.ui.pages.dashboard    import DashboardPage
from server.ui.pages.site_monitor import SiteMonitorPage
from server.ui.pages.global_model import GlobalModelPage
from server.ui.pages.graphs       import GraphsPage
from server.ui.pages.settings     import SettingsPage
from server.ui.components.nav_rail import build_nav_rail
from server.config import get_settings


def main(page: ft.Page) -> None:
    page.title      = "Viral FL — Server Dashboard"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding    = 0

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


if __name__ == "__main__":
    s = get_settings()
    ft.run(main, port=s.flet_port, view=ft.AppView.WEB_BROWSER)

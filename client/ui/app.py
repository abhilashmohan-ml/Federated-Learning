"""Client Flet UI — simple operator dashboard (two tabs)."""
import flet as ft
from client.ui.pages.status        import StatusPage
from client.ui.pages.local_results import LocalResultsPage
from client.config                 import get_client_settings


def main(page: ft.Page) -> None:
    settings = get_client_settings()
    page.title      = f"Viral FL Client — {settings.site_id}"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding    = 20

    status_content  = StatusPage(page).build()
    results_content = LocalResultsPage(page).build()

    tab_bar  = ft.TabBar(tabs=[ft.Tab(label="Status"), ft.Tab(label="Local Results")])
    tab_view = ft.TabBarView(controls=[status_content, results_content], expand=True)

    tabs = ft.Tabs(
        content=ft.Column([tab_bar, tab_view], expand=True),
        length=2,
        selected_index=0,
        expand=True,
    )

    page.add(
        ft.Text(f"Site: {settings.site_id}", size=20, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        tabs,
    )

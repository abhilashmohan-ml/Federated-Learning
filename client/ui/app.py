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

    tabs = ft.Tabs(
        selected_index=0,
        expand=True,
        tabs=[
            ft.Tab(text="Status",        content=StatusPage(page).build()),
            ft.Tab(text="Local Results", content=LocalResultsPage(page).build()),
        ],
    )

    page.add(
        ft.Text(f"Site: {settings.site_id}", size=20, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        tabs,
    )

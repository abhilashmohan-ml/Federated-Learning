"""Dashboard page — all sites overview + round progress."""
import flet as ft

from server.config import get_settings
from server.ui.components.site_card      import SiteCard
from server.ui.components.round_timeline import RoundTimeline


class DashboardPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        settings = get_settings()
        self.cards    = [SiteCard(f"site_{i}") for i in range(1, 6)]
        self.timeline = RoundTimeline(total_rounds=settings.fl_rounds)

    def build(self) -> ft.Control:
        site_cards = ft.Row(
            [c.build() for c in self.cards],
            wrap=True, spacing=14,
        )
        return ft.Container(
            content=ft.Column([
                ft.Text("Federation Dashboard", size=26, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Text("Site Status", size=17),
                site_cards,
                ft.Divider(),
                ft.Text("Current Round", size=17),
                self.timeline.build(),
            ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=16),
            padding=24,
            expand=True,
        )

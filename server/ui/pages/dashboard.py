"""Dashboard page — all sites overview + round progress."""
import flet as ft

from server.config import get_settings
from server.ui.components.site_card      import SiteCard
from server.ui.components.round_timeline import RoundTimeline


class DashboardPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        settings = get_settings()
        # Cards keyed by site_id — populated dynamically as sites register/connect.
        self.cards: dict[str, SiteCard] = {}
        self._cards_row = ft.Row([], wrap=True, spacing=14)
        self.timeline = RoundTimeline(total_rounds=settings.fl_rounds)

    def update_sites(self, sites: dict[str, str]) -> bool:
        """Update site card statuses; create a card on first sight of a site_id.

        Returns True when at least one new card was created (caller should
        call page.update() regardless, but the flag lets it know layout grew).
        """
        added = False
        for site_id, status in sites.items():
            if site_id not in self.cards:
                card = SiteCard(site_id)
                self.cards[site_id] = card
                self._cards_row.controls.append(card.build())
                added = True
            self.cards[site_id].set_status(status)
        return added

    def build(self) -> ft.Control:
        return ft.Container(
            content=ft.Column([
                ft.Text("Federation Dashboard", size=26, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Text("Site Status", size=17),
                self._cards_row,
                ft.Divider(),
                ft.Text("Current Round", size=17),
                self.timeline.build(),
            ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=16),
            padding=24,
            expand=True,
        )

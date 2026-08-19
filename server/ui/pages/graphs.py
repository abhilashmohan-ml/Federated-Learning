"""Graphs page — comparative charts across all 5 sites, updated from poll loop."""
from __future__ import annotations

import flet as ft
from server.ui.components.flux_chart import FluxChart
from server.ui.components.lrv_chart  import LRVChart
from shared.utils.theme import LC


class GraphsPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self._flux_chart = FluxChart(multi_site=True)
        self._lrv_chart  = LRVChart(multi_site=True)
        self._built: ft.Control | None = None

    def update(self, site_metrics: dict[str, dict[str, float]]) -> None:
        """Refresh charts with latest per-site metrics. Called by server poll loop."""
        self._flux_chart.update_data(site_metrics)
        self._lrv_chart.update_data(site_metrics)

    def build(self) -> ft.Control:
        if self._built is not None:
            return self._built
        self._built = ft.Container(
            content=ft.Column([
                ft.Text("Comparative Results — All Sites", size=26,
                        weight=ft.FontWeight.BOLD, color=LC.TEXT_PRIMARY),
                ft.Divider(color=LC.BORDER),
                ft.Text("Min Filter Area  Amin (m²) — All Sites", size=17,
                        color=LC.TEXT_PRIMARY),
                self._flux_chart.build(),
                ft.Divider(color=LC.BORDER),
                ft.Text("Flux Ratio Distribution Across Sites", size=17,
                        color=LC.TEXT_PRIMARY),
                self._lrv_chart.build(),
                ft.Divider(color=LC.BORDER),
                ft.Text("Hermia Model Consensus", size=17, color=LC.TEXT_PRIMARY),
                ft.Text(
                    "(bar chart — dominant blocking model per site — populates after round 1)",
                    size=12, color=LC.TEXT_MUTED,
                ),
            ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=16),
            padding=24,
            expand=True,
            bgcolor=LC.BG_PRIMARY,
        )
        return self._built

"""Site Monitor page — per-site model metrics and live charts."""
import flet as ft
from server.ui.components.metric_tile import MetricTile
from server.ui.components.flux_chart  import FluxChart
from server.ui.components.lrv_chart   import LRVChart
from shared.utils.theme import LC


class SiteMonitorPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page

    def build(self) -> ft.Control:
        site_options = [ft.dropdown.Option(f"site_{i}") for i in range(1, 6)]
        site_dd = ft.Dropdown(
            label="Select Site",
            options=site_options,
            value=site_options[0].key if site_options else None,
            width=200,
            border_color=LC.BORDER,
            focused_border_color=LC.PRIMARY,
            bgcolor=LC.SURFACE,
            fill_color=LC.SURFACE,
            color=LC.TEXT_PRIMARY,
            label_style=ft.TextStyle(color=LC.TEXT_MUTED),
            border_radius=LC.RADIUS_MD,
        )
        metrics = ft.Row([
            MetricTile("LRV",          "--", "log10").build(),
            MetricTile("Amin",         "--", "m2").build(),
            MetricTile("Flux Ratio",   "--", "").build(),
            MetricTile("Best Model",   "--", "").build(),
            MetricTile("Round",        "--", "").build(),
        ], spacing=12, wrap=True)

        return ft.Container(
            content=ft.Column([
                ft.Text("Site Monitor", size=26, weight=ft.FontWeight.BOLD,
                        color=LC.TEXT_PRIMARY),
                site_dd,
                ft.Divider(color=LC.BORDER),
                metrics,
                ft.Text("Flux Decline J(t)", size=16, color=LC.TEXT_PRIMARY),
                FluxChart().build(),
                ft.Text("LRV vs Flux", size=16, color=LC.TEXT_PRIMARY),
                LRVChart().build(),
            ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=16),
            padding=24,
            expand=True,
            bgcolor=LC.BG_PRIMARY,
        )

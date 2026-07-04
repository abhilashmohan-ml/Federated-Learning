"""Site Monitor page — per-site model metrics and live charts."""
import flet as ft
from server.ui.components.metric_tile import MetricTile
from server.ui.components.flux_chart  import FluxChart
from server.ui.components.lrv_chart   import LRVChart


class SiteMonitorPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page

    def build(self) -> ft.Control:
        site_dd = ft.Dropdown(
            label="Select Site",
            options=[ft.dropdown.Option(f"site_{i}") for i in range(1, 6)],
            value="site_1",
            width=200,
        )
        metrics = ft.Row([
            MetricTile("LRV",          "--", "log10").build(),
            MetricTile("Amin",         "--", "m2").build(),
            MetricTile("Flux Ratio",   "--", "").build(),
            MetricTile("Best Model",   "--", "").build(),
            MetricTile("Round",        "--", "").build(),
        ], spacing=12, wrap=True)

        return ft.Column([
            ft.Text("Site Monitor", size=26, weight=ft.FontWeight.BOLD),
            site_dd,
            ft.Divider(),
            metrics,
            ft.Text("Flux Decline J(t)", size=16),
            FluxChart().build(),
            ft.Text("LRV vs Flux", size=16),
            LRVChart().build(),
        ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=16, padding=24)

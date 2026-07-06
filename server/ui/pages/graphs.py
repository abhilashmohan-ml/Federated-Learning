"""Graphs page — comparative charts across all 5 sites."""
import flet as ft
from server.ui.components.flux_chart import FluxChart
from server.ui.components.lrv_chart  import LRVChart


class GraphsPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page

    def build(self) -> ft.Control:
        return ft.Column([
            ft.Text("Comparative Results — All Sites", size=26, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Text("Flux Decline Overlay  (all 5 sites)", size=17),
            FluxChart(multi_site=True).build(),
            ft.Divider(),
            ft.Text("LRV Distribution Across Sites", size=17),
            LRVChart(multi_site=True).build(),
            ft.Divider(),
            ft.Text("Amin vs Throughput  (populate after first complete round)",
                    size=14, color=ft.Colors.GREY_500),
            ft.Divider(),
            ft.Text("Hermia Model Consensus",  size=17),
            ft.Text("(bar chart showing most-selected blocking model per site — populate after round 1)",
                    size=12, color=ft.Colors.GREY_500),
        ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=16, padding=24)

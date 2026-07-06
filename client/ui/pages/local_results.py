"""Local results page — J(t) chart and summary metrics."""
import flet as ft


class LocalResultsPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page

    def build(self) -> ft.Control:
        flux_chart = ft.Container(
            content=ft.Column([
                ft.Text("Flux (LMH) vs Time (min)", size=13, weight=ft.FontWeight.BOLD),
                ft.Text("Flux Decline J(t)  (data populates after first local training)",
                        size=12, color=ft.Colors.CYAN),
            ], spacing=8),
            height=260,
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.CYAN),
            border_radius=8,
            padding=16,
        )

        metrics = ft.Row([
            ft.Card(content=ft.Container(ft.Column([
                ft.Text("LRV",  size=11, color=ft.Colors.GREY_500),
                ft.Text("—",    size=22, weight=ft.FontWeight.BOLD),
            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=14, width=110, alignment=ft.Alignment(0, 0))),

            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Amin (m2)", size=11, color=ft.Colors.GREY_500),
                ft.Text("—",        size=22, weight=ft.FontWeight.BOLD),
            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=14, width=120, alignment=ft.Alignment(0, 0))),

            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Flux Ratio",  size=11, color=ft.Colors.GREY_500),
                ft.Text("—",           size=22, weight=ft.FontWeight.BOLD),
            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=14, width=120, alignment=ft.Alignment(0, 0))),

            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Best Model", size=11, color=ft.Colors.GREY_500),
                ft.Text("—",          size=16, weight=ft.FontWeight.BOLD),
            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=14, width=140, alignment=ft.Alignment(0, 0))),
        ], spacing=10, wrap=True)

        return ft.Column([
            ft.Text("Local Flux Decline  J(t)", size=18, weight=ft.FontWeight.BOLD),
            flux_chart,
            ft.Divider(),
            ft.Text("Local Metrics", size=16),
            metrics,
        ], spacing=14, scroll=ft.ScrollMode.AUTO)

"""Local results page — J(t) chart and summary metrics."""
import math
import flet as ft


class LocalResultsPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page

    def build(self) -> ft.Control:
        flux_chart = ft.LineChart(
            data_series=[ft.LineChartData(
                data_points=[
                    ft.LineChartDataPoint(x=t, y=100 * math.exp(-0.022 * t))
                    for t in range(0, 61, 2)
                ],
                stroke_width=2, color=ft.colors.CYAN, curved=True,
            )],
            left_axis=ft.ChartAxis(title=ft.Text("Flux (LMH)"), title_size=13),
            bottom_axis=ft.ChartAxis(title=ft.Text("Time (min)"), title_size=13),
            height=260, expand=True,
        )

        metrics = ft.Row([
            ft.Card(content=ft.Container(ft.Column([
                ft.Text("LRV",  size=11, color=ft.colors.GREY_500),
                ft.Text("—",    size=22, weight=ft.FontWeight.BOLD),
            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=14, width=110, alignment=ft.alignment.center)),

            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Amin (m2)", size=11, color=ft.colors.GREY_500),
                ft.Text("—",        size=22, weight=ft.FontWeight.BOLD),
            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=14, width=120, alignment=ft.alignment.center)),

            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Flux Ratio",  size=11, color=ft.colors.GREY_500),
                ft.Text("—",           size=22, weight=ft.FontWeight.BOLD),
            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=14, width=120, alignment=ft.alignment.center)),

            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Best Model", size=11, color=ft.colors.GREY_500),
                ft.Text("—",          size=16, weight=ft.FontWeight.BOLD),
            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=14, width=140, alignment=ft.alignment.center)),
        ], spacing=10, wrap=True)

        return ft.Column([
            ft.Text("Local Flux Decline  J(t)", size=18, weight=ft.FontWeight.BOLD),
            flux_chart,
            ft.Divider(),
            ft.Text("Local Metrics", size=16),
            metrics,
        ], spacing=14, scroll=ft.ScrollMode.AUTO)

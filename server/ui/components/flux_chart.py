"""Flux decline J(t) line chart component."""
import math
import flet as ft

SITE_COLORS = [
    ft.Colors.BLUE, ft.Colors.GREEN, ft.Colors.ORANGE,
    ft.Colors.PINK, ft.Colors.PURPLE,
]


class FluxChart:
    def __init__(self, multi_site: bool = False) -> None:
        self.multi_site = multi_site

    def build(self) -> ft.Control:
        if self.multi_site:
            # Placeholder: one series per site
            series = [
                ft.LineChartData(
                    data_points=[
                        ft.LineChartDataPoint(x=t, y=100 * math.exp(-0.015 * (1 + i * 0.3) * t))
                        for t in range(0, 61, 3)
                    ],
                    stroke_width=2,
                    color=SITE_COLORS[i],
                    curved=True,
                )
                for i in range(5)
            ]
        else:
            series = [
                ft.LineChartData(
                    data_points=[
                        ft.LineChartDataPoint(x=t, y=100 * math.exp(-0.02 * t))
                        for t in range(0, 61, 2)
                    ],
                    stroke_width=2,
                    color=ft.Colors.CYAN,
                    curved=True,
                )
            ]

        return ft.LineChart(
            data_series=series,
            left_axis=ft.ChartAxis(title=ft.Text("Flux (LMH)"), title_size=13),
            bottom_axis=ft.ChartAxis(title=ft.Text("Time (min)"), title_size=13),
            height=270,
            expand=True,
        )

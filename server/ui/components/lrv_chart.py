"""LRV bar chart across sites."""
import flet as ft


class LRVChart:
    def __init__(self, multi_site: bool = False) -> None:
        self.multi_site = multi_site

    def build(self) -> ft.Control:
        lrvs  = [4.8, 5.1, 4.6, 5.3, 4.9]
        sites = [f"site_{i}" for i in range(1, 6)]

        bar_groups = [
            ft.BarChartGroup(
                x=i,
                bar_rods=[ft.BarChartRod(from_y=0, to_y=lrvs[i], width=28, color=ft.Colors.TEAL)],
            )
            for i in range(5)
        ]

        # Regulatory minimum line
        return ft.Column([
            ft.BarChart(
                bar_groups=bar_groups,
                left_axis=ft.ChartAxis(title=ft.Text("LRV"), title_size=13),
                bottom_axis=ft.ChartAxis(
                    labels=[ft.ChartAxisLabel(value=i, label=ft.Text(s, size=10))
                            for i, s in enumerate(sites)],
                    title_size=13,
                ),
                max_y=7.0,
                height=240,
                expand=True,
                interactive=True,
            ),
            ft.Text("Dashed line at LRV=4.0 represents regulatory minimum",
                    size=10, color=ft.Colors.GREY_500),
        ])

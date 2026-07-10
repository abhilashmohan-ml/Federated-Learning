"""LRV bar chart across sites."""
import flet as ft

_PLACEHOLDER_LRVS = [4.8, 5.1, 4.6, 5.3, 4.9]
_SITES = [f"site_{i}" for i in range(1, 6)]


class LRVChart:
    def __init__(self, multi_site: bool = False) -> None:
        self.multi_site = multi_site

    def build(self) -> ft.Control:
        bars = ft.Row([
            ft.Column([
                ft.Container(
                    width=28,
                    height=int(_PLACEHOLDER_LRVS[i] * 30),
                    bgcolor=ft.Colors.TEAL,
                    border_radius=ft.BorderRadius(3, 3, 0, 0),
                    content=ft.Text(""),
                ),
                ft.Text(_SITES[i], size=10, color=ft.Colors.GREY_500),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2)
            for i in range(5)
        ], spacing=12, alignment=ft.MainAxisAlignment.CENTER)

        return ft.Column([
            bars,
            ft.Text("Dashed line at LRV=4.0 represents regulatory minimum",
                    size=10, color=ft.Colors.GREY_500),
        ])

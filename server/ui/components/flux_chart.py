"""Flux decline J(t) line chart component."""
import flet as ft

SITE_COLORS = [
    ft.Colors.BLUE, ft.Colors.GREEN, ft.Colors.ORANGE,
    ft.Colors.PINK, ft.Colors.PURPLE,
]

_SITE_LABELS = [f"site_{i}" for i in range(1, 6)]


class FluxChart:
    def __init__(self, multi_site: bool = False) -> None:
        self.multi_site = multi_site

    def build(self) -> ft.Control:
        if self.multi_site:
            legend = ft.Row([
                ft.Container(width=14, height=14, bgcolor=SITE_COLORS[i],
                             border_radius=3,
                             content=ft.Text(""))
                for i in range(5)
            ], spacing=6)
            subtitle = ft.Text("Flux Decline — All 5 Sites  (data populates after round 1)",
                               size=12, color=ft.Colors.GREY_400)
        else:
            legend = ft.Container(width=14, height=14, bgcolor=ft.Colors.CYAN,
                                  border_radius=3, content=ft.Text(""))
            subtitle = ft.Text("Flux Decline J(t)  (data populates after first local training)",
                               size=12, color=ft.Colors.GREY_400)

        return ft.Container(
            content=ft.Column([
                ft.Text("Flux (LMH) vs Time (min)", size=13, weight=ft.FontWeight.BOLD),
                subtitle,
                legend,
            ], spacing=8),
            height=270,
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.BLUE),
            border_radius=8,
            padding=16,
        )

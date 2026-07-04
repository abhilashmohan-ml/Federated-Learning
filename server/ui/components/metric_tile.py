"""KPI metric tile widget."""
import flet as ft


class MetricTile:
    def __init__(self, label: str, value: str, unit: str) -> None:
        self.label = label
        self.value = value
        self.unit  = unit

    def build(self) -> ft.Control:
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text(self.label, size=11, color=ft.colors.GREY_400),
                    ft.Text(self.value, size=22, weight=ft.FontWeight.BOLD),
                    ft.Text(self.unit,  size=10, color=ft.colors.GREY_600),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
                padding=14, width=135,
                alignment=ft.alignment.center,
            )
        )

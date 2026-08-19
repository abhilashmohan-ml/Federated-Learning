"""KPI metric tile widget."""
import flet as ft

from shared.utils.theme import LC


class MetricTile:
    def __init__(self, label: str, value: str, unit: str) -> None:
        self.label = label
        self.value = value
        self.unit  = unit

    def build(self) -> ft.Control:
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text(self.label, size=11, color=LC.TEXT_MUTED),
                    ft.Text(self.value, size=22, weight=ft.FontWeight.BOLD,
                            color=LC.TEXT_PRIMARY),
                    ft.Text(self.unit,  size=10, color=LC.TEXT_SECONDARY),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
                padding=14, width=135,
                alignment=ft.Alignment(0, 0),
            )
        )

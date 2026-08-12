"""Site status card widget — holds a mutable status Text so the polling loop can update it."""
import flet as ft

STATUS_COLORS = {
    "IDLE":      ft.Colors.GREY,
    "TRAINING":  ft.Colors.BLUE,
    "UPLOADING": ft.Colors.ORANGE,
    "DONE":      ft.Colors.GREEN,
    "ERROR":     ft.Colors.RED,
}


class SiteCard:
    def __init__(self, site_id: str) -> None:
        self.site_id = site_id
        self._status_text = ft.Text("IDLE", size=12, color=ft.Colors.GREY)

    def build(self) -> ft.Control:
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text(self.site_id, size=15, weight=ft.FontWeight.BOLD),
                    self._status_text,
                    ft.Text("LRV: --",    size=11, color=ft.Colors.GREY_400),
                    ft.Text("Amin: -- m2", size=11, color=ft.Colors.GREY_400),
                ], spacing=3),
                padding=14,
                width=155,
            )
        )

    def set_status(self, status: str) -> None:
        """Update the displayed status. Called by the polling loop."""
        upper = status.upper()
        self._status_text.value = upper
        self._status_text.color = STATUS_COLORS.get(upper, ft.Colors.GREY)

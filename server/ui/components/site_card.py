"""Site status card widget."""
import flet as ft

STATUS_COLORS = {
    "IDLE":      ft.Colors.GREY,
    "TRAINING":  ft.Colors.BLUE,
    "UPLOADING": ft.Colors.ORANGE,
    "DONE":      ft.Colors.GREEN,
    "ERROR":     ft.Colors.RED,
}


class SiteCard:
    def __init__(self, site_id: str, status: str = "IDLE",
                 lrv: str = "--", amin: str = "--") -> None:
        self.site_id = site_id
        self.status  = status
        self.lrv     = lrv
        self.amin    = amin

    def build(self) -> ft.Control:
        color = STATUS_COLORS.get(self.status, ft.Colors.GREY)
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text(self.site_id, size=15, weight=ft.FontWeight.BOLD),
                    ft.Text(self.status,  size=12, color=color),
                    ft.Text(f"LRV: {self.lrv}", size=11, color=ft.Colors.GREY_400),
                    ft.Text(f"Amin: {self.amin} m2", size=11, color=ft.Colors.GREY_400),
                ], spacing=3),
                padding=14,
                width=155,
            )
        )

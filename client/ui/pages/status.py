"""Status page — connection info, current round, training progress."""
import flet as ft
from client.config import get_client_settings


class StatusPage:
    def __init__(self, page: ft.Page) -> None:
        self.page     = page
        self.settings = get_client_settings()

    def build(self) -> ft.Control:
        return ft.Column([
            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Connection", size=15, weight=ft.FontWeight.BOLD),
                ft.Text(f"Server : {self.settings.server_url}", size=13),
                ft.Text(f"Site ID: {self.settings.site_id}",   size=13),
                ft.Text("Status : IDLE", size=13, color=ft.colors.GREY_400),
            ], spacing=4), padding=16)),

            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Current Round", size=15, weight=ft.FontWeight.BOLD),
                ft.Text("Round  : —", size=13),
                ft.Text("Phase  : —", size=13),
                ft.ProgressBar(value=0.0, height=10, color=ft.colors.BLUE),
            ], spacing=6), padding=16)),

            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Local Training", size=15, weight=ft.FontWeight.BOLD),
                ft.Text("Last model  : —",    size=13),
                ft.Text("Flux RMSE   : —",    size=13),
                ft.Text("Best Hermia : —",    size=13),
                ft.Text("DP noise σ  : " + str(self.settings.dp_noise_sigma), size=13),
            ], spacing=4), padding=16)),

            ft.ElevatedButton(
                "Trigger Manual Round",
                icon=ft.icons.PLAY_ARROW,
                on_click=lambda e: None,   # TODO: call fl_client.upload_update()
            ),
        ], spacing=14, scroll=ft.ScrollMode.AUTO)

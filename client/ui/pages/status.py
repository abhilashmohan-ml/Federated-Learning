"""Status page — connection info, current round, training progress."""

from __future__ import annotations

import threading
from typing import Any

import flet as ft

from client.comms.fl_client import FLClient
from client.config import get_client_settings


class StatusPage:
    def __init__(self, page: ft.Page, fl_client: FLClient) -> None:
        self.page = page
        self.settings = get_client_settings()
        self.fl_client = fl_client
        self._round_text = ft.Text("Round  : —", size=13)
        self._phase_text = ft.Text("Phase  : —", size=13)

    def _run_round(self) -> None:
        """Network call — always invoked on a background daemon thread, never on the UI thread."""
        try:
            round_info = self.fl_client.start_round()
            self._round_text.value = f"Round  : {round_info.round_id}"
            self._phase_text.value = f"Phase  : {round_info.status.value}"
        except Exception as exc:  # noqa: BLE001
            self._round_text.value = "Round  : ERROR"
            self._phase_text.value = f"Phase  : {str(exc)[:40]}"
        self.page.update()

    def _handle_round_click(self, e: Any) -> None:  # noqa: ANN401
        """Button on_click — spawns daemon thread so the UI never blocks."""
        threading.Thread(target=self._run_round, daemon=True, name="fl-manual-round").start()

    def build(self) -> ft.Control:
        return ft.Column(
            [
                ft.Card(
                    content=ft.Container(
                        ft.Column(
                            [
                                ft.Text("Connection", size=15, weight=ft.FontWeight.BOLD),
                                ft.Text(f"Server : {self.settings.server_url}", size=13),
                                ft.Text(f"Site ID: {self.settings.site_id}", size=13),
                                ft.Text("Status : IDLE", size=13, color=ft.Colors.GREY_400),
                            ],
                            spacing=4,
                        ),
                        padding=16,
                    )
                ),
                ft.Card(
                    content=ft.Container(
                        ft.Column(
                            [
                                ft.Text("Current Round", size=15, weight=ft.FontWeight.BOLD),
                                self._round_text,
                                self._phase_text,
                                ft.ProgressBar(value=0.0, height=10, color=ft.Colors.BLUE),
                            ],
                            spacing=6,
                        ),
                        padding=16,
                    )
                ),
                ft.Card(
                    content=ft.Container(
                        ft.Column(
                            [
                                ft.Text("Local Training", size=15, weight=ft.FontWeight.BOLD),
                                ft.Text("Last model  : —", size=13),
                                ft.Text("Flux RMSE   : —", size=13),
                                ft.Text("Best Hermia : —", size=13),
                                ft.Text(
                                    "DP noise σ  : " + str(self.settings.dp_noise_sigma),
                                    size=13,
                                ),
                            ],
                            spacing=4,
                        ),
                        padding=16,
                    )
                ),
                ft.Button(
                    "Trigger Manual Round",
                    icon=ft.Icons.PLAY_ARROW,
                    on_click=self._handle_round_click,
                ),
            ],
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
        )

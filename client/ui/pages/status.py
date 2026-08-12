"""Status page — connection info, current round, training progress."""
from __future__ import annotations

import threading
from typing import Any

import flet as ft

from client.comms.fl_client import FLClient
from client.config import get_client_settings
from client.engine.state import TrainingState

_PHASE_COLORS = {
    "idle":      ft.Colors.GREY_400,
    "training":  ft.Colors.BLUE,
    "uploading": ft.Colors.ORANGE,
    "done":      ft.Colors.GREEN,
    "error":     ft.Colors.RED,
}


class StatusPage:
    def __init__(self, page: ft.Page, fl_client: FLClient) -> None:
        self.page      = page
        self.settings  = get_client_settings()
        self.fl_client = fl_client

        self._status_text = ft.Text("Status : IDLE", size=13,
                                    color=ft.Colors.GREY_400)
        self._round_text  = ft.Text("Round  : -", size=13)
        self._phase_text  = ft.Text("Phase  : -", size=13)
        self._round_button = ft.Button(
            "Trigger Manual Round",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._handle_round_click,
        )

    def update_from_state(self, state: TrainingState) -> None:
        """Refresh controls from a TrainingState snapshot. Called by poll loop."""
        phase = state.phase
        self._status_text.value = f"Status : {phase.upper()}"
        self._status_text.color = _PHASE_COLORS.get(phase, ft.Colors.GREY_400)
        if state.current_round_id > 0:
            self._round_text.value = f"Round  : {state.current_round_id}"
            self._phase_text.value = f"Phase  : {phase}"

    def _run_round(self) -> None:
        try:
            round_info = self.fl_client.start_round()
            self._round_text.value = f"Round  : {round_info.round_id}"
            self._phase_text.value = f"Phase  : {round_info.status.value}"
        except Exception as exc:
            self._round_text.value = "Round  : ERROR"
            self._phase_text.value = f"Phase  : {str(exc)[:40]}"
        self._round_button.disabled = False
        self.page.update()

    def _handle_round_click(self, e: Any) -> None:
        self._round_button.disabled = True
        self.page.update()
        threading.Thread(target=self._run_round, daemon=True,
                         name="fl-manual-round").start()

    def build(self) -> ft.Control:
        return ft.Column(
            [
                ft.Card(content=ft.Container(
                    ft.Column([
                        ft.Text("Connection", size=15,
                                weight=ft.FontWeight.BOLD),
                        ft.Text(f"Server : {self.settings.server_url}",
                                size=13),
                        ft.Text(f"Site ID: {self.settings.site_id}",
                                size=13),
                        self._status_text,
                    ], spacing=4),
                    padding=16,
                )),
                ft.Card(content=ft.Container(
                    ft.Column([
                        ft.Text("Current Round", size=15,
                                weight=ft.FontWeight.BOLD),
                        self._round_text,
                        self._phase_text,
                        ft.ProgressBar(value=0.0, height=10,
                                       color=ft.Colors.BLUE),
                    ], spacing=6),
                    padding=16,
                )),
                self._round_button,
            ],
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
        )

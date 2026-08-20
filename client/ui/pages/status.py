"""Status page — connection info, current round, training progress."""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any

import flet as ft

from client.comms.fl_client import FLClient
from client.config import get_client_settings
from client.engine.state import TrainingState, get_state, update_state
from shared.utils.theme import LC

_PHASE_COLORS = {
    "idle":      LC.TEXT_MUTED,
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
                                    color=LC.TEXT_MUTED)
        self._round_text  = ft.Text("Round  : -", size=13, color=LC.TEXT_PRIMARY)
        self._phase_text  = ft.Text("Phase  : -", size=13, color=LC.TEXT_SECONDARY)
        self._spinner = ft.ProgressRing(
            width=18, height=18, stroke_width=2.5,
            color=LC.PRIMARY, visible=False,
        )
        self._round_button = ft.Button(
            "Trigger Manual Round",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._handle_round_click,
            style=ft.ButtonStyle(
                bgcolor=LC.PRIMARY,
                color=LC.SURFACE,
            ),
        )

    def update_from_state(self, state: TrainingState) -> None:
        """Refresh controls from a TrainingState snapshot. Called by poll loop."""
        phase = state.phase
        self._status_text.value = f"Status : {phase.upper()}"
        self._status_text.color = _PHASE_COLORS.get(phase, LC.TEXT_MUTED)
        if phase in ("training", "uploading"):
            run_num = state.run_count + 1
        else:
            run_num = state.run_count
        if run_num > 0 or phase not in ("idle",):
            self._round_text.value = f"Round  : {run_num if run_num > 0 else '-'}"
            self._phase_text.value = f"Phase  : {phase}"

    def _set_button_state(self, label: str, icon: str, busy: bool) -> None:
        self._round_button.text     = label
        self._round_button.icon     = icon
        self._round_button.disabled = busy
        self._spinner.visible       = busy

    def _run_round(self) -> None:
        try:
            from client.engine.data_source import DevDataSource, ProdDataSource
            from client.engine.local_trainer import LocalTrainer

            cfg = self.settings
            if cfg.dev_mode:
                physics = {
                    "J0": cfg.dev_j0, "k1": cfg.dev_k1, "k2": cfg.dev_k2,
                    "noise": cfg.dev_noise, "tmp_base": cfg.dev_tmp_base,
                }
                ds = DevDataSource(physics, jitter=cfg.dev_jitter_fraction)
            else:
                data_dir = os.path.dirname(cfg.local_data_path) or f"data/{cfg.site_id}"
                ds = ProdDataSource(data_dir)

            trainer = LocalTrainer(data_source=ds)

            self._set_button_state("Connecting…", ft.Icons.SYNC, busy=True)
            self.page.update()

            round_info = self.fl_client.get_current_round()
            this_run = get_state().run_count + 1
            self._round_text.value = f"Round  : {this_run}"
            self._set_button_state("Training…", ft.Icons.MEMORY, busy=True)
            self._phase_text.value = "Phase  : training"
            self.page.update()

            update_state(phase="training", current_round_id=round_info.round_id)
            update = trainer.train_and_prepare_update(round_info.round_id)

            self._set_button_state("Uploading…", ft.Icons.CLOUD_UPLOAD, busy=True)
            update_state(phase="uploading")
            self._phase_text.value = "Phase  : uploading"
            self.page.update()

            self.fl_client.upload_update(update)

            now = datetime.now(timezone.utc).isoformat()
            state = get_state()
            update_state(
                phase="done",
                last_round_completed=round_info.round_id,
                run_count=state.run_count + 1,
                last_run_at=now,
                last_lrv=update.local_metrics.get("lrv"),
                last_flux_ratio=update.local_metrics.get("flux_ratio"),
                last_amin=update.local_metrics.get("amin_m2"),
                last_hermia_model=update.hermia_best_model,
            )
            self._phase_text.value = "Phase  : done"
        except Exception as exc:
            update_state(phase="error")
            self._round_text.value = "Round  : ERROR"
            self._phase_text.value = f"Phase  : {str(exc)[:40]}"
        self._set_button_state("Trigger Manual Round", ft.Icons.PLAY_ARROW, busy=False)
        self.page.update()

    def _handle_round_click(self, e: Any) -> None:
        self._set_button_state("Starting…", ft.Icons.HOURGLASS_TOP, busy=True)
        self.page.update()
        threading.Thread(target=self._run_round, daemon=True,
                         name="fl-manual-round").start()

    def build(self) -> ft.Control:
        return ft.Column(
            [
                ft.Card(
                    content=ft.Container(
                        ft.Column([
                            ft.Text("Connection", size=15,
                                    weight=ft.FontWeight.BOLD,
                                    color=LC.TEXT_PRIMARY),
                            ft.Text(f"Server : {self.settings.server_url}",
                                    size=13, color=LC.TEXT_SECONDARY),
                            ft.Text(f"Site ID: {self.settings.site_id}",
                                    size=13, color=LC.TEXT_SECONDARY),
                            self._status_text,
                        ], spacing=4),
                        padding=16,
                    )),
                ft.Card(
                    content=ft.Container(
                        ft.Column([
                            ft.Text("Current Round", size=15,
                                    weight=ft.FontWeight.BOLD,
                                    color=LC.TEXT_PRIMARY),
                            self._round_text,
                            self._phase_text,
                            ft.ProgressBar(value=0.0, height=10,
                                           color=ft.Colors.BLUE),
                        ], spacing=6),
                        padding=16,
                    )),
                ft.Row(
                    [self._round_button, self._spinner],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
        )

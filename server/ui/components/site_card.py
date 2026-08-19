"""Site status card widget — shows status, run count, and last run timestamp."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import flet as ft

from shared.utils.theme import LC

STATUS_COLORS = {
    "IDLE":      ft.Colors.GREY,
    "TRAINING":  ft.Colors.BLUE,
    "UPLOADING": ft.Colors.ORANGE,
    "DONE":      ft.Colors.GREEN,
    "ERROR":     ft.Colors.RED,
}


class SiteCard:
    def __init__(self, site_id: str) -> None:
        self.site_id       = site_id
        self._status_text  = ft.Text("IDLE",    size=12, color=ft.Colors.GREY)
        self._runs_text    = ft.Text("Runs: --", size=11, color=LC.TEXT_MUTED)
        self._last_text    = ft.Text("Last: --",  size=11, color=LC.TEXT_MUTED)

    def build(self) -> ft.Control:
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text(self.site_id, size=15, weight=ft.FontWeight.BOLD,
                            color=LC.TEXT_PRIMARY),
                    self._status_text,
                    self._runs_text,
                    self._last_text,
                ], spacing=3),
                padding=14,
                width=155,
            )
        )

    def set_status(self, status: str) -> None:
        upper = status.upper()
        self._status_text.value = upper
        self._status_text.color = STATUS_COLORS.get(upper, ft.Colors.GREY)

    def set_run_info(self, run_count: int, last_run_at: Optional[str]) -> None:
        """Update run count and last-run timestamp. Called by dashboard poll loop."""
        self._runs_text.value = f"Runs: {run_count}"
        if last_run_at:
            try:
                dt = datetime.fromisoformat(last_run_at)
                now = datetime.now(timezone.utc)
                if dt.date() == now.date():
                    self._last_text.value = f"Last: {dt.strftime('%H:%M')}"
                else:
                    self._last_text.value = f"Last: {dt.strftime('%d %b')}"
            except ValueError:
                self._last_text.value = f"Last: {last_run_at[:16]}"
        else:
            self._last_text.value = "Last: --"

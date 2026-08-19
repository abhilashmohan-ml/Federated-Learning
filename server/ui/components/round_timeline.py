"""Federation round progress timeline widget — mutable controls updated by polling loop."""
from __future__ import annotations

import flet as ft

from shared.utils.theme import LC

_STATUS_COLOR = {
    "collecting":  ft.Colors.BLUE,
    "aggregating": ft.Colors.ORANGE,
    "complete":    ft.Colors.GREEN,
    "failed":      ft.Colors.RED,
    "idle":        LC.TEXT_MUTED,
}

_SITE_STATUS_COLOR = {
    "done":      ft.Colors.GREEN,
    "training":  ft.Colors.BLUE,
    "uploading": ft.Colors.ORANGE,
    "error":     ft.Colors.RED,
}


class RoundTimeline:
    def __init__(self, total_rounds: int = 50) -> None:
        self.total_rounds = total_rounds

        self._progress_text = ft.Text(
            f"Round 0 / {total_rounds}", size=14, weight=ft.FontWeight.BOLD,
            color=LC.TEXT_PRIMARY,
        )
        self._progress_bar = ft.ProgressBar(value=0.0, color=ft.Colors.BLUE, height=8)

        self._status_badge  = ft.Text("IDLE", size=13, weight=ft.FontWeight.BOLD,
                                      color=LC.TEXT_MUTED)
        self._sites_text    = ft.Text("", size=12, color=LC.TEXT_SECONDARY)
        self._started_text  = ft.Text("", size=12, color=LC.TEXT_MUTED)
        self._completed_text = ft.Text("", size=12, color=LC.TEXT_MUTED)
        self._chips_row     = ft.Row([], spacing=8, wrap=True)

    def build(self) -> ft.Control:
        return ft.Column([
            ft.Row([self._progress_text], spacing=8),
            self._progress_bar,
            ft.Divider(height=6, color=ft.Colors.TRANSPARENT),

            ft.Card(
                content=ft.Container(
                    ft.Column([
                        ft.Row([
                            ft.Text("Current Round Status:", size=13,
                                    color=LC.TEXT_SECONDARY),
                            self._status_badge,
                        ], spacing=8),
                        self._sites_text,
                        self._started_text,
                        self._completed_text,
                        ft.Divider(height=4, color=ft.Colors.TRANSPARENT),
                        self._chips_row,
                    ], spacing=4),
                    padding=14,
                )),
        ], spacing=6)

    def update(
        self,
        current_round: int,
        site_statuses: dict,
        round_status: str = "idle",
        participating: list | None = None,
        min_sites: int = 1,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        """Refresh timeline with full round details."""
        participating = participating or []

        pct = current_round / max(self.total_rounds, 1)
        self._progress_text.value = f"Round {current_round} / {self.total_rounds}"
        self._progress_bar.value  = pct

        status_label = round_status.upper()
        self._status_badge.value = status_label
        self._status_badge.color = _STATUS_COLOR.get(round_status, LC.TEXT_MUTED)

        n_submitted = len(participating)
        if round_status == "collecting":
            self._sites_text.value = (
                f"{n_submitted} of {min_sites} required sites submitted"
            )
        elif round_status == "complete":
            self._sites_text.value = f"{n_submitted} site(s) contributed this round"
        elif round_status == "aggregating":
            self._sites_text.value = f"Aggregating {n_submitted} site update(s)..."
        elif round_status == "failed":
            self._sites_text.value = "Round failed — no updates received"
        else:
            self._sites_text.value = ""

        self._started_text.value = (
            f"Started:   {_fmt_ts(started_at)}" if started_at else ""
        )
        self._completed_text.value = (
            f"Completed: {_fmt_ts(completed_at)}" if completed_at else ""
        )

        self._chips_row.controls = [
            ft.Chip(
                label=ft.Text(
                    f"{sid}  {'✓ submitted' if st == 'done' else st}",
                    size=11,
                    color=LC.TEXT_PRIMARY,
                ),
                bgcolor=_SITE_STATUS_COLOR.get(st, LC.SURFACE_ELEVATED),
            )
            for sid, st in site_statuses.items()
        ]


def _fmt_ts(iso: str) -> str:
    """Format ISO-8601 UTC string to readable HH:MM:SS UTC for display."""
    try:
        time_part = iso.split("T")[1][:8]
        return f"{time_part} UTC"
    except Exception:
        return iso

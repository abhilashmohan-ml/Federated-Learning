"""Federation round progress timeline widget — mutable controls updated by polling loop."""
import flet as ft


class RoundTimeline:
    def __init__(self, total_rounds: int = 50) -> None:
        self.total_rounds = total_rounds
        self._round_text = ft.Text(
            f"Round 0 / {total_rounds}", size=14, weight=ft.FontWeight.BOLD
        )
        self._pct_text   = ft.Text("0% complete", size=12, color=ft.Colors.GREY_400)
        self._progress   = ft.ProgressBar(value=0.0, color=ft.Colors.BLUE, height=10)
        self._chips_row  = ft.Row([], spacing=8, wrap=True)

    def build(self) -> ft.Control:
        return ft.Column([
            ft.Row([self._round_text, self._pct_text], spacing=16),
            self._progress,
            self._chips_row,
        ], spacing=8)

    def update(self, current_round: int, site_statuses: dict) -> None:
        """Refresh timeline with latest round number and per-site statuses."""
        pct = current_round / max(self.total_rounds, 1)
        self._round_text.value = f"Round {current_round} / {self.total_rounds}"
        self._pct_text.value   = f"{pct * 100:.0f}% complete"
        self._progress.value   = pct
        self._chips_row.controls = [
            ft.Chip(
                label=ft.Text(f"{sid}: {st}", size=11),
                bgcolor=ft.Colors.BLUE if st == "done" else ft.Colors.GREY_800,
            )
            for sid, st in site_statuses.items()
        ]

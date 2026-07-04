"""Federation round progress timeline widget."""
import flet as ft


class RoundTimeline:
    def __init__(self, current_round: int = 1, total_rounds: int = 50,
                 site_statuses: dict | None = None) -> None:
        self.current_round  = current_round
        self.total_rounds   = total_rounds
        self.site_statuses  = site_statuses or {f"site_{i}": "IDLE" for i in range(1, 6)}

    def build(self) -> ft.Control:
        progress = self.current_round / max(self.total_rounds, 1)
        status_chips = ft.Row([
            ft.Chip(
                label=ft.Text(f"{sid}: {st}", size=11),
                bgcolor=ft.colors.BLUE if st == "DONE" else ft.colors.GREY_800,
            )
            for sid, st in self.site_statuses.items()
        ], spacing=8, wrap=True)

        return ft.Column([
            ft.Row([
                ft.Text(f"Round {self.current_round} / {self.total_rounds}",
                        size=14, weight=ft.FontWeight.BOLD),
                ft.Text(f"{progress*100:.0f}% complete",
                        size=12, color=ft.colors.GREY_400),
            ], spacing=16),
            ft.ProgressBar(value=progress, color=ft.colors.BLUE, height=10),
            status_chips,
        ], spacing=8)

"""Global Model page — FL global metrics + convergence status."""
from __future__ import annotations

import flet as ft

from shared.utils.theme import LC


_METRIC_META: list[tuple[str, str, str]] = [
    ("flux_rmse", "Flux RMSE",  "LMH"),
    ("lrv_rmse",  "LRV RMSE",   "-"),
    ("flux_ratio","Flux Ratio",  "-"),
    ("amin_m2",   "Min Area Amin", "m²"),
]


class GlobalModelPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page

        self._version_text    = ft.Text("-", size=28, weight=ft.FontWeight.BOLD,
                                        color=LC.TEXT_PRIMARY)
        self._rounds_text     = ft.Text("-", size=28, weight=ft.FontWeight.BOLD,
                                        color=LC.TEXT_PRIMARY)
        self._sites_text      = ft.Text("-", size=28, weight=ft.FontWeight.BOLD,
                                        color=LC.TEXT_PRIMARY)

        self._metric_vals: dict[str, ft.Text] = {
            key: ft.Text("-", size=22, weight=ft.FontWeight.BOLD, color=LC.TEXT_PRIMARY)
            for key, _, _ in _METRIC_META
        }

        self._body = ft.Column(
            [self._waiting_view()],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=16,
        )
        self._showing_data = False

    def _waiting_view(self) -> ft.Control:
        return ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.HOURGLASS_EMPTY, size=48, color=LC.TEXT_MUTED),
                ft.Text(
                    "Global model not available yet",
                    size=16, color=LC.TEXT_SECONDARY,
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.Text(
                    "Complete at least one FL round to see consolidated results.",
                    size=13, color=LC.TEXT_MUTED,
                    text_align=ft.TextAlign.CENTER,
                ),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
            alignment=ft.Alignment(0, 0),
            expand=True,
            padding=40,
        )

    def _data_view(self) -> ft.Control:
        metric_cards = ft.Row(
            [
                ft.Card(
                    content=ft.Container(
                        ft.Column([
                            ft.Text(label, size=12, color=LC.TEXT_MUTED),
                            self._metric_vals[key],
                            ft.Text(unit, size=11, color=LC.TEXT_SECONDARY),
                        ], spacing=2),
                        padding=14, width=145,
                    ))
                for key, label, unit in _METRIC_META
            ],
            spacing=12,
            wrap=True,
        )

        return ft.Column([
            ft.Text("Aggregated Performance Metrics", size=17, color=LC.TEXT_PRIMARY),
            ft.Text(
                "Mean across all sites that participated in the latest completed round.",
                size=12, color=LC.TEXT_MUTED,
            ),
            metric_cards,
        ], spacing=10)

    def build(self) -> ft.Control:
        return ft.Container(
            content=ft.Column([
                ft.Text("Global Consolidated Model", size=26,
                        weight=ft.FontWeight.BOLD, color=LC.TEXT_PRIMARY),
                ft.Text(
                    "FedProx-aggregated Physics-Informed Neural Network",
                    size=13, color=LC.TEXT_SECONDARY,
                ),
                ft.Divider(color=LC.BORDER),
                ft.Row([
                    ft.Card(content=ft.Container(ft.Column([
                        ft.Text("Model Version", size=12, color=LC.TEXT_MUTED),
                        self._version_text,
                    ], spacing=2), padding=14, width=130)),
                    ft.Card(content=ft.Container(ft.Column([
                        ft.Text("Rounds Completed", size=12, color=LC.TEXT_MUTED),
                        self._rounds_text,
                    ], spacing=2), padding=14, width=160)),
                    ft.Card(content=ft.Container(ft.Column([
                        ft.Text("Sites Last Round", size=12, color=LC.TEXT_MUTED),
                        self._sites_text,
                    ], spacing=2), padding=14, width=155)),
                ], spacing=12),
                ft.Divider(color=LC.BORDER),
                self._body,
            ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=16),
            padding=24,
            expand=True,
            bgcolor=LC.BG_PRIMARY,
        )

    def update_model_data(
        self,
        model_version: int,
        rounds_completed: int,
        sites_last_round: int,
        global_metrics: dict[str, float],
    ) -> None:
        """Refresh tiles and metric cards. Called by polling loop when model version changes."""
        self._version_text.value = str(model_version) if model_version > 0 else "-"
        self._rounds_text.value  = str(rounds_completed) if rounds_completed > 0 else "-"
        self._sites_text.value   = str(sites_last_round) if sites_last_round > 0 else "-"

        if model_version > 0:
            for key, _, _ in _METRIC_META:
                val = global_metrics.get(key)
                self._metric_vals[key].value = f"{val:.4f}" if val is not None else "N/A"

            if not self._showing_data:
                self._body.controls = [self._data_view()]
                self._showing_data = True
        else:
            if self._showing_data:
                self._body.controls = [self._waiting_view()]
                self._showing_data = False

"""Local results page — J(t) chart placeholder and live summary metrics."""
from __future__ import annotations

import flet as ft

from client.engine.state import TrainingState


class LocalResultsPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self._lrv_text        = ft.Text("-", size=22, weight=ft.FontWeight.BOLD)
        self._amin_text       = ft.Text("-", size=22, weight=ft.FontWeight.BOLD)
        self._flux_ratio_text = ft.Text("-", size=22, weight=ft.FontWeight.BOLD)
        self._hermia_text     = ft.Text("-", size=16, weight=ft.FontWeight.BOLD)

    def update_from_state(self, state: TrainingState) -> None:
        """Refresh metric tiles from a TrainingState snapshot. Called by poll loop."""
        # last_lrv remains None until LRV extraction via Manabe model is implemented
        # (same scope note as GlobalModelPage parameter table — see design spec Section 3)
        self._lrv_text.value = (
            f"{state.last_lrv:.3f}" if state.last_lrv is not None else "-"
        )
        self._amin_text.value = (
            f"{state.last_amin:.4f}" if state.last_amin is not None else "-"
        )
        self._flux_ratio_text.value = (
            f"{state.last_flux_ratio:.3f}" if state.last_flux_ratio is not None else "-"
        )
        self._hermia_text.value = state.last_hermia_model or "-"

    def build(self) -> ft.Control:
        flux_chart = ft.Container(
            content=ft.Column([
                ft.Text("Flux (LMH) vs Time (min)", size=13,
                        weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Flux Decline J(t)  (data populates after first local training)",
                    size=12, color=ft.Colors.CYAN,
                ),
            ], spacing=8),
            height=260,
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.CYAN),
            border_radius=8,
            padding=16,
        )

        metrics = ft.Row([
            ft.Card(content=ft.Container(ft.Column([
                ft.Text("LRV",      size=11, color=ft.Colors.GREY_500),
                self._lrv_text,
            ], spacing=2,
               horizontal_alignment=ft.CrossAxisAlignment.CENTER),
               padding=14, width=110, alignment=ft.Alignment(0, 0))),

            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Amin (m2)", size=11, color=ft.Colors.GREY_500),
                self._amin_text,
            ], spacing=2,
               horizontal_alignment=ft.CrossAxisAlignment.CENTER),
               padding=14, width=120, alignment=ft.Alignment(0, 0))),

            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Flux Ratio", size=11, color=ft.Colors.GREY_500),
                self._flux_ratio_text,
            ], spacing=2,
               horizontal_alignment=ft.CrossAxisAlignment.CENTER),
               padding=14, width=120, alignment=ft.Alignment(0, 0))),

            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Best Model", size=11, color=ft.Colors.GREY_500),
                self._hermia_text,
            ], spacing=2,
               horizontal_alignment=ft.CrossAxisAlignment.CENTER),
               padding=14, width=140, alignment=ft.Alignment(0, 0))),
        ], spacing=10, wrap=True)

        return ft.Column([
            ft.Text("Local Flux Decline  J(t)", size=18,
                    weight=ft.FontWeight.BOLD),
            flux_chart,
            ft.Divider(),
            ft.Text("Local Metrics", size=16),
            metrics,
        ], spacing=14, scroll=ft.ScrollMode.AUTO)

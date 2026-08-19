"""Local results page — J(t) chart and live summary metrics."""
from __future__ import annotations

import io

import flet as ft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from client.engine.state import TrainingState
from shared.utils.theme import LC

_PLACEHOLDER_TEXT = "Flux Decline J(t) — data populates after first local training"


def _render_flux_png(times: list, vals: list) -> bytes:
    """Render J(t) line chart as PNG bytes using LC light theme."""
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor=LC.BG_PRIMARY)
    ax.set_facecolor(LC.SURFACE)
    ax.plot(times, vals, color=LC.PRIMARY, linewidth=2)
    ax.set_xlabel("Time (min)", color=LC.TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Flux (LMH)", color=LC.TEXT_SECONDARY, fontsize=9)
    ax.tick_params(colors=LC.TEXT_MUTED, labelsize=8)
    ax.spines[:].set_color(LC.BORDER_DARK)
    ax.grid(True, color=LC.BORDER, linestyle="--", linewidth=0.5)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


class LocalResultsPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self._lrv_text        = ft.Text("-", size=22, weight=ft.FontWeight.BOLD,
                                        color=LC.TEXT_PRIMARY)
        self._amin_text       = ft.Text("-", size=22, weight=ft.FontWeight.BOLD,
                                        color=LC.TEXT_PRIMARY)
        self._flux_ratio_text = ft.Text("-", size=22, weight=ft.FontWeight.BOLD,
                                        color=LC.TEXT_PRIMARY)
        self._hermia_text     = ft.Text("-", size=16, weight=ft.FontWeight.BOLD,
                                        color=LC.TEXT_PRIMARY)

        self._chart_placeholder = ft.Text(
            _PLACEHOLDER_TEXT, size=12, color=LC.ACCENT,
        )
        self._flux_img = ft.Image(
            src=b"",
            expand=True,
            fit=ft.BoxFit.CONTAIN,
            visible=False,
        )
        self._chart_container = ft.Container(
            content=ft.Column([
                ft.Text("Flux (LMH) vs Time (min)", size=13,
                        weight=ft.FontWeight.BOLD, color=LC.TEXT_PRIMARY),
                self._chart_placeholder,
                self._flux_img,
            ], spacing=8),
            height=280,
            expand=True,
            bgcolor=LC.ACCENT_LIGHT,
            border_radius=8,
            padding=16,
        )

    def update_from_state(self, state: TrainingState) -> None:
        """Refresh metrics and flux chart from a TrainingState snapshot."""
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

        if state.flux_times and state.flux_vals:
            self._flux_img.src = _render_flux_png(state.flux_times, state.flux_vals)
            self._flux_img.visible = True
            self._chart_placeholder.visible = False

    def build(self) -> ft.Control:
        metrics = ft.Row([
            ft.Card(
                    content=ft.Container(ft.Column([
                        ft.Text("LRV",      size=11, color=LC.TEXT_MUTED),
                        self._lrv_text,
                    ], spacing=2,
                       horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                       padding=14, width=110, alignment=ft.Alignment(0, 0))),

            ft.Card(
                    content=ft.Container(ft.Column([
                        ft.Text("Amin (m2)", size=11, color=LC.TEXT_MUTED),
                        self._amin_text,
                    ], spacing=2,
                       horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                       padding=14, width=120, alignment=ft.Alignment(0, 0))),

            ft.Card(
                    content=ft.Container(ft.Column([
                        ft.Text("Flux Ratio", size=11, color=LC.TEXT_MUTED),
                        self._flux_ratio_text,
                    ], spacing=2,
                       horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                       padding=14, width=120, alignment=ft.Alignment(0, 0))),

            ft.Card(
                    content=ft.Container(ft.Column([
                        ft.Text("Best Model", size=11, color=LC.TEXT_MUTED),
                        self._hermia_text,
                    ], spacing=2,
                       horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                       padding=14, width=140, alignment=ft.Alignment(0, 0))),
        ], spacing=10, wrap=True)

        return ft.Column([
            ft.Text("Local Flux Decline  J(t)", size=18,
                    weight=ft.FontWeight.BOLD, color=LC.TEXT_PRIMARY),
            self._chart_container,
            ft.Divider(color=LC.BORDER),
            ft.Text("Local Metrics", size=16, color=LC.TEXT_PRIMARY),
            metrics,
        ], spacing=14, scroll=ft.ScrollMode.AUTO)

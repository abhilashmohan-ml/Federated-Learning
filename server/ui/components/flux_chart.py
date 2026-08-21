"""Flux chart — J(t) line (single site) or Amin bars (multi-site comparative)."""
from __future__ import annotations

import io

import flet as ft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.utils.theme import LC

_SITES = [f"site_{i}" for i in range(1, 6)]
SITE_COLORS = LC.CHART_COLORS[:5]


def _render_amin_png(site_metrics: dict[str, dict[str, float]]) -> bytes:
    """Amin bar chart for all sites — used by multi-site (Graphs page)."""
    values = [site_metrics.get(s, {}).get("amin_m2", 0.0) for s in _SITES]
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor=LC.BG_PRIMARY)
    ax.set_facecolor(LC.SURFACE)
    bars = ax.bar(_SITES, values, color=SITE_COLORS, width=0.5)
    ax.set_ylabel("Amin (m²)", color=LC.TEXT_SECONDARY, fontsize=9)
    ax.tick_params(colors=LC.TEXT_MUTED, labelsize=8)
    ax.spines[:].set_color(LC.BORDER_DARK)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                    f"{val:.4f}", ha="center", va="bottom",
                    color=LC.TEXT_PRIMARY, fontsize=7)
    ax.grid(True, axis="y", color=LC.BORDER, linestyle="--", linewidth=0.5)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _render_jt_png(
    t: list[float],
    j: list[float],
    site_id: str,
    model_name: str,
) -> bytes:
    """J(t) fitted-curve line chart for a single site."""
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor=LC.BG_PRIMARY)
    ax.set_facecolor(LC.SURFACE)
    ax.plot(t, j, color=LC.PRIMARY, linewidth=2)
    ax.set_xlabel("Time (min)", color=LC.TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Flux J(t) (LMH)", color=LC.TEXT_SECONDARY, fontsize=9)
    ax.set_title(f"{site_id}  |  {model_name}", color=LC.TEXT_PRIMARY,
                 fontsize=9, pad=4)
    ax.tick_params(colors=LC.TEXT_MUTED, labelsize=8)
    ax.spines[:].set_color(LC.BORDER_DARK)
    ax.grid(True, color=LC.BORDER, linestyle="--", linewidth=0.5)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


class FluxChart:
    def __init__(self, multi_site: bool = False) -> None:
        self.multi_site = multi_site
        self._img = ft.Image(
            src=b"",
            expand=True,
            fit=ft.BoxFit.CONTAIN,
            visible=False,
        )
        placeholder_text = (
            "Amin (m²) per site — populates after first round"
            if multi_site
            else "Flux Decline J(t) — populates after first training round"
        )
        self._placeholder = ft.Text(placeholder_text, size=12, color=LC.TEXT_MUTED)

    def update_data(self, site_metrics: dict[str, dict[str, float]]) -> None:
        """Re-render Amin bar chart — multi-site Graphs page only."""
        if not any(m.get("amin_m2", 0) for m in site_metrics.values()):
            return
        self._img.src = _render_amin_png(site_metrics)
        self._img.visible = True
        self._placeholder.visible = False

    def update_single_site(
        self,
        t: list[float],
        j: list[float],
        site_id: str,
        model_name: str,
    ) -> None:
        """Re-render J(t) line chart for the selected site (Sites page)."""
        if not t or not j:
            return
        self._img.src = _render_jt_png(t, j, site_id, model_name)
        self._img.visible = True
        self._placeholder.visible = False

    def build(self) -> ft.Control:
        title_text = (
            "Min Filter Area  Amin (m²) — All Sites"
            if self.multi_site
            else "Flux Decline  J(t)  (LMH vs min)"
        )
        legend: ft.Control = (
            ft.Row(
                [
                    ft.Row(
                        [
                            ft.Container(
                                width=12, height=12,
                                bgcolor=SITE_COLORS[i],
                                border_radius=3,
                                content=ft.Text(""),
                            ),
                            ft.Text(_SITES[i], size=10, color=LC.TEXT_SECONDARY),
                        ],
                        spacing=4,
                    )
                    for i in range(5)
                ],
                spacing=12,
            )
            if self.multi_site
            else ft.Container(
                width=14, height=14, bgcolor=LC.PRIMARY, border_radius=3,
                content=ft.Text(""),
            )
        )

        if self.multi_site:
            return ft.Column(
                [
                    ft.Text(title_text, size=13, weight=ft.FontWeight.BOLD,
                            color=LC.TEXT_PRIMARY),
                    self._placeholder,
                    legend,
                    ft.Container(content=self._img, height=220, expand=True),
                ],
                spacing=8,
            )
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(title_text, size=13, weight=ft.FontWeight.BOLD,
                            color=LC.TEXT_PRIMARY),
                    self._placeholder,
                    legend,
                    ft.Container(content=self._img, expand=True),
                ],
                spacing=8,
            ),
            height=270,
            expand=True,
            bgcolor=LC.ACCENT_LIGHT,
            border_radius=8,
            padding=16,
        )

"""Flux / Amin chart — matplotlib PNG rendered into a Flet Image widget."""
from __future__ import annotations

import io

import flet as ft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SITES = [f"site_{i}" for i in range(1, 6)]
SITE_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]


def _render_amin_png(site_metrics: dict[str, dict[str, float]]) -> bytes:
    """Render Amin bar chart for all sites, return PNG bytes."""
    values = [site_metrics.get(s, {}).get("amin_m2", 0.0) for s in _SITES]
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(_SITES, values, color=SITE_COLORS, width=0.5)
    ax.set_ylabel("Amin (m²)", color="#cccccc", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.spines[:].set_color("#444444")
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                    f"{val:.4f}", ha="center", va="bottom", color="#ffffff", fontsize=7)
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
        self._placeholder = ft.Text(
            "Amin (m²) per site — data populates after first round" if multi_site
            else "Flux Decline J(t) — data populates after first local training",
            size=12,
            color=ft.Colors.GREY_400,
        )

    def update_data(self, site_metrics: dict[str, dict[str, float]]) -> None:
        """Re-render Amin bar chart from latest per-site metrics (server graphs page)."""
        if not any(m.get("amin_m2", 0) for m in site_metrics.values()):
            return
        self._img.src = _render_amin_png(site_metrics)
        self._img.visible = True
        self._placeholder.visible = False

    def build(self) -> ft.Control:
        title_text = (
            "Min Filter Area  Amin (m²) — All Sites" if self.multi_site
            else "Flux (LMH) vs Time (min)"
        )
        legend = ft.Row([
            ft.Row([
                ft.Container(width=12, height=12, bgcolor=SITE_COLORS[i],
                             border_radius=3, content=ft.Text("")),
                ft.Text(_SITES[i], size=10, color=ft.Colors.GREY_400),
            ], spacing=4)
            for i in range(5)
        ], spacing=12) if self.multi_site else ft.Container(
            width=14, height=14, bgcolor="#00BCD4", border_radius=3, content=ft.Text(""),
        )

        if self.multi_site:
            return ft.Column([
                ft.Text(title_text, size=13, weight=ft.FontWeight.BOLD),
                self._placeholder,
                legend,
                ft.Container(content=self._img, height=220, expand=True),
            ], spacing=8)
        else:
            return ft.Container(
                content=ft.Column([
                    ft.Text(title_text, size=13, weight=ft.FontWeight.BOLD),
                    self._placeholder,
                    legend,
                    ft.Container(content=self._img, expand=True),
                ], spacing=8),
                height=270,
                expand=True,
                bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.BLUE),
                border_radius=8,
                padding=16,
            )

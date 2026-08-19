"""Flux-ratio bar chart — matplotlib PNG rendered into a Flet Image widget."""
from __future__ import annotations

import io

import flet as ft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.utils.theme import LC

_SITES = [f"site_{i}" for i in range(1, 6)]
_COLORS = LC.CHART_COLORS[:5]


def _render_flux_ratio_png(site_metrics: dict[str, dict[str, float]]) -> bytes:
    """Render flux_ratio bar chart for all sites, return PNG bytes."""
    values = [site_metrics.get(s, {}).get("flux_ratio", 0.0) for s in _SITES]
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor=LC.BG_PRIMARY)
    ax.set_facecolor(LC.SURFACE)
    bars = ax.bar(_SITES, values, color=_COLORS, width=0.5)
    ax.set_ylabel("Flux ratio (J_f/J_0)", color=LC.TEXT_SECONDARY, fontsize=9)
    ax.set_ylim(0, max(1.1, max(values) * 1.2) if values else 1.1)
    ax.tick_params(colors=LC.TEXT_MUTED, labelsize=8)
    ax.spines[:].set_color(LC.BORDER_DARK)
    ax.grid(True, axis="y", color=LC.BORDER, linestyle="--", linewidth=0.5)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    color=LC.TEXT_PRIMARY, fontsize=7)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


class LRVChart:
    def __init__(self, multi_site: bool = False) -> None:
        self.multi_site = multi_site
        self._img = ft.Image(
            src=b"",
            expand=True,
            fit=ft.BoxFit.CONTAIN,
            visible=False,
        )
        self._placeholder = ft.Text(
            "Flux ratio (J_final/J_initial) per site — data populates after first round",
            size=11,
            color=LC.TEXT_MUTED,
        )

    def update_data(self, site_metrics: dict[str, dict[str, float]]) -> None:
        """Re-render flux_ratio bar chart from latest per-site metrics."""
        if not any(m.get("flux_ratio", 0) for m in site_metrics.values()):
            return
        self._img.src = _render_flux_ratio_png(site_metrics)
        self._img.visible = True
        self._placeholder.visible = False

    def build(self) -> ft.Column:
        return ft.Column([
            self._placeholder,
            ft.Container(content=self._img, height=220, expand=True),
            ft.Text(
                "Flux ratio = J_final / J_initial  (lower → more fouling)",
                size=10,
                color=LC.TEXT_MUTED,
            ),
        ], spacing=6)

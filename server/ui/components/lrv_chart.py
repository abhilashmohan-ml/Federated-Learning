"""LRV vs flux-ratio scatter chart — all sites overlaid."""
from __future__ import annotations

import io

import flet as ft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.utils.theme import LC

_SITES = [f"site_{i}" for i in range(1, 6)]
_COLORS = LC.CHART_COLORS[:5]


def _render_lrv_scatter_png(site_metrics: dict[str, dict[str, float]]) -> bytes:
    """Scatter: LRV (y) vs flux_ratio (x), one labelled point per site."""
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor=LC.BG_PRIMARY)
    ax.set_facecolor(LC.SURFACE)

    plotted = False
    for i, site in enumerate(_SITES):
        m = site_metrics.get(site, {})
        lrv = m.get("lrv", 0.0)
        fr  = m.get("flux_ratio", 0.0)
        if lrv > 0 and fr > 0:
            ax.scatter(fr, lrv, color=_COLORS[i], s=70, zorder=5, label=site)
            ax.annotate(
                site, (fr, lrv),
                fontsize=7, color=LC.TEXT_SECONDARY,
                xytext=(5, 4), textcoords="offset points",
            )
            plotted = True

    if plotted:
        ax.set_xlabel("Flux Ratio (J_f / J_0)", color=LC.TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel("LRV (log₁₀)", color=LC.TEXT_SECONDARY, fontsize=9)
    ax.tick_params(colors=LC.TEXT_MUTED, labelsize=8)
    ax.spines[:].set_color(LC.BORDER_DARK)
    ax.grid(True, color=LC.BORDER, linestyle="--", linewidth=0.5)
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
            "LRV vs Flux Ratio — populates after first round",
            size=11,
            color=LC.TEXT_MUTED,
        )

    def update_data(self, site_metrics: dict[str, dict[str, float]]) -> None:
        """Re-render scatter from latest per-site metrics."""
        has_data = any(
            m.get("lrv", 0) > 0 and m.get("flux_ratio", 0) > 0
            for m in site_metrics.values()
        )
        if not has_data:
            return
        self._img.src = _render_lrv_scatter_png(site_metrics)
        self._img.visible = True
        self._placeholder.visible = False

    def build(self) -> ft.Column:
        return ft.Column(
            [
                self._placeholder,
                ft.Container(content=self._img, height=220, expand=True),
                ft.Text(
                    "Lower flux ratio → more fouling.  Higher LRV → better virus removal.",
                    size=10,
                    color=LC.TEXT_MUTED,
                ),
            ],
            spacing=6,
        )

"""Hermia model comparison — RMSE bar chart + ranked DataTable for the Sites page."""
from __future__ import annotations

import io

import flet as ft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.utils.theme import LC

_MODEL_LABELS: dict[str, str] = {
    "standard":     "Standard",
    "complete":     "Complete",
    "intermediate": "Intermediate",
    "cake":         "Cake",
    "combined_1a":  "Combined\n1-A",
}


def _render_hermia_bar_png(
    model_scores: dict[str, dict[str, float]],
    best_model: str,
) -> bytes:
    """Bar chart: model name (x) vs RMSE (y). Best model bar in SUCCESS green."""
    models  = list(model_scores.keys())
    rmse    = [model_scores[m].get("rmse", 0.0) for m in models]
    colors  = [LC.SUCCESS if m == best_model else LC.ACCENT for m in models]
    xlabels = [_MODEL_LABELS.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor=LC.BG_PRIMARY)
    ax.set_facecolor(LC.SURFACE)
    bars = ax.bar(range(len(models)), rmse, color=colors, width=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("RMSE (LMH)", color=LC.TEXT_SECONDARY, fontsize=9)
    ax.tick_params(colors=LC.TEXT_MUTED, labelsize=8)
    ax.spines[:].set_color(LC.BORDER_DARK)
    ax.grid(True, axis="y", color=LC.BORDER, linestyle="--", linewidth=0.5)
    for bar, val in zip(bars, rmse):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val * 1.02,
                f"{val:.2f}",
                ha="center", va="bottom",
                color=LC.TEXT_PRIMARY, fontsize=7,
            )
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


class HermiaComparisonChart:
    """RMSE bar chart + ranked summary DataTable for the per-site view."""

    def __init__(self) -> None:
        self._best_model: str = ""
        self._model_scores: dict[str, dict[str, float]] = {}
        self._img = ft.Image(
            src=b"",
            expand=True,
            fit=ft.BoxFit.CONTAIN,
            visible=False,
        )
        self._placeholder = ft.Text(
            "Hermia model comparison — populates after first training round",
            size=12,
            color=LC.TEXT_MUTED,
        )
        self._table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Model",      size=11, color=LC.TEXT_SECONDARY)),
                ft.DataColumn(ft.Text("RMSE (LMH)", size=11, color=LC.TEXT_SECONDARY), numeric=True),
                ft.DataColumn(ft.Text("AIC",        size=11, color=LC.TEXT_SECONDARY), numeric=True),
                ft.DataColumn(ft.Text("BIC",        size=11, color=LC.TEXT_SECONDARY), numeric=True),
                ft.DataColumn(ft.Text("Winner",     size=11, color=LC.TEXT_SECONDARY)),
            ],
            rows=[],
            heading_row_color={ft.ControlState.DEFAULT: LC.BG_SECONDARY},
            border=ft.Border(
                left=ft.BorderSide(1, LC.BORDER), right=ft.BorderSide(1, LC.BORDER),
                top=ft.BorderSide(1, LC.BORDER),  bottom=ft.BorderSide(1, LC.BORDER),
            ),
            border_radius=8,
            show_bottom_border=True,
            visible=False,
        )

    def update_data(
        self,
        model_scores: dict[str, dict[str, float]],
        best_model: str,
    ) -> None:
        """Refresh bar chart and table with latest model scores."""
        if not model_scores:
            return
        self._best_model  = best_model
        self._model_scores = model_scores

        # Re-render bar chart
        self._img.src     = _render_hermia_bar_png(model_scores, best_model)
        self._img.visible = True
        self._placeholder.visible = False

        # Re-build DataTable rows, sorted by AIC ascending (lower = better)
        sorted_models = sorted(
            model_scores.items(),
            key=lambda kv: kv[1].get("aic", float("inf")),
        )
        rows: list[ft.DataRow] = []
        for name, scores in sorted_models:
            is_best = name == best_model
            label   = _MODEL_LABELS.get(name, name)
            rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.Text(
                                label,
                                size=11,
                                weight=ft.FontWeight.BOLD if is_best else ft.FontWeight.NORMAL,
                                color=LC.SUCCESS if is_best else LC.TEXT_PRIMARY,
                            )
                        ),
                        ft.DataCell(ft.Text(f"{scores.get('rmse', 0):.3f}", size=11,
                                            color=LC.TEXT_PRIMARY)),
                        ft.DataCell(ft.Text(f"{scores.get('aic', 0):.1f}",  size=11,
                                            color=LC.TEXT_PRIMARY)),
                        ft.DataCell(ft.Text(f"{scores.get('bic', 0):.1f}",  size=11,
                                            color=LC.TEXT_PRIMARY)),
                        ft.DataCell(
                            ft.Text(
                                "✓" if is_best else "",
                                size=12,
                                color=LC.SUCCESS,
                            )
                        ),
                    ],
                    color={ft.ControlState.DEFAULT: LC.ACCENT_LIGHT if is_best else LC.SURFACE},
                )
            )
        self._table.rows   = rows
        self._table.visible = True

    def build(self) -> ft.Control:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "Hermia Model Comparison — RMSE (lower is better, green = AIC winner)",
                        size=13,
                        weight=ft.FontWeight.BOLD,
                        color=LC.TEXT_PRIMARY,
                    ),
                    self._placeholder,
                    ft.Container(content=self._img, height=220, expand=True),
                    ft.Text(
                        "Sorted by AIC ascending — lower is better, winner highlighted",
                        size=10,
                        color=LC.TEXT_MUTED,
                    ),
                    self._table,
                ],
                spacing=10,
            ),
            padding=16,
            bgcolor=LC.SURFACE,
            border_radius=LC.RADIUS_MD,
            border=ft.Border(
                left=ft.BorderSide(1, LC.BORDER), right=ft.BorderSide(1, LC.BORDER),
                top=ft.BorderSide(1, LC.BORDER),  bottom=ft.BorderSide(1, LC.BORDER),
            ),
            expand=True,
        )

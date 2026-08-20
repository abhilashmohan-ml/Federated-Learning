"""Site Monitor page — per-site model metrics and live charts."""
from __future__ import annotations

import flet as ft
from server.ui.components.flux_chart import FluxChart
from server.ui.components.lrv_chart  import LRVChart
from shared.utils.theme import LC


class SiteMonitorPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self._site_metrics:     dict[str, dict[str, float]] = {}
        self._site_best_models: dict[str, str]              = {}
        self._current_round_id: int                         = 0
        self._selected_site:    str                         = "site_1"
        self._built: ft.Control | None                      = None

        # Persistent value-text refs so poll loop can mutate them in-place.
        self._val_lrv        = ft.Text("--", size=22, weight=ft.FontWeight.BOLD,
                                       color=LC.TEXT_PRIMARY)
        self._val_amin       = ft.Text("--", size=22, weight=ft.FontWeight.BOLD,
                                       color=LC.TEXT_PRIMARY)
        self._val_flux_ratio = ft.Text("--", size=22, weight=ft.FontWeight.BOLD,
                                       color=LC.TEXT_PRIMARY)
        self._val_best_model = ft.Text("--", size=22, weight=ft.FontWeight.BOLD,
                                       color=LC.TEXT_PRIMARY)
        self._val_round      = ft.Text("--", size=22, weight=ft.FontWeight.BOLD,
                                       color=LC.TEXT_PRIMARY)

        # Persistent chart instances — updated in-place by poll loop.
        self._flux_chart = FluxChart(multi_site=False)
        self._lrv_chart  = LRVChart(multi_site=False)

    # ------------------------------------------------------------------
    # Poll-loop entry point
    # ------------------------------------------------------------------

    def update_data(
        self,
        site_metrics:     dict[str, dict[str, float]],
        site_best_models: dict[str, str],
        current_round_id: int,
    ) -> None:
        """Refresh tiles and charts with latest server data. Called every 5 s."""
        self._site_metrics     = site_metrics
        self._site_best_models = site_best_models
        self._current_round_id = current_round_id
        self._refresh_tiles()
        self._flux_chart.update_data(site_metrics)
        self._lrv_chart.update_data(site_metrics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_tiles(self) -> None:
        m    = self._site_metrics.get(self._selected_site, {})
        amin = m.get("amin_m2")
        fr   = m.get("flux_ratio")
        lrv  = m.get("lrv")
        self._val_amin.value       = f"{amin:.4f}" if amin is not None else "--"
        self._val_flux_ratio.value = f"{fr:.3f}"   if fr   is not None else "--"
        self._val_lrv.value        = f"{lrv:.3f}"  if lrv  is not None else "--"
        self._val_best_model.value = self._site_best_models.get(self._selected_site, "--")
        self._val_round.value      = str(self._current_round_id) if self._current_round_id else "--"

    def _on_site_change(self, e: ft.ControlEvent) -> None:
        self._selected_site = e.control.value
        self._refresh_tiles()
        self.page.update()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> ft.Control:
        if self._built is not None:
            return self._built

        site_options = [ft.dropdown.Option(f"site_{i}") for i in range(1, 6)]
        site_dd = ft.Dropdown(
            label="Select Site",
            options=site_options,
            value=site_options[0].key if site_options else None,
            width=200,
            border_color=LC.BORDER,
            focused_border_color=LC.PRIMARY,
            bgcolor=LC.SURFACE,
            fill_color=LC.SURFACE,
            color=LC.TEXT_PRIMARY,
            label_style=ft.TextStyle(color=LC.TEXT_MUTED),
            border_radius=LC.RADIUS_MD,
            on_select=self._on_site_change,
        )

        def _tile(label: str, val_text: ft.Text, unit: str) -> ft.Control:
            return ft.Card(
                content=ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(label, size=11, color=LC.TEXT_MUTED),
                            val_text,
                            ft.Text(unit, size=10, color=LC.TEXT_SECONDARY),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=2,
                    ),
                    padding=14,
                    width=135,
                    alignment=ft.Alignment(0, 0),
                )
            )

        metrics_row = ft.Row(
            [
                _tile("LRV",        self._val_lrv,        "log10"),
                _tile("Amin",       self._val_amin,       "m2"),
                _tile("Flux Ratio", self._val_flux_ratio, ""),
                _tile("Best Model", self._val_best_model, ""),
                _tile("Round",      self._val_round,      ""),
            ],
            spacing=12,
            wrap=True,
        )

        self._built = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Site Monitor", size=26, weight=ft.FontWeight.BOLD,
                            color=LC.TEXT_PRIMARY),
                    site_dd,
                    ft.Divider(color=LC.BORDER),
                    metrics_row,
                    ft.Text("Flux Decline J(t)", size=16, color=LC.TEXT_PRIMARY),
                    self._flux_chart.build(),
                    ft.Text("LRV vs Flux", size=16, color=LC.TEXT_PRIMARY),
                    self._lrv_chart.build(),
                ],
                scroll=ft.ScrollMode.AUTO,
                expand=True,
                spacing=16,
            ),
            padding=24,
            expand=True,
            bgcolor=LC.BG_PRIMARY,
        )
        return self._built

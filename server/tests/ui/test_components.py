"""Unit tests for server/ui/components — 100% line + branch coverage."""
import pytest
import flet as ft

from server.ui.components.site_card      import SiteCard, STATUS_COLORS
from server.ui.components.flux_chart              import FluxChart, SITE_COLORS, _SITES, _render_amin_png, _render_jt_png
from server.ui.components.lrv_chart               import LRVChart, _SITES as _LRV_SITES, _COLORS, _render_lrv_scatter_png
from server.ui.components.hermia_comparison_chart import HermiaComparisonChart, _render_hermia_bar_png
from server.ui.components.metric_tile    import MetricTile
from server.ui.components.nav_rail       import build_nav_rail
from server.ui.components.round_timeline import RoundTimeline
from shared.utils.theme import LC


# ---------------------------------------------------------------------------
# site_card
# ---------------------------------------------------------------------------

class TestStatusColors:
    def test_all_keys_present(self) -> None:
        assert set(STATUS_COLORS) == {"IDLE", "TRAINING", "UPLOADING", "DONE", "ERROR"}

    def test_correct_color_values(self) -> None:
        assert STATUS_COLORS["IDLE"]      == ft.Colors.GREY
        assert STATUS_COLORS["TRAINING"]  == ft.Colors.BLUE
        assert STATUS_COLORS["UPLOADING"] == ft.Colors.ORANGE
        assert STATUS_COLORS["DONE"]      == ft.Colors.GREEN
        assert STATUS_COLORS["ERROR"]     == ft.Colors.RED


class TestSiteCard:
    def test_init_defaults(self) -> None:
        card = SiteCard("site_1")
        assert card.site_id == "site_1"
        assert card._status_text.value == "IDLE"
        assert card._runs_text.value == "Runs: --"
        assert card._last_text.value == "Last: --"

    def test_init_custom(self) -> None:
        card = SiteCard("site_2")
        card.set_status("TRAINING")
        assert card._status_text.value == "TRAINING"

    def test_build_returns_card(self) -> None:
        assert isinstance(SiteCard("site_1").build(), ft.Card)

    @pytest.mark.parametrize("status", ["IDLE", "TRAINING", "UPLOADING", "DONE", "ERROR"])
    def test_build_known_statuses(self, status: str) -> None:
        card = SiteCard("site_1")
        card.set_status(status)
        assert isinstance(card.build(), ft.Card)

    def test_build_unknown_status_falls_back_to_grey(self) -> None:
        # branch: STATUS_COLORS.get(upper, ft.Colors.GREY) for unknown key
        card = SiteCard("site_1")
        card.set_status("UNKNOWN")
        assert card._status_text.color == ft.Colors.GREY

    def test_set_run_info_shows_count(self) -> None:
        card = SiteCard("test_site")
        card.set_run_info(5, None)
        assert "5" in card._runs_text.value

    def test_set_run_info_no_timestamp_shows_dash(self) -> None:
        card = SiteCard("test_site")
        card.set_run_info(0, None)
        assert card._last_text.value == "Last: --"

    def test_set_run_info_today_shows_time(self) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        card = SiteCard("test_site")
        card.set_run_info(1, now)
        # Today's date — should show HH:MM format
        assert ":" in card._last_text.value
        assert "Last:" in card._last_text.value

    def test_set_run_info_old_date_shows_day_month(self) -> None:
        card = SiteCard("test_site")
        card.set_run_info(3, "2025-01-15T10:30:00+00:00")
        # Old date — should show "15 Jan" format
        assert "Jan" in card._last_text.value or "15" in card._last_text.value

    def test_set_run_info_invalid_timestamp_truncates(self) -> None:
        card = SiteCard("test_site")
        card.set_run_info(1, "not-a-date")
        # Falls back to last_run_at[:16]
        assert card._last_text.value == "Last: not-a-date"

    def test_build_site_id_in_text(self) -> None:
        col = SiteCard("site_3").build().content.content
        texts = [c.value for c in col.controls if isinstance(c, ft.Text)]
        assert "site_3" in texts

    def test_build_runs_and_last_in_text(self) -> None:
        col = SiteCard("site_1").build().content.content
        texts = [c.value for c in col.controls if isinstance(c, ft.Text)]
        assert any("Runs: --" in v for v in texts)
        assert any("Last: --" in v for v in texts)


# ---------------------------------------------------------------------------
# flux_chart
# ---------------------------------------------------------------------------

class TestSiteColors:
    def test_five_colors(self) -> None:
        assert len(SITE_COLORS) == 5

    def test_colors_are_strings(self) -> None:
        assert all(isinstance(c, str) for c in SITE_COLORS)


class TestFluxChart:
    def test_init_default(self) -> None:
        assert FluxChart().multi_site is False

    def test_init_multi_site(self) -> None:
        assert FluxChart(multi_site=True).multi_site is True

    def test_build_single_returns_container(self) -> None:
        assert isinstance(FluxChart(multi_site=False).build(), ft.Container)

    def test_build_multi_returns_column(self) -> None:
        assert isinstance(FluxChart(multi_site=True).build(), ft.Column)

    def test_build_single_height_270(self) -> None:
        assert FluxChart(multi_site=False).build().height == 270

    def test_build_multi_legend_is_row_with_five_items(self) -> None:
        col = FluxChart(multi_site=True).build()
        legend = col.controls[2]  # title, placeholder, legend, chart_container
        assert isinstance(legend, ft.Row)
        assert len(legend.controls) == 5

    def test_build_multi_legend_colors_match_site_colors(self) -> None:
        col = FluxChart(multi_site=True).build()
        legend = col.controls[2]
        for i, item in enumerate(legend.controls):
            assert item.controls[0].bgcolor == SITE_COLORS[i]

    def test_build_single_contains_column(self) -> None:
        container = FluxChart(multi_site=False).build()
        assert isinstance(container.content, ft.Column)

    def test_build_multi_site_colors_correct(self) -> None:
        assert len(SITE_COLORS) == 5

    def test_sites_list_correct(self) -> None:
        assert _SITES == [f"site_{i}" for i in range(1, 6)]

    def test_render_amin_png_returns_png_bytes(self) -> None:
        data = _render_amin_png({"site_1": {"amin_m2": 0.05}})
        assert isinstance(data, bytes)
        assert data[:4] == b"\x89PNG"

    def test_update_data_sets_img_visible(self) -> None:
        fc = FluxChart(multi_site=True)
        assert fc._img.visible is False
        fc.update_data({"site_1": {"amin_m2": 0.05}})
        assert fc._img.visible is True
        assert fc._img.src != b""

    def test_update_data_no_values_no_render(self) -> None:
        fc = FluxChart(multi_site=True)
        fc.update_data({"site_1": {"amin_m2": 0.0}})
        assert fc._img.visible is False  # all zeros → skip render

    def test_render_jt_png_returns_png_bytes(self) -> None:
        t = [0.0, 5.0, 10.0]
        j = [100.0, 80.0, 65.0]
        data = _render_jt_png(t, j, "site_1", "intermediate")
        assert isinstance(data, bytes)
        assert data[:4] == b"\x89PNG"

    def test_update_single_site_sets_img_visible(self) -> None:
        fc = FluxChart(multi_site=False)
        assert fc._img.visible is False
        fc.update_single_site([0.0, 5.0, 10.0], [100.0, 80.0, 65.0], "site_1", "cake")
        assert fc._img.visible is True
        assert fc._img.src != b""

    def test_update_single_site_empty_t_skips_render(self) -> None:
        fc = FluxChart(multi_site=False)
        fc.update_single_site([], [], "site_1", "cake")
        assert fc._img.visible is False

    def test_build_single_legend_is_container(self) -> None:
        container = FluxChart(multi_site=False).build()
        col = container.content
        legend = col.controls[2]
        assert isinstance(legend, ft.Container)
        assert legend.bgcolor == LC.PRIMARY


# ---------------------------------------------------------------------------
# lrv_chart
# ---------------------------------------------------------------------------

class TestLRVChartConstants:
    def test_sites_count(self) -> None:
        assert len(_LRV_SITES) == 5

    def test_sites_names(self) -> None:
        assert _LRV_SITES == [f"site_{i}" for i in range(1, 6)]

    def test_colors_count(self) -> None:
        assert len(_COLORS) == 5


class TestLRVChart:
    def test_init_default(self) -> None:
        assert LRVChart().multi_site is False

    def test_init_multi_site(self) -> None:
        assert LRVChart(multi_site=True).multi_site is True

    def test_build_returns_column(self) -> None:
        assert isinstance(LRVChart().build(), ft.Column)

    def test_build_has_placeholder_container_and_footnote(self) -> None:
        col = LRVChart().build()
        assert isinstance(col.controls[0], ft.Text)   # placeholder
        assert isinstance(col.controls[1], ft.Container)  # chart container
        assert isinstance(col.controls[2], ft.Text)   # footnote

    def test_build_footnote_color_muted(self) -> None:
        col = LRVChart().build()
        assert col.controls[2].color == LC.TEXT_MUTED

    def test_render_lrv_scatter_png_returns_png_bytes(self) -> None:
        data = _render_lrv_scatter_png({"site_1": {"lrv": 4.5, "flux_ratio": 0.5}})
        assert isinstance(data, bytes)
        assert data[:4] == b"\x89PNG"

    def test_update_data_sets_img_visible(self) -> None:
        lrv = LRVChart()
        assert lrv._img.visible is False
        lrv.update_data({"site_1": {"lrv": 4.5, "flux_ratio": 0.45}})
        assert lrv._img.visible is True
        assert lrv._img.src != b""

    def test_update_data_no_values_no_render(self) -> None:
        lrv = LRVChart()
        lrv.update_data({"site_1": {"lrv": 0.0, "flux_ratio": 0.0}})
        assert lrv._img.visible is False


# ---------------------------------------------------------------------------
# hermia_comparison_chart
# ---------------------------------------------------------------------------

_SAMPLE_SCORES: dict[str, dict[str, float]] = {
    "standard":     {"rmse": 2.5,  "aic": 55.0, "bic": 58.0},
    "complete":     {"rmse": 3.1,  "aic": 60.0, "bic": 63.0},
    "intermediate": {"rmse": 1.8,  "aic": 50.0, "bic": 53.0},
    "cake":         {"rmse": 2.0,  "aic": 52.0, "bic": 55.0},
    "combined_1a":  {"rmse": 4.0,  "aic": 70.0, "bic": 74.0},
}


class TestHermiaComparisonChart:
    def test_init_img_not_visible(self) -> None:
        hc = HermiaComparisonChart()
        assert hc._img.visible is False

    def test_init_table_not_visible(self) -> None:
        hc = HermiaComparisonChart()
        assert hc._table.visible is False

    def test_build_returns_container(self) -> None:
        hc = HermiaComparisonChart()
        assert isinstance(hc.build(), ft.Container)

    def test_render_bar_png_returns_png_bytes(self) -> None:
        data = _render_hermia_bar_png(_SAMPLE_SCORES, "intermediate")
        assert isinstance(data, bytes)
        assert data[:4] == b"\x89PNG"

    def test_update_data_sets_img_and_table_visible(self) -> None:
        hc = HermiaComparisonChart()
        hc.update_data(_SAMPLE_SCORES, "intermediate")
        assert hc._img.visible is True
        assert hc._img.src != b""
        assert hc._table.visible is True

    def test_update_data_builds_five_rows(self) -> None:
        hc = HermiaComparisonChart()
        hc.update_data(_SAMPLE_SCORES, "intermediate")
        assert len(hc._table.rows) == 5

    def test_update_data_table_sorted_by_aic(self) -> None:
        hc = HermiaComparisonChart()
        hc.update_data(_SAMPLE_SCORES, "intermediate")
        # First row must be model with lowest AIC (intermediate=50.0)
        first_cell_text = hc._table.rows[0].cells[0].content.value
        assert "Intermediate" in first_cell_text

    def test_update_data_winner_row_uses_success_color(self) -> None:
        hc = HermiaComparisonChart()
        hc.update_data(_SAMPLE_SCORES, "intermediate")
        # Find the best-model row
        best_row = next(
            r for r in hc._table.rows
            if r.cells[4].content.value == "✓"
        )
        assert best_row.cells[0].content.color == LC.SUCCESS

    def test_update_data_non_winner_rows_use_primary_color(self) -> None:
        hc = HermiaComparisonChart()
        hc.update_data(_SAMPLE_SCORES, "intermediate")
        non_best = [r for r in hc._table.rows if r.cells[4].content.value != "✓"]
        for row in non_best:
            assert row.cells[0].content.color == LC.TEXT_PRIMARY

    def test_update_data_empty_scores_is_noop(self) -> None:
        hc = HermiaComparisonChart()
        hc.update_data({}, "intermediate")
        assert hc._img.visible is False
        assert hc._table.visible is False

    def test_update_data_stores_best_model(self) -> None:
        hc = HermiaComparisonChart()
        hc.update_data(_SAMPLE_SCORES, "cake")
        assert hc._best_model == "cake"


# ---------------------------------------------------------------------------
# metric_tile
# ---------------------------------------------------------------------------

class TestMetricTile:
    def test_init(self) -> None:
        tile = MetricTile("Flux", "4.8", "LMH")
        assert tile.label == "Flux"
        assert tile.value == "4.8"
        assert tile.unit  == "LMH"

    def test_build_returns_card(self) -> None:
        assert isinstance(MetricTile("Flux", "4.8", "LMH").build(), ft.Card)

    def test_build_label_color_muted(self) -> None:
        col = MetricTile("Flux", "4.8", "LMH").build().content.content
        assert col.controls[0].color == LC.TEXT_MUTED

    def test_build_unit_color_secondary(self) -> None:
        col = MetricTile("Flux", "4.8", "LMH").build().content.content
        assert col.controls[2].color == LC.TEXT_SECONDARY

    def test_build_value_bold(self) -> None:
        col = MetricTile("Flux", "4.8", "LMH").build().content.content
        assert col.controls[1].weight == ft.FontWeight.BOLD

    def test_build_text_values(self) -> None:
        col = MetricTile("LRV", "5.1", "—").build().content.content
        assert [c.value for c in col.controls] == ["LRV", "5.1", "—"]


# ---------------------------------------------------------------------------
# nav_rail
# ---------------------------------------------------------------------------

class TestBuildNavRail:
    def test_returns_navigation_rail(self) -> None:
        assert isinstance(build_nav_rail(on_change=lambda e: None), ft.NavigationRail)

    def test_has_five_destinations(self) -> None:
        assert len(build_nav_rail(on_change=lambda e: None).destinations) == 5

    def test_destination_icons(self) -> None:
        rail = build_nav_rail(on_change=lambda e: None)
        expected = [
            ft.Icons.DASHBOARD, ft.Icons.MONITOR, ft.Icons.MODEL_TRAINING,
            ft.Icons.SHOW_CHART, ft.Icons.SETTINGS,
        ]
        for dest, icon in zip(rail.destinations, expected):
            assert dest.icon == icon

    def test_destination_labels(self) -> None:
        rail = build_nav_rail(on_change=lambda e: None)
        assert [d.label for d in rail.destinations] == [
            "Dashboard", "Sites", "Global Model", "Graphs", "Settings"
        ]

    def test_on_change_wired(self) -> None:
        called = []
        rail = build_nav_rail(on_change=lambda e: called.append(e))
        assert rail.on_change is not None


# ---------------------------------------------------------------------------
# round_timeline
# ---------------------------------------------------------------------------

class TestRoundTimeline:
    def test_init_defaults(self) -> None:
        rt = RoundTimeline()
        assert rt.total_rounds == 50
        assert rt._progress_bar.value == 0.0

    def test_init_default_chips_row_empty(self) -> None:
        rt = RoundTimeline()
        rt.build()
        assert rt._chips_row.controls == []

    def test_init_custom_progress_value(self) -> None:
        rt = RoundTimeline(total_rounds=30)
        rt.update(10, {"alpha": "done"})
        assert rt._progress_bar.value == pytest.approx(10 / 30)

    def test_init_custom_total_rounds(self) -> None:
        rt = RoundTimeline(total_rounds=100)
        assert rt.total_rounds == 100

    def test_build_returns_column(self) -> None:
        assert isinstance(RoundTimeline().build(), ft.Column)

    def test_build_progress_bar_value(self) -> None:
        rt = RoundTimeline(50)
        rt.update(25, {})
        col = rt.build()
        bar = next(c for c in col.controls if isinstance(c, ft.ProgressBar))
        assert bar.value == pytest.approx(0.5)

    def test_build_progress_bar_color_blue(self) -> None:
        col = RoundTimeline().build()
        bar = next(c for c in col.controls if isinstance(c, ft.ProgressBar))
        assert bar.color == ft.Colors.BLUE

    def test_build_chip_bgcolor_done_is_green(self) -> None:
        rt = RoundTimeline()
        rt.update(1, {"alpha": "done"})
        rt.build()
        assert rt._chips_row.controls[0].bgcolor == ft.Colors.GREEN

    def test_build_chip_bgcolor_training_is_blue(self) -> None:
        rt = RoundTimeline()
        rt.update(1, {"alpha": "training"})
        rt.build()
        assert rt._chips_row.controls[0].bgcolor == ft.Colors.BLUE

    def test_build_chip_bgcolor_idle_is_surface_elevated(self) -> None:
        rt = RoundTimeline()
        rt.update(1, {"alpha": "idle"})
        rt.build()
        assert rt._chips_row.controls[0].bgcolor == LC.SURFACE_ELEVATED

    def test_build_total_rounds_zero_no_division_error(self) -> None:
        rt = RoundTimeline(0)
        rt.update(0, {})
        col = rt.build()
        bar = next(c for c in col.controls if isinstance(c, ft.ProgressBar))
        assert bar.value == pytest.approx(0.0)

    def test_update_status_badge_collecting(self) -> None:
        rt = RoundTimeline()
        rt.update(1, {}, round_status="collecting", min_sites=3,
                  participating=["site_1"])
        assert rt._status_badge.value == "COLLECTING"

    def test_update_sites_text_collecting(self) -> None:
        rt = RoundTimeline()
        rt.update(1, {}, round_status="collecting", min_sites=3,
                  participating=["alpha", "beta"])
        assert "2 of 3" in rt._sites_text.value

    def test_update_status_badge_complete(self) -> None:
        rt = RoundTimeline()
        rt.update(1, {}, round_status="complete", participating=["site_1"])
        assert rt._status_badge.value == "COMPLETE"

    def test_update_started_at_shown(self) -> None:
        rt = RoundTimeline()
        rt.update(1, {}, started_at="2026-08-18T10:00:00+00:00")
        assert "10:00:00 UTC" in rt._started_text.value

    def test_update_completed_at_shown(self) -> None:
        rt = RoundTimeline()
        rt.update(1, {}, completed_at="2026-08-18T10:05:30+00:00")
        assert "10:05:30 UTC" in rt._completed_text.value

    def test_update_no_timestamps_empty_strings(self) -> None:
        rt = RoundTimeline()
        rt.update(1, {})
        assert rt._started_text.value == ""
        assert rt._completed_text.value == ""

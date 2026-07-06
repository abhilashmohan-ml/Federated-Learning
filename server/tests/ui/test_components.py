"""Unit tests for server/ui/components — 100% line + branch coverage."""
import pytest
import flet as ft

from server.ui.components.site_card      import SiteCard, STATUS_COLORS
from server.ui.components.flux_chart     import FluxChart, SITE_COLORS
from server.ui.components.lrv_chart      import LRVChart, _PLACEHOLDER_LRVS, _SITES
from server.ui.components.metric_tile    import MetricTile
from server.ui.components.nav_rail       import build_nav_rail
from server.ui.components.round_timeline import RoundTimeline


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
        assert card.status  == "IDLE"
        assert card.lrv     == "--"
        assert card.amin    == "--"

    def test_init_custom(self) -> None:
        card = SiteCard("site_2", status="TRAINING", lrv="4.8", amin="0.12")
        assert card.status == "TRAINING"
        assert card.lrv    == "4.8"
        assert card.amin   == "0.12"

    def test_build_returns_card(self) -> None:
        assert isinstance(SiteCard("site_1").build(), ft.Card)

    @pytest.mark.parametrize("status", ["IDLE", "TRAINING", "UPLOADING", "DONE", "ERROR"])
    def test_build_known_statuses(self, status: str) -> None:
        assert isinstance(SiteCard("site_1", status=status).build(), ft.Card)

    def test_build_unknown_status_falls_back_to_grey(self) -> None:
        # branch: STATUS_COLORS.get(self.status, ft.Colors.GREY) for unknown key
        assert isinstance(SiteCard("site_1", status="UNKNOWN").build(), ft.Card)

    def test_build_site_id_in_text(self) -> None:
        col = SiteCard("site_3", lrv="5.1", amin="0.25").build().content.content
        texts = [c.value for c in col.controls if isinstance(c, ft.Text)]
        assert "site_3" in texts

    def test_build_lrv_amin_in_text(self) -> None:
        col = SiteCard("site_1", lrv="4.9", amin="0.30").build().content.content
        texts = [c.value for c in col.controls if isinstance(c, ft.Text)]
        assert any("4.9" in v for v in texts)
        assert any("0.30" in v for v in texts)


# ---------------------------------------------------------------------------
# flux_chart
# ---------------------------------------------------------------------------

class TestSiteColors:
    def test_five_colors(self) -> None:
        assert len(SITE_COLORS) == 5

    def test_correct_color_values(self) -> None:
        assert SITE_COLORS == [
            ft.Colors.BLUE, ft.Colors.GREEN, ft.Colors.ORANGE,
            ft.Colors.PINK, ft.Colors.PURPLE,
        ]


class TestFluxChart:
    def test_init_default(self) -> None:
        assert FluxChart().multi_site is False

    def test_init_multi_site(self) -> None:
        assert FluxChart(multi_site=True).multi_site is True

    def test_build_single_returns_container(self) -> None:
        assert isinstance(FluxChart(multi_site=False).build(), ft.Container)

    def test_build_multi_returns_container(self) -> None:
        assert isinstance(FluxChart(multi_site=True).build(), ft.Container)

    def test_build_single_height_270(self) -> None:
        assert FluxChart(multi_site=False).build().height == 270

    def test_build_multi_height_270(self) -> None:
        assert FluxChart(multi_site=True).build().height == 270

    def test_build_single_subtitle_contains_single_legend(self) -> None:
        container = FluxChart(multi_site=False).build()
        col = container.content
        # legend is a single Container (not a Row)
        assert isinstance(col.controls[2], ft.Container)

    def test_build_multi_legend_is_row_with_five_items(self) -> None:
        container = FluxChart(multi_site=True).build()
        col = container.content
        legend = col.controls[2]
        assert isinstance(legend, ft.Row)
        assert len(legend.controls) == 5

    def test_build_multi_legend_colors_match_site_colors(self) -> None:
        container = FluxChart(multi_site=True).build()
        col = container.content
        legend = col.controls[2]
        for i, item in enumerate(legend.controls):
            assert item.bgcolor == SITE_COLORS[i]

    def test_build_single_legend_color_cyan(self) -> None:
        container = FluxChart(multi_site=False).build()
        col = container.content
        legend = col.controls[2]
        assert legend.bgcolor == ft.Colors.CYAN

    def test_build_single_subtitle_color_grey_400(self) -> None:
        container = FluxChart(multi_site=False).build()
        subtitle = container.content.controls[1]
        assert subtitle.color == ft.Colors.GREY_400

    def test_build_multi_subtitle_color_grey_400(self) -> None:
        container = FluxChart(multi_site=True).build()
        subtitle = container.content.controls[1]
        assert subtitle.color == ft.Colors.GREY_400


# ---------------------------------------------------------------------------
# lrv_chart
# ---------------------------------------------------------------------------

class TestLRVChartConstants:
    def test_placeholder_lrvs_count(self) -> None:
        assert len(_PLACEHOLDER_LRVS) == 5

    def test_sites_count(self) -> None:
        assert len(_SITES) == 5

    def test_sites_names(self) -> None:
        assert _SITES == [f"site_{i}" for i in range(1, 6)]


class TestLRVChart:
    def test_init_default(self) -> None:
        assert LRVChart().multi_site is False

    def test_init_multi_site(self) -> None:
        assert LRVChart(multi_site=True).multi_site is True

    def test_build_returns_column(self) -> None:
        assert isinstance(LRVChart().build(), ft.Column)

    def test_build_has_bars_row_and_footnote(self) -> None:
        col = LRVChart().build()
        assert isinstance(col.controls[0], ft.Row)
        assert isinstance(col.controls[1], ft.Text)

    def test_build_bars_row_has_five_columns(self) -> None:
        col = LRVChart().build()
        bars = col.controls[0]
        assert len(bars.controls) == 5

    def test_build_bar_colors_teal(self) -> None:
        col = LRVChart().build()
        bars = col.controls[0]
        for site_col in bars.controls:
            bar = site_col.controls[0]
            assert bar.bgcolor == ft.Colors.TEAL

    def test_build_site_labels_in_text(self) -> None:
        col = LRVChart().build()
        bars = col.controls[0]
        labels = [site_col.controls[1].value for site_col in bars.controls]
        assert labels == _SITES

    def test_build_footnote_color_grey_500(self) -> None:
        col = LRVChart().build()
        assert col.controls[1].color == ft.Colors.GREY_500


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

    def test_build_label_color_grey_400(self) -> None:
        col = MetricTile("Flux", "4.8", "LMH").build().content.content
        assert col.controls[0].color == ft.Colors.GREY_400

    def test_build_unit_color_grey_600(self) -> None:
        col = MetricTile("Flux", "4.8", "LMH").build().content.content
        assert col.controls[2].color == ft.Colors.GREY_600

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
        assert rt.current_round == 1
        assert rt.total_rounds  == 50
        assert len(rt.site_statuses) == 5

    def test_init_default_site_statuses_all_idle(self) -> None:
        assert all(v == "IDLE" for v in RoundTimeline().site_statuses.values())

    def test_init_custom(self) -> None:
        statuses = {"site_1": "DONE", "site_2": "TRAINING"}
        rt = RoundTimeline(current_round=10, total_rounds=50, site_statuses=statuses)
        assert rt.current_round == 10
        assert rt.site_statuses == statuses

    def test_init_none_site_statuses_generates_defaults(self) -> None:
        rt = RoundTimeline(site_statuses=None)
        assert set(rt.site_statuses.keys()) == {f"site_{i}" for i in range(1, 6)}

    def test_build_returns_column(self) -> None:
        assert isinstance(RoundTimeline().build(), ft.Column)

    def test_build_progress_bar_value(self) -> None:
        col = RoundTimeline(current_round=25, total_rounds=50).build()
        bar = next(c for c in col.controls if isinstance(c, ft.ProgressBar))
        assert bar.value == pytest.approx(0.5)

    def test_build_progress_bar_color_blue(self) -> None:
        col = RoundTimeline().build()
        bar = next(c for c in col.controls if isinstance(c, ft.ProgressBar))
        assert bar.color == ft.Colors.BLUE

    def test_build_chip_bgcolor_done_is_blue(self) -> None:
        rt = RoundTimeline(site_statuses={"site_1": "DONE"})
        col = rt.build()
        # status_chips Row is the last control (after header Row and ProgressBar)
        chip_row = col.controls[2]
        assert isinstance(chip_row, ft.Row)
        assert chip_row.controls[0].bgcolor == ft.Colors.BLUE

    def test_build_chip_bgcolor_non_done_is_grey_800(self) -> None:
        rt = RoundTimeline(site_statuses={"site_1": "TRAINING"})
        col = rt.build()
        chip_row = col.controls[2]
        assert chip_row.controls[0].bgcolor == ft.Colors.GREY_800

    def test_build_total_rounds_zero_no_division_error(self) -> None:
        # max(total_rounds, 1) guards against ZeroDivisionError
        col = RoundTimeline(current_round=0, total_rounds=0).build()
        bar = next(c for c in col.controls if isinstance(c, ft.ProgressBar))
        assert bar.value == pytest.approx(0.0)

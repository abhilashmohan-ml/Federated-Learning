"""Unit tests for server/ui/pages — 100% line + branch coverage."""
from unittest.mock import MagicMock
import flet as ft

from server.ui.pages.dashboard     import DashboardPage
from server.ui.pages.site_monitor  import SiteMonitorPage
from server.ui.pages.global_model  import GlobalModelPage, _PARAM_ROWS
from server.ui.pages.graphs        import GraphsPage
from server.ui.pages.settings      import SettingsPage


def _mock_page() -> MagicMock:
    return MagicMock(spec=ft.Page)


# ---------------------------------------------------------------------------
# global_model
# ---------------------------------------------------------------------------

class TestParamRows:
    def test_count(self) -> None:
        assert len(_PARAM_ROWS) == 10

    def test_each_has_three_fields(self) -> None:
        assert all(len(r) == 3 for r in _PARAM_ROWS)

    def test_known_params_present(self) -> None:
        names = {r[0] for r in _PARAM_ROWS}
        for p in ("J0", "k1", "k2", "ks", "ki", "kc", "kcf", "Pc", "J_crit", "Dv"):
            assert p in names


class TestGlobalModelPage:
    def test_init_stores_page(self) -> None:
        page = _mock_page()
        assert GlobalModelPage(page).page is page

    def test_build_returns_column(self) -> None:
        assert isinstance(GlobalModelPage(_mock_page()).build(), ft.Column)

    def test_build_contains_data_table(self) -> None:
        col = GlobalModelPage(_mock_page()).build()
        assert any(isinstance(c, ft.DataTable) for c in col.controls)

    def test_build_data_table_row_count(self) -> None:
        col = GlobalModelPage(_mock_page()).build()
        table = next(c for c in col.controls if isinstance(c, ft.DataTable))
        assert len(table.rows) == len(_PARAM_ROWS)

    def test_build_data_table_column_count(self) -> None:
        col = GlobalModelPage(_mock_page()).build()
        table = next(c for c in col.controls if isinstance(c, ft.DataTable))
        assert len(table.columns) == 5

    def test_build_subtitle_color_grey_400(self) -> None:
        col = GlobalModelPage(_mock_page()).build()
        subtitle = next(
            c for c in col.controls
            if isinstance(c, ft.Text) and c.color == ft.Colors.GREY_400
        )
        assert "Physics-Informed" in subtitle.value

    def test_build_stat_cards_label_color_grey_500(self) -> None:
        col = GlobalModelPage(_mock_page()).build()
        stat_row = next(c for c in col.controls if isinstance(c, ft.Row))
        for card in stat_row.controls:
            label = card.content.content.controls[0]
            assert label.color == ft.Colors.GREY_500


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------

class TestGraphsPage:
    def test_init_stores_page(self) -> None:
        page = _mock_page()
        assert GraphsPage(page).page is page

    def test_build_returns_column(self) -> None:
        assert isinstance(GraphsPage(_mock_page()).build(), ft.Column)

    def test_build_contains_flux_and_lrv_charts(self) -> None:
        col = GraphsPage(_mock_page()).build()
        types = {type(c) for c in col.controls}
        # FluxChart.build() → ft.Container; LRVChart.build() → ft.Column
        assert ft.Container in types
        assert ft.Column in types

    def test_build_placeholder_texts_grey_500(self) -> None:
        col = GraphsPage(_mock_page()).build()
        grey_texts = [
            c for c in col.controls
            if isinstance(c, ft.Text) and c.color == ft.Colors.GREY_500
        ]
        assert len(grey_texts) == 2


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

class TestSettingsPage:
    def test_init_stores_page(self) -> None:
        page = _mock_page()
        assert SettingsPage(page).page is page

    def test_build_returns_column(self) -> None:
        assert isinstance(SettingsPage(_mock_page()).build(), ft.Column)

    def test_build_has_button(self) -> None:
        col = SettingsPage(_mock_page()).build()
        assert any(isinstance(c, ft.Button) for c in col.controls)

    def test_build_button_icon_save(self) -> None:
        col = SettingsPage(_mock_page()).build()
        btn = next(c for c in col.controls if isinstance(c, ft.Button))
        assert btn.icon == ft.Icons.SAVE

    def test_build_has_data_table(self) -> None:
        col = SettingsPage(_mock_page()).build()
        assert any(isinstance(c, ft.DataTable) for c in col.controls)

    def test_build_data_table_five_rows(self) -> None:
        col = SettingsPage(_mock_page()).build()
        table = next(c for c in col.controls if isinstance(c, ft.DataTable))
        assert len(table.rows) == 5

    def test_build_delete_buttons_icon_and_color(self) -> None:
        col = SettingsPage(_mock_page()).build()
        table = next(c for c in col.controls if isinstance(c, ft.DataTable))
        for row in table.rows:
            btn = row.cells[3].content
            assert isinstance(btn, ft.IconButton)
            assert btn.icon == ft.Icons.DELETE_OUTLINE
            assert btn.icon_color == ft.Colors.RED_300

    def test_build_hyperparameter_row_count(self) -> None:
        col = SettingsPage(_mock_page()).build()
        param_row = next(c for c in col.controls if isinstance(c, ft.Row))
        assert len(param_row.controls) == 5


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

class TestDashboardPage:
    def test_init_stores_page(self) -> None:
        page = _mock_page()
        assert DashboardPage(page).page is page

    def test_build_returns_container(self) -> None:
        ctrl = DashboardPage(_mock_page()).build()
        assert isinstance(ctrl, ft.Container)

    def test_build_container_padding_24(self) -> None:
        ctrl = DashboardPage(_mock_page()).build()
        assert ctrl.padding == 24

    def test_build_inner_column_scrollable(self) -> None:
        ctrl = DashboardPage(_mock_page()).build()
        inner = ctrl.content
        assert isinstance(inner, ft.Column)
        assert inner.scroll == ft.ScrollMode.AUTO

    def test_build_inner_column_expand(self) -> None:
        ctrl = DashboardPage(_mock_page()).build()
        assert ctrl.content.expand is True

    def test_build_contains_heading_text(self) -> None:
        ctrl = DashboardPage(_mock_page()).build()
        col = ctrl.content
        heading = next(
            c for c in col.controls
            if isinstance(c, ft.Text) and "Dashboard" in c.value
        )
        assert heading.size == 26

    def test_build_contains_site_cards_row(self) -> None:
        ctrl = DashboardPage(_mock_page()).build()
        col = ctrl.content
        site_row = next(c for c in col.controls if isinstance(c, ft.Row))
        assert len(site_row.controls) == 5

    def test_build_contains_round_timeline(self) -> None:
        ctrl = DashboardPage(_mock_page()).build()
        col = ctrl.content
        # RoundTimeline.build() returns ft.Column
        timeline_cols = [c for c in col.controls if isinstance(c, ft.Column)]
        assert len(timeline_cols) >= 1


# ---------------------------------------------------------------------------
# site_monitor
# ---------------------------------------------------------------------------

class TestSiteMonitorPage:
    def test_init_stores_page(self) -> None:
        page = _mock_page()
        assert SiteMonitorPage(page).page is page

    def test_build_returns_container(self) -> None:
        ctrl = SiteMonitorPage(_mock_page()).build()
        assert isinstance(ctrl, ft.Container)

    def test_build_container_padding_24(self) -> None:
        ctrl = SiteMonitorPage(_mock_page()).build()
        assert ctrl.padding == 24

    def test_build_inner_column_scrollable(self) -> None:
        ctrl = SiteMonitorPage(_mock_page()).build()
        inner = ctrl.content
        assert isinstance(inner, ft.Column)
        assert inner.scroll == ft.ScrollMode.AUTO

    def test_build_contains_site_dropdown(self) -> None:
        ctrl = SiteMonitorPage(_mock_page()).build()
        col = ctrl.content
        dd = next(c for c in col.controls if isinstance(c, ft.Dropdown))
        assert dd.value == "site_1"
        assert len(dd.options) == 5

    def test_build_contains_metrics_row(self) -> None:
        ctrl = SiteMonitorPage(_mock_page()).build()
        col = ctrl.content
        metrics_row = next(c for c in col.controls if isinstance(c, ft.Row))
        assert len(metrics_row.controls) == 5

    def test_build_contains_flux_chart(self) -> None:
        ctrl = SiteMonitorPage(_mock_page()).build()
        col = ctrl.content
        # FluxChart.build() → ft.Container
        chart_containers = [
            c for c in col.controls
            if isinstance(c, ft.Container) and c.height == 270
        ]
        assert len(chart_containers) == 1

    def test_build_heading_text(self) -> None:
        ctrl = SiteMonitorPage(_mock_page()).build()
        col = ctrl.content
        heading = next(
            c for c in col.controls
            if isinstance(c, ft.Text) and "Monitor" in c.value
        )
        assert heading.size == 26

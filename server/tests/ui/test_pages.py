"""Unit tests for server/ui/pages — 100% line + branch coverage."""
from unittest.mock import MagicMock
import flet as ft

from server.ui.pages.dashboard     import DashboardPage
from server.ui.pages.site_monitor  import SiteMonitorPage
from server.ui.pages.global_model  import GlobalModelPage, _METRIC_META
from server.ui.pages.graphs        import GraphsPage
from server.ui.pages.settings      import SettingsPage
from shared.utils.theme import LC


def _mock_page() -> MagicMock:
    return MagicMock(spec=ft.Page)


# ---------------------------------------------------------------------------
# global_model
# ---------------------------------------------------------------------------

class TestMetricMeta:
    def test_count(self) -> None:
        assert len(_METRIC_META) == 4

    def test_each_has_three_fields(self) -> None:
        assert all(len(r) == 3 for r in _METRIC_META)

    def test_known_keys_present(self) -> None:
        keys = {r[0] for r in _METRIC_META}
        for k in ("flux_rmse", "lrv_rmse", "flux_ratio", "amin_m2"):
            assert k in keys


class TestGlobalModelPage:
    def test_init_stores_page(self) -> None:
        page = _mock_page()
        assert GlobalModelPage(page).page is page

    def test_build_returns_container(self) -> None:
        assert isinstance(GlobalModelPage(_mock_page()).build(), ft.Container)

    def test_build_inner_column_scrollable(self) -> None:
        ctrl = GlobalModelPage(_mock_page()).build()
        assert isinstance(ctrl.content, ft.Column)
        assert ctrl.content.scroll == ft.ScrollMode.AUTO

    def test_build_subtitle_color_secondary(self) -> None:
        ctrl = GlobalModelPage(_mock_page()).build()
        col = ctrl.content
        subtitle = next(
            c for c in col.controls
            if isinstance(c, ft.Text) and c.color == LC.TEXT_SECONDARY
        )
        assert "Physics-Informed" in subtitle.value

    def test_build_stat_cards_label_color_muted(self) -> None:
        ctrl = GlobalModelPage(_mock_page()).build()
        col = ctrl.content
        stat_row = next(c for c in col.controls if isinstance(c, ft.Row))
        for card in stat_row.controls:
            label = card.content.content.controls[0]
            assert label.color == LC.TEXT_MUTED

    def test_build_no_data_table(self) -> None:
        ctrl = GlobalModelPage(_mock_page()).build()

        def _walk(ctrl: object) -> bool:
            if isinstance(ctrl, ft.DataTable):
                return True
            for attr in ("controls", "content"):
                child = getattr(ctrl, attr, None)
                if child is None:
                    continue
                if isinstance(child, list):
                    if any(_walk(c) for c in child):
                        return True
                else:
                    if _walk(child):
                        return True
            return False

        assert not _walk(ctrl)

    def test_build_starts_with_waiting_view(self) -> None:
        gm = GlobalModelPage(_mock_page())
        gm.build()
        assert not gm._showing_data

    def test_update_model_data_populates_version_tile(self) -> None:
        gm = GlobalModelPage(_mock_page())
        gm.build()
        gm.update_model_data(3, 3, 2, {"flux_rmse": 0.05, "lrv_rmse": 0.1,
                                        "flux_ratio": 0.8, "amin_m2": 0.002})
        assert gm._version_text.value == "3"

    def test_update_model_data_switches_to_data_view(self) -> None:
        gm = GlobalModelPage(_mock_page())
        gm.build()
        gm.update_model_data(1, 1, 1, {"flux_rmse": 0.05, "lrv_rmse": 0.1,
                                        "flux_ratio": 0.8, "amin_m2": 0.002})
        assert gm._showing_data

    def test_update_model_data_zero_version_keeps_waiting(self) -> None:
        gm = GlobalModelPage(_mock_page())
        gm.build()
        gm.update_model_data(0, 0, 0, {})
        assert not gm._showing_data

    def test_update_model_data_tiles_show_dash_when_zero(self) -> None:
        gm = GlobalModelPage(_mock_page())
        gm.build()
        gm.update_model_data(0, 0, 0, {})
        assert gm._version_text.value == "-"
        assert gm._rounds_text.value == "-"
        assert gm._sites_text.value == "-"

    def test_update_model_data_metric_vals_populated(self) -> None:
        gm = GlobalModelPage(_mock_page())
        gm.build()
        gm.update_model_data(1, 1, 1, {"flux_rmse": 0.123, "lrv_rmse": 0.456,
                                        "flux_ratio": 0.789, "amin_m2": 0.001})
        assert gm._metric_vals["flux_rmse"].value == "0.1230"

    def test_update_model_data_resets_to_waiting_view(self) -> None:
        gm = GlobalModelPage(_mock_page())
        gm.build()
        gm.update_model_data(1, 1, 1, {"flux_rmse": 0.1, "lrv_rmse": 0.1,
                                        "flux_ratio": 0.8, "amin_m2": 0.002})
        assert gm._showing_data
        gm.update_model_data(0, 0, 0, {})
        assert not gm._showing_data


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------

class TestGraphsPage:
    def test_init_stores_page(self) -> None:
        page = _mock_page()
        assert GraphsPage(page).page is page

    def test_build_returns_container(self) -> None:
        assert isinstance(GraphsPage(_mock_page()).build(), ft.Container)

    def test_build_contains_flux_and_lrv_charts(self) -> None:
        container = GraphsPage(_mock_page()).build()
        col = container.content
        column_controls = [c for c in col.controls if isinstance(c, ft.Column)]
        # FluxChart(multi_site=True) → ft.Column; LRVChart → ft.Column
        assert len(column_controls) == 2

    def test_build_placeholder_texts_muted(self) -> None:
        container = GraphsPage(_mock_page()).build()
        col = container.content
        muted_texts = [
            c for c in col.controls
            if isinstance(c, ft.Text) and c.color == LC.TEXT_MUTED
        ]
        # Hermia placeholder text is a direct TEXT_MUTED child
        assert len(muted_texts) >= 1


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

class TestSettingsPage:
    def test_init_stores_page(self) -> None:
        page = _mock_page()
        assert SettingsPage(page).page is page

    def test_init_refs_are_none(self) -> None:
        page = _mock_page()
        sp = SettingsPage(page)
        assert sp._mode_radio is None
        assert sp._quorum_field is None
        assert sp._window_field is None
        assert sp._heartbeat_field is None
        assert sp._policy_status is None

    def test_build_returns_container(self) -> None:
        assert isinstance(SettingsPage(_mock_page()).build(), ft.Container)

    def test_build_has_button(self) -> None:
        col = SettingsPage(_mock_page()).build().content
        assert any(isinstance(c, ft.Button) for c in col.controls)

    def test_build_buttons_have_save_icon(self) -> None:
        col = SettingsPage(_mock_page()).build().content
        buttons = [c for c in col.controls if isinstance(c, ft.Button)]
        assert all(b.icon == ft.Icons.SAVE for b in buttons)

    def test_build_has_mode_radio_after_build(self) -> None:
        page = _mock_page()
        sp = SettingsPage(page)
        sp.build()
        assert sp._mode_radio is not None

    def test_build_initializes_all_refs(self) -> None:
        page = _mock_page()
        sp = SettingsPage(page)
        sp.build()
        assert sp._mode_radio is not None
        assert sp._quorum_field is not None
        assert sp._window_field is not None
        assert sp._heartbeat_field is not None
        assert sp._policy_status is not None

    def test_on_mode_change_shows_window_field(self) -> None:
        page = _mock_page()
        page.update = MagicMock()
        sp = SettingsPage(page)
        sp.build()
        event = MagicMock()
        event.control.value = "time_window"
        sp._on_mode_change(event)
        assert sp._window_field.visible is True
        assert sp._quorum_field.visible is False

    def test_on_mode_change_shows_quorum_field(self) -> None:
        page = _mock_page()
        page.update = MagicMock()
        sp = SettingsPage(page)
        sp.build()
        event = MagicMock()
        event.control.value = "quorum"
        sp._on_mode_change(event)
        assert sp._quorum_field.visible is True
        assert sp._window_field.visible is False

    def test_build_hyperparameter_row_count(self) -> None:
        col = SettingsPage(_mock_page()).build().content
        param_rows = [c for c in col.controls if isinstance(c, ft.Row)]
        # One for hyperparameters, one for quorum/window fields
        assert len(param_rows) >= 1
        # First row should have 5 hyperparameter fields
        first_row = param_rows[0]
        assert len(first_row.controls) == 5

    def test_build_no_hardcoded_site_rows(self) -> None:
        """Ensure there are no hardcoded site_1..site_5 rows."""
        col = SettingsPage(_mock_page()).build().content
        # There should be no DataTable in the new settings page
        assert not any(isinstance(c, ft.DataTable) for c in col.controls)


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
        """Cards row starts empty — populated dynamically as sites connect."""
        ctrl = DashboardPage(_mock_page()).build()
        col = ctrl.content
        site_row = next(c for c in col.controls if isinstance(c, ft.Row))
        assert len(site_row.controls) == 0

    def test_build_contains_round_timeline(self) -> None:
        ctrl = DashboardPage(_mock_page()).build()
        col = ctrl.content
        # RoundTimeline.build() returns ft.Column
        timeline_cols = [c for c in col.controls if isinstance(c, ft.Column)]
        assert len(timeline_cols) >= 1

    def test_update_sites_creates_cards_for_new_sites(self) -> None:
        dash = DashboardPage(_mock_page())
        added = dash.update_sites({"alpha": "idle", "beta": "training"})
        assert added is True
        assert set(dash.cards.keys()) == {"alpha", "beta"}

    def test_update_sites_adds_controls_to_row(self) -> None:
        dash = DashboardPage(_mock_page())
        dash.update_sites({"alpha": "idle", "beta": "training"})
        assert len(dash._cards_row.controls) == 2

    def test_update_sites_returns_false_for_existing_sites(self) -> None:
        dash = DashboardPage(_mock_page())
        dash.update_sites({"alpha": "idle"})
        added = dash.update_sites({"alpha": "done"})
        assert added is False

    def test_update_sites_updates_status_of_existing_card(self) -> None:
        dash = DashboardPage(_mock_page())
        dash.update_sites({"alpha": "idle"})
        dash.update_sites({"alpha": "training"})
        assert dash.cards["alpha"]._status_text.value == "TRAINING"

    def test_update_sites_does_not_duplicate_cards(self) -> None:
        dash = DashboardPage(_mock_page())
        dash.update_sites({"alpha": "idle"})
        dash.update_sites({"alpha": "done"})
        assert len(dash.cards) == 1
        assert len(dash._cards_row.controls) == 1

    def test_update_sites_accepts_arbitrary_site_names(self) -> None:
        dash = DashboardPage(_mock_page())
        dash.update_sites({"basel_plant": "idle", "singapore": "done"})
        assert "basel_plant" in dash.cards
        assert "singapore" in dash.cards


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
        assert dd.value == dd.options[0].key   # first option selected, not hardcoded
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

    def test_build_caches_result(self) -> None:
        sm = SiteMonitorPage(_mock_page())
        assert sm.build() is sm.build()

    def test_update_data_populates_amin_tile(self) -> None:
        sm = SiteMonitorPage(_mock_page())
        sm.update_data(
            {"site_1": {"amin_m2": 0.0025, "flux_ratio": 0.75}},
            {},
            3,
        )
        assert sm._val_amin.value == "0.0025"

    def test_update_data_populates_flux_ratio_tile(self) -> None:
        sm = SiteMonitorPage(_mock_page())
        sm.update_data(
            {"site_1": {"amin_m2": 0.0025, "flux_ratio": 0.75}},
            {},
            3,
        )
        assert sm._val_flux_ratio.value == "0.750"

    def test_update_data_populates_round_tile(self) -> None:
        sm = SiteMonitorPage(_mock_page())
        sm.update_data({}, {}, 7)
        assert sm._val_round.value == "7"

    def test_update_data_populates_best_model_tile(self) -> None:
        sm = SiteMonitorPage(_mock_page())
        sm.update_data({}, {"site_1": "combined_1a"}, 1)
        assert sm._val_best_model.value == "combined_1a"

    def test_update_data_no_metrics_shows_dashes(self) -> None:
        sm = SiteMonitorPage(_mock_page())
        sm.update_data({}, {}, 0)
        assert sm._val_amin.value == "--"
        assert sm._val_flux_ratio.value == "--"
        assert sm._val_round.value == "--"

    def test_on_site_change_updates_selected_site_and_calls_page_update(self) -> None:
        page = _mock_page()
        sm   = SiteMonitorPage(page)
        sm.update_data(
            {"site_3": {"amin_m2": 0.003, "flux_ratio": 0.6}},
            {"site_3": "cake"},
            2,
        )
        e = MagicMock()
        e.control.value = "site_3"
        page.update.reset_mock()
        sm._on_site_change(e)
        assert sm._selected_site == "site_3"
        assert sm._val_amin.value == "0.0030"
        page.update.assert_called_once()

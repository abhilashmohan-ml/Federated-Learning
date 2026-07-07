"""Unit tests for client/ui/app — covers main() and TabBar/TabBarView construction."""
from unittest.mock import MagicMock, patch
import flet as ft

from client.ui.app import main


def _mock_page() -> MagicMock:
    return MagicMock(spec=ft.Page)


def _mock_settings(site_id: str = "site_1") -> MagicMock:
    s = MagicMock()
    s.site_id = site_id
    return s


def _run_main(**setting_kwargs):
    """Run main() with mocked page and settings; return (page, tab_bar, tab_view)."""
    page = _mock_page()
    settings = _mock_settings(**setting_kwargs)
    captured_tabs: list[ft.Tabs] = []

    def capture_add(*controls: ft.Control) -> None:
        for ctrl in controls:
            if isinstance(ctrl, ft.Tabs):
                captured_tabs.append(ctrl)

    page.add.side_effect = capture_add

    with patch("client.ui.app.get_client_settings", return_value=settings), \
         patch("client.ui.app.StatusPage") as MockStatus, \
         patch("client.ui.app.LocalResultsPage") as MockResults:
        MockStatus.return_value.build.return_value = ft.Column()
        MockResults.return_value.build.return_value = ft.Column()
        main(page)

    tab_bar = tab_view = None
    tabs_ctrl = captured_tabs[0] if captured_tabs else None
    if tabs_ctrl is not None:
        inner_col: ft.Column = tabs_ctrl.content
        tab_bar  = next((c for c in inner_col.controls if isinstance(c, ft.TabBar)),  None)
        tab_view = next((c for c in inner_col.controls if isinstance(c, ft.TabBarView)), None)

    return page, tabs_ctrl, tab_bar, tab_view


class TestMain:
    def test_main_runs_without_error(self) -> None:
        _run_main()

    def test_page_title_contains_site_id(self) -> None:
        page, *_ = _run_main(site_id="site_3")
        assert "site_3" in page.title

    def test_page_theme_dark(self) -> None:
        page, *_ = _run_main()
        assert page.theme_mode == ft.ThemeMode.DARK

    def test_page_add_called(self) -> None:
        page, *_ = _run_main()
        page.add.assert_called_once()

    def test_tabs_controller_created(self) -> None:
        _, tabs_ctrl, _, _ = _run_main()
        assert isinstance(tabs_ctrl, ft.Tabs)

    def test_tabs_length_two(self) -> None:
        _, tabs_ctrl, _, _ = _run_main()
        assert tabs_ctrl.length == 2

    def test_tabs_selected_index_zero(self) -> None:
        _, tabs_ctrl, _, _ = _run_main()
        assert tabs_ctrl.selected_index == 0

    # ------------------------------------------------------------------
    # TabBar regression: must use label= (Flet 0.85 API, not text=)
    # ------------------------------------------------------------------

    def test_tab_bar_exists(self) -> None:
        _, _, tab_bar, _ = _run_main()
        assert isinstance(tab_bar, ft.TabBar)

    def test_tab_bar_has_two_tabs(self) -> None:
        _, _, tab_bar, _ = _run_main()
        assert len(tab_bar.tabs) == 2

    def test_tab_labels(self) -> None:
        """Regression: Tab must use label=, not text=."""
        _, _, tab_bar, _ = _run_main()
        labels = [t.label for t in tab_bar.tabs]
        assert labels == ["Status", "Local Results"]

    # ------------------------------------------------------------------
    # TabBarView regression: content in controls list, not Tab.content
    # ------------------------------------------------------------------

    def test_tab_bar_view_exists(self) -> None:
        _, _, _, tab_view = _run_main()
        assert isinstance(tab_view, ft.TabBarView)

    def test_tab_bar_view_has_two_controls(self) -> None:
        _, _, _, tab_view = _run_main()
        assert len(tab_view.controls) == 2

    def test_tab_bar_view_controls_are_status_and_results(self) -> None:
        page = _mock_page()
        settings = _mock_settings()
        status_col  = ft.Column([ft.Text("status")])
        results_col = ft.Column([ft.Text("results")])
        captured_tabs: list[ft.Tabs] = []

        def capture_add(*controls: ft.Control) -> None:
            for ctrl in controls:
                if isinstance(ctrl, ft.Tabs):
                    captured_tabs.append(ctrl)

        page.add.side_effect = capture_add

        with patch("client.ui.app.get_client_settings", return_value=settings), \
             patch("client.ui.app.StatusPage") as MockStatus, \
             patch("client.ui.app.LocalResultsPage") as MockResults:
            MockStatus.return_value.build.return_value = status_col
            MockResults.return_value.build.return_value = results_col
            main(page)

        inner_col = captured_tabs[0].content
        tab_view = next(c for c in inner_col.controls if isinstance(c, ft.TabBarView))
        assert tab_view.controls[0] is status_col
        assert tab_view.controls[1] is results_col

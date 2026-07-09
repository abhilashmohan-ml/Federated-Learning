"""Unit tests for client/ui/app — covers main() and TabBar/TabBarView construction."""

from unittest.mock import MagicMock, patch
import flet as ft

from client.ui.app import main

# ---------------------------------------------------------------------------
# Flet layout constraint checker
#
# Flet's layout engine is Dart/C++ — it never runs in Python unit tests.
# However, the *structural conditions* that cause runtime layout errors are
# purely Python-level: wrong attributes on control objects.  This helper
# walks the control tree and flags known anti-patterns so tests fail before
# the browser is opened.
#
# Known anti-patterns covered:
#   1. "height is unbounded"  — TabBarView / ListView / GridView inside a
#      Column without expand=True or a fixed height
#   2. "width is unbounded"   — (same family) inside a Row without
#      expand=True or a fixed width
# ---------------------------------------------------------------------------

# Controls that require bounded height when placed inside ft.Column
_NEEDS_BOUNDED_HEIGHT: tuple[type, ...] = (ft.TabBarView, ft.ListView, ft.GridView)
# Controls that require bounded width when placed inside ft.Row
_NEEDS_BOUNDED_WIDTH: tuple[type, ...] = (ft.TabBarView, ft.ListView, ft.GridView)


def _iter_children(ctrl: ft.Control):
    """Yield all direct Flet children of *ctrl*, regardless of which attribute holds them."""
    if hasattr(ctrl, "controls") and ctrl.controls:
        yield from (c for c in ctrl.controls if isinstance(c, ft.Control))
    if hasattr(ctrl, "content") and isinstance(ctrl.content, ft.Control):
        yield ctrl.content
    if hasattr(ctrl, "tabs") and ctrl.tabs:
        yield from (t for t in ctrl.tabs if isinstance(t, ft.Control))


def find_layout_violations(
    ctrl: ft.Control,
    *,
    inside_col: bool = False,
    inside_row: bool = False,
) -> list[str]:
    """Recursively walk *ctrl* and return human-readable layout violation strings.

    Parameters
    ----------
    ctrl:
        Root control to inspect.
    inside_col / inside_row:
        Whether an ancestor Column / Row is already in the path (unbounded
        height / width flows down through all nested containers).
    """
    violations: list[str] = []

    if inside_col and isinstance(ctrl, _NEEDS_BOUNDED_HEIGHT):
        if not (ctrl.expand or ctrl.height):
            violations.append(
                f"{type(ctrl).__name__} inside Column: needs expand=True or height=<n> "
                "— will throw 'height is unbounded' at runtime"
            )

    if inside_row and isinstance(ctrl, _NEEDS_BOUNDED_WIDTH):
        if not (ctrl.expand or ctrl.width):
            violations.append(
                f"{type(ctrl).__name__} inside Row: needs expand=True or width=<n> "
                "— will throw 'width is unbounded' at runtime"
            )

    child_col = inside_col or isinstance(ctrl, ft.Column)
    child_row = inside_row or isinstance(ctrl, ft.Row)

    for child in _iter_children(ctrl):
        violations.extend(find_layout_violations(child, inside_col=child_col, inside_row=child_row))
    return violations


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

    with (
        patch("client.ui.app.get_client_settings", return_value=settings),
        patch("client.ui.app.FLClient"),
        patch("client.ui.app.StatusPage") as MockStatus,
        patch("client.ui.app.LocalResultsPage") as MockResults,
    ):
        MockStatus.return_value.build.return_value = ft.Column()
        MockResults.return_value.build.return_value = ft.Column()
        main(page)

    tab_bar = tab_view = None
    tabs_ctrl = captured_tabs[0] if captured_tabs else None
    if tabs_ctrl is not None:
        inner_col: ft.Column = tabs_ctrl.content
        tab_bar = next((c for c in inner_col.controls if isinstance(c, ft.TabBar)), None)
        tab_view = next((c for c in inner_col.controls if isinstance(c, ft.TabBarView)), None)

    return page, tabs_ctrl, tab_bar, tab_view


class TestLayoutConstraints:
    """Walk the full control tree produced by main() and assert no layout violations.

    This class is the catch-all guard: any future change that introduces a
    scroll-container without a bounded dimension will fail here immediately,
    without needing to open a browser.
    """

    def _collect_controls(self) -> list[ft.Control]:
        """Return all top-level controls passed to page.add() by main()."""
        page = _mock_page()
        controls_added: list[ft.Control] = []
        page.add.side_effect = lambda *args: controls_added.extend(args)
        with (
            patch("client.ui.app.get_client_settings", return_value=_mock_settings()),
            patch("client.ui.app.FLClient"),
            patch("client.ui.app.StatusPage") as MockStatus,
            patch("client.ui.app.LocalResultsPage") as MockResults,
        ):
            MockStatus.return_value.build.return_value = ft.Column()
            MockResults.return_value.build.return_value = ft.Column()
            main(page)
        return controls_added

    def test_no_unbounded_height_violations(self) -> None:
        """No scroll/fill control inside a Column is missing expand=True or height."""
        violations = [
            v
            for ctrl in self._collect_controls()
            for v in find_layout_violations(ctrl)
            if "height" in v
        ]
        assert not violations, "\n".join(violations)

    def test_no_unbounded_width_violations(self) -> None:
        """No scroll/fill control inside a Row is missing expand=True or width."""
        violations = [
            v
            for ctrl in self._collect_controls()
            for v in find_layout_violations(ctrl)
            if "width" in v
        ]
        assert not violations, "\n".join(violations)

    def test_layout_checker_catches_missing_expand(self) -> None:
        """Verify the checker itself flags a TabBarView-in-Column without expand."""
        col = ft.Column([ft.TabBarView(controls=[])])  # intentionally broken
        violations = find_layout_violations(col)
        assert any("height is unbounded" in v for v in violations)

    def test_layout_checker_passes_with_expand(self) -> None:
        """Checker must NOT flag a TabBarView that correctly has expand=True."""
        col = ft.Column([ft.TabBarView(controls=[], expand=True)])
        violations = find_layout_violations(col)
        assert not violations

    def test_layout_checker_passes_with_fixed_height(self) -> None:
        """Checker must NOT flag a TabBarView that has an explicit height."""
        col = ft.Column([ft.TabBarView(controls=[], height=400)])
        violations = find_layout_violations(col)
        assert not violations

    def test_layout_checker_catches_missing_width(self) -> None:
        """Checker flags TabBarView inside Row without expand or width."""
        row = ft.Row([ft.TabBarView(controls=[])])
        violations = find_layout_violations(row)
        assert any("width is unbounded" in v for v in violations)


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

    def test_tab_bar_view_expand_true(self) -> None:
        """Regression: TabBarView must have expand=True; unbounded height crashes Flet."""
        _, _, _, tab_view = _run_main()
        assert tab_view.expand is True

    def test_tab_bar_view_controls_are_status_and_results(self) -> None:
        page = _mock_page()
        settings = _mock_settings()
        status_col = ft.Column([ft.Text("status")])
        results_col = ft.Column([ft.Text("results")])
        captured_tabs: list[ft.Tabs] = []

        def capture_add(*controls: ft.Control) -> None:
            for ctrl in controls:
                if isinstance(ctrl, ft.Tabs):
                    captured_tabs.append(ctrl)

        page.add.side_effect = capture_add

        with (
            patch("client.ui.app.get_client_settings", return_value=settings),
            patch("client.ui.app.FLClient"),
            patch("client.ui.app.StatusPage") as MockStatus,
            patch("client.ui.app.LocalResultsPage") as MockResults,
        ):
            MockStatus.return_value.build.return_value = status_col
            MockResults.return_value.build.return_value = results_col
            main(page)

        inner_col = captured_tabs[0].content
        tab_view = next(c for c in inner_col.controls if isinstance(c, ft.TabBarView))
        assert tab_view.controls[0] is status_col
        assert tab_view.controls[1] is results_col


def _run_main_full(**setting_kwargs):
    page = _mock_page()
    settings = _mock_settings(**setting_kwargs)
    mock_fl = MagicMock()
    page.add.side_effect = lambda *args: None

    with (
        patch("client.ui.app.get_client_settings", return_value=settings),
        patch("client.ui.app.FLClient", return_value=mock_fl) as MockFLC,
        patch("client.ui.app.StatusPage") as MockStatus,
        patch("client.ui.app.LocalResultsPage") as MockResults,
    ):
        MockStatus.return_value.build.return_value = ft.Column()
        MockResults.return_value.build.return_value = ft.Column()
        main(page)

    return page, mock_fl, MockFLC, MockStatus


class TestFLClientIntegration:
    def test_fl_client_instantiated_once(self) -> None:
        _, _, MockFLC, _ = _run_main_full()
        MockFLC.assert_called_once_with()

    def test_authenticate_called_on_fl_client(self) -> None:
        _, mock_fl, _, _ = _run_main_full()
        mock_fl.authenticate.assert_called_once_with()

    def test_status_page_receives_fl_client(self) -> None:
        _, mock_fl, _, MockStatus = _run_main_full()
        assert MockStatus.call_args.kwargs.get("fl_client") is mock_fl

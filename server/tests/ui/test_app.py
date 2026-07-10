"""Unit tests for server/ui/app.py — 100% coverage."""
import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch

import flet as ft

from server.ui.app import main

# ---------------------------------------------------------------------------
# Flet layout constraint checker (same logic as client/tests/ui/test_app.py)
# ---------------------------------------------------------------------------

_NEEDS_BOUNDED_HEIGHT: tuple[type, ...] = (ft.TabBarView, ft.ListView, ft.GridView)
_NEEDS_BOUNDED_WIDTH: tuple[type, ...]  = (ft.TabBarView, ft.ListView, ft.GridView)


def _iter_children(ctrl: ft.Control):
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
    violations: list[str] = []
    if inside_col and isinstance(ctrl, _NEEDS_BOUNDED_HEIGHT):
        if not (ctrl.expand or ctrl.height):
            violations.append(
                f"{type(ctrl).__name__} inside Column: needs expand=True or height=<n>"
            )
    if inside_row and isinstance(ctrl, _NEEDS_BOUNDED_WIDTH):
        if not (ctrl.expand or ctrl.width):
            violations.append(
                f"{type(ctrl).__name__} inside Row: needs expand=True or width=<n>"
            )
    child_col = inside_col or isinstance(ctrl, ft.Column)
    child_row = inside_row or isinstance(ctrl, ft.Row)
    for child in _iter_children(ctrl):
        violations.extend(find_layout_violations(child, inside_col=child_col, inside_row=child_row))
    return violations


class TestServerLayoutConstraints:
    """Walk the real control tree produced by server main() and assert no violations.

    Uses real page/component instances (no mocking) so the actual Flet tree
    is populated — a MagicMock build() return would hide any controls inside.
    """

    def _collect_controls(self) -> list[ft.Control]:
        page = MagicMock(spec=ft.Page)
        added: list[ft.Control] = []
        page.add.side_effect = lambda *args: added.extend(args)
        main(page)
        return added

    def test_no_unbounded_height_violations(self) -> None:
        violations = [
            v
            for ctrl in self._collect_controls()
            for v in find_layout_violations(ctrl)
            if "height" in v
        ]
        assert not violations, "\n".join(violations)

    def test_no_unbounded_width_violations(self) -> None:
        violations = [
            v
            for ctrl in self._collect_controls()
            for v in find_layout_violations(ctrl)
            if "width" in v
        ]
        assert not violations, "\n".join(violations)

    def test_checker_catches_height_violation(self) -> None:
        """Helper self-test: Column with TabBarView and no expand is flagged."""
        col = ft.Column([ft.TabBarView(controls=[])])
        assert any("height" in v for v in find_layout_violations(col))

    def test_checker_catches_width_violation(self) -> None:
        """Helper self-test: Row with TabBarView and no expand is flagged."""
        row = ft.Row([ft.TabBarView(controls=[])])
        assert any("width" in v for v in find_layout_violations(row))

    def test_checker_iterates_tabs_attribute(self) -> None:
        """Helper self-test: _iter_children traverses the tabs= list."""
        bar = ft.TabBar(tabs=[ft.Tab(label="A"), ft.Tab(label="B")])
        violations = find_layout_violations(bar)
        assert isinstance(violations, list)  # no crash — tabs branch exercised

# Absolute path used by the __main__ block test.
_APP_PY = str(Path(__file__).resolve().parents[3] / "server" / "ui" / "app.py")

# Convenience: target strings for all five page classes.
_DASH    = "server.ui.app.DashboardPage"
_SITE    = "server.ui.app.SiteMonitorPage"
_GLOBAL  = "server.ui.app.GlobalModelPage"
_GRAPHS  = "server.ui.app.GraphsPage"
_SETTINGS = "server.ui.app.SettingsPage"
_NAV     = "server.ui.app.build_nav_rail"


def _page() -> MagicMock:
    """Return a fresh MagicMock that satisfies ft.Page's interface."""
    return MagicMock(spec=ft.Page)


class TestMainPageProperties:
    """page.title / page.theme_mode / page.padding / page.add are all set."""

    def test_title_set(self) -> None:
        page = _page()
        with patch(_DASH), patch(_SITE), patch(_GLOBAL), \
             patch(_GRAPHS), patch(_SETTINGS), patch(_NAV):
            main(page)
        assert page.title == "Viral FL — Server Dashboard"

    def test_theme_mode_dark(self) -> None:
        page = _page()
        with patch(_DASH), patch(_SITE), patch(_GLOBAL), \
             patch(_GRAPHS), patch(_SETTINGS), patch(_NAV):
            main(page)
        assert page.theme_mode == ft.ThemeMode.DARK

    def test_padding_zero(self) -> None:
        page = _page()
        with patch(_DASH), patch(_SITE), patch(_GLOBAL), \
             patch(_GRAPHS), patch(_SETTINGS), patch(_NAV):
            main(page)
        assert page.padding == 0

    def test_page_add_called_exactly_once(self) -> None:
        page = _page()
        with patch(_DASH), patch(_SITE), patch(_GLOBAL), \
             patch(_GRAPHS), patch(_SETTINGS), patch(_NAV):
            main(page)
        page.add.assert_called_once()


class TestMainRowLayout:
    """The ft.Row passed to page.add contains the nav-rail result as its first control."""

    def test_row_first_control_is_nav_rail(self) -> None:
        page = _page()
        added: list = []
        page.add.side_effect = lambda *args: added.extend(args)

        mock_rail = MagicMock()

        with patch(_DASH), patch(_SITE), patch(_GLOBAL), \
             patch(_GRAPHS), patch(_SETTINGS), \
             patch(_NAV, return_value=mock_rail):
            main(page)

        row = added[0]
        assert isinstance(row, ft.Row), "page.add must receive a ft.Row"
        assert row.controls[0] is mock_rail, (
            "first control in the Row must be the build_nav_rail() result"
        )


class TestOnNav:
    """The on_nav closure switches body.content and calls page.update()."""

    def test_on_nav_index_2_updates_body_to_global_model_and_calls_update(self) -> None:
        page = _page()
        added: list = []
        page.add.side_effect = lambda *args: added.extend(args)

        # Sentinel: the value we expect body.content to be set to.
        expected_content = MagicMock()

        mock_global_instance = MagicMock()
        mock_global_instance.build.return_value = expected_content
        mock_global_cls = MagicMock(return_value=mock_global_instance)

        # build_nav_rail receives on_nav as its only positional argument.
        # Capture it here so we can call it directly in the test.
        mock_build_nav_rail = MagicMock()

        with patch(_DASH), patch(_SITE), \
             patch(_GLOBAL, mock_global_cls), \
             patch(_GRAPHS), patch(_SETTINGS), \
             patch(_NAV, mock_build_nav_rail):
            main(page)

        # Recover on_nav from the first call to build_nav_rail.
        on_nav = mock_build_nav_rail.call_args[0][0]
        assert callable(on_nav), "build_nav_rail must be called with the on_nav callable"

        # body is the third element of the Row's controls list:
        # [nav_rail, ft.VerticalDivider, body]
        row = added[0]
        body = row.controls[2]

        # Simulate a navigation event pointing at page index 2 (GlobalModelPage).
        e = MagicMock()
        e.control.selected_index = 2

        page.update.reset_mock()
        on_nav(e)

        assert body.content is expected_content, (
            "body.content must be updated to GlobalModelPage().build() result"
        )
        page.update.assert_called_once()


class TestMainBlock:
    """The __main__ guard calls ft.run with the configured port and WEB_BROWSER view."""

    def test_ft_run_called_with_flet_port_and_web_browser_view(self) -> None:
        mock_settings = MagicMock()
        mock_settings.flet_port = 8550

        with patch("flet.run") as mock_run, \
             patch("server.config.get_settings", return_value=mock_settings):
            runpy.run_path(_APP_PY, run_name="__main__")

        mock_run.assert_called_once()
        _, kw = mock_run.call_args
        assert kw["port"] == 8550
        assert kw["view"] == ft.AppView.WEB_BROWSER

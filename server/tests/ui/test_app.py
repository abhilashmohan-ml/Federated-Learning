"""Unit tests for server/ui/app.py — 100% coverage."""
import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch

import flet as ft

from server.ui.app import main

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

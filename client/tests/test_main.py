"""Unit tests for client/main.py — 100% coverage."""
import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import flet as ft
import pytest

# Absolute path to the entry-point module so runpy can execute it directly.
_MAIN_PATH = str(Path(__file__).resolve().parent.parent / "main.py")


# ── 1. Module-level code ──────────────────────────────────────────────────────


class TestModuleLevelCode:
    """Verify side effects that fire when client.main is first imported."""

    def test_configure_logging_and_get_settings_called_on_import(self) -> None:
        """configure_logging() and get_client_settings() are called at import time.

        Strategy: evict client.main from sys.modules so the module body
        re-executes under our patches, then restore the original entry.
        """
        # Ensure a pre-existing entry so the restore branch in `finally` is always reached.
        import client.main as _  # noqa: PLC0415, F401
        original = sys.modules.pop("client.main", None)
        try:
            mock_s = MagicMock()
            mock_s.flet_client_port = 8551

            with patch(
                "shared.utils.logging_config.configure_logging"
            ) as mock_log, patch(
                "client.config.get_client_settings", return_value=mock_s
            ) as mock_cfg:
                import client.main  # noqa: PLC0415  (intentional deferred import)

                mock_log.assert_called_once()
                mock_cfg.assert_called_once()
                # settings attribute must be the object returned by get_client_settings
                assert client.main.settings is mock_s
        finally:
            # Discard the patched copy and restore the original module.
            # Also sync the `client` package attribute so later tests resolve correctly.
            sys.modules.pop("client.main", None)
            if original is not None:
                sys.modules["client.main"] = original
                import client as _client  # noqa: PLC0415
                _client.main = original


# ── 2. _background() ─────────────────────────────────────────────────────────


class TestBackground:
    """_background() must invoke both service starters exactly once."""

    def test_calls_start_heartbeat_and_start_scheduler(self) -> None:
        """_background() calls start_heartbeat() then start_scheduler(), each once."""
        import client.main  # noqa: PLC0415

        with patch("client.main.start_heartbeat") as mock_hb, patch(
            "client.main.start_scheduler"
        ) as mock_sched:
            client.main._background()

        mock_hb.assert_called_once_with()
        mock_sched.assert_called_once_with()


# ── 3. __main__ block ─────────────────────────────────────────────────────────


class TestMainBlock:
    """if __name__ == '__main__': block creates a daemon thread and launches Flet."""

    def test_thread_created_with_daemon_flag_and_flet_run_called(self) -> None:
        """__main__ block: daemon Thread started; ft.run called with correct args."""
        mock_settings = MagicMock()
        mock_settings.flet_client_port = 8551

        with patch("shared.utils.logging_config.configure_logging"), patch(
            "client.config.get_client_settings", return_value=mock_settings
        ), patch("threading.Thread") as mock_thread, patch(
            "flet.run"
        ) as mock_ft_run:

            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            # Execute client/main.py as __main__ — only the guarded block runs.
            runpy.run_path(_MAIN_PATH, run_name="__main__")

        # threading.Thread must be called with daemon=True
        thread_kwargs = mock_thread.call_args.kwargs
        assert thread_kwargs.get("daemon") is True, (
            "Thread must be created with daemon=True so it dies when main exits"
        )

        # .start() must be invoked on the thread instance
        mock_thread_instance.start.assert_called_once()

        # ft.run must be called exactly once with the right port and view
        mock_ft_run.assert_called_once()
        ft_kwargs = mock_ft_run.call_args.kwargs
        assert ft_kwargs.get("port") == 8551, (
            f"Expected port=8551, got port={ft_kwargs.get('port')}"
        )
        assert ft_kwargs.get("view") == ft.AppView.WEB_BROWSER, (
            "Expected view=ft.AppView.WEB_BROWSER"
        )

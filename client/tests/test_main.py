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
    """_background() must invoke all service starters exactly once."""

    def _mock_cfg(self, dev_mode: bool = True) -> MagicMock:
        cfg = MagicMock()
        cfg.dev_mode = dev_mode
        cfg.dev_j0 = 150.0
        cfg.dev_k1 = 0.015
        cfg.dev_k2 = 0.002
        cfg.dev_noise = 2.0
        cfg.dev_tmp_base = 1.0
        cfg.dev_jitter_fraction = 0.05
        cfg.client_status_port = 9001
        cfg.local_data_path = "data/site_1/filtration.csv"
        cfg.site_id = "site_1"
        return cfg

    def test_dev_mode_creates_dev_data_source_and_starts_services(self) -> None:
        """dev_mode=True: DevDataSource constructed; heartbeat, status server, scheduler started."""
        import client.main  # noqa: PLC0415

        mock_cfg = self._mock_cfg(dev_mode=True)
        mock_ds = MagicMock()
        mock_fl = MagicMock()

        with (
            patch("client.main.start_heartbeat") as mock_hb,
            patch("client.main.start_scheduler") as mock_sched,
            patch("client.main.start_status_server") as mock_status,
            patch("client.main.get_client_settings", return_value=mock_cfg),
            patch("client.main.DevDataSource", return_value=mock_ds) as mock_dev_cls,
        ):
            client.main._background(mock_fl)

        mock_hb.assert_called_once_with()
        mock_dev_cls.assert_called_once_with(
            {
                "J0": 150.0, "k1": 0.015, "k2": 0.002,
                "noise": 2.0, "tmp_base": 1.0,
            },
            jitter=0.05,
        )
        mock_status.assert_called_once_with(9001)
        mock_sched.assert_called_once_with(data_source=mock_ds, fl_client=mock_fl)

    def test_prod_mode_creates_prod_data_source_and_starts_services(self) -> None:
        """dev_mode=False: ProdDataSource constructed with data_dir; services started."""
        import client.main  # noqa: PLC0415

        mock_cfg = self._mock_cfg(dev_mode=False)
        mock_cfg.local_data_path = "data/site_1/filtration.csv"
        mock_ds = MagicMock()
        mock_fl = MagicMock()

        with (
            patch("client.main.start_heartbeat"),
            patch("client.main.start_scheduler"),
            patch("client.main.start_status_server"),
            patch("client.main.get_client_settings", return_value=mock_cfg),
            patch("client.main.ProdDataSource", return_value=mock_ds) as mock_prod_cls,
        ):
            client.main._background(mock_fl)

        # data_dir should be "data/site_1" (dirname of the local_data_path)
        mock_prod_cls.assert_called_once_with("data/site_1")

    def test_prod_mode_falls_back_to_site_dir_when_path_has_no_dir(self) -> None:
        """When dirname(local_data_path) is empty, fall back to data/<site_id>."""
        import client.main  # noqa: PLC0415

        mock_cfg = self._mock_cfg(dev_mode=False)
        mock_cfg.local_data_path = "filtration.csv"   # no directory component
        mock_cfg.site_id = "singapore"
        mock_ds = MagicMock()
        mock_fl = MagicMock()

        with (
            patch("client.main.start_heartbeat"),
            patch("client.main.start_scheduler"),
            patch("client.main.start_status_server"),
            patch("client.main.get_client_settings", return_value=mock_cfg),
            patch("client.main.ProdDataSource", return_value=mock_ds) as mock_prod_cls,
        ):
            client.main._background(mock_fl)

        mock_prod_cls.assert_called_once_with("data/singapore")


# ── 3. __main__ block ─────────────────────────────────────────────────────────


class TestMainBlock:
    """if __name__ == '__main__': block creates a daemon thread and launches Flet."""

    def test_thread_created_with_daemon_flag_and_flet_run_called(self) -> None:
        """__main__ block: FLClient created+authenticated; daemon Thread started; ft.run called."""
        mock_settings = MagicMock()
        mock_settings.flet_client_port = 8551
        mock_fl_instance = MagicMock()

        with patch("shared.utils.logging_config.configure_logging"), patch(
            "client.config.get_client_settings", return_value=mock_settings
        ), patch("threading.Thread") as mock_thread, patch(
            "flet.run"
        ) as mock_ft_run, patch(
            "client.comms.fl_client.FLClient", return_value=mock_fl_instance
        ) as mock_fl_cls:

            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            # Execute client/main.py as __main__ — only the guarded block runs.
            runpy.run_path(_MAIN_PATH, run_name="__main__")

        # FLClient must be instantiated exactly once and authenticated
        mock_fl_cls.assert_called_once_with()
        mock_fl_instance.authenticate.assert_called_once_with()

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

    def test_authenticate_retries_on_transient_failure(self) -> None:
        """Startup retry loop: if authenticate() raises, sleep and retry until it succeeds."""
        mock_settings = MagicMock()
        mock_settings.flet_client_port = 8551
        mock_fl_instance = MagicMock()
        mock_fl_instance.authenticate.side_effect = [Exception("server not ready"), None]

        with patch("shared.utils.logging_config.configure_logging"), patch(
            "client.config.get_client_settings", return_value=mock_settings
        ), patch("threading.Thread"), patch("flet.run"), patch(
            "time.sleep"
        ) as mock_sleep, patch(
            "client.comms.fl_client.FLClient", return_value=mock_fl_instance
        ):
            runpy.run_path(_MAIN_PATH, run_name="__main__")

        assert mock_fl_instance.authenticate.call_count == 2
        mock_sleep.assert_called_once_with(5.0)

    def test_flet_run_target_passes_fl_to_flet_main(self) -> None:
        """The lambda passed to ft.run must forward the shared fl instance to flet_main."""
        mock_settings = MagicMock()
        mock_settings.flet_client_port = 8551
        mock_fl_instance = MagicMock()

        with patch("shared.utils.logging_config.configure_logging"), patch(
            "client.config.get_client_settings", return_value=mock_settings
        ), patch("threading.Thread"), patch(
            "flet.run"
        ) as mock_ft_run, patch(
            "client.comms.fl_client.FLClient", return_value=mock_fl_instance
        ), patch(
            "client.ui.app.main"
        ) as mock_flet_main:
            runpy.run_path(_MAIN_PATH, run_name="__main__")

        # The first positional arg must be a callable (the closure lambda)
        run_target = mock_ft_run.call_args.args[0]
        assert callable(run_target), "ft.run first arg must be callable"

        # Calling the lambda must forward (page, fl) to flet_main
        mock_page = MagicMock()
        run_target(mock_page)
        mock_flet_main.assert_called_once_with(mock_page, mock_fl_instance)

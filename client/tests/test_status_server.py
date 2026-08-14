"""Unit tests for client/comms/status_server.py — 100% coverage."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ─── helpers ────────────────────────────────────────────────────────────────────


def _mock_settings(site_id: str = "test_site") -> MagicMock:
    s = MagicMock()
    s.site_id = site_id
    return s


def _mock_state(
    run_count: int = 0,
    last_run_at: str | None = None,
    phase: str = "idle",
) -> MagicMock:
    s = MagicMock()
    s.run_count = run_count
    s.last_run_at = last_run_at
    s.phase = phase
    return s


# ─── /site/status endpoint ──────────────────────────────────────────────────────


class TestSiteStatusEndpoint:
    """GET /site/status returns correct fields from settings and state."""

    def _client(
        self,
        site_id: str = "site_1",
        run_count: int = 0,
        last_run_at: str | None = None,
        phase: str = "idle",
    ) -> TestClient:
        from client.comms.status_server import _app
        return TestClient(_app, raise_server_exceptions=True)

    def test_returns_200_ok(self) -> None:
        from client.comms.status_server import _app
        with (
            patch("client.comms.status_server.get_client_settings",
                  return_value=_mock_settings("site_1")),
            patch("client.comms.status_server.get_state",
                  return_value=_mock_state()),
        ):
            resp = TestClient(_app).get("/site/status")
        assert resp.status_code == 200

    def test_returns_site_id(self) -> None:
        from client.comms.status_server import _app
        with (
            patch("client.comms.status_server.get_client_settings",
                  return_value=_mock_settings("basel")),
            patch("client.comms.status_server.get_state",
                  return_value=_mock_state()),
        ):
            resp = TestClient(_app).get("/site/status")
        assert resp.json()["site_id"] == "basel"

    def test_returns_run_count(self) -> None:
        from client.comms.status_server import _app
        with (
            patch("client.comms.status_server.get_client_settings",
                  return_value=_mock_settings()),
            patch("client.comms.status_server.get_state",
                  return_value=_mock_state(run_count=7)),
        ):
            resp = TestClient(_app).get("/site/status")
        assert resp.json()["run_count"] == 7

    def test_returns_last_run_at(self) -> None:
        from client.comms.status_server import _app
        ts = "2026-08-14T14:32:00+00:00"
        with (
            patch("client.comms.status_server.get_client_settings",
                  return_value=_mock_settings()),
            patch("client.comms.status_server.get_state",
                  return_value=_mock_state(last_run_at=ts)),
        ):
            resp = TestClient(_app).get("/site/status")
        assert resp.json()["last_run_at"] == ts

    def test_returns_none_last_run_at_when_not_set(self) -> None:
        from client.comms.status_server import _app
        with (
            patch("client.comms.status_server.get_client_settings",
                  return_value=_mock_settings()),
            patch("client.comms.status_server.get_state",
                  return_value=_mock_state(last_run_at=None)),
        ):
            resp = TestClient(_app).get("/site/status")
        assert resp.json()["last_run_at"] is None

    def test_returns_phase(self) -> None:
        from client.comms.status_server import _app
        with (
            patch("client.comms.status_server.get_client_settings",
                  return_value=_mock_settings()),
            patch("client.comms.status_server.get_state",
                  return_value=_mock_state(phase="training")),
        ):
            resp = TestClient(_app).get("/site/status")
        assert resp.json()["phase"] == "training"

    def test_response_contains_all_four_keys(self) -> None:
        from client.comms.status_server import _app
        with (
            patch("client.comms.status_server.get_client_settings",
                  return_value=_mock_settings("singapore")),
            patch("client.comms.status_server.get_state",
                  return_value=_mock_state(run_count=3, phase="done")),
        ):
            resp = TestClient(_app).get("/site/status")
        data = resp.json()
        assert set(data.keys()) == {"site_id", "run_count", "last_run_at", "phase"}


# ─── start_status_server ────────────────────────────────────────────────────────


class TestStartStatusServer:
    """start_status_server() launches a named daemon thread targeting uvicorn.run."""

    def test_creates_daemon_thread(self) -> None:
        mock_thread = MagicMock()
        with patch(
            "client.comms.status_server.threading.Thread", return_value=mock_thread
        ) as mock_cls:
            from client.comms.status_server import start_status_server
            start_status_server(9001)

        assert mock_cls.call_args.kwargs["daemon"] is True
        mock_thread.start.assert_called_once()

    def test_thread_name_is_fl_status_server(self) -> None:
        mock_thread = MagicMock()
        with patch(
            "client.comms.status_server.threading.Thread", return_value=mock_thread
        ) as mock_cls:
            from client.comms.status_server import start_status_server
            start_status_server(9001)

        assert mock_cls.call_args.kwargs["name"] == "fl-status-server"

    def test_thread_target_is_uvicorn_run(self) -> None:
        import uvicorn
        mock_thread = MagicMock()
        with patch(
            "client.comms.status_server.threading.Thread", return_value=mock_thread
        ) as mock_cls:
            from client.comms.status_server import start_status_server
            start_status_server(9002)

        assert mock_cls.call_args.kwargs["target"] is uvicorn.run

    def test_thread_kwargs_include_port(self) -> None:
        mock_thread = MagicMock()
        with patch(
            "client.comms.status_server.threading.Thread", return_value=mock_thread
        ) as mock_cls:
            from client.comms.status_server import start_status_server
            start_status_server(9003)

        thread_kwargs = mock_cls.call_args.kwargs["kwargs"]
        assert thread_kwargs["port"] == 9003
        assert thread_kwargs["host"] == "0.0.0.0"
        assert thread_kwargs["log_level"] == "warning"

    def test_thread_kwargs_include_app(self) -> None:
        mock_thread = MagicMock()
        with patch(
            "client.comms.status_server.threading.Thread", return_value=mock_thread
        ) as mock_cls:
            from client.comms import status_server as ss
            ss.start_status_server(9004)

        thread_kwargs = mock_cls.call_args.kwargs["kwargs"]
        assert thread_kwargs["app"] is ss._app

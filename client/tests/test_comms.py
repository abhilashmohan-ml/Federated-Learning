"""Unit tests for client/comms — fl_client and heartbeat. 100% coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from shared.schemas.federation import ModelUpdate

# ─── sentinel ────────────────────────────────────────────────────────────────


class _BreakLoop(Exception):
    """Raised by a mocked time.sleep to escape an infinite loop under test."""


# ─── shared helpers ───────────────────────────────────────────────────────────


def _mock_settings() -> MagicMock:
    s = MagicMock()
    s.server_url = "http://localhost:8000"
    s.site_id = "site_1"
    s.site_secret = "secret"
    s.verify_ssl = False
    s.connect_timeout = 10
    s.request_timeout = 60
    s.retry_attempts = 3
    return s


def _token_json(access: str = "access_abc", refresh: str = "refresh_xyz") -> dict:
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "expires_in": 900,
    }


def _mock_resp(status: int = 200, json_data: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data if json_data is not None else {}
    return r


def _make_update() -> ModelUpdate:
    return ModelUpdate(
        site_id="site_1",
        round_id=1,
        n_samples=100,
        delta_W={"layer_1": [0.1, 0.2]},
    )


def _make_round_json(round_id: int = 1, status: str = "collecting") -> dict:
    from datetime import datetime, timezone

    return {
        "round_id": round_id,
        "status": status,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "participating_sites": [],
        "global_model_version": 0,
    }


def _build_fl_client(settings: MagicMock | None = None) -> object:
    """
    Instantiate FLClient with all external dependencies patched.

    The returned instance has _http replaced with a plain MagicMock so each
    test controls return values and side effects independently.
    """
    if settings is None:
        settings = _mock_settings()
    with (
        patch("client.comms.fl_client.get_client_settings", return_value=settings),
        patch("client.comms.fl_client.httpx.Client"),
    ):
        from client.comms.fl_client import FLClient

        fl = FLClient()
    # Patch _http after construction so tests own all request behaviour.
    fl._http = MagicMock()
    fl.settings = settings
    return fl


# ─────────────────────────────────────────────────────────────────────────────
# FLClient — construction
# ─────────────────────────────────────────────────────────────────────────────


class TestFLClientInit:
    def test_settings_stored_on_instance(self) -> None:
        settings = _mock_settings()
        fl = _build_fl_client(settings)
        assert fl.settings is settings

    def test_tokens_start_empty(self) -> None:
        fl = _build_fl_client()
        assert fl._access_token == ""
        assert fl._refresh_token == ""

    def test_httpx_client_created_with_ssl_and_timeout(self) -> None:
        settings = _mock_settings()
        with (
            patch("client.comms.fl_client.get_client_settings", return_value=settings),
            patch("client.comms.fl_client.httpx.Client") as mock_cls,
        ):
            from client.comms.fl_client import FLClient

            FLClient()

        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs.get("verify") is False
        assert "timeout" in kwargs


# ─────────────────────────────────────────────────────────────────────────────
# FLClient — close / context manager
# ─────────────────────────────────────────────────────────────────────────────


class TestFLClientContextManager:
    def test_close_calls_http_close(self) -> None:
        fl = _build_fl_client()
        fl.close()
        fl._http.close.assert_called_once()

    def test_enter_returns_self(self) -> None:
        fl = _build_fl_client()
        assert fl.__enter__() is fl

    def test_exit_calls_close(self) -> None:
        fl = _build_fl_client()
        fl.__exit__(None, None, None)
        fl._http.close.assert_called_once()

    def test_with_statement_closes_on_block_exit(self) -> None:
        settings = _mock_settings()
        with (
            patch("client.comms.fl_client.get_client_settings", return_value=settings),
            patch("client.comms.fl_client.httpx.Client"),
        ):
            from client.comms.fl_client import FLClient

            with FLClient() as fl:
                fl._http = MagicMock()

        fl._http.close.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# FLClient — auth_headers property
# ─────────────────────────────────────────────────────────────────────────────


class TestAuthHeaders:
    def test_empty_token_yields_empty_bearer(self) -> None:
        fl = _build_fl_client()
        assert fl.auth_headers == {"Authorization": "Bearer "}

    def test_populated_token_included_in_header(self) -> None:
        fl = _build_fl_client()
        fl._access_token = "tok_xyz"
        assert fl.auth_headers == {"Authorization": "Bearer tok_xyz"}


# ─────────────────────────────────────────────────────────────────────────────
# FLClient — _request retry logic
# ─────────────────────────────────────────────────────────────────────────────


class TestRequest:
    def test_success_on_first_attempt_returns_response(self) -> None:
        fl = _build_fl_client()
        expected = _mock_resp(200)
        fl._http.request.return_value = expected

        result = fl._request("GET", "http://localhost:8000/test")

        assert result is expected
        fl._http.request.assert_called_once_with("GET", "http://localhost:8000/test")

    def test_connect_error_retried_then_success(self) -> None:
        fl = _build_fl_client()
        good = _mock_resp(200)
        fl._http.request.side_effect = [httpx.ConnectError("refused"), good]

        with patch("client.comms.fl_client.time.sleep") as mock_sleep:
            result = fl._request("GET", "http://localhost:8000/test")

        assert result is good
        assert fl._http.request.call_count == 2
        mock_sleep.assert_called_once_with(2.0)

    def test_timeout_exception_retried_then_success(self) -> None:
        fl = _build_fl_client()
        good = _mock_resp(200)
        fl._http.request.side_effect = [httpx.TimeoutException("timeout"), good]

        with patch("client.comms.fl_client.time.sleep"):
            result = fl._request("POST", "http://localhost:8000/test")

        assert result is good

    def test_remote_protocol_error_retried_then_success(self) -> None:
        fl = _build_fl_client()
        good = _mock_resp(200)
        fl._http.request.side_effect = [httpx.RemoteProtocolError("closed"), good]

        with patch("client.comms.fl_client.time.sleep"):
            result = fl._request("POST", "http://localhost:8000/test")

        assert result is good

    def test_all_attempts_exhausted_raises_runtime_error(self) -> None:
        fl = _build_fl_client()
        fl.settings.retry_attempts = 3
        fl._http.request.side_effect = httpx.ConnectError("refused")

        with patch("client.comms.fl_client.time.sleep"):
            with pytest.raises(RuntimeError, match="failed after 3 attempts"):
                fl._request("GET", "http://localhost:8000/test")

        assert fl._http.request.call_count == 3

    def test_exponential_backoff_delays_double_each_attempt(self) -> None:
        """sleep called with 2.0 then 4.0; never after the final attempt."""
        fl = _build_fl_client()
        fl.settings.retry_attempts = 3
        fl._http.request.side_effect = httpx.TimeoutException("timeout")

        with patch("client.comms.fl_client.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                fl._request("GET", "http://localhost:8000/test")

        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2.0)
        mock_sleep.assert_any_call(4.0)

    def test_no_sleep_after_final_attempt(self) -> None:
        """With retry_attempts=1 the single failure must not sleep."""
        fl = _build_fl_client()
        fl.settings.retry_attempts = 1
        fl._http.request.side_effect = httpx.ConnectError("refused")

        with patch("client.comms.fl_client.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                fl._request("GET", "http://localhost:8000/test")

        mock_sleep.assert_not_called()

    def test_extra_kwargs_forwarded_to_http_request(self) -> None:
        fl = _build_fl_client()
        fl._http.request.return_value = _mock_resp(200)

        fl._request(
            "POST",
            "http://localhost:8000/test",
            json={"k": "v"},
            headers={"X-H": "1"},
        )

        fl._http.request.assert_called_once_with(
            "POST",
            "http://localhost:8000/test",
            json={"k": "v"},
            headers={"X-H": "1"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# FLClient — authenticate
# ─────────────────────────────────────────────────────────────────────────────


class TestAuthenticate:
    def test_stores_access_and_refresh_tokens(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200, _token_json("acc_tok", "ref_tok"))

        with patch.object(fl, "_request", return_value=resp):
            fl.authenticate()

        assert fl._access_token == "acc_tok"
        assert fl._refresh_token == "ref_tok"

    def test_raises_for_status_on_response(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200, _token_json())

        with patch.object(fl, "_request", return_value=resp):
            fl.authenticate()

        resp.raise_for_status.assert_called_once()

    def test_posts_to_auth_token_endpoint(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200, _token_json())

        with patch.object(fl, "_request", return_value=resp) as mock_req:
            fl.authenticate()

        args = mock_req.call_args.args
        assert args[0] == "POST"
        assert args[1] == "http://localhost:8000/auth/token"

    def test_sends_site_id_and_secret_in_json_body(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200, _token_json())

        with patch.object(fl, "_request", return_value=resp) as mock_req:
            fl.authenticate()

        body = mock_req.call_args.kwargs["json"]
        assert body["site_id"] == "site_1"
        assert body["site_secret"] == "secret"


# ─────────────────────────────────────────────────────────────────────────────
# FLClient — upload_update
# ─────────────────────────────────────────────────────────────────────────────


class TestUploadUpdate:
    def test_success_does_not_call_refresh(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200)

        with (
            patch.object(fl, "_request", return_value=resp) as mock_req,
            patch.object(fl, "_do_refresh") as mock_refresh,
        ):
            fl.upload_update(_make_update())

        mock_req.assert_called_once()
        mock_refresh.assert_not_called()
        resp.raise_for_status.assert_called_once()

    def test_401_triggers_do_refresh_then_retries(self) -> None:
        fl = _build_fl_client()
        resp_401 = _mock_resp(401)
        resp_200 = _mock_resp(200)

        with (
            patch.object(fl, "_request", side_effect=[resp_401, resp_200]) as mock_req,
            patch.object(fl, "_do_refresh") as mock_refresh,
        ):
            fl.upload_update(_make_update())

        assert mock_req.call_count == 2
        mock_refresh.assert_called_once()
        resp_200.raise_for_status.assert_called_once()

    def test_posts_to_federation_update_url(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200)

        with patch.object(fl, "_request", return_value=resp) as mock_req:
            fl.upload_update(_make_update())

        assert mock_req.call_args.args[1] == "http://localhost:8000/federation/update"

    def test_auth_headers_included_in_request(self) -> None:
        fl = _build_fl_client()
        fl._access_token = "bearer_tok"
        resp = _mock_resp(200)

        with patch.object(fl, "_request", return_value=resp) as mock_req:
            fl.upload_update(_make_update())

        headers = mock_req.call_args.kwargs["headers"]
        assert headers == {"Authorization": "Bearer bearer_tok"}


# ─────────────────────────────────────────────────────────────────────────────
# FLClient — get_global_model
# ─────────────────────────────────────────────────────────────────────────────


class TestGetGlobalModel:
    def test_returns_parsed_json(self) -> None:
        fl = _build_fl_client()
        payload = {"layer_1": [0.1, 0.2, 0.3]}
        resp = _mock_resp(200, payload)

        with patch.object(fl, "_request", return_value=resp):
            result = fl.get_global_model()

        assert result == payload

    def test_calls_raise_for_status(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200, {})

        with patch.object(fl, "_request", return_value=resp):
            fl.get_global_model()

        resp.raise_for_status.assert_called_once()

    def test_gets_global_model_endpoint(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200, {})

        with patch.object(fl, "_request", return_value=resp) as mock_req:
            fl.get_global_model()

        args = mock_req.call_args.args
        assert args[0] == "GET"
        assert "global-model" in args[1]

    def test_auth_headers_included(self) -> None:
        fl = _build_fl_client()
        fl._access_token = "my_tok"
        resp = _mock_resp(200, {})

        with patch.object(fl, "_request", return_value=resp) as mock_req:
            fl.get_global_model()

        headers = mock_req.call_args.kwargs["headers"]
        assert headers.get("Authorization") == "Bearer my_tok"


# ─────────────────────────────────────────────────────────────────────────────
# FLClient — _do_refresh
# ─────────────────────────────────────────────────────────────────────────────


class TestDoRefresh:
    def test_stores_new_access_and_refresh_tokens(self) -> None:
        fl = _build_fl_client()
        fl._refresh_token = "old_ref"
        resp = _mock_resp(200, _token_json("new_acc", "new_ref"))

        with patch.object(fl, "_request", return_value=resp):
            fl._do_refresh()

        assert fl._access_token == "new_acc"
        assert fl._refresh_token == "new_ref"

    def test_calls_raise_for_status(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200, _token_json())

        with patch.object(fl, "_request", return_value=resp):
            fl._do_refresh()

        resp.raise_for_status.assert_called_once()

    def test_posts_to_auth_refresh_endpoint(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200, _token_json())

        with patch.object(fl, "_request", return_value=resp) as mock_req:
            fl._do_refresh()

        args = mock_req.call_args.args
        assert args[0] == "POST"
        assert "auth/refresh" in args[1]

    def test_sends_current_refresh_token_in_body(self) -> None:
        fl = _build_fl_client()
        fl._refresh_token = "curr_ref_tok"
        resp = _mock_resp(200, _token_json())

        with patch.object(fl, "_request", return_value=resp) as mock_req:
            fl._do_refresh()

        body = mock_req.call_args.kwargs["json"]
        assert body["refresh_token"] == "curr_ref_tok"


# ─────────────────────────────────────────────────────────────────────────────
# heartbeat — INTERVAL constant
# ─────────────────────────────────────────────────────────────────────────────


class TestHeartbeatInterval:
    def test_interval_is_30_seconds(self) -> None:
        from client.comms.heartbeat import INTERVAL

        assert INTERVAL == 30


# ─────────────────────────────────────────────────────────────────────────────
# heartbeat — _beat
# ─────────────────────────────────────────────────────────────────────────────


class TestBeat:
    """
    _beat() runs an infinite loop; tests break it by making time.sleep raise
    _BreakLoop, which propagates past the inner try/except (because time.sleep
    is called *outside* the inner block) through the outer try/finally so that
    client.close() is always called before the exception surfaces.
    """

    def test_successful_ping_logs_debug_with_status_code(self) -> None:
        settings = _mock_settings()
        mock_http = MagicMock()
        mock_http.get.return_value = MagicMock(status_code=200)

        with (
            patch("client.comms.heartbeat.get_client_settings", return_value=settings),
            patch("client.comms.heartbeat.httpx.Client", return_value=mock_http),
            patch("client.comms.heartbeat.time.sleep", side_effect=_BreakLoop),
            patch("client.comms.heartbeat.log") as mock_log,
        ):
            from client.comms.heartbeat import _beat

            with pytest.raises(_BreakLoop):
                _beat()

        mock_log.debug.assert_called_once_with("heartbeat", site="site_1", status=200)

    def test_failed_ping_logs_warning_with_error_message(self) -> None:
        settings = _mock_settings()
        mock_http = MagicMock()
        mock_http.get.side_effect = Exception("connection refused")

        with (
            patch("client.comms.heartbeat.get_client_settings", return_value=settings),
            patch("client.comms.heartbeat.httpx.Client", return_value=mock_http),
            patch("client.comms.heartbeat.time.sleep", side_effect=_BreakLoop),
            patch("client.comms.heartbeat.log") as mock_log,
        ):
            from client.comms.heartbeat import _beat

            with pytest.raises(_BreakLoop):
                _beat()

        mock_log.warning.assert_called_once_with(
            "heartbeat_failed", site="site_1", error="connection refused"
        )

    def test_finally_always_closes_http_client(self) -> None:
        settings = _mock_settings()
        mock_http = MagicMock()
        mock_http.get.return_value = MagicMock(status_code=200)

        with (
            patch("client.comms.heartbeat.get_client_settings", return_value=settings),
            patch("client.comms.heartbeat.httpx.Client", return_value=mock_http),
            patch("client.comms.heartbeat.time.sleep", side_effect=_BreakLoop),
        ):
            from client.comms.heartbeat import _beat

            with pytest.raises(_BreakLoop):
                _beat()

        mock_http.close.assert_called_once()

    def test_finally_closes_client_even_after_failed_ping(self) -> None:
        settings = _mock_settings()
        mock_http = MagicMock()
        mock_http.get.side_effect = Exception("timeout")

        with (
            patch("client.comms.heartbeat.get_client_settings", return_value=settings),
            patch("client.comms.heartbeat.httpx.Client", return_value=mock_http),
            patch("client.comms.heartbeat.time.sleep", side_effect=_BreakLoop),
        ):
            from client.comms.heartbeat import _beat

            with pytest.raises(_BreakLoop):
                _beat()

        mock_http.close.assert_called_once()

    def test_http_client_created_with_verify_ssl_and_connect_timeout(self) -> None:
        settings = _mock_settings()
        mock_http = MagicMock()
        mock_http.get.return_value = MagicMock(status_code=200)

        with (
            patch("client.comms.heartbeat.get_client_settings", return_value=settings),
            patch("client.comms.heartbeat.httpx.Client", return_value=mock_http) as mock_cls,
            patch("client.comms.heartbeat.time.sleep", side_effect=_BreakLoop),
        ):
            from client.comms.heartbeat import _beat

            with pytest.raises(_BreakLoop):
                _beat()

        mock_cls.assert_called_once_with(verify=False, timeout=10.0)

    def test_health_endpoint_url_is_correct(self) -> None:
        settings = _mock_settings()
        mock_http = MagicMock()
        mock_http.get.return_value = MagicMock(status_code=200)

        with (
            patch("client.comms.heartbeat.get_client_settings", return_value=settings),
            patch("client.comms.heartbeat.httpx.Client", return_value=mock_http),
            patch("client.comms.heartbeat.time.sleep", side_effect=_BreakLoop),
        ):
            from client.comms.heartbeat import _beat

            with pytest.raises(_BreakLoop):
                _beat()

        mock_http.get.assert_called_once_with("http://localhost:8000/health/")

    def test_sleep_called_with_interval_constant(self) -> None:
        settings = _mock_settings()
        mock_http = MagicMock()
        mock_http.get.return_value = MagicMock(status_code=200)

        with (
            patch("client.comms.heartbeat.get_client_settings", return_value=settings),
            patch("client.comms.heartbeat.httpx.Client", return_value=mock_http),
            patch("client.comms.heartbeat.time.sleep", side_effect=_BreakLoop) as mock_sleep,
        ):
            from client.comms.heartbeat import INTERVAL, _beat

            with pytest.raises(_BreakLoop):
                _beat()

        mock_sleep.assert_called_once_with(INTERVAL)


# ─────────────────────────────────────────────────────────────────────────────
# heartbeat — start_heartbeat
# ─────────────────────────────────────────────────────────────────────────────


class TestStartHeartbeat:
    def test_creates_thread_with_correct_target_name_and_daemon(self) -> None:
        mock_thread = MagicMock()

        with patch("client.comms.heartbeat.threading.Thread", return_value=mock_thread) as mock_cls:
            from client.comms.heartbeat import _beat, start_heartbeat

            start_heartbeat()

        mock_cls.assert_called_once_with(target=_beat, daemon=True, name="fl-heartbeat")

    def test_thread_start_is_called(self) -> None:
        mock_thread = MagicMock()

        with patch("client.comms.heartbeat.threading.Thread", return_value=mock_thread):
            from client.comms.heartbeat import start_heartbeat

            start_heartbeat()

        mock_thread.start.assert_called_once()

    def test_thread_daemon_flag_is_true(self) -> None:
        mock_thread = MagicMock()

        with patch("client.comms.heartbeat.threading.Thread", return_value=mock_thread) as mock_cls:
            from client.comms.heartbeat import start_heartbeat

            start_heartbeat()

        assert mock_cls.call_args.kwargs["daemon"] is True

    def test_thread_name_is_fl_heartbeat(self) -> None:
        mock_thread = MagicMock()

        with patch("client.comms.heartbeat.threading.Thread", return_value=mock_thread) as mock_cls:
            from client.comms.heartbeat import start_heartbeat

            start_heartbeat()

        assert mock_cls.call_args.kwargs["name"] == "fl-heartbeat"


# ─────────────────────────────────────────────────────────────────────────────
# FLClient — start_round
# ─────────────────────────────────────────────────────────────────────────────


class TestStartRound:
    def test_success_returns_federation_round(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200, _make_round_json(round_id=3, status="collecting"))
        from shared.schemas.federation import FederationRound

        with patch.object(fl, "_request", return_value=resp):
            result = fl.start_round()

        assert isinstance(result, FederationRound)
        assert result.round_id == 3

    def test_401_triggers_do_refresh_then_retries(self) -> None:
        fl = _build_fl_client()
        resp_401 = _mock_resp(401)
        resp_200 = _mock_resp(200, _make_round_json())

        with (
            patch.object(fl, "_request", side_effect=[resp_401, resp_200]) as mock_req,
            patch.object(fl, "_do_refresh") as mock_refresh,
        ):
            fl.start_round()

        assert mock_req.call_count == 2
        mock_refresh.assert_called_once()
        resp_200.raise_for_status.assert_called_once()

    def test_no_refresh_on_200(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200, _make_round_json())

        with (
            patch.object(fl, "_request", return_value=resp),
            patch.object(fl, "_do_refresh") as mock_refresh,
        ):
            fl.start_round()

        mock_refresh.assert_not_called()

    def test_posts_to_federation_round_start_url(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200, _make_round_json())

        with patch.object(fl, "_request", return_value=resp) as mock_req:
            fl.start_round()

        args = mock_req.call_args.args
        assert args[0] == "POST"
        assert args[1] == "http://localhost:8000/federation/round/start"

    def test_auth_headers_included_in_request(self) -> None:
        fl = _build_fl_client()
        fl._access_token = "tok_abc"
        resp = _mock_resp(200, _make_round_json())

        with patch.object(fl, "_request", return_value=resp) as mock_req:
            fl.start_round()

        headers = mock_req.call_args.kwargs["headers"]
        assert headers == {"Authorization": "Bearer tok_abc"}

    def test_calls_raise_for_status(self) -> None:
        fl = _build_fl_client()
        resp = _mock_resp(200, _make_round_json())

        with patch.object(fl, "_request", return_value=resp):
            fl.start_round()

        resp.raise_for_status.assert_called_once()

"""Unit tests for client/engine — data_loader, local_trainer, scheduler. 100% coverage."""
import csv
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from client.engine.data_loader import load_filtration_csv, REQUIRED
from client.engine.scheduler import _watch, start_scheduler, POLL_SECONDS


# ── helpers ────────────────────────────────────────────────────────────────────

def _write_csv(path: str, rows, header=None) -> None:
    header = header or ["time_min", "flux_lmh", "tmp_bar"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _mock_client_settings(**kwargs) -> MagicMock:
    s = MagicMock()
    s.site_id = kwargs.get("site_id", "site_1")
    s.local_data_path = kwargs.get("local_data_path", "./data/filtration.csv")
    s.dp_noise_sigma = kwargs.get("dp_noise_sigma", 0.01)
    s.server_url = kwargs.get("server_url", "http://localhost:8000")
    return s


# ── load_filtration_csv ────────────────────────────────────────────────────────

class TestLoadFiltrationCSV:
    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_filtration_csv("/nonexistent/path/data.csv")

    def test_missing_column_raises_value_error(self, tmp_path) -> None:
        p = tmp_path / "bad.csv"
        # Missing tmp_bar
        _write_csv(str(p), [[0, 1]], header=["time_min", "flux_lmh"])
        with pytest.raises(ValueError, match="missing required columns"):
            load_filtration_csv(str(p))

    def test_normal_load_returns_three_arrays(self, tmp_path) -> None:
        p = tmp_path / "good.csv"
        rows = [(i * 5.0, 100.0 - i * 2.0, 1.0) for i in range(10)]
        _write_csv(str(p), rows)
        time, flux, tmp = load_filtration_csv(str(p))
        assert len(time) == 10
        assert len(flux) == 10
        assert len(tmp) == 10

    def test_nan_rows_dropped(self, tmp_path) -> None:
        p = tmp_path / "nan.csv"
        content = "time_min,flux_lmh,tmp_bar\n0,100.0,1.0\n,,\n5,90.0,1.0\n"
        p.write_text(content)
        time, flux, tmp = load_filtration_csv(str(p))
        assert len(time) == 2

    def test_returns_float64_arrays(self, tmp_path) -> None:
        p = tmp_path / "dtype.csv"
        _write_csv(str(p), [(0, 100, 1), (5, 90, 1)])
        time, flux, tmp = load_filtration_csv(str(p))
        assert time.dtype == np.float64
        assert flux.dtype == np.float64
        assert tmp.dtype == np.float64

    def test_required_constant(self) -> None:
        assert REQUIRED == {"time_min", "flux_lmh", "tmp_bar"}

    def test_extra_columns_allowed(self, tmp_path) -> None:
        p = tmp_path / "extra.csv"
        _write_csv(str(p), [(0, 100.0, 1.0, 7.0)],
                   header=["time_min", "flux_lmh", "tmp_bar", "lrv_obs"])
        time, flux, tmp = load_filtration_csv(str(p))
        assert len(time) == 1


# ── LocalTrainer ──────────────────────────────────────────────────────────────

def _make_hermia_result(selected: bool = True, model_name: str = "combined_1a"):
    from shared.models.hermia import HermiaResult
    return HermiaResult(
        model_name=model_name,
        params={"J0": 100.0, "k1": 0.05, "k2": 0.005},
        aic=-50.0, bic=-45.0, rmse=1.2, selected=selected,
    )


class TestLocalTrainer:
    def _build_csv(self, tmp_path) -> str:
        p = tmp_path / "data.csv"
        rows = [(i * 3.0, 100.0 - i * 1.5, 1.0) for i in range(20)]
        _write_csv(str(p), rows)
        return str(p)

    def test_train_basic_returns_model_update(self, tmp_path) -> None:
        csv_path = self._build_csv(tmp_path)
        mock_result = _make_hermia_result(selected=True)

        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings(local_data_path=csv_path)), \
             patch("client.engine.local_trainer.fit_all_models",
                   return_value={"combined_1a": mock_result}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.8), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.05), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=lambda w, sigma: w):
            from client.engine.local_trainer import LocalTrainer
            trainer = LocalTrainer()
            update = trainer.train_and_prepare_update(round_id=1)

        from shared.schemas.federation import ModelUpdate
        assert isinstance(update, ModelUpdate)
        assert update.round_id == 1
        assert update.site_id == "site_1"
        assert update.hermia_best_model == "combined_1a"
        assert update.n_samples == 20

    def test_train_local_metrics_set(self, tmp_path) -> None:
        csv_path = self._build_csv(tmp_path)
        mock_result = _make_hermia_result(selected=True)
        mock_result.rmse = 2.5

        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings(local_data_path=csv_path)), \
             patch("client.engine.local_trainer.fit_all_models",
                   return_value={"combined_1a": mock_result}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.75), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.1), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=lambda w, sigma: w):
            from client.engine.local_trainer import LocalTrainer
            trainer = LocalTrainer()
            update = trainer.train_and_prepare_update(round_id=3)

        assert "flux_rmse" in update.local_metrics
        assert "flux_ratio" in update.local_metrics
        assert "amin_m2" in update.local_metrics
        assert update.local_metrics["flux_ratio"] == pytest.approx(0.75)

    def test_train_fallback_to_first_when_no_selected(self, tmp_path) -> None:
        """When no HermiaResult has selected=True, falls back to first in dict."""
        csv_path = self._build_csv(tmp_path)
        mock_result = _make_hermia_result(selected=False, model_name="standard")

        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings(local_data_path=csv_path)), \
             patch("client.engine.local_trainer.fit_all_models",
                   return_value={"standard": mock_result}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.8), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.05), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=lambda w, sigma: w):
            from client.engine.local_trainer import LocalTrainer
            trainer = LocalTrainer()
            update = trainer.train_and_prepare_update(round_id=2)

        assert update.hermia_best_model == "standard"

    def test_empty_hermia_results_raises_index_error(self, tmp_path) -> None:
        """fit_all_models returning {} → list({})[0] raises IndexError (documented behavior)."""
        csv_path = self._build_csv(tmp_path)
        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings(local_data_path=csv_path)), \
             patch("client.engine.local_trainer.fit_all_models", return_value={}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.8), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.05), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=lambda w, sigma: w):
            from client.engine.local_trainer import LocalTrainer
            with pytest.raises(IndexError):
                LocalTrainer().train_and_prepare_update(round_id=1)

    def test_all_five_local_metric_keys_present(self, tmp_path) -> None:
        csv_path = self._build_csv(tmp_path)
        mock_result = _make_hermia_result(selected=True)
        mock_result.aic = -60.0
        mock_result.bic = -55.0

        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings(local_data_path=csv_path)), \
             patch("client.engine.local_trainer.fit_all_models",
                   return_value={"combined_1a": mock_result}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.75), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.1), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=lambda w, sigma: w):
            from client.engine.local_trainer import LocalTrainer
            update = LocalTrainer().train_and_prepare_update(round_id=1)

        assert "flux_rmse" in update.local_metrics
        assert "flux_ratio" in update.local_metrics
        assert "amin_m2" in update.local_metrics
        assert "best_aic" in update.local_metrics
        assert "best_bic" in update.local_metrics
        assert update.local_metrics["best_aic"] == pytest.approx(-60.0)
        assert update.local_metrics["best_bic"] == pytest.approx(-55.0)

    def test_dp_noise_applied(self, tmp_path) -> None:
        csv_path = self._build_csv(tmp_path)
        mock_result = _make_hermia_result(selected=True)
        noise_called_with = {}

        def capture_noise(w, sigma):
            noise_called_with["sigma"] = sigma
            return w

        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings(
                       local_data_path=csv_path, dp_noise_sigma=0.05)), \
             patch("client.engine.local_trainer.fit_all_models",
                   return_value={"combined_1a": mock_result}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.8), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.05), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=capture_noise):
            from client.engine.local_trainer import LocalTrainer
            LocalTrainer().train_and_prepare_update(round_id=1)

        assert noise_called_with.get("sigma") == pytest.approx(0.05)


# ── scheduler._watch ──────────────────────────────────────────────────────────

class TestWatch:
    def _mock_fl(self, auth_raises: bool = False) -> MagicMock:
        fl = MagicMock()
        if auth_raises:
            fl.authenticate.side_effect = Exception("auth failed")
        fl.auth_headers = {"Authorization": "Bearer fake"}
        return fl

    def _mock_resp(self, status: int = 200, data: dict | None = None) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = data or {}
        return resp

    def test_auth_failure_returns_early(self) -> None:
        fl = self._mock_fl(auth_raises=True)
        mock_trainer = MagicMock()
        with patch("client.engine.scheduler.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer):
            _watch()
        fl.upload_update.assert_not_called()
        mock_trainer.train_and_prepare_update.assert_not_called()

    def test_non_200_response_no_training(self) -> None:
        fl = self._mock_fl()
        mock_trainer = MagicMock()
        resp = self._mock_resp(status=404)
        with patch("client.engine.scheduler.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.httpx.get", return_value=resp), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()
        mock_trainer.train_and_prepare_update.assert_not_called()

    def test_collecting_round_triggers_training_and_upload(self) -> None:
        fl = self._mock_fl()
        mock_update = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.return_value = mock_update
        resp = self._mock_resp(data={"round_id": 1, "status": "collecting"})

        with patch("client.engine.scheduler.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.httpx.get", return_value=resp), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()

        mock_trainer.train_and_prepare_update.assert_called_once_with(1)
        fl.upload_update.assert_called_once_with(mock_update)

    def test_already_seen_round_no_training(self) -> None:
        """round_id=0 <= last_seen_round=0 → no training."""
        fl = self._mock_fl()
        mock_trainer = MagicMock()
        resp = self._mock_resp(data={"round_id": 0, "status": "collecting"})

        with patch("client.engine.scheduler.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.httpx.get", return_value=resp), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()

        mock_trainer.train_and_prepare_update.assert_not_called()

    def test_non_collecting_status_no_training(self) -> None:
        """status="complete" → no training."""
        fl = self._mock_fl()
        mock_trainer = MagicMock()
        resp = self._mock_resp(data={"round_id": 1, "status": "complete"})

        with patch("client.engine.scheduler.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.httpx.get", return_value=resp), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()

        mock_trainer.train_and_prepare_update.assert_not_called()

    def test_second_poll_same_round_no_retraining(self) -> None:
        """After training round 1, a second poll returning round_id=1 must not retrain."""
        fl = self._mock_fl()
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.return_value = MagicMock()
        # First poll: collecting → trains. Second poll: same round_id → skip.
        poll_resp_collecting = self._mock_resp(data={"round_id": 1, "status": "collecting"})
        sleep_calls = {"count": 0}

        def sleep_side_effect(t):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                raise SystemExit(0)

        with patch("client.engine.scheduler.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.httpx.get",
                   return_value=poll_resp_collecting), \
             patch("client.engine.scheduler.time.sleep",
                   side_effect=sleep_side_effect):
            with pytest.raises(SystemExit):
                _watch()

        # Trained exactly once despite two polls returning round_id=1
        assert mock_trainer.train_and_prepare_update.call_count == 1

    def test_exception_in_loop_caught_warning_logged(self) -> None:
        """Network error inside loop is caught; loop continues until sleep exits."""
        fl = self._mock_fl()
        with patch("client.engine.scheduler.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer"), \
             patch("client.engine.scheduler.httpx.get",
                   side_effect=ConnectionError("unreachable")), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()
        # No exception propagated out of _watch (ConnectionError was caught)


# ── scheduler.start_scheduler ─────────────────────────────────────────────────

class TestStartScheduler:
    def test_creates_named_daemon_thread(self) -> None:
        with patch("client.engine.scheduler.threading.Thread") as mock_thread_cls:
            start_scheduler()
        mock_thread_cls.assert_called_once_with(
            target=_watch, daemon=True, name="fl-scheduler"
        )
        mock_thread_cls.return_value.start.assert_called_once()

    def test_poll_seconds_constant(self) -> None:
        assert POLL_SECONDS == 15

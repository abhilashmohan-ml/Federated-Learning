"""Unit tests for client/engine — data_loader, local_trainer, scheduler. 100% coverage."""
import csv
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from client.engine.data_loader import load_filtration_csv, REQUIRED
from client.engine.scheduler import _watch_dev, _watch_prod, start_scheduler, POLL_SECONDS


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
    def _mock_ds(self, n: int = 20) -> MagicMock:
        """Return a mock DataSource that produces n-row filtration arrays."""
        ds = MagicMock()
        ds.get_data.return_value = (
            np.arange(0, n, dtype=np.float64) * 3.0,
            np.array([100.0 - i * 1.5 for i in range(n)], dtype=np.float64),
            np.ones(n, dtype=np.float64),
        )
        return ds

    def test_train_basic_returns_model_update(self) -> None:
        mock_result = _make_hermia_result(selected=True)

        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.local_trainer.fit_all_models",
                   return_value={"combined_1a": mock_result}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.8), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.05), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=lambda w, sigma: w):
            from client.engine.local_trainer import LocalTrainer
            trainer = LocalTrainer(data_source=self._mock_ds())
            update = trainer.train_and_prepare_update(round_id=1)

        from shared.schemas.federation import ModelUpdate
        assert isinstance(update, ModelUpdate)
        assert update.round_id == 1
        assert update.site_id == "site_1"
        assert update.hermia_best_model == "combined_1a"
        assert update.n_samples == 20

    def test_train_local_metrics_set(self) -> None:
        mock_result = _make_hermia_result(selected=True)
        mock_result.rmse = 2.5

        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.local_trainer.fit_all_models",
                   return_value={"combined_1a": mock_result}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.75), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.1), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=lambda w, sigma: w):
            from client.engine.local_trainer import LocalTrainer
            trainer = LocalTrainer(data_source=self._mock_ds())
            update = trainer.train_and_prepare_update(round_id=3)

        assert "flux_rmse" in update.local_metrics
        assert "flux_ratio" in update.local_metrics
        assert "amin_m2" in update.local_metrics
        assert update.local_metrics["flux_ratio"] == pytest.approx(0.75)

    def test_train_fallback_to_first_when_no_selected(self) -> None:
        """When no HermiaResult has selected=True, falls back to first in dict."""
        mock_result = _make_hermia_result(selected=False, model_name="standard")

        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.local_trainer.fit_all_models",
                   return_value={"standard": mock_result}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.8), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.05), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=lambda w, sigma: w):
            from client.engine.local_trainer import LocalTrainer
            trainer = LocalTrainer(data_source=self._mock_ds())
            update = trainer.train_and_prepare_update(round_id=2)

        assert update.hermia_best_model == "standard"

    def test_empty_hermia_results_raises_runtime_error(self) -> None:
        """fit_all_models returning {} → explicit RuntimeError with descriptive message."""
        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.local_trainer.fit_all_models", return_value={}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.8), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.05), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=lambda w, sigma: w):
            from client.engine.local_trainer import LocalTrainer
            with pytest.raises(RuntimeError, match="All Hermia models failed"):
                LocalTrainer(data_source=self._mock_ds()).train_and_prepare_update(round_id=1)

    def test_all_five_local_metric_keys_present(self) -> None:
        mock_result = _make_hermia_result(selected=True)
        mock_result.aic = -60.0
        mock_result.bic = -55.0

        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.local_trainer.fit_all_models",
                   return_value={"combined_1a": mock_result}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.75), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.1), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=lambda w, sigma: w):
            from client.engine.local_trainer import LocalTrainer
            update = LocalTrainer(data_source=self._mock_ds()).train_and_prepare_update(round_id=1)

        assert "flux_rmse" in update.local_metrics
        assert "flux_ratio" in update.local_metrics
        assert "amin_m2" in update.local_metrics
        assert "best_aic" in update.local_metrics
        assert "best_bic" in update.local_metrics
        assert update.local_metrics["best_aic"] == pytest.approx(-60.0)
        assert update.local_metrics["best_bic"] == pytest.approx(-55.0)

    def test_dp_noise_applied(self) -> None:
        mock_result = _make_hermia_result(selected=True)
        noise_called_with = {}

        def capture_noise(w, sigma):
            noise_called_with["sigma"] = sigma
            return w

        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings(dp_noise_sigma=0.05)), \
             patch("client.engine.local_trainer.fit_all_models",
                   return_value={"combined_1a": mock_result}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.8), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.05), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=capture_noise):
            from client.engine.local_trainer import LocalTrainer
            LocalTrainer(data_source=self._mock_ds()).train_and_prepare_update(round_id=1)

        assert noise_called_with.get("sigma") == pytest.approx(0.05)


# ── scheduler._watch_dev ──────────────────────────────────────────────────────

class TestWatchDev:
    """Tests for the dev-mode scheduler loop."""

    def _mock_fl(self) -> MagicMock:
        return MagicMock()

    def _mock_update(
        self, flux_ratio: float = 0.88, amin: float = 0.04, model: str = "combined_1a"
    ) -> MagicMock:
        u = MagicMock()
        u.local_metrics = {"flux_ratio": flux_ratio, "amin_m2": amin}
        u.hermia_best_model = model
        return u

    def test_none_response_no_training(self) -> None:
        """get_round_status returns None (404/not-yet-existing) → no training."""
        fl = self._mock_fl()
        fl.get_round_status.return_value = None
        mock_trainer = MagicMock()

        with patch("client.engine.scheduler.time.sleep", side_effect=StopIteration):
            with pytest.raises(StopIteration):
                _watch_dev(fl, mock_trainer)

        mock_trainer.train_and_prepare_update.assert_not_called()

    def test_collecting_round_triggers_training_and_upload(self) -> None:
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 1, "status": "collecting"}
        mock_update = self._mock_update()
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.return_value = mock_update

        with patch("client.engine.scheduler.time.sleep", side_effect=StopIteration), \
             patch("client.engine.scheduler.get_state", return_value=MagicMock(run_count=0)):
            with pytest.raises(StopIteration):
                _watch_dev(fl, mock_trainer)

        mock_trainer.train_and_prepare_update.assert_called_once_with(1)
        fl.upload_update.assert_called_once_with(mock_update)

    def test_already_seen_round_no_training(self) -> None:
        """round_id=0 <= last_seen_round=0 → no training."""
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 0, "status": "collecting"}
        mock_trainer = MagicMock()

        with patch("client.engine.scheduler.time.sleep", side_effect=StopIteration):
            with pytest.raises(StopIteration):
                _watch_dev(fl, mock_trainer)

        mock_trainer.train_and_prepare_update.assert_not_called()

    def test_non_collecting_status_no_training(self) -> None:
        """status="complete" → no training."""
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 1, "status": "complete"}
        mock_trainer = MagicMock()

        with patch("client.engine.scheduler.time.sleep", side_effect=StopIteration):
            with pytest.raises(StopIteration):
                _watch_dev(fl, mock_trainer)

        mock_trainer.train_and_prepare_update.assert_not_called()

    def test_second_poll_same_round_no_retraining(self) -> None:
        """After training round 1, a second poll returning round_id=1 must not retrain."""
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 1, "status": "collecting"}
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.return_value = self._mock_update()
        sleep_count = {"n": 0}

        def sleep_side(t: float) -> None:
            sleep_count["n"] += 1
            if sleep_count["n"] >= 2:
                raise StopIteration

        with patch("client.engine.scheduler.time.sleep", side_effect=sleep_side), \
             patch("client.engine.scheduler.get_state", return_value=MagicMock(run_count=0)):
            with pytest.raises(StopIteration):
                _watch_dev(fl, mock_trainer)

        # Trained exactly once despite two polls returning round_id=1
        assert mock_trainer.train_and_prepare_update.call_count == 1

    def test_poll_error_sets_phase_error_and_logs_warning(self) -> None:
        """Network error inside loop is caught; error phase set; warning logged."""
        fl = self._mock_fl()
        fl.get_round_status.side_effect = ConnectionError("unreachable")

        with patch("client.engine.scheduler.time.sleep", side_effect=StopIteration), \
             patch("client.engine.scheduler.log") as mock_log, \
             patch("client.engine.scheduler.update_state") as mock_update_state:
            with pytest.raises(StopIteration):
                _watch_dev(fl, MagicMock())

        mock_log.warning.assert_called_once_with(
            "scheduler_poll_error", error="unreachable"
        )
        mock_update_state.assert_any_call(phase="error")

    def test_done_state_carries_run_count_and_metrics(self) -> None:
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 2, "status": "collecting"}
        mock_update = self._mock_update(flux_ratio=0.75, amin=0.03, model="cake")
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.return_value = mock_update
        calls: list[dict] = []

        with patch("client.engine.scheduler.time.sleep", side_effect=StopIteration), \
             patch("client.engine.scheduler.update_state",
                   side_effect=lambda **kw: calls.append(kw)), \
             patch("client.engine.scheduler.get_state",
                   return_value=MagicMock(run_count=0)):
            with pytest.raises(StopIteration):
                _watch_dev(fl, mock_trainer)

        done_call = next((c for c in calls if c.get("phase") == "done"), None)
        assert done_call is not None
        assert done_call["run_count"] == 1
        assert done_call["last_round_completed"] == 2
        assert done_call["last_flux_ratio"] == pytest.approx(0.75)
        assert done_call["last_amin"] == pytest.approx(0.03)
        assert done_call["last_hermia_model"] == "cake"

    def test_phase_transitions_training_uploading_done_in_order(self) -> None:
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 1, "status": "collecting"}
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.return_value = self._mock_update()
        calls: list[dict] = []

        with patch("client.engine.scheduler.time.sleep", side_effect=StopIteration), \
             patch("client.engine.scheduler.update_state",
                   side_effect=lambda **kw: calls.append(kw)), \
             patch("client.engine.scheduler.get_state",
                   return_value=MagicMock(run_count=0)):
            with pytest.raises(StopIteration):
                _watch_dev(fl, mock_trainer)

        phases = [c["phase"] for c in calls if "phase" in c]
        assert "training" in phases
        assert "uploading" in phases
        assert "done" in phases
        assert phases.index("training") < phases.index("uploading") < phases.index("done")


# ── scheduler._watch_prod ─────────────────────────────────────────────────────

class TestWatchProd:
    """Tests for the prod-mode scheduler loop."""

    def _mock_fl(self) -> MagicMock:
        return MagicMock()

    def _mock_prod_source(self, has_new: bool = False) -> MagicMock:
        ps = MagicMock()
        ps.has_new_data.return_value = has_new
        return ps

    def _mock_round(self, round_id: int = 1) -> MagicMock:
        r = MagicMock()
        r.round_id = round_id
        return r

    def _mock_update(
        self, flux_ratio: float = 0.8, amin: float = 0.04, model: str = "combined_1a"
    ) -> MagicMock:
        u = MagicMock()
        u.local_metrics = {"flux_ratio": flux_ratio, "amin_m2": amin}
        u.hermia_best_model = model
        return u

    def test_no_new_data_skips_training(self) -> None:
        fl = self._mock_fl()
        mock_trainer = MagicMock()
        ps = self._mock_prod_source(has_new=False)

        with patch("client.engine.scheduler.time.sleep", side_effect=StopIteration):
            with pytest.raises(StopIteration):
                _watch_prod(fl, mock_trainer, ps, 30)

        mock_trainer.train_and_prepare_update.assert_not_called()

    def test_has_new_data_trains_and_uploads(self) -> None:
        fl = self._mock_fl()
        fl.get_current_round.return_value = self._mock_round(round_id=5)
        ps = self._mock_prod_source(has_new=True)
        mock_update = self._mock_update(flux_ratio=0.7, amin=0.02, model="intermediate")
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.return_value = mock_update

        with patch("client.engine.scheduler.time.sleep", side_effect=StopIteration), \
             patch("client.engine.scheduler.get_state",
                   return_value=MagicMock(run_count=0)):
            with pytest.raises(StopIteration):
                _watch_prod(fl, mock_trainer, ps, 30)

        mock_trainer.train_and_prepare_update.assert_called_once_with(5)
        fl.upload_update.assert_called_once_with(mock_update)

    def test_no_new_data_error_silently_caught(self) -> None:
        """NoNewDataError raised during training is caught without setting error state."""
        from client.engine.data_source import NoNewDataError
        fl = self._mock_fl()
        fl.get_current_round.return_value = self._mock_round()
        ps = self._mock_prod_source(has_new=True)
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.side_effect = NoNewDataError("no data")
        calls: list[dict] = []

        with patch("client.engine.scheduler.time.sleep", side_effect=StopIteration), \
             patch("client.engine.scheduler.update_state",
                   side_effect=lambda **kw: calls.append(kw)):
            with pytest.raises(StopIteration):
                _watch_prod(fl, mock_trainer, ps, 30)

        # NoNewDataError must NOT set error phase
        assert not any(c.get("phase") == "error" for c in calls)

    def test_generic_exception_sets_error_state_and_logs(self) -> None:
        fl = self._mock_fl()
        fl.get_current_round.return_value = self._mock_round()
        ps = self._mock_prod_source(has_new=True)
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.side_effect = RuntimeError("model exploded")
        calls: list[dict] = []

        with patch("client.engine.scheduler.time.sleep", side_effect=StopIteration), \
             patch("client.engine.scheduler.update_state",
                   side_effect=lambda **kw: calls.append(kw)), \
             patch("client.engine.scheduler.log") as mock_log:
            with pytest.raises(StopIteration):
                _watch_prod(fl, mock_trainer, ps, 30)

        assert any(c.get("phase") == "error" for c in calls)
        mock_log.warning.assert_called_once_with(
            "prod_poll_error", error="model exploded"
        )

    def test_done_state_carries_metrics_and_run_count(self) -> None:
        fl = self._mock_fl()
        fl.get_current_round.return_value = self._mock_round(round_id=3)
        ps = self._mock_prod_source(has_new=True)
        mock_update = self._mock_update(flux_ratio=0.65, amin=0.02, model="cake")
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.return_value = mock_update
        calls: list[dict] = []

        with patch("client.engine.scheduler.time.sleep", side_effect=StopIteration), \
             patch("client.engine.scheduler.update_state",
                   side_effect=lambda **kw: calls.append(kw)), \
             patch("client.engine.scheduler.get_state",
                   return_value=MagicMock(run_count=1)):
            with pytest.raises(StopIteration):
                _watch_prod(fl, mock_trainer, ps, 30)

        done_call = next((c for c in calls if c.get("phase") == "done"), None)
        assert done_call is not None
        assert done_call["run_count"] == 2
        assert done_call["last_round_completed"] == 3
        assert done_call["last_flux_ratio"] == pytest.approx(0.65)
        assert done_call["last_amin"] == pytest.approx(0.02)
        assert done_call["last_hermia_model"] == "cake"


# ── scheduler.start_scheduler ─────────────────────────────────────────────────

class TestStartScheduler:
    """Tests for start_scheduler() — thread creation and lambda coverage."""

    def _mock_settings(self, auto_schedule: bool = True) -> MagicMock:
        s = MagicMock()
        s.data_poll_seconds = 60
        s.auto_schedule = auto_schedule
        return s

    def test_dev_mode_spawns_dev_thread(self) -> None:
        from client.engine.data_source import DevDataSource, PHYSICS_DEFAULTS
        ds = DevDataSource(PHYSICS_DEFAULTS)
        mock_thread = MagicMock()
        mock_fl = MagicMock()
        settings = self._mock_settings()

        with patch("client.engine.scheduler.threading.Thread",
                   return_value=mock_thread) as mock_cls, \
             patch("client.engine.scheduler.LocalTrainer"), \
             patch("client.config.get_client_settings", return_value=settings):
            start_scheduler(ds, fl_client=mock_fl)

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["name"] == "fl-scheduler-dev"
        assert call_kwargs["daemon"] is True
        mock_thread.start.assert_called_once()

    def test_prod_mode_spawns_prod_thread(self, tmp_path) -> None:
        from client.engine.data_source import ProdDataSource
        ds = ProdDataSource(str(tmp_path))
        mock_thread = MagicMock()
        mock_fl = MagicMock()
        settings = self._mock_settings()

        with patch("client.engine.scheduler.threading.Thread",
                   return_value=mock_thread) as mock_cls, \
             patch("client.engine.scheduler.LocalTrainer"), \
             patch("client.config.get_client_settings", return_value=settings):
            start_scheduler(ds, fl_client=mock_fl)

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["name"] == "fl-scheduler-prod"
        assert call_kwargs["daemon"] is True
        mock_thread.start.assert_called_once()

    def test_dev_lambda_body_calls_watch_dev(self) -> None:
        """Cover the lambda body: lambda: _watch_dev(fl, trainer)."""
        from client.engine.data_source import DevDataSource, PHYSICS_DEFAULTS
        ds = DevDataSource(PHYSICS_DEFAULTS)
        mock_thread = MagicMock()
        mock_fl = MagicMock()
        settings = self._mock_settings()

        with patch("client.engine.scheduler.threading.Thread",
                   return_value=mock_thread) as mock_cls, \
             patch("client.engine.scheduler.LocalTrainer") as mock_trainer_cls, \
             patch("client.engine.scheduler._watch_dev") as mock_watch, \
             patch("client.config.get_client_settings", return_value=settings):
            start_scheduler(ds, fl_client=mock_fl)
            target = mock_cls.call_args.kwargs["target"]
            target()

        mock_watch.assert_called_once_with(mock_fl, mock_trainer_cls.return_value)

    def test_prod_lambda_body_calls_watch_prod(self, tmp_path) -> None:
        """Cover the lambda body: lambda: _watch_prod(fl, trainer, data_source, poll_seconds)."""
        from client.engine.data_source import ProdDataSource
        ds = ProdDataSource(str(tmp_path))
        mock_thread = MagicMock()
        mock_fl = MagicMock()
        settings = self._mock_settings()

        with patch("client.engine.scheduler.threading.Thread",
                   return_value=mock_thread) as mock_cls, \
             patch("client.engine.scheduler.LocalTrainer") as mock_trainer_cls, \
             patch("client.engine.scheduler._watch_prod") as mock_watch, \
             patch("client.config.get_client_settings", return_value=settings):
            start_scheduler(ds, fl_client=mock_fl)
            target = mock_cls.call_args.kwargs["target"]
            target()

        mock_watch.assert_called_once_with(mock_fl, mock_trainer_cls.return_value, ds, 60)

    def test_prod_mode_auto_schedule_false_still_spawns_thread(self, tmp_path) -> None:
        """auto_schedule=False must NOT suppress the prod-mode scheduler."""
        from client.engine.data_source import ProdDataSource
        ds = ProdDataSource(str(tmp_path))
        mock_thread = MagicMock()
        mock_fl = MagicMock()
        settings = self._mock_settings(auto_schedule=False)

        with patch("client.engine.scheduler.threading.Thread",
                   return_value=mock_thread) as mock_cls, \
             patch("client.engine.scheduler.LocalTrainer"), \
             patch("client.config.get_client_settings", return_value=settings):
            start_scheduler(ds, fl_client=mock_fl)

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["name"] == "fl-scheduler-prod"
        mock_thread.start.assert_called_once()

    def test_dev_mode_auto_schedule_false_skips_thread(self) -> None:
        """auto_schedule=False in dev mode must NOT spawn a scheduler thread."""
        from client.engine.data_source import DevDataSource, PHYSICS_DEFAULTS
        ds = DevDataSource(PHYSICS_DEFAULTS)
        mock_fl = MagicMock()
        settings = self._mock_settings(auto_schedule=False)

        with patch("client.engine.scheduler.threading.Thread") as mock_thread_cls, \
             patch("client.engine.scheduler.LocalTrainer"), \
             patch("client.config.get_client_settings", return_value=settings):
            start_scheduler(ds, fl_client=mock_fl)

        mock_thread_cls.assert_not_called()

    def test_poll_seconds_constant(self) -> None:
        assert POLL_SECONDS == 15


# ── TrainingState ─────────────────────────────────────────────────────────────

class TestTrainingState:
    def setup_method(self) -> None:
        """Reset shared state to defaults before each test."""
        from client.engine import state as _s
        _s.update_state(
            current_round_id=0, phase="idle",
            last_lrv=None, last_amin=None,
            last_flux_ratio=None, last_hermia_model=None,
            last_round_completed=0,
            run_count=0,
            last_run_at=None,
        )

    def test_initial_defaults(self) -> None:
        from client.engine.state import get_state
        s = get_state()
        assert s.current_round_id == 0
        assert s.phase == "idle"
        assert s.last_lrv is None
        assert s.last_amin is None
        assert s.last_flux_ratio is None
        assert s.last_hermia_model is None
        assert s.last_round_completed == 0
        assert s.run_count == 0
        assert s.last_run_at is None

    def test_update_single_field(self) -> None:
        from client.engine.state import get_state, update_state
        update_state(current_round_id=7)
        assert get_state().current_round_id == 7

    def test_update_multiple_fields(self) -> None:
        from client.engine.state import get_state, update_state
        update_state(phase="training", last_lrv=4.2, last_amin=0.05)
        s = get_state()
        assert s.phase == "training"
        assert s.last_lrv == pytest.approx(4.2)
        assert s.last_amin == pytest.approx(0.05)

    def test_get_state_returns_snapshot_not_reference(self) -> None:
        """Mutating the snapshot must not affect shared state."""
        from client.engine.state import get_state
        snap = get_state()
        snap.phase = "mutated"
        assert get_state().phase != "mutated"

    def test_all_phase_transitions(self) -> None:
        from client.engine.state import get_state, update_state
        for phase in ("idle", "training", "uploading", "done", "error"):
            update_state(phase=phase)
            assert get_state().phase == phase

    def test_thread_safety_concurrent_writes(self) -> None:
        from client.engine.state import get_state, update_state
        errors: list[Exception] = []

        def writer(val: int) -> None:
            try:
                for _ in range(200):
                    update_state(current_round_id=val)
                    get_state()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety violation: {errors}"
        assert get_state().current_round_id in range(6)

    def test_update_no_args_is_noop(self) -> None:
        from client.engine.state import get_state, update_state
        update_state(phase="training")
        before = get_state().phase
        update_state()  # empty kwargs — must be a no-op
        assert get_state().phase == before

    def test_update_invalid_field_raises(self) -> None:
        from client.engine.state import update_state
        with pytest.raises(AttributeError, match="has no field"):
            update_state(nonexistent_field=99)



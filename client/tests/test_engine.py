"""Unit tests for client/engine — data_loader, local_trainer, scheduler. 100% coverage."""
import csv
import threading
from unittest.mock import MagicMock, patch

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

    def test_empty_hermia_results_raises_index_error(self) -> None:
        """fit_all_models returning {} → list({})[0] raises IndexError (documented behavior)."""
        with patch("client.engine.local_trainer.get_client_settings",
                   return_value=_mock_client_settings()), \
             patch("client.engine.local_trainer.fit_all_models", return_value={}), \
             patch("client.engine.local_trainer.compute_flux_ratio", return_value=0.8), \
             patch("client.engine.local_trainer.compute_amin", return_value=0.05), \
             patch("client.engine.local_trainer.add_gaussian_noise",
                   side_effect=lambda w, sigma: w):
            from client.engine.local_trainer import LocalTrainer
            with pytest.raises(IndexError):
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


# ── scheduler._watch ──────────────────────────────────────────────────────────

class TestWatch:
    def _mock_fl(self, auth_raises: bool = False) -> MagicMock:
        fl = MagicMock()
        if auth_raises:
            fl.authenticate.side_effect = Exception("auth failed")
        return fl

    def test_auth_failure_returns_early(self) -> None:
        fl = self._mock_fl(auth_raises=True)
        mock_trainer = MagicMock()
        with patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer):
            _watch()
        fl.upload_update.assert_not_called()
        mock_trainer.train_and_prepare_update.assert_not_called()

    def test_none_response_no_training(self) -> None:
        """get_round_status returns None (404/not-yet-existing) → no training."""
        fl = self._mock_fl()
        fl.get_round_status.return_value = None
        mock_trainer = MagicMock()

        with patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()

        mock_trainer.train_and_prepare_update.assert_not_called()

    def test_collecting_round_triggers_training_and_upload(self) -> None:
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 1, "status": "collecting"}
        mock_update = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.return_value = mock_update

        with patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()

        mock_trainer.train_and_prepare_update.assert_called_once_with(1)
        fl.upload_update.assert_called_once_with(mock_update)

    def test_already_seen_round_no_training(self) -> None:
        """round_id=0 <= last_seen_round=0 → no training."""
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 0, "status": "collecting"}
        mock_trainer = MagicMock()

        with patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()

        mock_trainer.train_and_prepare_update.assert_not_called()

    def test_non_collecting_status_no_training(self) -> None:
        """status="complete" → no training."""
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 1, "status": "complete"}
        mock_trainer = MagicMock()

        with patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()

        mock_trainer.train_and_prepare_update.assert_not_called()

    def test_second_poll_same_round_no_retraining(self) -> None:
        """After training round 1, a second poll returning round_id=1 must not retrain."""
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 1, "status": "collecting"}
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.return_value = MagicMock()
        sleep_calls = {"count": 0}

        def sleep_side_effect(t):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                raise SystemExit(0)

        with patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.time.sleep", side_effect=sleep_side_effect):
            with pytest.raises(SystemExit):
                _watch()

        # Trained exactly once despite two polls returning round_id=1
        assert mock_trainer.train_and_prepare_update.call_count == 1

    def test_exception_in_loop_caught_warning_logged(self) -> None:
        """Network error inside loop is caught; warning logged; loop continues."""
        fl = self._mock_fl()
        fl.get_round_status.side_effect = ConnectionError("unreachable")

        with patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer"), \
             patch("client.engine.scheduler.log") as mock_log, \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()

        mock_log.warning.assert_called_once_with(
            "scheduler_poll_error", error="unreachable"
        )


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


# ── scheduler state updates ───────────────────────────────────────────────────

class TestWatchStateUpdates:
    """Verify _watch() calls update_state() at the correct phase transitions."""

    def _mock_fl(self) -> MagicMock:
        fl = MagicMock()
        return fl

    def _mock_update(self) -> MagicMock:
        u = MagicMock()
        u.local_metrics = {"flux_ratio": 0.88, "amin_m2": 0.04}
        u.hermia_best_model = "combined_1a"
        return u

    def test_training_phase_set_before_train_call(self) -> None:
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 1, "status": "collecting"}
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.return_value = self._mock_update()
        calls: list[dict] = []

        with patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.update_state", side_effect=lambda **kw: calls.append(kw)), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()

        phases = [c["phase"] for c in calls if "phase" in c]
        assert "training" in phases
        training_idx = phases.index("training")
        # training phase must appear before done
        assert "done" in phases
        done_idx = phases.index("done")
        assert training_idx < done_idx

    def test_done_phase_carries_metrics(self) -> None:
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 2, "status": "collecting"}
        mock_trainer = MagicMock()
        update = self._mock_update()
        mock_trainer.train_and_prepare_update.return_value = update
        calls: list[dict] = []

        with patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.update_state", side_effect=lambda **kw: calls.append(kw)), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()

        done_call = next((c for c in calls if c.get("phase") == "done"), None)
        assert done_call is not None
        assert done_call.get("last_hermia_model") == "combined_1a"
        assert done_call.get("last_flux_ratio") == pytest.approx(0.88)
        assert done_call.get("last_amin") == pytest.approx(0.04)
        assert done_call.get("last_round_completed") == 2

    def test_error_phase_set_on_exception(self) -> None:
        fl = self._mock_fl()
        fl.get_round_status.side_effect = ConnectionError("unreachable")
        calls: list[dict] = []

        with patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer"), \
             patch("client.engine.scheduler.update_state", side_effect=lambda **kw: calls.append(kw)), \
             patch("client.engine.scheduler.log"), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()

        assert any(c.get("phase") == "error" for c in calls)

    def test_uploading_phase_set_before_upload(self) -> None:
        fl = self._mock_fl()
        fl.get_round_status.return_value = {"round_id": 1, "status": "collecting"}
        mock_trainer = MagicMock()
        mock_trainer.train_and_prepare_update.return_value = self._mock_update()
        calls: list[dict] = []

        with patch("client.engine.scheduler.FLClient", return_value=fl), \
             patch("client.engine.scheduler.LocalTrainer", return_value=mock_trainer), \
             patch("client.engine.scheduler.update_state", side_effect=lambda **kw: calls.append(kw)), \
             patch("client.engine.scheduler.time.sleep", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                _watch()

        phases = [c["phase"] for c in calls if "phase" in c]
        assert "uploading" in phases
        assert phases.index("uploading") < phases.index("done")

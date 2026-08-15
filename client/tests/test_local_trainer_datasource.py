"""Tests for LocalTrainer DataSource injection — Task 3."""
from __future__ import annotations
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from client.engine.data_source import DevDataSource, NoNewDataError
from client.engine.local_trainer import LocalTrainer


def _make_mock_source(n: int = 121) -> MagicMock:
    src = MagicMock()
    src.get_data.return_value = (
        np.arange(0, n, dtype=np.float64),
        np.linspace(100.0, 50.0, n),
        np.full(n, 1.0),
    )
    return src


def test_trainer_calls_get_data_once() -> None:
    src = _make_mock_source()
    trainer = LocalTrainer(data_source=src)
    trainer.train_and_prepare_update(round_id=1)
    src.get_data.assert_called_once()


def test_trainer_returns_model_update_with_correct_round() -> None:
    src = _make_mock_source()
    trainer = LocalTrainer(data_source=src)
    update = trainer.train_and_prepare_update(round_id=7)
    assert update.round_id == 7


def test_trainer_n_samples_matches_data_length() -> None:
    src = _make_mock_source(n=80)
    trainer = LocalTrainer(data_source=src)
    update = trainer.train_and_prepare_update(round_id=1)
    assert update.n_samples == 80


def test_trainer_propagates_no_new_data_error() -> None:
    src = MagicMock()
    src.get_data.side_effect = NoNewDataError("no files")
    trainer = LocalTrainer(data_source=src)
    with pytest.raises(NoNewDataError):
        trainer.train_and_prepare_update(round_id=1)


def test_trainer_works_with_dev_source() -> None:
    from client.engine.data_source import PHYSICS_DEFAULTS
    from client.config import get_client_settings
    get_client_settings.cache_clear()
    try:
        with patch.dict("os.environ", {"SITE_ID": "test_site"}):
            ds = DevDataSource(PHYSICS_DEFAULTS)
            trainer = LocalTrainer(data_source=ds)
            update = trainer.train_and_prepare_update(round_id=1)
            assert update.round_id == 1
            assert "hermia_params" in update.delta_W
            assert update.local_metrics["flux_ratio"] > 0
    finally:
        get_client_settings.cache_clear()

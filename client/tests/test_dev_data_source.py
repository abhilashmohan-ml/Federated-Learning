from __future__ import annotations
import numpy as np
import pytest
from client.engine.data_source import DevDataSource, NoNewDataError, PHYSICS_DEFAULTS

_PHYSICS_A = {"J0": 150.0, "k1": 0.015, "k2": 0.0020, "noise": 2.0, "tmp_base": 1.0}
_PHYSICS_B = {"J0": 100.0, "k1": 0.025, "k2": 0.0040, "noise": 2.5, "tmp_base": 1.4}


def test_dev_returns_correct_shape() -> None:
    ds = DevDataSource(_PHYSICS_A)
    time, flux, tmp = ds.get_data()
    assert len(time) == 121
    assert len(flux) == 121
    assert len(tmp) == 121


def test_dev_time_is_0_to_120() -> None:
    ds = DevDataSource(_PHYSICS_A)
    time, _, _ = ds.get_data()
    assert time[0] == pytest.approx(0.0)
    assert time[-1] == pytest.approx(120.0)


def test_dev_flux_always_positive() -> None:
    ds = DevDataSource(_PHYSICS_A)
    _, flux, _ = ds.get_data()
    assert all(flux > 0)


def test_dev_each_call_different() -> None:
    """With non-zero jitter, successive calls return different flux arrays."""
    ds = DevDataSource(_PHYSICS_A, jitter=0.10)
    _, flux1, _ = ds.get_data()
    _, flux2, _ = ds.get_data()
    assert not np.allclose(flux1, flux2)


def test_dev_inter_physics_variance() -> None:
    """Different physics produce clearly different flux curves (zero jitter, base params only)."""
    ds_a = DevDataSource(_PHYSICS_A, jitter=0.0)
    ds_b = DevDataSource(_PHYSICS_B, jitter=0.0)
    _, flux_a, _ = ds_a.get_data()
    _, flux_b, _ = ds_b.get_data()
    # J0 differs by 50% — should not be within 5% relative tolerance
    assert not np.allclose(flux_a, flux_b, rtol=0.05)


def test_dev_zero_jitter_initial_flux_near_j0() -> None:
    """With zero jitter, the first flux value should be approximately J0."""
    ds = DevDataSource(_PHYSICS_A, jitter=0.0)
    _, flux, _ = ds.get_data()
    # flux[0] = J0/(1+k1*0)^2 * exp(-k2*0) + noise ≈ J0 ± noise
    assert flux[0] == pytest.approx(_PHYSICS_A["J0"], abs=10.0)


def test_dev_accepts_arbitrary_physics() -> None:
    """DevDataSource works with any valid physics dict — not tied to site names."""
    custom = {"J0": 200.0, "k1": 0.008, "k2": 0.0005, "noise": 1.0, "tmp_base": 0.9}
    ds = DevDataSource(custom)
    time, flux, tmp = ds.get_data()
    assert len(time) == 121


def test_no_new_data_error_is_exception() -> None:
    assert issubclass(NoNewDataError, Exception)


def test_physics_defaults_has_required_keys() -> None:
    required = {"J0", "k1", "k2", "noise", "tmp_base"}
    assert required.issubset(PHYSICS_DEFAULTS.keys())


def test_physics_defaults_values_are_positive() -> None:
    assert all(v > 0 for v in PHYSICS_DEFAULTS.values())

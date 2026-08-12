"""Unit tests for shared/models/manabe.py — 100% coverage."""
import numpy as np
import pytest

from shared.models.manabe import (
    ManabeResult, capture_probability, compute_lrv, fit_manabe,
)


class TestCaptureProb:
    def test_high_flux_high_capture(self) -> None:
        pc = capture_probability(J=500.0, lambda_=3.0, J_crit=50.0)
        assert pc > 0.99

    def test_zero_flux_zero_capture(self) -> None:
        pc = capture_probability(J=0.0, lambda_=2.5, J_crit=50.0)
        assert pc == pytest.approx(0.0)

    def test_in_range_zero_to_one(self) -> None:
        for J in [10.0, 50.0, 100.0, 300.0]:
            pc = capture_probability(J, lambda_=1.5, J_crit=80.0)
            assert 0.0 <= pc <= 1.0

    def test_returns_float(self) -> None:
        assert isinstance(capture_probability(100.0, 2.0, 80.0), float)

    def test_higher_lambda_higher_capture(self) -> None:
        pc1 = capture_probability(100.0, lambda_=1.0, J_crit=80.0)
        pc2 = capture_probability(100.0, lambda_=3.0, J_crit=80.0)
        assert pc2 > pc1


class TestComputeLRV:
    def test_basic_positive(self) -> None:
        lrv = compute_lrv(Pc=0.9999)
        assert lrv > 4.0

    def test_two_layers_doubles(self) -> None:
        lrv1 = compute_lrv(Pc=0.9, n_layers=1)
        lrv2 = compute_lrv(Pc=0.9, n_layers=2)
        assert lrv2 == pytest.approx(lrv1 * 2)

    def test_pc_exactly_one_clipped_finite(self) -> None:
        """Pc=1.0 would give log(inf); clipping must keep it finite."""
        assert np.isfinite(compute_lrv(Pc=1.0))

    def test_pc_exactly_zero_clipped_finite(self) -> None:
        """Pc=0.0 → log(1/1)=0; clipped value must give a finite result."""
        lrv = compute_lrv(Pc=0.0)
        assert np.isfinite(lrv)
        assert lrv >= 0.0

    def test_returns_float(self) -> None:
        assert isinstance(compute_lrv(0.5), float)

    def test_n_layers_one_is_default(self) -> None:
        assert compute_lrv(0.9, n_layers=1) == compute_lrv(0.9)


class TestFitManabe:
    def _synthetic_data(self):
        """Exact Manabe data with lambda_=2.0, J_crit=80.0, n_layers=1."""
        lambda_, J_crit = 2.0, 80.0
        flux = np.array([50.0, 75.0, 100.0, 125.0, 150.0])
        Pc = np.clip(1.0 - np.exp(-lambda_ * flux / J_crit), 1e-9, 1 - 1e-9)
        lrv = np.log10(1.0 / (1.0 - Pc))
        return flux, lrv

    def test_returns_manabe_result(self) -> None:
        flux, lrv = self._synthetic_data()
        result = fit_manabe(flux, lrv)
        assert isinstance(result, ManabeResult)

    def test_lambda_positive(self) -> None:
        flux, lrv = self._synthetic_data()
        assert fit_manabe(flux, lrv).lambda_ > 0

    def test_j_crit_positive(self) -> None:
        flux, lrv = self._synthetic_data()
        assert fit_manabe(flux, lrv).J_crit > 0

    def test_pc_in_zero_one(self) -> None:
        flux, lrv = self._synthetic_data()
        r = fit_manabe(flux, lrv)
        assert 0.0 <= r.Pc <= 1.0

    def test_lrv_field_positive(self) -> None:
        flux, lrv = self._synthetic_data()
        assert fit_manabe(flux, lrv).LRV > 0

    def test_compliant_false_with_impossible_threshold(self) -> None:
        flux, lrv = self._synthetic_data()
        result = fit_manabe(flux, lrv, lrv_required=100.0)
        assert result.compliant is False
        assert result.LRV_required == 100.0

    def test_compliant_depends_on_lrv(self) -> None:
        flux, lrv = self._synthetic_data()
        result = fit_manabe(flux, lrv, lrv_required=4.0)
        assert result.compliant == (result.LRV >= 4.0)

    def test_n_layers_param_accepted(self) -> None:
        flux, lrv = self._synthetic_data()
        result = fit_manabe(flux, lrv * 2, n_layers=2)
        assert result.LRV > 0

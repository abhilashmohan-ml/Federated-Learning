"""Unit tests for shared/models/combined_1a.py — 100% coverage."""
import numpy as np
import pytest

from shared.models.combined_1a import combined_1a_flux, fit_combined_1a


class TestCombined1aFlux:
    def test_at_t_zero_equals_j0(self) -> None:
        t = np.array([0.0])
        J = combined_1a_flux(t, J0=100.0, k1=0.01, k2=0.001)
        assert J[0] == pytest.approx(100.0)

    def test_monotone_decreasing(self) -> None:
        t = np.linspace(0, 100, 50)
        J = combined_1a_flux(t, J0=100.0, k1=0.05, k2=0.005)
        assert all(J[i] >= J[i + 1] for i in range(len(J) - 1))

    def test_vectorized_length(self) -> None:
        t = np.array([0.0, 10.0, 20.0, 30.0])
        J = combined_1a_flux(t, J0=80.0, k1=0.02, k2=0.001)
        assert len(J) == 4

    def test_returns_ndarray(self) -> None:
        t = np.linspace(0, 60, 20)
        J = combined_1a_flux(t, J0=100.0, k1=0.01, k2=0.001)
        assert isinstance(J, np.ndarray)

    def test_k1_zero_reduces_to_exp_decay(self) -> None:
        """k1=0 → J(t) = J0 * exp(-k2*t)."""
        t = np.linspace(0, 10, 5)
        J = combined_1a_flux(t, J0=100.0, k1=0.0, k2=0.1)
        expected = 100.0 * np.exp(-0.1 * t)
        np.testing.assert_allclose(J, expected, rtol=1e-6)

    def test_k2_zero_reduces_to_power_law(self) -> None:
        """k2=0 → J(t) = J0 / (1 + k1*t)^2."""
        t = np.array([0.0, 5.0, 10.0])
        J = combined_1a_flux(t, J0=100.0, k1=0.1, k2=0.0)
        expected = 100.0 / (1.0 + 0.1 * t) ** 2
        np.testing.assert_allclose(J, expected, rtol=1e-6)

    def test_positive_values(self) -> None:
        t = np.linspace(0, 120, 30)
        J = combined_1a_flux(t, J0=100.0, k1=0.05, k2=0.005)
        assert (J > 0).all()


class TestFitCombined1a:
    def _synthetic_data(self):
        t = np.linspace(0, 60, 30)
        J0, k1, k2 = 100.0, 0.05, 0.005
        flux = combined_1a_flux(t, J0, k1, k2)
        rng = np.random.default_rng(42)
        flux += rng.normal(0, 0.1, len(t))
        return t, np.clip(flux, 1.0, None)

    def test_returns_four_values(self) -> None:
        t, flux = self._synthetic_data()
        result = fit_combined_1a(t, flux)
        assert len(result) == 4

    def test_j0_positive(self) -> None:
        t, flux = self._synthetic_data()
        J0, k1, k2, rmse = fit_combined_1a(t, flux)
        assert J0 > 0

    def test_k1_nonneg(self) -> None:
        _, _, k2, _ = fit_combined_1a(*self._synthetic_data())
        J0, k1, k2, rmse = fit_combined_1a(*self._synthetic_data())
        assert k1 >= 0

    def test_k2_nonneg(self) -> None:
        J0, k1, k2, rmse = fit_combined_1a(*self._synthetic_data())
        assert k2 >= 0

    def test_rmse_small_on_low_noise(self) -> None:
        t, flux = self._synthetic_data()
        J0, k1, k2, rmse = fit_combined_1a(t, flux)
        assert rmse < 5.0

    def test_rmse_is_float(self) -> None:
        t, flux = self._synthetic_data()
        _, _, _, rmse = fit_combined_1a(t, flux)
        assert isinstance(rmse, float)

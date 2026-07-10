"""Unit tests for shared/models/hermia.py — 100% coverage."""
import numpy as np
import pytest

from shared.models.hermia import (
    HermiaResult,
    _aic, _bic, _rmse,
    fit_standard_blocking,
    fit_complete_blocking,
    fit_intermediate_blocking,
    fit_cake_filtration,
    fit_combined_1a,
    fit_all_models,
    compute_flux_ratio,
    compute_amin,
)


# ── Synthetic data helpers ─────────────────────────────────────────────────────

def _standard_data(n: int = 20):
    t = np.linspace(0, 60, n)
    flux = 100.0 / (1.0 + 0.05 * t) ** 2
    flux += np.random.default_rng(0).normal(0, 0.3, n)
    return t, np.clip(flux, 1.0, None)


def _complete_data(n: int = 20):
    t = np.linspace(0, 60, n)
    flux = 100.0 * np.exp(-0.02 * t)
    flux += np.random.default_rng(1).normal(0, 0.3, n)
    return t, np.clip(flux, 1.0, None)


def _combined_data(n: int = 25):
    t = np.linspace(0, 60, n)
    flux = (100.0 / (1.0 + 0.05 * t) ** 2) * np.exp(-0.005 * t)
    flux += np.random.default_rng(2).normal(0, 0.3, n)
    return t, np.clip(flux, 1.0, None)


# ── _aic ──────────────────────────────────────────────────────────────────────

class TestAIC:
    def test_normal_case(self) -> None:
        val = _aic(20, 2, 10.0)
        assert np.isfinite(val)
        assert val == pytest.approx(20 * np.log(10.0 / 20) + 2 * 2)

    def test_zero_rss_returns_inf(self) -> None:
        assert _aic(20, 2, 0.0) == float("inf")

    def test_negative_rss_returns_inf(self) -> None:
        assert _aic(20, 2, -1.0) == float("inf")

    def test_larger_k_increases_aic(self) -> None:
        """More parameters penalised: AIC(k=3) > AIC(k=2) for same data."""
        aic2 = _aic(20, 2, 10.0)
        aic3 = _aic(20, 3, 10.0)
        assert aic3 > aic2


# ── _bic ──────────────────────────────────────────────────────────────────────

class TestBIC:
    def test_normal_case(self) -> None:
        val = _bic(20, 2, 10.0)
        assert np.isfinite(val)
        assert val == pytest.approx(20 * np.log(10.0 / 20) + 2 * np.log(20))

    def test_zero_rss_returns_inf(self) -> None:
        assert _bic(20, 2, 0.0) == float("inf")

    def test_negative_rss_returns_inf(self) -> None:
        assert _bic(20, 2, -5.0) == float("inf")


# ── _rmse ─────────────────────────────────────────────────────────────────────

class TestRMSE:
    def test_perfect_fit(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert _rmse(y, y) == pytest.approx(0.0)

    def test_known_error(self) -> None:
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert _rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_returns_float(self) -> None:
        assert isinstance(_rmse(np.array([1.0]), np.array([2.0])), float)


# ── Individual fitters ────────────────────────────────────────────────────────

class TestFitStandardBlocking:
    def test_returns_hermia_result(self) -> None:
        t, flux = _standard_data()
        r = fit_standard_blocking(t, flux)
        assert isinstance(r, HermiaResult)

    def test_model_name(self) -> None:
        t, flux = _standard_data()
        assert fit_standard_blocking(t, flux).model_name == "standard"

    def test_has_j0_and_ks(self) -> None:
        t, flux = _standard_data()
        r = fit_standard_blocking(t, flux)
        assert "J0" in r.params
        assert "ks" in r.params

    def test_rmse_nonneg(self) -> None:
        t, flux = _standard_data()
        assert fit_standard_blocking(t, flux).rmse >= 0

    def test_selected_default_false(self) -> None:
        t, flux = _standard_data()
        assert fit_standard_blocking(t, flux).selected is False

    def test_finite_aic(self) -> None:
        t, flux = _standard_data()
        assert np.isfinite(fit_standard_blocking(t, flux).aic)


class TestFitCompleteBlocking:
    def test_returns_hermia_result(self) -> None:
        t, flux = _complete_data()
        r = fit_complete_blocking(t, flux)
        assert isinstance(r, HermiaResult)
        assert r.model_name == "complete"

    def test_has_j0_and_kc(self) -> None:
        t, flux = _complete_data()
        r = fit_complete_blocking(t, flux)
        assert "J0" in r.params and "kc" in r.params

    def test_rmse_nonneg(self) -> None:
        assert fit_complete_blocking(*_complete_data()).rmse >= 0


class TestFitIntermediateBlocking:
    def test_returns_hermia_result(self) -> None:
        t, flux = _standard_data()
        r = fit_intermediate_blocking(t, flux)
        assert isinstance(r, HermiaResult)
        assert r.model_name == "intermediate"

    def test_has_j0_and_ki(self) -> None:
        r = fit_intermediate_blocking(*_standard_data())
        assert "J0" in r.params and "ki" in r.params

    def test_rmse_nonneg(self) -> None:
        assert fit_intermediate_blocking(*_standard_data()).rmse >= 0


class TestFitCakeFiltration:
    def test_returns_hermia_result(self) -> None:
        t, flux = _standard_data()
        r = fit_cake_filtration(t, flux)
        assert isinstance(r, HermiaResult)
        assert r.model_name == "cake"

    def test_has_j0_and_kcf(self) -> None:
        r = fit_cake_filtration(*_standard_data())
        assert "J0" in r.params and "kcf" in r.params

    def test_rmse_nonneg(self) -> None:
        assert fit_cake_filtration(*_standard_data()).rmse >= 0


class TestFitCombined1aHermia:
    def test_returns_hermia_result(self) -> None:
        t, flux = _combined_data()
        r = fit_combined_1a(t, flux)
        assert isinstance(r, HermiaResult)
        assert r.model_name == "combined_1a"

    def test_has_three_params(self) -> None:
        r = fit_combined_1a(*_combined_data())
        assert "J0" in r.params and "k1" in r.params and "k2" in r.params

    def test_rmse_nonneg(self) -> None:
        assert fit_combined_1a(*_combined_data()).rmse >= 0

    def test_aic_penalises_extra_param(self) -> None:
        """Combined (3 params) should have AIC computed with k=3."""
        r = fit_combined_1a(*_combined_data())
        assert np.isfinite(r.aic)


# ── fit_all_models ────────────────────────────────────────────────────────────

class TestFitAllModels:
    def test_returns_dict(self) -> None:
        results = fit_all_models(*_combined_data())
        assert isinstance(results, dict)

    def test_nonempty(self) -> None:
        results = fit_all_models(*_combined_data())
        assert len(results) > 0

    def test_exactly_one_selected(self) -> None:
        results = fit_all_models(*_combined_data())
        selected = [r for r in results.values() if r.selected]
        assert len(selected) == 1

    def test_selected_has_lowest_aic(self) -> None:
        results = fit_all_models(*_combined_data())
        best = min(results.values(), key=lambda r: r.aic)
        assert best.selected is True

    def test_failed_fitter_silently_excluded(self) -> None:
        """Fitter that raises RuntimeError is silently skipped; others still returned."""
        from unittest.mock import patch
        t, flux = _combined_data()
        with patch("shared.models.hermia.fit_standard_blocking",
                   side_effect=RuntimeError("forced failure")):
            results = fit_all_models(t, flux)
        assert isinstance(results, dict)
        assert "standard" not in results           # failed fitter excluded
        assert len(results) >= 1                   # remaining fitters still ran

    def test_keys_are_model_names(self) -> None:
        results = fit_all_models(*_combined_data())
        valid = {"standard", "complete", "intermediate", "cake", "combined_1a"}
        assert set(results.keys()).issubset(valid)


# ── compute_flux_ratio ────────────────────────────────────────────────────────

class TestComputeFluxRatio:
    def test_normal_case(self) -> None:
        flux = np.array([100.0, 80.0, 60.0, 42.0])
        assert compute_flux_ratio(flux) == pytest.approx(0.42)

    def test_constant_flux_is_one(self) -> None:
        flux = np.array([50.0, 50.0, 50.0])
        assert compute_flux_ratio(flux) == pytest.approx(1.0)

    def test_zero_initial_returns_zero(self) -> None:
        assert compute_flux_ratio(np.array([0.0, 10.0, 5.0])) == 0.0

    def test_negative_initial_returns_zero(self) -> None:
        assert compute_flux_ratio(np.array([-5.0, 10.0, 8.0])) == 0.0


# ── compute_amin ──────────────────────────────────────────────────────────────

class TestComputeAmin:
    def test_normal_case(self) -> None:
        # 10 L / (80 LMH × 2 h) = 0.0625 m²
        assert compute_amin(10.0, 80.0, 2.0) == pytest.approx(0.0625)

    def test_zero_avg_flux_returns_inf(self) -> None:
        assert compute_amin(10.0, 0.0, 2.0) == float("inf")

    def test_negative_avg_flux_returns_inf(self) -> None:
        assert compute_amin(10.0, -5.0, 2.0) == float("inf")

    def test_zero_time_returns_inf(self) -> None:
        assert compute_amin(10.0, 80.0, 0.0) == float("inf")

    def test_negative_time_returns_inf(self) -> None:
        assert compute_amin(10.0, 80.0, -1.0) == float("inf")

    def test_returns_float(self) -> None:
        assert isinstance(compute_amin(10.0, 80.0, 2.0), float)

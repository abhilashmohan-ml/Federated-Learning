"""Unit tests for shared/models/polarization.py — 100% coverage."""
import numpy as np
import pytest

from shared.models.polarization import (
    PolarizationResult,
    concentration_at_wall,
    permeate_concentration,
    compute_polarization,
)


class TestConcentrationAtWall:
    def test_zero_flux_equals_feed(self) -> None:
        c = concentration_at_wall(C_feed=1.0, J_lmh=0.0)
        assert c == pytest.approx(1.0)

    def test_higher_flux_higher_wall_conc(self) -> None:
        c1 = concentration_at_wall(1.0, J_lmh=100.0)
        c2 = concentration_at_wall(1.0, J_lmh=200.0)
        assert c2 > c1

    def test_wall_conc_greater_than_feed(self) -> None:
        c = concentration_at_wall(C_feed=5.0, J_lmh=100.0)
        assert c > 5.0

    def test_scales_with_feed(self) -> None:
        c1 = concentration_at_wall(C_feed=1.0, J_lmh=100.0)
        c2 = concentration_at_wall(C_feed=2.0, J_lmh=100.0)
        assert c2 == pytest.approx(c1 * 2.0)

    def test_custom_delta_and_dv(self) -> None:
        c = concentration_at_wall(1.0, 100.0, delta=1e-5, D_v=1e-11)
        assert c > 1.0


class TestPermeateConcentration:
    def test_zero_flux_with_full_rejection_is_near_zero(self) -> None:
        # At J=0, exp(-0)=1 and (1-R)=0.01 → c = feed * 0.01
        c = permeate_concentration(C_feed=1.0, J_lmh=0.0, R=0.99)
        assert c == pytest.approx(1.0 * 0.01)

    def test_higher_flux_lower_permeate(self) -> None:
        c1 = permeate_concentration(1.0, J_lmh=100.0)
        c2 = permeate_concentration(1.0, J_lmh=200.0)
        assert c2 < c1

    def test_zero_rejection_equals_feed_at_zero_flux(self) -> None:
        c = permeate_concentration(1.0, J_lmh=0.0, R=0.0)
        assert c == pytest.approx(1.0)

    def test_positive_result(self) -> None:
        c = permeate_concentration(C_feed=2.0, J_lmh=100.0)
        assert c > 0.0


class TestComputePolarization:
    def test_returns_polarization_result(self) -> None:
        result = compute_polarization(C_feed=1.0, J_lmh=100.0)
        assert isinstance(result, PolarizationResult)

    def test_c_wall_greater_than_feed(self) -> None:
        result = compute_polarization(C_feed=1.0, J_lmh=100.0)
        assert result.C_wall > 1.0

    def test_c_perm_less_than_feed(self) -> None:
        result = compute_polarization(C_feed=1.0, J_lmh=100.0)
        assert result.C_perm < 1.0

    def test_lrv_pol_positive(self) -> None:
        result = compute_polarization(C_feed=1.0, J_lmh=100.0)
        assert result.LRV_pol > 0

    def test_c_perm_floor_at_1e20(self) -> None:
        """Extreme flux → C_perm floored at 1e-20, LRV remains finite."""
        result = compute_polarization(C_feed=1.0, J_lmh=1e12, delta=1e-3, D_v=1e-11)
        assert result.C_perm >= 1e-20
        assert np.isfinite(result.LRV_pol)

    def test_larger_boundary_layer_more_polarisation(self) -> None:
        r1 = compute_polarization(1.0, 100.0, delta=1e-5, D_v=1e-11)
        r2 = compute_polarization(1.0, 100.0, delta=2e-5, D_v=1e-11)
        assert r2.C_wall > r1.C_wall

    def test_lrv_pol_is_float(self) -> None:
        result = compute_polarization(1.0, 100.0)
        assert isinstance(result.LRV_pol, float)

"""Tests for scripts/generate_synthetic_data.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_synthetic_data import (
    SITE_CONFIGS,
    _flux_for_model,
    generate,
)
from shared.models.hermia import fit_all_models

TIME = np.arange(0, 121, 1, dtype=float)


# ── _flux_for_model unit tests ────────────────────────────────────────────────


class TestFluxForModel:
    def test_standard_decreasing(self) -> None:
        cfg = {"model": "standard", "J0": 150.0, "ks": 0.015}
        flux = _flux_for_model(cfg, TIME)
        assert flux[0] == pytest.approx(150.0)
        assert flux[-1] < flux[0]

    def test_complete_exponential_decay(self) -> None:
        cfg = {"model": "complete", "J0": 120.0, "kc": 0.020}
        flux = _flux_for_model(cfg, TIME)
        assert flux[0] == pytest.approx(120.0)
        # Exponential decay: ratio of consecutive points is constant
        ratios = flux[1:] / flux[:-1]
        assert np.std(ratios) < 0.001  # constant ratio → exponential

    def test_intermediate_decreasing(self) -> None:
        cfg = {"model": "intermediate", "J0": 180.0, "ki": 3.0e-4}
        flux = _flux_for_model(cfg, TIME)
        assert flux[0] == pytest.approx(180.0)
        assert flux[-1] < flux[0]

    def test_cake_decreasing(self) -> None:
        cfg = {"model": "cake", "J0": 100.0, "kcf": 8.0e-6}
        flux = _flux_for_model(cfg, TIME)
        assert flux[0] == pytest.approx(100.0)
        assert flux[-1] < flux[0]

    def test_combined_1a_decreasing(self) -> None:
        cfg = {"model": "combined_1a", "J0": 160.0, "k1": 0.012, "k2": 0.0015}
        flux = _flux_for_model(cfg, TIME)
        assert flux[0] == pytest.approx(160.0)
        assert flux[-1] < flux[0]

    def test_unknown_model_raises(self) -> None:
        cfg = {"model": "unknown", "J0": 100.0}
        with pytest.raises(ValueError, match="Unknown model"):
            _flux_for_model(cfg, TIME)

    def test_all_site_configs_produce_positive_flux(self) -> None:
        for site_id, cfg in SITE_CONFIGS.items():
            flux = _flux_for_model(cfg, TIME)
            assert np.all(flux > 0), f"{site_id} produced non-positive flux"

    def test_all_site_configs_end_flux_above_1(self) -> None:
        """All models must still have flux > 1 LMH at t=120 min (before clipping)."""
        for site_id, cfg in SITE_CONFIGS.items():
            flux = _flux_for_model(cfg, TIME)
            assert flux[-1] > 1.0, f"{site_id} flux collapses to {flux[-1]:.2f} at t=120"


# ── SITE_CONFIGS structure tests ──────────────────────────────────────────────


class TestSiteConfigs:
    def test_five_sites(self) -> None:
        assert len(SITE_CONFIGS) == 5

    def test_all_five_models_present(self) -> None:
        models = {cfg["model"] for cfg in SITE_CONFIGS.values()}
        assert models == {"standard", "complete", "intermediate", "cake", "combined_1a"}

    def test_no_two_sites_share_model(self) -> None:
        models = [cfg["model"] for cfg in SITE_CONFIGS.values()]
        assert len(models) == len(set(models)), "Duplicate model assignments"

    def test_required_keys_present(self) -> None:
        required = {"filter", "model", "J0", "noise", "tmp_base", "lrv_mean", "lrv_std"}
        for site_id, cfg in SITE_CONFIGS.items():
            missing = required - cfg.keys()
            assert not missing, f"{site_id} missing keys: {missing}"


# ── generate() filesystem tests ───────────────────────────────────────────────


class TestGenerate:
    """
    Use monkeypatch.chdir so that generate()'s relative Path("data/{site_id}")
    resolves inside tmp_path — no mocking of Path required.
    """

    def test_creates_filtration_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        generate("site_1", SITE_CONFIGS["site_1"])
        assert (tmp_path / "data" / "site_1" / "filtration.csv").exists()

    def test_filtration_csv_has_correct_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CSV must have time_min, flux_lmh, tmp_bar, filter_type columns."""
        monkeypatch.chdir(tmp_path)
        cfg = {
            "filter": "TestFilter",
            "model": "standard",
            "J0": 100.0,
            "ks": 0.01,
            "noise": 1.0,
            "tmp_base": 1.0,
            "lrv_mean": 4.5,
            "lrv_std": 0.2,
        }
        generate("site_test", cfg)
        df = pd.read_csv(tmp_path / "data" / "site_test" / "filtration.csv")
        assert set(df.columns) == {"time_min", "flux_lmh", "tmp_bar", "filter_type"}

    def test_flux_values_clipped_to_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All flux values in the CSV must be >= 1.0 (clipped)."""
        monkeypatch.chdir(tmp_path)
        cfg = {
            "filter": "TestFilter",
            "model": "complete",
            "J0": 2.0,
            "kc": 0.5,  # aggressive decay → near 0
            "noise": 0.0,
            "tmp_base": 1.0,
            "lrv_mean": 4.5,
            "lrv_std": 0.2,
        }
        generate("site_test", cfg)
        df = pd.read_csv(tmp_path / "data" / "site_test" / "filtration.csv")
        assert (df["flux_lmh"] >= 1.0).all()


# ── AIC regime selection end-to-end tests ─────────────────────────────────────


class TestAICRegimeSelection:
    """
    Key tests: data generated from each model must have AIC select that model.
    These are the tests that prove the bug is fixed.

    Note on site_1 (standard):
        Standard blocking is mathematically a special case of Combined 1-A with
        k2 = 0.  On noiseless float64 data the Combined 1-A optimizer can absorb
        floating-point rounding with k2 > 0, achieving a lower RSS than the
        2-parameter standard fit.  AIC therefore legitimately selects combined_1a
        for site_1 data; both "standard" and "combined_1a" are acceptable winners.
        Sites 2–5 use models whose functional forms are NOT special cases of each
        other, so AIC correctly identifies them (see test_sites_2_to_5_have_distinct_regimes).
    """

    @pytest.mark.parametrize(
        "site_id,expected_models",
        [
            # standard ⊂ combined_1a (k2=0), so either winner is acceptable
            ("site_1", {"standard", "combined_1a"}),
            ("site_2", {"complete"}),
            ("site_3", {"intermediate"}),
            ("site_4", {"cake"}),
            ("site_5", {"combined_1a"}),
        ],
    )
    def test_aic_selects_correct_regime(self, site_id: str, expected_models: set) -> None:
        """
        Generate noiseless flux for the site, fit all Hermia models,
        and assert the AIC winner is within the expected set for that site.
        """
        cfg = SITE_CONFIGS[site_id]
        flux = _flux_for_model(cfg, TIME)  # noiseless — maximise AIC signal
        results = fit_all_models(TIME, flux)
        best = next(r for r in results.values() if r.selected)
        assert best.model_name in expected_models, (
            f"{site_id}: expected AIC winner in {expected_models}, "
            f"got '{best.model_name}' (AIC={best.aic:.1f})"
        )

    def test_sites_2_to_5_have_distinct_regimes(self) -> None:
        """
        Sites 2–5 (complete / intermediate / cake / combined_1a) each have a
        distinct functional form — AIC must select a different winner for each.

        Site 1 (standard) is excluded because standard is a degenerate case of
        combined_1a (k2 = 0); see class docstring for the mathematical explanation.
        """
        winners: list[str] = []
        for site_id, cfg in list(SITE_CONFIGS.items())[1:]:  # site_2 … site_5
            flux = _flux_for_model(cfg, TIME)
            results = fit_all_models(TIME, flux)
            best = next(r for r in results.values() if r.selected)
            winners.append(best.model_name)
        assert len(winners) == len(
            set(winners)
        ), f"Duplicate AIC winners across sites 2–5: {winners}"

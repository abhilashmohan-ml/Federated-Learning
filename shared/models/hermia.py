"""
Hermia blocking models for membrane filtration.

Models implemented:
  - Standard blocking     (pore constriction)
  - Complete blocking     (pore sealing)
  - Intermediate blocking (partial pore sealing)
  - Cake filtration       (surface cake layer)
  - Combined 1-A          (most general; recommended for mAb VF)

Each fit_*() returns a HermiaResult with params, AIC, BIC, RMSE.
fit_all_models() fits all and flags the best by AIC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from scipy.optimize import curve_fit

from shared.utils.constants import (
    J_MIN, J_MAX,
    KS_BOUNDS, KI_BOUNDS, KC_BOUNDS, KCF_BOUNDS, K1_BOUNDS, K2_BOUNDS,
)


@dataclass
class HermiaResult:
    model_name: str
    params: Dict[str, float]
    aic: float
    bic: float
    rmse: float
    selected: bool = False    # True => lowest AIC across all models


# ── Information criteria ──────────────────────────────────────────────────────

def _aic(n: int, k: int, rss: float) -> float:
    if rss <= 0:
        return float("inf")
    return n * np.log(rss / n) + 2.0 * k


def _bic(n: int, k: int, rss: float) -> float:
    if rss <= 0:
        return float("inf")
    return n * np.log(rss / n) + k * np.log(n)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ── Individual model fitters ──────────────────────────────────────────────────

def fit_standard_blocking(time: np.ndarray, flux: np.ndarray) -> HermiaResult:
    """1/sqrt(J) = 1/sqrt(J0) + ks*t  =>  J(t) = J0 / (1 + ks*t)^2"""
    def _model(t: np.ndarray, J0: float, ks: float) -> np.ndarray:
        return J0 / (1.0 + ks * t) ** 2

    popt, _ = curve_fit(
        _model, time, flux,
        p0=[flux[0], 0.01],
        bounds=([J_MIN, KS_BOUNDS[0]], [J_MAX, KS_BOUNDS[1]]),
        maxfev=5000,
    )
    J0, ks = popt
    pred = _model(time, J0, ks)
    rss = float(np.sum((flux - pred) ** 2))
    n = len(flux)
    return HermiaResult(
        model_name="standard",
        params={"J0": J0, "ks": ks},
        aic=_aic(n, 2, rss), bic=_bic(n, 2, rss), rmse=_rmse(flux, pred),
    )


def fit_complete_blocking(time: np.ndarray, flux: np.ndarray) -> HermiaResult:
    """J = J0 * exp(-kc * t)"""
    def _model(t: np.ndarray, J0: float, kc: float) -> np.ndarray:
        return J0 * np.exp(-kc * t)

    popt, _ = curve_fit(
        _model, time, flux,
        p0=[flux[0], 0.01],
        bounds=([J_MIN, KC_BOUNDS[0]], [J_MAX, KC_BOUNDS[1]]),
        maxfev=5000,
    )
    J0, kc = popt
    pred = _model(time, J0, kc)
    rss = float(np.sum((flux - pred) ** 2))
    n = len(flux)
    return HermiaResult(
        model_name="complete",
        params={"J0": J0, "kc": kc},
        aic=_aic(n, 2, rss), bic=_bic(n, 2, rss), rmse=_rmse(flux, pred),
    )


def fit_intermediate_blocking(time: np.ndarray, flux: np.ndarray) -> HermiaResult:
    """1/J = 1/J0 + ki*t  =>  J(t) = J0 / (1 + J0*ki*t)"""
    def _model(t: np.ndarray, J0: float, ki: float) -> np.ndarray:
        return J0 / (1.0 + J0 * ki * t)

    popt, _ = curve_fit(
        _model, time, flux,
        p0=[flux[0], 1e-4],
        bounds=([J_MIN, KI_BOUNDS[0]], [J_MAX, KI_BOUNDS[1]]),
        maxfev=5000,
    )
    J0, ki = popt
    pred = _model(time, J0, ki)
    rss = float(np.sum((flux - pred) ** 2))
    n = len(flux)
    return HermiaResult(
        model_name="intermediate",
        params={"J0": J0, "ki": ki},
        aic=_aic(n, 2, rss), bic=_bic(n, 2, rss), rmse=_rmse(flux, pred),
    )


def fit_cake_filtration(time: np.ndarray, flux: np.ndarray) -> HermiaResult:
    """1/J^2 = 1/J0^2 + kcf*t  =>  J(t) = J0 / sqrt(1 + J0^2*kcf*t)"""
    def _model(t: np.ndarray, J0: float, kcf: float) -> np.ndarray:
        return J0 / np.sqrt(1.0 + J0 ** 2 * kcf * t)

    popt, _ = curve_fit(
        _model, time, flux,
        p0=[flux[0], 1e-6],
        bounds=([J_MIN, KCF_BOUNDS[0]], [J_MAX, KCF_BOUNDS[1]]),
        maxfev=5000,
    )
    J0, kcf = popt
    pred = _model(time, J0, kcf)
    rss = float(np.sum((flux - pred) ** 2))
    n = len(flux)
    return HermiaResult(
        model_name="cake",
        params={"J0": J0, "kcf": kcf},
        aic=_aic(n, 2, rss), bic=_bic(n, 2, rss), rmse=_rmse(flux, pred),
    )


def fit_combined_1a(time: np.ndarray, flux: np.ndarray) -> HermiaResult:
    """J(t) = J0 / (1 + k1*t)^2 * exp(-k2*t)  [Combined 1-A — most general]"""
    def _model(t: np.ndarray, J0: float, k1: float, k2: float) -> np.ndarray:
        return (J0 / (1.0 + k1 * t) ** 2) * np.exp(-k2 * t)

    popt, _ = curve_fit(
        _model, time, flux,
        p0=[flux[0], 0.01, 0.001],
        bounds=(
            [J_MIN, K1_BOUNDS[0], K2_BOUNDS[0]],
            [J_MAX, K1_BOUNDS[1], K2_BOUNDS[1]],
        ),
        maxfev=5000,
    )
    J0, k1, k2 = popt
    pred = _model(time, J0, k1, k2)
    rss = float(np.sum((flux - pred) ** 2))
    n = len(flux)
    return HermiaResult(
        model_name="combined_1a",
        params={"J0": J0, "k1": k1, "k2": k2},
        aic=_aic(n, 3, rss), bic=_bic(n, 3, rss), rmse=_rmse(flux, pred),
    )


# ── Master fitter ─────────────────────────────────────────────────────────────

def fit_all_models(time: np.ndarray, flux: np.ndarray) -> Dict[str, HermiaResult]:
    """Fit all 5 Hermia models.  Best model (lowest AIC) has .selected = True."""
    fitters = [
        fit_standard_blocking,
        fit_complete_blocking,
        fit_intermediate_blocking,
        fit_cake_filtration,
        fit_combined_1a,
    ]
    results: Dict[str, HermiaResult] = {}
    for fitter in fitters:
        try:
            r = fitter(time, flux)
            results[r.model_name] = r
        except Exception:
            pass  # skip failed fits silently

    if results:
        best = min(results.values(), key=lambda r: r.aic)
        best.selected = True
    return results


# ── Derived metrics ───────────────────────────────────────────────────────────

def compute_flux_ratio(flux: np.ndarray) -> float:
    """J_final / J_initial  (fouling severity indicator)."""
    if flux[0] <= 0:
        return 0.0
    return float(flux[-1] / flux[0])


def compute_amin(
    target_throughput_L: float,
    avg_flux_lmh: float,
    operation_time_h: float,
) -> float:
    """Minimum filter area [m^2] = throughput / (avg_flux * time)."""
    if avg_flux_lmh <= 0 or operation_time_h <= 0:
        return float("inf")
    return target_throughput_L / (avg_flux_lmh * operation_time_h)

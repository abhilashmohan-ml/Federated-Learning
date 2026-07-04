"""
Hermia blocking models for membrane filtration.

OVERVIEW — THE HERMIA FRAMEWORK
---------------------------------
Professor Hermia (1982) proposed a unified framework for describing how a
membrane filter "fouls" — i.e. how its performance degrades over time as the
drug product blocks the filter.

There are four fundamental fouling mechanisms, each with a different mathematical
form. In practice, real filtration runs often show a mix of mechanisms, so we
also implement the Combined 1-A model (Zydney, Bolton et al.) which blends two.

The five models this file implements:
  1. Standard blocking   — pores gradually narrow (constriction)
  2. Complete blocking   — pores are entirely sealed one by one
  3. Intermediate blocking — pores are partially sealed
  4. Cake filtration     — a filtration "cake" builds up on the surface
  5. Combined 1-A        — simultaneous pore constriction + cake (most general)

MODEL SELECTION BY AIC
-----------------------
We fit ALL five models to the same data and select the best one using the
Akaike Information Criterion (AIC). The AIC rewards models that fit the data
well but penalises models with more parameters (to prevent overfitting).

    AIC = n × ln(RSS / n) + 2k

    n   = number of data points
    k   = number of parameters in the model
    RSS = residual sum of squares (total squared error)

LOWER AIC = BETTER MODEL. A model that fits equally well with fewer parameters
wins because it is a more parsimonious description of the data.

PYTHON CONCEPT: @dataclass
  Automatically creates an __init__ method from the annotated fields.

PYTHON CONCEPT: Dict[str, float]
  A dictionary mapping parameter names (strings) to their fitted values (floats).
  For Standard: {"J0": 42.3, "ks": 0.012}
  For Combined: {"J0": 42.3, "k1": 0.012, "k2": 0.0003}

PYTHON CONCEPT: curve_fit
  `scipy.optimize.curve_fit` solves the non-linear least squares problem:
  find parameter values p* such that Σ(y_measured - model(x, p*))² is minimised.
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
    """
    Results from fitting one Hermia model to flux-decline data.

    Attributes
    ----------
    model_name : str   — which model this is: "standard", "complete", etc.
    params     : dict  — fitted parameter values, e.g. {"J0": 42.3, "ks": 0.012}
    aic        : float — Akaike Information Criterion (lower is better)
    bic        : float — Bayesian Information Criterion (lower is better; penalises
                         more parameters more strongly than AIC for large datasets)
    rmse       : float — Root Mean Square Error in LMH (lower is better)
    selected   : bool  — True if this model has the lowest AIC across all models
                         (set by fit_all_models, not by the individual fitters)
    """
    model_name: str
    params:     Dict[str, float]
    aic:        float
    bic:        float
    rmse:       float
    selected:   bool = False   # default False; set to True by fit_all_models()


# ── Information criteria ───────────────────────────────────────────────────────
#
# These helper functions compute model quality metrics.
# They are prefixed with _ to indicate they are private (internal to this module).
# By convention, functions/variables starting with _ are not meant to be imported
# by other modules.

def _aic(n: int, k: int, rss: float) -> float:
    """
    Akaike Information Criterion.

    AIC = n × ln(RSS / n) + 2k

    Parameters
    ----------
    n   : int   — number of data points (time/flux pairs)
    k   : int   — number of model parameters (2 for most Hermia models, 3 for Combined 1-A)
    rss : float — residual sum of squares: Σ(y_measured - y_predicted)²

    Returns
    -------
    float — AIC value; lower is better. Returns infinity if rss <= 0.
    """
    if rss <= 0:
        return float("inf")  # invalid RSS means the model cannot be compared
    return n * np.log(rss / n) + 2.0 * k


def _bic(n: int, k: int, rss: float) -> float:
    """
    Bayesian Information Criterion.

    BIC = n × ln(RSS / n) + k × ln(n)

    BIC penalises parameters more strongly than AIC when n (data count) is large.
    We compute and store BIC alongside AIC; model selection is always by AIC.

    Parameters
    ----------
    n   : int   — number of data points
    k   : int   — number of parameters
    rss : float — residual sum of squares

    Returns
    -------
    float — BIC value; lower is better.
    """
    if rss <= 0:
        return float("inf")
    return n * np.log(rss / n) + k * np.log(n)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Error.

    RMSE = sqrt( mean( (y_true - y_pred)² ) )

    Gives the typical magnitude of error in the same units as y (LMH here).
    An RMSE of 2.0 LMH means the model is typically off by about 2 LMH.

    Parameters
    ----------
    y_true : np.ndarray — measured values
    y_pred : np.ndarray — model-predicted values (must be same length)

    Returns
    -------
    float — RMSE in the same units as the input arrays
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ── Individual model fitters ───────────────────────────────────────────────────
#
# Each fitter below follows the same pattern:
#   1. Define the model function (what J(t) looks like)
#   2. Call curve_fit to find the best parameters
#   3. Calculate AIC, BIC, RMSE
#   4. Return a HermiaResult

def fit_standard_blocking(time: np.ndarray, flux: np.ndarray) -> HermiaResult:
    """
    Fit the Standard Blocking (pore constriction) model.

    Physical interpretation:
      Particles deposit on the pore walls, gradually narrowing the pores.
      The flux decreases as (1 + ks·t)⁻² — a power-law decay.

    Equation: J(t) = J0 / (1 + ks·t)²

    Parameters
    ----------
    time : np.ndarray — time in minutes
    flux : np.ndarray — measured flux in LMH (same length as time)

    Returns
    -------
    HermiaResult with model_name="standard", params={"J0":..., "ks":...}
    """
    # Define the model inline using a "nested function" (also called a closure)
    def _model(t: np.ndarray, J0: float, ks: float) -> np.ndarray:
        return J0 / (1.0 + ks * t) ** 2

    # curve_fit finds J0 and ks that minimise the sum of squared residuals
    popt, _ = curve_fit(
        _model, time, flux,
        p0=[flux[0], 0.01],   # initial guess: J0 ≈ first measured flux
        bounds=([J_MIN, KS_BOUNDS[0]], [J_MAX, KS_BOUNDS[1]]),
        maxfev=5000,
    )
    J0, ks = popt   # unpack the two fitted values

    pred = _model(time, J0, ks)                   # predicted flux values
    rss  = float(np.sum((flux - pred) ** 2))      # residual sum of squares
    n    = len(flux)                               # number of data points (for AIC/BIC)

    return HermiaResult(
        model_name="standard",
        params={"J0": J0, "ks": ks},
        aic=_aic(n, 2, rss),       # 2 parameters: J0 and ks
        bic=_bic(n, 2, rss),
        rmse=_rmse(flux, pred),
    )


def fit_complete_blocking(time: np.ndarray, flux: np.ndarray) -> HermiaResult:
    """
    Fit the Complete Blocking (pore sealing) model.

    Physical interpretation:
      Each particle that arrives at the membrane completely seals one pore.
      The number of open pores — and hence flux — decays exponentially.

    Equation: J(t) = J0 × exp(-kc·t)

    The exponential form is the simplest fouling model and applies when
    every particle independently and permanently blocks a pore it lands on.
    """
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
    rss  = float(np.sum((flux - pred) ** 2))
    n    = len(flux)

    return HermiaResult(
        model_name="complete",
        params={"J0": J0, "kc": kc},
        aic=_aic(n, 2, rss),
        bic=_bic(n, 2, rss),
        rmse=_rmse(flux, pred),
    )


def fit_intermediate_blocking(time: np.ndarray, flux: np.ndarray) -> HermiaResult:
    """
    Fit the Intermediate Blocking model.

    Physical interpretation:
      Particles land on both open membrane surface AND already-blocked areas.
      Not every particle blocks a pore — blocking is probabilistic.
      The reciprocal of flux (1/J) grows linearly with time.

    Equation: J(t) = J0 / (1 + J0·ki·t)

    This is derived from:  1/J(t) = 1/J0 + ki·t
    (linear growth of hydraulic resistance with time)
    """
    def _model(t: np.ndarray, J0: float, ki: float) -> np.ndarray:
        return J0 / (1.0 + J0 * ki * t)

    popt, _ = curve_fit(
        _model, time, flux,
        p0=[flux[0], 1e-4],   # smaller initial guess for ki — intermediate is slow
        bounds=([J_MIN, KI_BOUNDS[0]], [J_MAX, KI_BOUNDS[1]]),
        maxfev=5000,
    )
    J0, ki = popt
    pred = _model(time, J0, ki)
    rss  = float(np.sum((flux - pred) ** 2))
    n    = len(flux)

    return HermiaResult(
        model_name="intermediate",
        params={"J0": J0, "ki": ki},
        aic=_aic(n, 2, rss),
        bic=_bic(n, 2, rss),
        rmse=_rmse(flux, pred),
    )


def fit_cake_filtration(time: np.ndarray, flux: np.ndarray) -> HermiaResult:
    """
    Fit the Cake Filtration model.

    Physical interpretation:
      A compressible or incompressible "cake" layer accumulates on the membrane
      surface. The cake adds resistance proportional to its thickness, which
      grows linearly with volume filtered (and hence with time at constant flux).
      The reciprocal of J² grows linearly with time.

    Equation: J(t) = J0 / sqrt(1 + J0²·kcf·t)

    This is derived from: 1/J(t)² = 1/J0² + kcf·t
    Cake filtration is common when the mAb product itself is the main fouling agent.
    """
    def _model(t: np.ndarray, J0: float, kcf: float) -> np.ndarray:
        return J0 / np.sqrt(1.0 + J0 ** 2 * kcf * t)

    popt, _ = curve_fit(
        _model, time, flux,
        p0=[flux[0], 1e-6],   # very small initial kcf — cake build-up is slow
        bounds=([J_MIN, KCF_BOUNDS[0]], [J_MAX, KCF_BOUNDS[1]]),
        maxfev=5000,
    )
    J0, kcf = popt
    pred = _model(time, J0, kcf)
    rss  = float(np.sum((flux - pred) ** 2))
    n    = len(flux)

    return HermiaResult(
        model_name="cake",
        params={"J0": J0, "kcf": kcf},
        aic=_aic(n, 2, rss),
        bic=_bic(n, 2, rss),
        rmse=_rmse(flux, pred),
    )


def fit_combined_1a(time: np.ndarray, flux: np.ndarray) -> HermiaResult:
    """
    Fit the Combined 1-A model — the most general Hermia-family model.

    Physical interpretation:
      Both pore constriction (k1 term) and surface cake deposition (k2 term)
      occur simultaneously. This model is recommended for mAb viral filtration
      because real filters typically show both mechanisms at different time scales.

    Equation: J(t) = J0 / (1 + k1·t)² × exp(-k2·t)

    The model has THREE parameters (J0, k1, k2), so AIC will penalise it
    slightly vs the 2-parameter models — it only wins if the better fit justifies
    the extra parameter.
    """
    def _model(t: np.ndarray, J0: float, k1: float, k2: float) -> np.ndarray:
        return (J0 / (1.0 + k1 * t) ** 2) * np.exp(-k2 * t)

    popt, _ = curve_fit(
        _model, time, flux,
        p0=[flux[0], 0.01, 0.001],   # three initial guesses
        bounds=(
            [J_MIN, K1_BOUNDS[0], K2_BOUNDS[0]],
            [J_MAX, K1_BOUNDS[1], K2_BOUNDS[1]],
        ),
        maxfev=5000,
    )
    J0, k1, k2 = popt
    pred = _model(time, J0, k1, k2)
    rss  = float(np.sum((flux - pred) ** 2))
    n    = len(flux)

    return HermiaResult(
        model_name="combined_1a",
        params={"J0": J0, "k1": k1, "k2": k2},
        aic=_aic(n, 3, rss),   # 3 parameters — AIC penalises the extra one
        bic=_bic(n, 3, rss),
        rmse=_rmse(flux, pred),
    )


# ── Master fitter ──────────────────────────────────────────────────────────────

def fit_all_models(time: np.ndarray, flux: np.ndarray) -> Dict[str, HermiaResult]:
    """
    Fit all 5 Hermia models and return results, flagging the best by AIC.

    This is the main entry point used by `LocalTrainer`. It:
      1. Runs all five model fitters
      2. Collects their results into a dict keyed by model name
      3. Identifies the model with the lowest AIC and sets its .selected = True

    Parameters
    ----------
    time : np.ndarray — time series in minutes
    flux : np.ndarray — flux series in LMH (same length as time)

    Returns
    -------
    Dict[str, HermiaResult]
        Keys: "standard", "complete", "intermediate", "cake", "combined_1a"
        The HermiaResult with .selected=True has the lowest AIC.
        Models that failed to converge are excluded from the dict.

    Example
    -------
    >>> results = fit_all_models(time_array, flux_array)
    >>> best = next(r for r in results.values() if r.selected)
    >>> print(f"Best model: {best.model_name}, AIC={best.aic:.1f}")
    """
    # List of all fitter functions to try
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
            # curve_fit may raise RuntimeError if it fails to converge.
            # We silently skip failed models — the remaining ones still compete.
            pass

    if results:
        # `min(iterable, key=...)` returns the item with the smallest key value.
        # Here we find the HermiaResult with the minimum AIC.
        best = min(results.values(), key=lambda r: r.aic)
        best.selected = True   # flag this as the winner

    return results


# ── Derived process metrics ────────────────────────────────────────────────────

def compute_flux_ratio(flux: np.ndarray) -> float:
    """
    Calculate the flux ratio: J_final / J_initial.

    This is a simple fouling severity indicator:
      - Ratio close to 1.0 → very little fouling, filter still has lots of capacity
      - Ratio = 0.5 → filter operating at half its initial flux (moderate fouling)
      - Ratio < 0.2 → FLUX_RATIO_MIN threshold — filter is considered exhausted

    Parameters
    ----------
    flux : np.ndarray — flux time series (assumes flux[0] is the initial value
                        and flux[-1] is the final value)

    Returns
    -------
    float — flux ratio in [0, 1]; 0.0 returned if initial flux is zero or negative

    Example
    -------
    >>> flux = np.array([100.0, 85.0, 70.0, 55.0, 42.0])
    >>> compute_flux_ratio(flux)   # 42.0 / 100.0 = 0.42
    """
    if flux[0] <= 0:
        return 0.0   # protect against division by zero
    return float(flux[-1] / flux[0])


def compute_amin(
    target_throughput_L: float,
    avg_flux_lmh: float,
    operation_time_h: float,
) -> float:
    """
    Calculate the minimum filter area (A_min) needed for a target throughput.

    FORMULA
    -------
    A_min [m²] = Throughput [L] / (avg_flux [LMH] × time [h])

    WHY IS THIS IMPORTANT?
    ----------------------
    In process design, engineers must choose how large a filter to use.
    Too small → not enough throughput, the drug product doesn't all pass through.
    Too large → wasted cost.

    A_min gives the theoretical minimum area needed. In practice, a safety factor
    of 1.5–2× is applied.

    Parameters
    ----------
    target_throughput_L  : float — total volume to filter in litres
    avg_flux_lmh         : float — average flux over the run in LMH
    operation_time_h     : float — available filtration time in hours

    Returns
    -------
    float — minimum filter area in square metres (m²)
            Returns infinity if avg_flux or time are zero or negative.

    Example
    -------
    >>> compute_amin(10.0, 80.0, 2.0)   # 10L / (80 LMH × 2h) = 0.0625 m²
    """
    if avg_flux_lmh <= 0 or operation_time_h <= 0:
        return float("inf")   # undefined — cannot filter with zero flux or time
    return target_throughput_L / (avg_flux_lmh * operation_time_h)

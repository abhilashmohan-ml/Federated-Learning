"""
Manabe model for virus capture probability and LRV.

Reference: Manabe S. (1981) virus removal by membrane filtration.

Capture probability (single layer):
    Pc = 1 - exp(-lambda * J / J_crit)

Log Reduction Value:
    LRV = log10(1 / (1 - Pc)) * N_layers
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from shared.utils.constants import PC_BOUNDS, JCRIT_BOUNDS, LAMBDA_BOUNDS


@dataclass
class ManabeResult:
    Pc: float           # capture probability  [0, 1]
    lambda_: float      # membrane affinity parameter
    J_crit: float       # critical flux  [LMH]
    LRV: float          # log reduction value at mean flux
    LRV_required: float # regulatory minimum
    compliant: bool     # LRV >= LRV_required


def capture_probability(J: float, lambda_: float, J_crit: float) -> float:
    """Single-layer capture probability."""
    return float(1.0 - np.exp(-lambda_ * J / J_crit))


def compute_lrv(Pc: float, n_layers: int = 1) -> float:
    """LRV from capture probability across n_layers."""
    Pc_c = float(np.clip(Pc, 1e-9, 1.0 - 1e-9))
    return float(np.log10(1.0 / (1.0 - Pc_c)) * n_layers)


def fit_manabe(
    flux_values: np.ndarray,
    lrv_values: np.ndarray,
    n_layers: int = 1,
    lrv_required: float = 4.0,
) -> ManabeResult:
    """Fit lambda_ and J_crit from (flux, LRV) measurement pairs."""

    def _model_lrv(J: np.ndarray, lambda_: float, J_crit: float) -> np.ndarray:
        Pc = 1.0 - np.exp(-lambda_ * J / J_crit)
        Pc = np.clip(Pc, 1e-9, 1.0 - 1e-9)
        return np.log10(1.0 / (1.0 - Pc)) * n_layers

    popt, _ = curve_fit(
        _model_lrv,
        flux_values,
        lrv_values,
        p0=[1.0, 100.0],
        bounds=(
            [LAMBDA_BOUNDS[0], JCRIT_BOUNDS[0]],
            [LAMBDA_BOUNDS[1], JCRIT_BOUNDS[1]],
        ),
        maxfev=5000,
    )
    lambda_, J_crit = popt
    J_mean = float(np.mean(flux_values))
    Pc = capture_probability(J_mean, lambda_, J_crit)
    LRV = compute_lrv(Pc, n_layers)

    return ManabeResult(
        Pc=Pc,
        lambda_=lambda_,
        J_crit=J_crit,
        LRV=LRV,
        LRV_required=lrv_required,
        compliant=LRV >= lrv_required,
    )

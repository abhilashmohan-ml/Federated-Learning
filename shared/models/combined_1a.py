"""
Combined 1-A flux decline model  (standalone reference implementation).

J(t) = J0 / (1 + k1*t)^2 * exp(-k2*t)

This model captures both pore constriction (k1 term) and
surface adsorption/cake deposition (k2 term).
It is the most general Hermia-family model for mAb viral filtration.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from shared.utils.constants import J_MIN, J_MAX, K1_BOUNDS, K2_BOUNDS


def combined_1a_flux(
    t: np.ndarray,
    J0: float,
    k1: float,
    k2: float,
) -> np.ndarray:
    """Evaluate J(t) for given parameters."""
    return (J0 / (1.0 + k1 * t) ** 2) * np.exp(-k2 * t)


def fit_combined_1a(
    time: np.ndarray,
    flux: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Fit Combined 1-A model to (time, flux) data.

    Returns
    -------
    J0, k1, k2, rmse
    """
    popt, _ = curve_fit(
        combined_1a_flux,
        time,
        flux,
        p0=[flux[0], 0.01, 0.001],
        bounds=(
            [J_MIN, K1_BOUNDS[0], K2_BOUNDS[0]],
            [J_MAX, K1_BOUNDS[1], K2_BOUNDS[1]],
        ),
        maxfev=10000,
    )
    J0, k1, k2 = popt
    pred = combined_1a_flux(time, J0, k1, k2)
    rmse = float(np.sqrt(np.mean((flux - pred) ** 2)))
    return J0, k1, k2, rmse

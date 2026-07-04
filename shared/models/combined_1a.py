"""
Combined 1-A flux decline model — standalone reference implementation.

WHAT IS THIS MODEL?
--------------------
The Combined 1-A model is the most general Hermia-family model for describing
flux decline during mAb viral filtration. It combines two separate fouling
mechanisms into a single equation:

  J(t) = J0 / (1 + k1·t)² × exp(-k2·t)

  J(t)  = flux at time t [LMH]
  J0    = initial flux at t=0 [LMH]
  k1    = pore constriction rate constant — describes how pores gradually narrow
           as the drug product adsorbs onto pore walls. Units: min⁻¹.
  k2    = surface fouling / cake deposition rate constant — describes how a
           layer of material builds up on the membrane surface over time. Units: min⁻¹.
  t     = time [minutes]

WHY TWO TERMS?
--------------
The (1 + k1·t)² term comes from the Standard Hermia model (pore narrowing):
  - At early times, pore constriction dominates
  - The denominator squared gives a rapid initial flux drop

The exp(-k2·t) term adds an exponential decay on top:
  - Models long-term surface fouling and cake build-up
  - At late times this term dominates when k2 > 0

Together they capture the sigmoidal-to-exponential shape often seen in
real mAb viral filtration runs.

NOTE: The same equation is implemented inside the PINN's PhysicsSolver
in `shared/models/pinn.py`. This standalone file exists as a reference
implementation and for use outside the neural network context.

PYTHON CONCEPT: np.ndarray
  Functions in this file accept NumPy arrays for `t` (time). This means
  they can compute flux at many time points simultaneously — much faster
  than looping in Python. This is called "vectorisation."
"""

from __future__ import annotations   # allows forward type references

import numpy as np                   # NumPy: vectorised mathematics
from scipy.optimize import curve_fit # scipy: least-squares curve fitting

from shared.utils.constants import J_MIN, J_MAX, K1_BOUNDS, K2_BOUNDS


def combined_1a_flux(
    t: np.ndarray,
    J0: float,
    k1: float,
    k2: float,
) -> np.ndarray:
    """
    Evaluate the Combined 1-A flux equation at given time points.

    Formula: J(t) = J0 / (1 + k1·t)² × exp(-k2·t)

    This is a pure mathematical evaluation — no fitting, no iteration.
    It takes parameters and time values and returns predicted flux values.

    Parameters
    ----------
    t  : np.ndarray — time points in minutes, e.g. np.array([0, 5, 10, 15, 20])
    J0 : float      — initial flux [LMH], must be > 0
    k1 : float      — pore constriction rate constant [min⁻¹], must be ≥ 0
    k2 : float      — surface fouling rate constant [min⁻¹], must be ≥ 0

    Returns
    -------
    np.ndarray — predicted flux values at each time point [LMH]
                 Same length as t.

    Example
    -------
    >>> t = np.linspace(0, 60, 100)    # 0 to 60 minutes, 100 points
    >>> J = combined_1a_flux(t, J0=100.0, k1=0.01, k2=0.001)
    >>> J[0]   # ≈ 100.0 (initial flux)
    >>> J[-1]  # much lower (fouled filter)
    """
    # (1 + k1·t)² grows over time, making the denominator larger → smaller J
    # exp(-k2·t) decreases over time → multiplies J by a shrinking factor
    return (J0 / (1.0 + k1 * t) ** 2) * np.exp(-k2 * t)


def fit_combined_1a(
    time: np.ndarray,
    flux: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Fit the Combined 1-A model to measured (time, flux) data.

    HOW FITTING WORKS
    -----------------
    `scipy.optimize.curve_fit` uses the Levenberg-Marquardt algorithm
    (a non-linear least-squares method) to find the values of J0, k1, k2
    that minimise:

        RSS = Σ (flux_measured[i] - combined_1a_flux(time[i], J0, k1, k2))²

    It starts from an initial guess (`p0`) and iteratively adjusts the parameters
    until the sum of squared residuals cannot be reduced further (or maxfev is reached).

    Parameters
    ----------
    time : np.ndarray — measured time points [minutes]
    flux : np.ndarray — measured flux at each time point [LMH]
                        Must be the same length as `time`.

    Returns
    -------
    tuple[float, float, float, float]
        (J0, k1, k2, rmse)
        J0   : fitted initial flux [LMH]
        k1   : fitted pore constriction rate [min⁻¹]
        k2   : fitted surface fouling rate [min⁻¹]
        rmse : root mean square error [LMH] — smaller = better fit
               rmse = sqrt(mean((flux_measured - flux_predicted)²))

    Raises
    ------
    RuntimeError (from scipy) if fitting fails to converge within maxfev evaluations.
    This is caught silently by `fit_all_models` in hermia.py.

    Example
    -------
    >>> J0, k1, k2, rmse = fit_combined_1a(time_array, flux_array)
    >>> print(f"J0={J0:.1f} LMH, k1={k1:.4f}, k2={k2:.5f}, RMSE={rmse:.2f}")
    """
    # curve_fit returns (popt, pcov):
    #   popt = array of optimal parameter values [J0, k1, k2]
    #   pcov = covariance matrix (uncertainty in parameters — not used here)
    popt, _ = curve_fit(
        combined_1a_flux,           # the model function to fit
        time,                       # x-axis data
        flux,                       # y-axis data (target to match)
        p0=[flux[0], 0.01, 0.001], # initial guess: J0 ≈ first data point, small k values
        bounds=(
            [J_MIN,        K1_BOUNDS[0], K2_BOUNDS[0]],   # lower bounds
            [J_MAX,        K1_BOUNDS[1], K2_BOUNDS[1]],   # upper bounds
        ),
        maxfev=10000,               # stop after 10,000 function evaluations if not converged
    )

    J0, k1, k2 = popt   # unpack the three fitted parameters

    # Evaluate the fitted model at all time points to compute the RMSE
    pred = combined_1a_flux(time, J0, k1, k2)
    rmse = float(np.sqrt(np.mean((flux - pred) ** 2)))

    return J0, k1, k2, rmse

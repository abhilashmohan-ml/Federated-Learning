"""
Manabe model for virus capture probability and LRV (Log Reduction Value).

BACKGROUND — WHAT IS LRV?
---------------------------
When a pharmaceutical manufacturer filters a drug product through a virus-retentive
membrane, they must PROVE that the filter removes viruses to a regulatory standard.
The standard measurement is the Log Reduction Value (LRV):

    LRV = log₁₀(C_feed / C_permeate)

    C_feed     = virus concentration in the liquid entering the filter
    C_permeate = virus concentration in the liquid that passed through

If LRV = 4, the filter achieved a 10,000-fold reduction (10^4 = 10,000).
Regulatory agencies (FDA, EMA) typically require LRV ≥ 4 for each virus family.

THE MANABE MODEL
-----------------
The Manabe model (Manabe, 1981) describes how the virus capture probability Pc
depends on the filtration flux J and properties of the membrane:

    Pc = 1 - exp(-λ · J / J_crit)

    λ     = membrane affinity parameter (how strongly the membrane retains viruses)
    J     = current filtration flux [LMH]
    J_crit = critical flux below which capture drops significantly [LMH]

Intuitively:
  - At J >> J_crit: Pc approaches 1 (near-complete capture)
  - At J << J_crit: Pc is small (poor capture — flow too slow)
  - Higher λ means better capture at the same flux

Once Pc is known, the LRV for N membrane layers is:
    LRV = log₁₀(1 / (1 - Pc)) × N_layers

PYTHON CONCEPT: @dataclass
  The `@dataclass` decorator automatically generates __init__, __repr__ and
  __eq__ methods based on the class's annotated fields. This saves writing
  boilerplate code for simple data-holding classes.
"""

from __future__ import annotations   # allows forward type references in annotations

from dataclasses import dataclass    # Python decorator for data classes

import numpy as np                   # NumPy for numerical operations
from scipy.optimize import curve_fit # scipy: scientific Python library for curve fitting

from shared.utils.constants import PC_BOUNDS, JCRIT_BOUNDS, LAMBDA_BOUNDS


@dataclass
class ManabeResult:
    """
    Results from fitting the Manabe model to (flux, LRV) measurement data.

    Attributes
    ----------
    Pc         : single-layer virus capture probability (0 = no capture, 1 = perfect)
    lambda_    : fitted membrane affinity parameter (dimensionless)
    J_crit     : fitted critical flux (LMH) — flux below this, capture degrades rapidly
    LRV        : calculated LRV at the mean flux of the experiment
    LRV_required: the regulatory minimum (default 4.0)
    compliant   : True if LRV >= LRV_required — the key pass/fail flag
    """
    Pc:           float   # capture probability [0, 1]
    lambda_:      float   # membrane affinity parameter (trailing underscore avoids
                          # conflict with Python's built-in `lambda` keyword)
    J_crit:       float   # critical flux [LMH]
    LRV:          float   # log reduction value at mean operating flux
    LRV_required: float   # regulatory minimum LRV (typically 4.0)
    compliant:    bool    # True if LRV >= LRV_required


def capture_probability(J: float, lambda_: float, J_crit: float) -> float:
    """
    Calculate the single-layer virus capture probability for a given flux.

    Uses the Manabe equation:  Pc = 1 - exp(-λ · J / J_crit)

    Parameters
    ----------
    J       : float — operating flux in LMH (Litres per square Metre per Hour)
    lambda_ : float — membrane affinity (fitted from validation data)
    J_crit  : float — critical flux in LMH (fitted from validation data)

    Returns
    -------
    float — capture probability in the range [0, 1]

    Example
    -------
    >>> capture_probability(J=100.0, lambda_=2.5, J_crit=80.0)
    # Returns approximately 0.956 (95.6% of viruses captured)
    """
    # np.exp(-x) = e^(-x); at x=0, result=1, so 1-exp(0)=0 (no capture at J=0)
    # As x → ∞, exp(-x) → 0, so Pc → 1 (perfect capture at very high flux)
    return float(1.0 - np.exp(-lambda_ * J / J_crit))


def compute_lrv(Pc: float, n_layers: int = 1) -> float:
    """
    Compute the Log Reduction Value from capture probability.

    LRV = log₁₀(1 / (1 - Pc)) × N_layers

    WHY CLIP Pc?
    ------------
    If Pc is exactly 1.0, then (1 - Pc) = 0, and log₁₀(1/0) = infinity.
    If Pc is exactly 0.0, then log₁₀(1/1) = 0.
    We clip to avoid division by zero and very large numbers.

    Parameters
    ----------
    Pc       : float — capture probability (0–1); clipped to (1e-9, 1 - 1e-9)
    n_layers : int   — number of membrane layers (default 1)
                       Some filter designs stack multiple membranes for higher LRV

    Returns
    -------
    float — LRV value (should be ≥ 4.0 to meet regulatory requirements)
    """
    # np.clip(x, a, b) constrains x to be between a and b.
    # We clip very close to (but not exactly at) 0 and 1 to avoid maths errors.
    Pc_c = float(np.clip(Pc, 1e-9, 1.0 - 1e-9))
    return float(np.log10(1.0 / (1.0 - Pc_c)) * n_layers)


def fit_manabe(
    flux_values: np.ndarray,
    lrv_values: np.ndarray,
    n_layers: int = 1,
    lrv_required: float = 4.0,
) -> ManabeResult:
    """
    Fit Manabe model parameters (λ and J_crit) from experimental data.

    HOW CURVE FITTING WORKS
    ------------------------
    `scipy.optimize.curve_fit` finds the parameter values (lambda_, J_crit)
    that make the model's predictions match the observed LRV data as closely
    as possible. It minimises the sum of squared differences between predicted
    and measured LRV values (least-squares fitting).

    The `p0` argument provides an initial guess. The `bounds` argument
    constrains the search space to physically meaningful ranges.

    Parameters
    ----------
    flux_values  : np.ndarray — flux values at which LRV was measured [LMH]
                   These are paired with lrv_values; index i in both arrays
                   correspond to the same measurement.
    lrv_values   : np.ndarray — measured LRV at each flux value
    n_layers     : int — number of membrane layers
    lrv_required : float — regulatory minimum LRV (default 4.0)

    Returns
    -------
    ManabeResult — fitted parameters plus compliance flag

    Example
    -------
    >>> flux = np.array([50.0, 75.0, 100.0, 125.0])
    >>> lrv  = np.array([3.2, 3.8, 4.3, 4.6])
    >>> result = fit_manabe(flux, lrv)
    >>> result.compliant   # True if result.LRV >= 4.0
    """

    def _model_lrv(J: np.ndarray, lambda_: float, J_crit: float) -> np.ndarray:
        """
        Internal helper: predict LRV from flux given model parameters.

        This is the function scipy tries to match to the measured data.
        It accepts arrays for J so scipy can evaluate it at all data points at once.
        """
        # Manabe capture probability
        Pc = 1.0 - np.exp(-lambda_ * J / J_crit)
        # Clip to avoid log(0) issues
        Pc = np.clip(Pc, 1e-9, 1.0 - 1e-9)
        # LRV for n_layers
        return np.log10(1.0 / (1.0 - Pc)) * n_layers

    # Fit the model. `popt` contains the optimal [lambda_, J_crit] values.
    # The underscore `_` captures the covariance matrix (not needed here).
    popt, _ = curve_fit(
        _model_lrv,
        flux_values,
        lrv_values,
        p0=[1.0, 100.0],           # initial guess: lambda_=1.0, J_crit=100 LMH
        bounds=(
            [LAMBDA_BOUNDS[0], JCRIT_BOUNDS[0]],   # lower bounds
            [LAMBDA_BOUNDS[1], JCRIT_BOUNDS[1]],   # upper bounds
        ),
        maxfev=5000,               # maximum number of function evaluations
    )
    lambda_, J_crit = popt   # unpack the two fitted parameters

    # Calculate Pc and LRV at the MEAN flux (representative operating point)
    J_mean = float(np.mean(flux_values))
    Pc = capture_probability(J_mean, lambda_, J_crit)
    LRV = compute_lrv(Pc, n_layers)

    return ManabeResult(
        Pc=Pc,
        lambda_=lambda_,
        J_crit=J_crit,
        LRV=LRV,
        LRV_required=lrv_required,
        compliant=LRV >= lrv_required,   # the critical compliance check
    )

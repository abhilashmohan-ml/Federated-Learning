"""
Virus internal concentration polarisation model.

WHAT IS CONCENTRATION POLARISATION?
-------------------------------------
During filtration, liquid flows through the membrane but viruses are (ideally)
stopped at the surface. This causes viruses to pile up at the membrane surface,
creating a region of very high concentration called the "polarisation layer."

The phenomenon is called "concentration polarisation" and it has two important effects:

  1. It INCREASES the local virus concentration at the wall (C_wall >> C_feed).
     This is important for predicting how the filter experiences virus loading.

  2. It DECREASES the concentration in the permeate (liquid that passed through),
     because the high local concentration actually IMPROVES the filter's retention.
     This means the LRV is BETTER than you would predict without polarisation.

THE PHYSICS
-----------
In a steady-state boundary layer of thickness δ (delta), two forces act on viruses:

  - Convective transport TOWARD the membrane: driven by the filtration flux J
    (liquid carries viruses toward the filter)

  - Diffusive transport AWAY from the membrane: driven by the concentration
    gradient; viruses diffuse back into the bulk liquid at rate D_v

At steady state, these forces balance, giving the exponential profiles:

    C_wall = C_feed × exp(J × δ / D_v)      [concentration at the membrane]
    C_perm = C_feed × (1 - R) × exp(-J × δ / D_v)  [concentration in permeate]

    where R = true membrane rejection (fraction blocked, e.g. 0.99 = 99%)

UNIT CONVERSIONS
-----------------
The flux J is measured in LMH (L/m²/h) in our system, but the exponential
formula requires J in m/s (SI units).

    1 LMH = 1/(3.6 × 10⁶) m/s
    So: J_m_per_s = J_LMH / 3,600,000

PYTHON CONCEPT: @dataclass
  Used here to create a simple data-holding class without writing __init__ manually.

PYTHON CONCEPT: default parameter values
  `delta: float = 1e-5` means if the caller doesn't pass delta, it defaults to
  1×10⁻⁵ = 0.00001 metres = 10 micrometres (a typical boundary layer thickness).
  Similarly D_v = 1×10⁻¹¹ m²/s is a typical value for a 25-nm parvovirus particle.
"""

from __future__ import annotations  # allows forward type references

from dataclasses import dataclass   # auto-generates __init__ etc.

import numpy as np                  # NumPy for mathematical operations


@dataclass
class PolarizationResult:
    """
    Results of the concentration polarisation calculation.

    Attributes
    ----------
    C_wall  : virus concentration at the membrane wall relative to feed
              Values > 1.0 mean more viruses at the wall than in the feed.
              Example: C_wall = 5.2 means 5.2× concentration at the wall.

    C_perm  : virus concentration in the permeate relative to feed
              Values < 1.0 mean fewer viruses passed through than entered.
              Example: C_perm = 0.0001 means 0.01% of viruses got through.

    LRV_pol : the LRV contribution attributable specifically to polarisation
              LRV_pol = log₁₀(C_feed / C_perm)
              This supplements the LRV calculated by the Manabe model.
    """
    C_wall:  float   # relative wall concentration (> 1.0 typical)
    C_perm:  float   # relative permeate concentration (< 1.0)
    LRV_pol: float   # LRV from polarisation contribution


def concentration_at_wall(
    C_feed: float,
    J_lmh: float,
    delta: float = 1e-5,    # boundary layer thickness [m], default 10 µm
    D_v:   float = 1e-11,   # virus diffusion coefficient [m²/s], default for ~25nm virus
) -> float:
    """
    Calculate virus concentration at the membrane wall.

    Formula: C_wall = C_feed × exp(J × δ / D_v)

    As flux J increases, more viruses are transported toward the membrane
    per second than can diffuse back, so C_wall grows exponentially with J.

    Parameters
    ----------
    C_feed : float — feed concentration (can be in any units; result is in same units)
    J_lmh  : float — filtration flux in LMH (Litres per square Metre per Hour)
    delta  : float — boundary layer thickness in metres (default 1×10⁻⁵ m = 10 µm)
    D_v    : float — virus diffusion coefficient in m²/s (default 1×10⁻¹¹)

    Returns
    -------
    float — wall concentration in the same units as C_feed
    """
    # Convert flux from LMH to m/s (SI units required for the formula)
    J_ms = J_lmh / 3.6e6   # 3.6e6 = 3,600,000 (seconds in an hour × litres to m³)

    # Apply the steady-state polarisation equation
    return C_feed * np.exp(J_ms * delta / D_v)


def permeate_concentration(
    C_feed: float,
    J_lmh: float,
    delta: float = 1e-5,
    D_v:   float = 1e-11,
    R:     float = 0.99,    # true membrane rejection: 0.99 = 99% blocked
) -> float:
    """
    Calculate virus concentration in the permeate (filtrate).

    Formula: C_perm = C_feed × (1 - R) × exp(-J × δ / D_v)

    The (1 - R) factor represents the fraction of viruses not blocked.
    The exponential term shows that higher flux actually REDUCES permeate
    concentration (counter-intuitive but correct: higher flux means more
    viruses pile up at the wall, making it harder for them to pass through).

    Parameters
    ----------
    C_feed : float — feed concentration
    J_lmh  : float — filtration flux in LMH
    delta  : float — boundary layer thickness [m]
    D_v    : float — virus diffusion coefficient [m²/s]
    R      : float — true membrane rejection [0–1]; 0.99 means 99% of viruses blocked

    Returns
    -------
    float — permeate concentration in the same units as C_feed
    """
    # Convert flux from LMH to m/s
    J_ms = J_lmh / 3.6e6

    # (1 - R) is the "leakage fraction" — viruses that sneak through
    # The negative exponent means: higher J → lower permeate concentration
    return C_feed * (1.0 - R) * np.exp(-J_ms * delta / D_v)


def compute_polarization(
    C_feed: float,
    J_lmh: float,
    delta: float = 1e-5,
    D_v:   float = 1e-11,
    R:     float = 0.99,
) -> PolarizationResult:
    """
    Compute the full polarisation profile: wall concentration, permeate, and LRV.

    This is the main entry point that callers should use. It calls the two
    helper functions above and bundles the results into a PolarizationResult.

    Parameters
    ----------
    C_feed : float — virus concentration in the feed (arbitrary units)
    J_lmh  : float — filtration flux [LMH]
    delta  : float — boundary layer thickness [m] (default 10 µm)
    D_v    : float — virus diffusion coefficient [m²/s]
    R      : float — true membrane rejection (default 0.99)

    Returns
    -------
    PolarizationResult — contains C_wall, C_perm, LRV_pol

    Example
    -------
    >>> result = compute_polarization(C_feed=1.0, J_lmh=100.0)
    >>> print(f"Wall: {result.C_wall:.2f}x, LRV: {result.LRV_pol:.2f}")
    """
    C_wall = concentration_at_wall(C_feed, J_lmh, delta, D_v)
    C_perm = permeate_concentration(C_feed, J_lmh, delta, D_v, R)

    # Guard against C_perm = 0 (would cause log10(infinity))
    # 1e-20 is effectively zero concentration — the filter is perfect
    C_perm = max(C_perm, 1e-20)

    # LRV = log₁₀(C_feed / C_perm)
    # e.g. if C_feed=1.0 and C_perm=0.0001, LRV = log₁₀(10000) = 4.0
    LRV_pol = float(np.log10(C_feed / C_perm))

    return PolarizationResult(C_wall=C_wall, C_perm=C_perm, LRV_pol=LRV_pol)

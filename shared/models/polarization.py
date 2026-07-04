"""
Virus internal concentration polarization model.

During normal-flow (dead-end) filtration viruses accumulate at the
membrane surface due to convective transport exceeding diffusion.

Wall concentration:
    C_wall = C_feed * exp(J * delta / D_v)

Permeate concentration (with true rejection R):
    C_perm = C_feed * (1 - R) * exp(-J * delta / D_v)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolarizationResult:
    C_wall: float    # virus concentration at membrane wall  [relative]
    C_perm: float    # virus concentration in permeate       [relative]
    LRV_pol: float   # LRV contribution from polarization


def concentration_at_wall(
    C_feed: float,
    J_lmh: float,
    delta: float = 1e-5,   # boundary layer thickness [m]
    D_v: float   = 1e-11,  # virus diffusion coefficient [m^2/s]
) -> float:
    """C_wall = C_feed * exp(J * delta / D_v).  J converted LMH -> m/s."""
    J_ms = J_lmh / 3.6e6
    return C_feed * np.exp(J_ms * delta / D_v)


def permeate_concentration(
    C_feed: float,
    J_lmh: float,
    delta: float = 1e-5,
    D_v: float   = 1e-11,
    R: float     = 0.99,   # true membrane rejection
) -> float:
    """C_perm = C_feed * (1-R) * exp(-J * delta / D_v)."""
    J_ms = J_lmh / 3.6e6
    return C_feed * (1.0 - R) * np.exp(-J_ms * delta / D_v)


def compute_polarization(
    C_feed: float,
    J_lmh: float,
    delta: float = 1e-5,
    D_v: float   = 1e-11,
    R: float     = 0.99,
) -> PolarizationResult:
    """Return wall concentration, permeate concentration, and LRV."""
    C_wall = concentration_at_wall(C_feed, J_lmh, delta, D_v)
    C_perm = permeate_concentration(C_feed, J_lmh, delta, D_v, R)
    C_perm = max(C_perm, 1e-20)
    LRV_pol = float(np.log10(C_feed / C_perm))
    return PolarizationResult(C_wall=C_wall, C_perm=C_perm, LRV_pol=LRV_pol)

"""DataSource abstraction — dev (in-memory simulation) and prod (CSV directory polling)."""
from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np

from shared.utils.logging_config import get_logger

log = get_logger(__name__)

# Sensible baseline physics for dev/testing — no site-name keys.
# Callers (scheduler, tests) use this as the default when no env-configured
# physics are available.  Combined 1-A flux model parameters:
#   J0       : initial flux (L m⁻² h⁻¹)
#   k1, k2   : Combined 1-A fouling rate constants
#   noise    : Gaussian noise σ on flux
#   tmp_base : initial TMP (bar)
PHYSICS_DEFAULTS: dict[str, float] = {
    "J0": 150.0, "k1": 0.015, "k2": 0.0020, "noise": 2.0, "tmp_base": 1.0
}


class NoNewDataError(Exception):
    """Raised by ProdDataSource when no new CSV files are found in the data directory."""


class DataSource(Protocol):
    """Protocol (interface) for all data sources. Returns (time_min, flux_lmh, tmp_bar)."""

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


class DevDataSource:
    """
    Generates fresh synthetic filtration data on each call using perturbed physics params.

    physics_cfg provides the base parameters (J0, k1, k2, noise, tmp_base).
    Each call jitters J0, k1, k2 by ±jitter * N(0,1), producing slightly
    different flux curves round-to-round.

    Different physics dicts produce clearly distinct flux curves — this is how
    inter-site variance is achieved without hardcoding site names.
    """

    def __init__(self, physics_cfg: dict[str, float], jitter: float = 0.05) -> None:
        self._cfg = physics_cfg
        self._jitter = jitter

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng()
        J0 = self._cfg["J0"] * (1.0 + self._jitter * rng.standard_normal())
        k1 = self._cfg["k1"] * (1.0 + self._jitter * rng.standard_normal())
        k2 = self._cfg["k2"] * (1.0 + self._jitter * rng.standard_normal())

        time = np.arange(0, 121, 1, dtype=np.float64)
        flux = (J0 / (1.0 + k1 * time) ** 2) * np.exp(-k2 * time)
        flux += rng.normal(0.0, self._cfg["noise"], len(time))
        flux = np.clip(flux, 1.0, None)
        tmp = self._cfg["tmp_base"] + 0.004 * time + rng.normal(0.0, 0.02, len(time))

        log.debug("dev_data_generated", J0=round(J0, 2), k1=round(k1, 5))
        return time, flux, tmp

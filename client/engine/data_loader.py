"""Load and validate local filtration CSV data.  No data leaves the site."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from shared.utils.logging_config import get_logger

log = get_logger(__name__)
REQUIRED = {"time_min", "flux_lmh", "tmp_bar"}


def load_filtration_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load filtration time-series from CSV.

    Returns
    -------
    time_min, flux_lmh, tmp_bar  as numpy float64 arrays.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(p)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df.dropna(subset=list(REQUIRED))
    log.info("data_loaded", path=str(path), n_rows=len(df))
    return (
        df["time_min"].to_numpy(dtype=np.float64),
        df["flux_lmh"].to_numpy(dtype=np.float64),
        df["tmp_bar"].to_numpy(dtype=np.float64),
    )

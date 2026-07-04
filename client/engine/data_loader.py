"""
Local filtration data loader — reads the site's CSV file for training.

PRIVACY GUARANTEE
------------------
This module reads the site's local measurement data from disk.
The data NEVER leaves this module or this container — it is used only to
train the local model. The only output of the training pipeline that is
transmitted to the server is the model update (gradient delta), not the
raw measurements.

The CSV file format is documented in docs/TECHNICAL_SPEC.md. A synthetic
example can be generated with: python scripts/generate_synthetic_data.py

REQUIRED CSV COLUMNS
----------------------
  time_min   : elapsed filtration time in minutes (float, e.g. 0, 5, 10, 15, ...)
  flux_lmh   : transmembrane flux in LMH — L/(m²·h) (float, must be > 0)
  tmp_bar    : transmembrane pressure in bar (float, must be > 0)

Optional columns (used for richer PINN training, ignored if absent):
  lrv_obs       : measured log reduction value (float, regulatory minimum 4.0)
  c_feed_mg_mL  : feed concentration of mAb product in mg/mL
  ph            : pH of the process stream
  ...etc (see schemas/filtration.py for full list)

PYTHON CONCEPT: pathlib.Path
  Path is the modern way to handle file paths in Python. It works on all
  operating systems (Windows uses backslashes, Linux uses forward slashes —
  Path handles this automatically). `Path(path).exists()` checks if the file
  exists without raising an exception.

PYTHON CONCEPT: pandas DataFrame
  pandas is Python's "spreadsheet in code." `pd.read_csv(p)` reads a CSV file
  into a DataFrame (a 2D table with named columns). We then use:
    - `set(df.columns)` to get all column names as a set
    - `df.dropna(subset=...)` to remove rows that have missing values in the key columns
    - `df["column"].to_numpy()` to extract a column as a raw NumPy array

PYTHON CONCEPT: tuple return value
  The function returns three NumPy arrays as a tuple: (time, flux, tmp).
  The caller unpacks them: `time, flux, tmp = load_filtration_csv(path)`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from shared.utils.logging_config import get_logger

log = get_logger(__name__)

# These three columns MUST be present in every filtration CSV file.
# The function raises ValueError if any are missing, rather than producing
# silently wrong results.
REQUIRED = {"time_min", "flux_lmh", "tmp_bar"}


def load_filtration_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load filtration time-series measurements from a CSV file.

    This is the ONLY place where raw site data is read from disk.
    The loaded arrays stay in memory on the site; they are passed to the
    local training pipeline and are never serialised or transmitted.

    Parameters
    ----------
    path : str — path to the CSV file (relative or absolute)
                 Typically set via the LOCAL_DATA_PATH environment variable.

    Returns
    -------
    tuple of three np.ndarray (dtype=float64):
        time_min : time points in minutes [0, 5, 10, ...]
        flux_lmh : measured flux at each time point in LMH
        tmp_bar  : transmembrane pressure at each time point in bar

    Raises
    ------
    FileNotFoundError : if the file doesn't exist at the given path
    ValueError        : if any of the required columns are missing from the CSV

    Example
    -------
    >>> time, flux, tmp = load_filtration_csv("./data/site_1/filtration.csv")
    >>> print(f"Loaded {len(time)} time points, initial flux: {flux[0]:.1f} LMH")
    """
    p = Path(path)

    # Check existence first — gives a clear error message rather than pandas' generic one
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Read the CSV into a pandas DataFrame — handles headers, type inference, etc.
    df = pd.read_csv(p)

    # Compute which required columns are absent using set difference
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Remove rows where any required column has NaN (missing value).
    # `subset=list(REQUIRED)` means: only drop rows where these specific columns are NaN;
    # rows with NaN in other (optional) columns are kept.
    df = df.dropna(subset=list(REQUIRED))

    # Log how many data points were loaded — useful for debugging data quality issues
    log.info("data_loaded", path=str(path), n_rows=len(df))

    # Convert each column to a 64-bit float NumPy array.
    # `dtype=np.float64` ensures consistent precision regardless of the CSV format.
    return (
        df["time_min"].to_numpy(dtype=np.float64),
        df["flux_lmh"].to_numpy(dtype=np.float64),
        df["tmp_bar"].to_numpy(dtype=np.float64),
    )

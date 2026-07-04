"""
Differential privacy: Gaussian noise mechanism.

Applies gradient clipping + Gaussian noise to model weight updates
before they are uploaded to the server.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def add_gaussian_noise(
    weights: Dict[str, List[float]],
    sigma: float,
    clip_norm: float = 1.0,
) -> Dict[str, List[float]]:
    """
    Clip each layer's L2 norm to clip_norm, then add Gaussian noise.

    Parameters
    ----------
    weights   : layer_name -> flat weight/gradient list
    sigma     : noise std dev  (larger => more privacy, less utility)
    clip_norm : L2 sensitivity clipping threshold

    Returns
    -------
    Perturbed weights dict (same structure as input).
    """
    noisy: Dict[str, List[float]] = {}
    for layer, vals in weights.items():
        arr = np.array(vals, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > clip_norm:
            arr = arr * (clip_norm / norm)
        arr += np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
        noisy[layer] = arr.tolist()
    return noisy

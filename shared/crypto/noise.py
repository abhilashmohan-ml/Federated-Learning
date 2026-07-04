"""
Differential Privacy: Gaussian noise mechanism.

WHY DIFFERENTIAL PRIVACY?
--------------------------
Even though we only send model weight UPDATES (not raw data), a sophisticated
attacker could potentially reverse-engineer information about the local training
data from the gradient values. Differential privacy (DP) adds carefully calibrated
mathematical noise to the gradients, making it provably difficult to reconstruct
anything about the individual data points.

HOW THE GAUSSIAN MECHANISM WORKS
----------------------------------
The "Gaussian mechanism" is one of the simplest and most widely used DP techniques:

  Step 1 — CLIPPING
    Limit the L2 norm (magnitude) of the gradient vector to `clip_norm`.
    This "clips" extreme values and bounds the "sensitivity" — the maximum
    amount any single training example can influence the update.

    WHY CLIPPING FIRST?
    Without clipping, a very unusual training example could produce a very large
    gradient, and the noise needed to hide it would also need to be very large
    (reducing the model's utility). Clipping ensures all gradients have the
    same maximum influence, so a fixed noise level is effective.

  Step 2 — NOISE ADDITION
    Add random noise drawn from a Gaussian (normal) distribution with
    standard deviation `sigma`.

    The Gaussian distribution N(0, sigma²) is centred on zero, so on average
    it does not change the gradient direction — but it blurs the exact values.

    A larger sigma provides more privacy (harder to reverse-engineer the data)
    but reduces model quality (the aggregated model learns more slowly).
    sigma is configurable via DP_NOISE_SIGMA in the site's .env file.

KNOWN LIMITATION
-----------------
This implements the basic Gaussian mechanism. For formal (ε, δ)-DP guarantees
under the Abadi et al. (2016) DP-SGD framework, a moments accountant is needed
to track privacy budget across many rounds. That is planned but not yet implemented.

PYTHON CONCEPT: Dict[str, List[float]]
  The `weights` parameter is a dictionary where:
    - key   = layer name (a string, e.g. "hermia_params")
    - value = list of float numbers (the weight values for that layer)
  We process each layer independently.

PYTHON CONCEPT: np.linalg.norm
  The L2 norm of a vector v is sqrt(v[0]² + v[1]² + ... + v[n]²).
  It measures the "length" of the vector in n-dimensional space.
  It is also called the Euclidean norm.
"""

from __future__ import annotations      # allows forward type references

from typing import Dict, List           # type hints for collections

import numpy as np                      # NumPy: fast numerical arrays


def add_gaussian_noise(
    weights: Dict[str, List[float]],
    sigma: float,
    clip_norm: float = 1.0,
) -> Dict[str, List[float]]:
    """
    Apply gradient clipping then Gaussian noise to model weight updates.

    This function is called by `LocalTrainer` on the local gradient update
    (delta_W) BEFORE it is included in the ModelUpdate sent to the server.

    Parameters
    ----------
    weights : Dict[str, List[float]]
        The raw weight updates to protect.
        Keys are layer names; values are flat lists of gradient values.
        Example: {"hermia_params": [42.3, 0.012, -0.0003]}

    sigma : float
        Standard deviation of the Gaussian noise. Larger = more privacy,
        less accuracy. Configured via DP_NOISE_SIGMA environment variable.
        Typical range: 0.001 (minimal noise) to 1.0 (strong privacy).

    clip_norm : float, default 1.0
        The maximum L2 norm allowed. Gradients larger than this are scaled
        down proportionally. 1.0 is the standard choice from the DP-SGD paper.

    Returns
    -------
    Dict[str, List[float]]
        The noisy weight updates — same structure as input but with clipping
        and noise applied. This is safe to send to the server.

    Example
    -------
    >>> raw_update = {"hermia_params": [100.0, 0.5, 0.001]}
    >>> noisy = add_gaussian_noise(raw_update, sigma=0.01)
    >>> # noisy["hermia_params"] is now clipped and slightly perturbed
    """
    # We will build a new dict to hold the perturbed values.
    # We do not modify the input dict in place (that would be surprising behaviour).
    noisy: Dict[str, List[float]] = {}

    for layer, vals in weights.items():
        # Convert the Python list to a NumPy array for fast numerical operations.
        # dtype=np.float32 uses 32-bit floats (same as PyTorch default).
        arr = np.array(vals, dtype=np.float32)

        # ── STEP 1: Gradient clipping ──────────────────────────────────────────
        # Compute the L2 norm (overall magnitude) of this layer's gradient vector.
        norm = float(np.linalg.norm(arr))

        if norm > clip_norm:
            # Scale the entire vector down so its norm equals clip_norm exactly.
            # This preserves the direction of the gradient but limits its magnitude.
            # Formula: arr_clipped = arr * (clip_norm / norm)
            arr = arr * (clip_norm / norm)

        # ── STEP 2: Gaussian noise addition ───────────────────────────────────
        # Draw noise from N(0, sigma²) — a Gaussian centred at 0.
        # `arr.shape` gives the dimensions of the array so the noise has the
        # same number of elements as the gradient.
        # .astype(np.float32) converts the noise to the same dtype as arr.
        arr += np.random.normal(0.0, sigma, arr.shape).astype(np.float32)

        # Convert back to a plain Python list for JSON serialisation.
        noisy[layer] = arr.tolist()

    return noisy

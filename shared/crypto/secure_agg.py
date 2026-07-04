"""
Secure aggregation stub — additive secret sharing.

WHAT IS SECURE AGGREGATION?
-----------------------------
Even after differential privacy noise, the server still sees each site's
individual weight update. An advanced threat model assumes the server itself
could be compromised or curious. Secure Aggregation (SecAgg) lets the server
compute the SUM of all site updates without ever seeing any individual update.

The protocol ensures:
  "The server learns ONLY the aggregated gradient, not which site contributed what."

HOW ADDITIVE SECRET SHARING WORKS
-----------------------------------
Suppose site_1 has a secret value v = 10 and there are 3 parties.

  1. Site_1 splits v into 3 random "shares" that add up to v:
       share_1 = 7    (random)
       share_2 = -2   (random)
       share_3 = 10 - 7 - (-2) = 5   (last share = v - sum of others)

  2. Each share goes to a different party (server, other sites, or a mix).

  3. Nobody who sees only one share can recover v (7 alone tells you nothing).
     But summing all three shares always gives back v: 7 + (-2) + 5 = 10.

In our multi-layer weight context, this happens independently for each element
of each layer's weight vector.

WHY THIS IS A STUB
-------------------
Full SecAgg requires careful key exchange and share routing so that:
  - Shares go to the right parties over encrypted channels
  - The protocol handles site dropouts gracefully
  - The whole thing is cryptographically secure under a threat model

This stub implements the mathematical core (split/reconstruct) for testing
and development. A production deployment would integrate a library like
OpenMined PySyft or the Bonawitz et al. (2017) SecAgg protocol.

PYTHON CONCEPT: List comprehension
  `[{} for _ in range(n_shares)]` creates a list of n_shares empty dictionaries.
  The underscore `_` is a conventional variable name meaning "I don't need this value."

PYTHON CONCEPT: sum() on a list of arrays
  `sum([a, b, c])` with numpy arrays computes a + b + c element-wise.
  Note: Python's built-in `sum` starts from 0, which fails for numpy arrays.
  For numpy arrays, `np.sum(list_of_arrays, axis=0)` is safer.
"""

from typing import Dict, List  # type hints
import numpy as np             # NumPy for fast numerical operations


def split_into_shares(
    weights: Dict[str, List[float]],
    n_shares: int,
) -> List[Dict[str, List[float]]]:
    """
    Split weight updates into additive shares.

    Given a weight dict, produce `n_shares` sub-dicts such that summing the
    corresponding values across all shares reconstructs the original weights.

    Parameters
    ----------
    weights : Dict[str, List[float]]
        The weight updates to split. Same format as ModelUpdate.delta_W.
        Example: {"layer_0": [0.1, -0.2, 0.3]}

    n_shares : int
        How many shares to produce. Typically equal to the number of parties
        (e.g. 5 for 5 sites, or 2 for a simple server+client scheme).

    Returns
    -------
    List[Dict[str, List[float]]]
        A list of n_shares dicts, each with the same keys as `weights`.
        `shares[i]["layer_0"]` is share i of the "layer_0" values.

    Example
    -------
    >>> w = {"params": [10.0, 20.0]}
    >>> shares = split_into_shares(w, 3)
    >>> # sum(shares[i]["params"][j] for i in range(3)) == 10.0 for j=0
    >>> # sum(shares[i]["params"][j] for i in range(3)) == 20.0 for j=1
    """
    # Initialise a list of empty dicts, one per share recipient.
    shares: List[Dict[str, List[float]]] = [{} for _ in range(n_shares)]

    for layer, vals in weights.items():
        # Convert to float64 for numerical precision during splitting.
        arr = np.array(vals, dtype=np.float64)

        # Generate (n_shares - 1) random shares. These can be any numbers —
        # the last share is determined by what's needed to make the sum correct.
        random_shares = [np.random.randn(*arr.shape) for _ in range(n_shares - 1)]

        # The last share is set so that all shares sum to the original value:
        #   last_share = arr - (share_0 + share_1 + ... + share_{n-2})
        last_share = arr - sum(random_shares)

        # Assign random shares to the first (n_shares - 1) recipients.
        for i, s in enumerate(random_shares):
            shares[i][layer] = s.tolist()

        # Assign the balancing share to the last recipient.
        shares[-1][layer] = last_share.tolist()

    return shares


def reconstruct_from_shares(
    shares: List[Dict[str, List[float]]],
) -> Dict[str, List[float]]:
    """
    Reconstruct original weights by summing all shares.

    This is the inverse of `split_into_shares`. The server calls this
    after collecting one share from each site to reveal only the sum.

    Parameters
    ----------
    shares : List[Dict[str, List[float]]]
        All shares collected — same format as returned by split_into_shares.

    Returns
    -------
    Dict[str, List[float]]
        The reconstructed weight updates, identical to the original input
        to split_into_shares (within floating-point precision).

    Example
    -------
    >>> w = {"params": [10.0, 20.0]}
    >>> shares = split_into_shares(w, 3)
    >>> reconstructed = reconstruct_from_shares(shares)
    >>> reconstructed["params"]   # ≈ [10.0, 20.0]
    """
    result: Dict[str, List[float]] = {}

    # Iterate over layer names (taken from the first share — all shares have the same keys).
    for layer in shares[0]:
        # Start with a zero vector of the correct size.
        total = np.zeros(len(shares[0][layer]), dtype=np.float64)

        # Add each share's contribution element-wise.
        for share in shares:
            total += np.array(share[layer], dtype=np.float64)

        # Convert back to a Python list for JSON serialisation.
        result[layer] = total.tolist()

    return result

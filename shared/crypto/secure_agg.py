"""
Secure aggregation stub.

In a full deployment this module implements the SecAgg protocol so
the aggregator never sees individual site updates in plaintext.

For the initial implementation we use additive secret sharing:
  - Each site splits its update into n shares
  - Each share goes to a different peer or the server
  - The server reconstructs only the sum

TODO: replace with a proper SecAgg library (e.g. OpenMined PySyft).
"""
from typing import Dict, List
import numpy as np


def split_into_shares(
    weights: Dict[str, List[float]],
    n_shares: int,
) -> List[Dict[str, List[float]]]:
    """Additive secret sharing: sum of shares == original weights."""
    shares: List[Dict[str, List[float]]] = [{} for _ in range(n_shares)]
    for layer, vals in weights.items():
        arr = np.array(vals, dtype=np.float64)
        random_shares = [np.random.randn(*arr.shape) for _ in range(n_shares - 1)]
        last_share = arr - sum(random_shares)
        for i, s in enumerate(random_shares):
            shares[i][layer] = s.tolist()
        shares[-1][layer] = last_share.tolist()
    return shares


def reconstruct_from_shares(
    shares: List[Dict[str, List[float]]],
) -> Dict[str, List[float]]:
    """Reconstruct original weights from additive shares."""
    result: Dict[str, List[float]] = {}
    for layer in shares[0]:
        total = np.zeros(len(shares[0][layer]), dtype=np.float64)
        for share in shares:
            total += np.array(share[layer], dtype=np.float64)
        result[layer] = total.tolist()
    return result

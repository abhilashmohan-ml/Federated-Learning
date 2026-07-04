"""
FedProx aggregation of model updates from multiple manufacturing sites.

FEDERATED AVERAGING (FedAvg) ALGORITHM
----------------------------------------
The core FL aggregation formula is a WEIGHTED AVERAGE of model updates,
where each site's contribution is weighted by how many data samples it has:

    W_new[layer] = Σᵢ  (nᵢ / N_total) × (W_current[layer] + ΔWᵢ[layer])

Where:
  nᵢ          = number of training samples at site i
  N_total     = Σᵢ nᵢ (total samples across all sites)
  W_current   = current global model weights (before this round)
  ΔWᵢ[layer]  = delta (change) in weights submitted by site i for that layer

WHY WEIGHTED BY n_samples?
---------------------------
Sites with more data should have more influence on the global model. A site
with 10,000 samples has seen more of the data distribution than a site with
100 samples, so its update should count for more.

  Example: 3 sites with n = [1000, 500, 500]  → weights = [0.5, 0.25, 0.25]
  Site 1 gets half the vote; sites 2 and 3 each get a quarter.

FEDPROX VS FEDAVG
-----------------
Standard FedAvg: sites train freely, then send updates → can diverge wildly
                 if sites have very different data distributions.
FedProx:         sites add a proximal penalty term during local training:
                     L_local += (μ/2) × ‖W_local - W_global‖²
                 This keeps each site's local model "close" to the global model,
                 improving convergence when data is non-IID (heterogeneous).

The aggregation step (this file) is IDENTICAL for FedAvg and FedProx —
the difference is in the local loss function (see shared/models/pinn.py).

METRIC AGGREGATION
------------------
In addition to model weights, we also aggregate local performance metrics
(RMSE, flux ratio, etc.) using a simple mean across sites. These become the
"global metrics" for the round — useful for tracking convergence.

PYTHON CONCEPT: numpy vectorised operations
  Instead of looping over each element of a weight vector, we use numpy arrays
  which perform operations on all elements simultaneously (much faster).
  `np.zeros_like(base)` creates a zero array with the same shape and dtype as `base`.
  `weighted += w * (base + delta)` adds the weighted contribution in-place.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from shared.schemas.federation import GlobalModel, ModelUpdate
from shared.utils.logging_config import get_logger

log = get_logger(__name__)


class FedProxAggregator:
    """
    Performs FedProx-style weighted aggregation of site model updates.

    This class is intentionally stateless — all state lives in RoundManager.
    The `aggregate` method takes inputs, performs the calculation, and returns
    a new GlobalModel. This makes it easy to test in isolation.
    """

    def aggregate(
        self,
        updates: List[ModelUpdate],
        current_global: Dict[str, List[float]],
        round_id: int,
        model_version: int,
    ) -> GlobalModel:
        """
        Aggregate model updates from all participating sites into a new global model.

        Parameters
        ----------
        updates        : list of ModelUpdate — one from each participating site.
                         Each contains delta_W (weight changes), n_samples, and metrics.
        current_global : dict of layer_name → list[float]
                         The global model weights BEFORE this round started.
                         Used as the "base" that delta_W is applied to.
        round_id       : int — the ID of the round being aggregated
        model_version  : int — the version number of current_global (result = version + 1)

        Returns
        -------
        GlobalModel — the new aggregated global model with updated weights and metrics

        Raises
        ------
        ValueError — if `updates` is empty (cannot compute a weighted average with no data)
        """
        if not updates:
            raise ValueError("No updates received — cannot aggregate")

        # Total samples across all sites — the denominator for weighting
        total_samples = sum(u.n_samples for u in updates)
        log.info(
            "aggregating",
            n_sites=len(updates),
            total_samples=total_samples,
            round_id=round_id,
        )

        aggregated: Dict[str, List[float]] = {}

        # Discover all layer names from the first update.
        # Assumption: all sites share the same model architecture (same layer names).
        all_layers = set(updates[0].delta_W.keys())

        for layer in all_layers:
            # Current global weight for this layer, as a numpy array.
            # If a layer doesn't exist in the global model (first round), default to zeros.
            base = np.array(
                current_global.get(layer, [0.0] * len(updates[0].delta_W[layer]))
            )

            # Accumulate weighted contributions from each site
            weighted = np.zeros_like(base, dtype=np.float64)
            for u in updates:
                if layer not in u.delta_W:
                    continue  # site didn't include this layer — skip
                w     = u.n_samples / total_samples                   # fractional weight
                delta = np.array(u.delta_W[layer], dtype=np.float64)  # site's weight change

                # W_new += (n_i / N) × (W_current + ΔW_i)
                # This applies the site's update delta ON TOP of the current global weights,
                # then takes the weighted average across sites.
                weighted += w * (base + delta)

            aggregated[layer] = weighted.tolist()   # convert back to plain Python list

        # ── Aggregate local metrics (simple mean across sites) ─────────────────
        # Collect each metric from sites that reported it, then average.
        # Sites that didn't compute a metric (older versions, failures) are excluded.
        global_metrics: Dict[str, float] = {}
        for key in ["flux_rmse", "lrv_rmse", "flux_ratio", "amin_m2"]:
            vals = [u.local_metrics[key] for u in updates if key in u.local_metrics]
            if vals:
                global_metrics[key] = float(np.mean(vals))

        return GlobalModel(
            version=model_version + 1,   # each successful round increments the version
            round_id=round_id,
            weights=aggregated,
            global_metrics=global_metrics,
        )

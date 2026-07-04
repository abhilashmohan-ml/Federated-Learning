"""
FedProx aggregation of model updates from multiple sites.

Weighted average by n_samples (FedAvg):
    W_new[l] = sum_i  (n_i / N_total) * W_i[l]

The FedProx proximal term is enforced client-side during local training.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from shared.schemas.federation import GlobalModel, ModelUpdate
from shared.utils.logging_config import get_logger

log = get_logger(__name__)


class FedProxAggregator:
    def aggregate(
        self,
        updates: List[ModelUpdate],
        current_global: Dict[str, List[float]],
        round_id: int,
        model_version: int,
    ) -> GlobalModel:
        if not updates:
            raise ValueError("No updates received — cannot aggregate")

        total_samples = sum(u.n_samples for u in updates)
        log.info("aggregating", n_sites=len(updates), total_samples=total_samples, round_id=round_id)

        aggregated: Dict[str, List[float]] = {}
        all_layers = set(updates[0].delta_W.keys())

        for layer in all_layers:
            base = np.array(current_global.get(layer, [0.0] * len(updates[0].delta_W[layer])))
            weighted = np.zeros_like(base, dtype=np.float64)
            for u in updates:
                if layer not in u.delta_W:
                    continue
                w = u.n_samples / total_samples
                delta = np.array(u.delta_W[layer], dtype=np.float64)
                weighted += w * (base + delta)
            aggregated[layer] = weighted.tolist()

        # Aggregate local metrics  (simple mean across sites)
        global_metrics: Dict[str, float] = {}
        for key in ["flux_rmse", "lrv_rmse", "flux_ratio", "amin_m2"]:
            vals = [u.local_metrics[key] for u in updates if key in u.local_metrics]
            if vals:
                global_metrics[key] = float(np.mean(vals))

        return GlobalModel(
            version=model_version + 1,
            round_id=round_id,
            weights=aggregated,
            global_metrics=global_metrics,
        )

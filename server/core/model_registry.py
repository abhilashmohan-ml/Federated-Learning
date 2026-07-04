"""
Global model version registry — tracks every version of the global model.

PURPOSE
-------
As FL rounds complete, the global model evolves: version 1, version 2, etc.
The ModelRegistry keeps a record of every version, including:
  - Which FL round produced it
  - How many parameters each layer has (summary, not the actual weights)
  - What the aggregated global metrics were for that round

This is useful for:
  - The dashboard UI: show convergence curves (metrics vs. round number)
  - Auditing: prove that model version N came from round M with those metrics
  - Rollback: if a round produces a worse model, you could revert to a prior version

CURRENT LIMITATION
------------------
This registry stores records IN MEMORY as a Python list. A server restart
erases all history. A production implementation would persist these records
to the `rounds` table in the PostgreSQL database (see server/db/models.py).

PYTHON CONCEPT: @dataclass
  `ModelRecord` uses @dataclass, which automatically generates:
    - __init__(self, version, round_id, weights_summary, global_metrics)
    - __repr__(self) — a readable string representation for debugging
    - __eq__(self, other) — equality comparison
  This saves writing boilerplate code for simple data-holding classes.

PYTHON CONCEPT: Dict[str, int] for weights_summary
  We store `{layer_name: n_params}` rather than the full weights for efficiency.
  Example: {"hermia_params": 3, "predictor.layer1.weight": 128}
  This lets the dashboard show the model architecture without transmitting
  the full weight arrays.

PYTHON CONCEPT: list[-1]
  `self._records[-1]` accesses the LAST element of the list.
  In Python, negative indices count from the end:
    list[-1]  = last element
    list[-2]  = second-to-last
    list[0]   = first element
"""
from __future__ import annotations

from dataclasses import dataclass, field   # field is imported for potential future use
from typing import Dict, List

from shared.schemas.federation import GlobalModel


@dataclass
class ModelRecord:
    """
    Metadata snapshot for one version of the global model.

    Attributes
    ----------
    version          : int — the model version number (1, 2, 3, ...)
    round_id         : int — the FL round that produced this model version
    weights_summary  : dict — layer_name → number of parameters in that layer
                       e.g. {"hermia_params": 3}
                       Stored as a count, NOT the actual weight values.
    global_metrics   : dict — aggregated performance metrics for this round
                       e.g. {"flux_rmse": 1.18, "flux_ratio": 0.73}
    """
    version:         int
    round_id:        int
    weights_summary: Dict[str, int]    # layer → parameter count
    global_metrics:  Dict[str, float]  # metric name → averaged value


class ModelRegistry:
    """
    In-memory version history of the global model.

    All ModelRecord objects are stored in a list in chronological order.
    The most recent version is always at the end of the list.

    Usage:
        registry = ModelRegistry()
        registry.register(global_model_from_round_1)
        latest = registry.latest()      # ModelRecord for round 1
        history = registry.history()    # [ModelRecord for round 1]
    """

    def __init__(self) -> None:
        # Private list accumulates records as rounds complete
        self._records: List[ModelRecord] = []

    def register(self, model: GlobalModel) -> None:
        """
        Record a newly completed global model in the registry.

        Called by RoundManager after each successful aggregation.

        Parameters
        ----------
        model : GlobalModel — the newly aggregated global model.
                We extract only a summary of the weights (param counts),
                not the full weight tensors. This keeps memory usage low.
        """
        record = ModelRecord(
            version=model.version,
            round_id=model.round_id,
            # {layer_name: len(weights_list)} — count params per layer
            weights_summary={k: len(v) for k, v in model.weights.items()},
            global_metrics=model.global_metrics,
        )
        self._records.append(record)

    def latest(self) -> ModelRecord | None:
        """
        Return the most recent model record, or None if no models registered.

        Uses `list[-1]` to access the last element in O(1) time.
        """
        return self._records[-1] if self._records else None

    def history(self) -> List[ModelRecord]:
        """
        Return a copy of all model records in chronological order.

        We return `list(self._records)` — a shallow copy — rather than
        the internal list directly. This prevents callers from accidentally
        modifying the registry's internal state.
        """
        return list(self._records)

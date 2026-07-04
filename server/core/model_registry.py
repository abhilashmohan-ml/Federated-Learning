"""
Global model version registry.

Tracks every version of the global model with its metrics and round info.
In production this would persist to the DB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from shared.schemas.federation import GlobalModel


@dataclass
class ModelRecord:
    version: int
    round_id: int
    weights_summary: Dict[str, int]     # layer_name -> n_params
    global_metrics: Dict[str, float]


class ModelRegistry:
    def __init__(self) -> None:
        self._records: List[ModelRecord] = []

    def register(self, model: GlobalModel) -> None:
        record = ModelRecord(
            version=model.version,
            round_id=model.round_id,
            weights_summary={k: len(v) for k, v in model.weights.items()},
            global_metrics=model.global_metrics,
        )
        self._records.append(record)

    def latest(self) -> ModelRecord | None:
        return self._records[-1] if self._records else None

    def history(self) -> List[ModelRecord]:
        return list(self._records)

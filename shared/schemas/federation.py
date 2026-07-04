"""Pydantic schemas for the FL federation protocol messages."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RoundStatus(str, Enum):
    PENDING     = "pending"
    COLLECTING  = "collecting"
    AGGREGATING = "aggregating"
    COMPLETE    = "complete"
    FAILED      = "failed"


class SiteStatus(str, Enum):
    REGISTERED = "registered"
    IDLE       = "idle"
    TRAINING   = "training"
    UPLOADING  = "uploading"
    DONE       = "done"
    ERROR      = "error"


class FederationRound(BaseModel):
    round_id: int
    status: RoundStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    participating_sites: List[str] = Field(default_factory=list)
    global_model_version: int = 0


class ModelUpdate(BaseModel):
    site_id: str
    round_id: int
    n_samples: int
    delta_W: Dict[str, List[float]]     # layer_name -> flattened gradient list
    dp_noise_sigma: float = 0.0
    hermia_best_model: str = "combined_1a"
    local_metrics: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GlobalModel(BaseModel):
    version: int
    round_id: int
    weights: Dict[str, List[float]]     # layer_name -> flattened weights
    global_metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

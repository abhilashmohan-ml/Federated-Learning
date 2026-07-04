"""
Pydantic schemas for the FL federation protocol messages.

OVERVIEW
--------
These schemas define the "language" that the server and sites speak to each other.
Every message sent over HTTP is validated against one of these schemas.

Key schemas:
  FederationRound  — describes the current state of one FL training round
  ModelUpdate      — what a site sends to the server after local training
  GlobalModel      — what the server sends back after aggregation
  RoundStatus      — enum: the stages a round goes through
  SiteStatus       — enum: what a site is currently doing

PYTHON CONCEPT: Enum
  An Enum is a set of named constants. Instead of using raw strings like
  "collecting" throughout the code (which is error-prone — a typo like
  "colecting" would not be caught), we use RoundStatus.COLLECTING.
  Python will raise an error if you use a value not in the enum.

PYTHON CONCEPT: Field(default_factory=list)
  For list fields, you MUST use `default_factory=list` instead of `default=[]`.
  This is because if you write `default=[]`, Python reuses the SAME list object
  for every instance of the class (a classic Python gotcha). `default_factory`
  creates a fresh empty list for each new instance.
"""

from __future__ import annotations   # allows using type names before they are defined

from datetime import datetime, timezone  # datetime: a point in time; timezone: timezone info
from enum import Enum                    # base class for enumerations
from typing import Dict, List, Optional  # typing helpers for Python < 3.10 compatibility

from pydantic import BaseModel, Field    # Field() adds metadata like default_factory


class RoundStatus(str, Enum):
    """
    The lifecycle stages of a single FL round.

    Inheriting from `str` makes each value a plain string as well as an enum
    member, so it serialises to JSON naturally (e.g. "collecting" not 2).

    PENDING     → The round has been created but not yet started collecting updates.
    COLLECTING  → Actively waiting for site updates (this is the main active state).
    AGGREGATING → Enough updates received; the server is computing the new global model.
    COMPLETE    → Aggregation finished successfully; new global model is available.
    FAILED      → Something went wrong (e.g. zero updates received at timeout).
    """
    PENDING     = "pending"
    COLLECTING  = "collecting"
    AGGREGATING = "aggregating"
    COMPLETE    = "complete"
    FAILED      = "failed"


class SiteStatus(str, Enum):
    """
    What a particular site is currently doing, as tracked by the server.

    REGISTERED → The site exists in the database but hasn't started anything yet.
    IDLE       → The site is connected and waiting for the next round to start.
    TRAINING   → The site is running local model fitting (Hermia, PINN training).
    UPLOADING  → The site has finished training and is sending its update to the server.
    DONE       → The site has successfully submitted its update for this round.
    ERROR      → Something went wrong on the site; it needs intervention.
    """
    REGISTERED = "registered"
    IDLE       = "idle"
    TRAINING   = "training"
    UPLOADING  = "uploading"
    DONE       = "done"
    ERROR      = "error"


class FederationRound(BaseModel):
    """
    Describes the state of one complete FL training round.

    This is returned by GET /federation/round/{id} and by
    POST /federation/round/start so clients know the round has begun.

    Fields
    ------
    round_id              : monotonically increasing integer (1, 2, 3, …)
    status                : current stage — see RoundStatus enum above
    started_at            : UTC timestamp when the round was created
    completed_at          : UTC timestamp when aggregation finished (None if not yet)
    participating_sites   : list of site_ids that have submitted updates so far
    global_model_version  : the version number of the global model produced by this round
    """
    round_id:             int
    status:               RoundStatus
    started_at:           datetime
    completed_at:         Optional[datetime] = None   # None until the round completes
    participating_sites:  List[str]          = Field(default_factory=list)
    global_model_version: int                = 0      # 0 until aggregation completes


class ModelUpdate(BaseModel):
    """
    The payload a site sends to POST /federation/update after local training.

    WHAT IS delta_W?
    ----------------
    In federated learning, sites share model "weight updates" (delta_W), not raw
    data. A neural network's "weights" are its internal numbers that it learned.
    delta_W is the CHANGE in those weights after training on local data.

    delta_W is structured as a dictionary:
      key   → layer name (e.g. "hermia_params", "predictor.net.0.weight")
      value → flat list of floats (the weight values, serialised as a list)

    WHY FLATTEN TENSORS TO LISTS?
    -----------------------------
    Neural network weight tensors are multi-dimensional arrays (matrices).
    JSON cannot represent multi-dimensional arrays natively, so we flatten
    them to one-dimensional lists before sending. The server knows the
    original shape from the global model and can reconstruct it.

    Fields
    ------
    site_id           : which site is sending this update — the server checks this
                        matches the JWT token to prevent spoofing
    round_id          : which round this update belongs to
    n_samples         : how many training samples this site used — used as the
                        weight in the FedProx weighted average
    delta_W           : the model weight updates (see above)
    dp_noise_sigma    : the sigma of the DP noise that was added — for audit
    hermia_best_model : which Hermia model had the lowest AIC on this site
    local_metrics     : performance numbers from this training run (flux_rmse etc.)
    timestamp         : when this update was created, in UTC
    """
    site_id:           str
    round_id:          int
    n_samples:         int                          # training sample count for FedProx weighting
    delta_W:           Dict[str, List[float]]       # layer_name -> flattened weight list
    dp_noise_sigma:    float = 0.0                  # DP noise level applied (for audit trail)
    hermia_best_model: str   = "combined_1a"        # best-fit Hermia model for this run
    local_metrics:     Dict[str, float] = Field(default_factory=dict)  # rmse, lrv, etc.
    timestamp:         datetime = Field(
        # default_factory calls datetime.now(timezone.utc) each time a new
        # ModelUpdate is created, so every update gets the current time.
        default_factory=lambda: datetime.now(timezone.utc)
    )


class GlobalModel(BaseModel):
    """
    The aggregated global model that the server produces after each round.

    Sites download this via GET /models/global-model and use the weights
    as the starting point for the NEXT round of local training.

    WHY DOES THIS LOOK LIKE ModelUpdate?
    -------------------------------------
    Both contain weights as Dict[str, List[float]]. The difference is:
      - ModelUpdate.delta_W   : a CHANGE to apply to the current global weights
      - GlobalModel.weights   : the COMPLETE current global model weights

    Fields
    ------
    version        : incremented by 1 every time aggregation succeeds
    round_id       : which round produced this model version
    weights        : the complete aggregated weights (layer_name -> flat list)
    global_metrics : average performance across all sites (flux_rmse etc.)
    created_at     : when this model version was produced
    """
    version:        int
    round_id:       int
    weights:        Dict[str, List[float]]      # complete global weights
    global_metrics: Dict[str, float] = Field(default_factory=dict)
    created_at:     datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

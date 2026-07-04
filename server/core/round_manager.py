"""
Federation round state machine — the brain of the FL server.

WHAT IS A STATE MACHINE?
--------------------------
A state machine is a system that can be in one of a finite set of states
and transitions between them based on events. Our FL round lifecycle has
the following states:

    PENDING  →  COLLECTING  →  AGGREGATING  →  COMPLETE
                                             →  FAILED

  COLLECTING: waiting for site updates to arrive
  AGGREGATING: calculating the new global model from received updates
  COMPLETE: aggregation finished; new global model is available
  FAILED: aggregation failed (no updates, or exception in aggregator)

  PENDING appears in the schema but is not used in the current code —
  rounds jump straight to COLLECTING when started.

HOW ROUNDS WORK IN PRACTICE
-----------------------------
1. Someone calls POST /federation/round/start
2. RoundManager creates a new FederationRound with status=COLLECTING
3. An asyncio background task starts a countdown (timeout guard)
4. Sites train locally and call POST /federation/update one by one
5. When min_sites_per_round updates arrive → trigger aggregation
   OR when the timeout fires → aggregate with whatever updates arrived
6. FedProxAggregator computes the new global weights
7. Round is marked COMPLETE; global weights updated in memory

IMPORTANT: Current implementation stores everything IN MEMORY.
This means a server restart loses all round history. A production
implementation would persist rounds and updates to the PostgreSQL
database after each state transition.

PYTHON CONCEPT: asyncio.create_task()
  Creates a concurrent background task that runs alongside the current
  coroutine. The timeout guard runs independently — the endpoint that called
  `start_new_round()` returns immediately while the timer runs in the background.

PYTHON CONCEPT: @lru_cache(maxsize=1)
  Used on `get_round_manager()` to create a singleton — one RoundManager
  instance for the entire server lifetime. FastAPI endpoints use this via
  `Depends(get_round_manager)`.

PYTHON CONCEPT: @property
  `current_global_weights` is decorated with @property, which means you can
  access it as `rm.current_global_weights` (like an attribute) rather than
  calling it as `rm.current_global_weights()` (like a method).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List, Optional

from shared.schemas.federation import (
    FederationRound, ModelUpdate, RoundStatus, SiteStatus,
)
from server.core.aggregator import FedProxAggregator
from server.config import get_settings
from shared.utils.logging_config import get_logger

log = get_logger(__name__)


class RoundManager:
    """
    Manages the complete lifecycle of FL rounds in memory.

    Attributes (all private, accessed via methods/properties):
    ---------------------------------------------------------
    _rounds         : dict of round_id → FederationRound (round metadata)
    _updates        : dict of round_id → list[ModelUpdate] (updates buffer)
    _current_global : the latest aggregated global model weights
    _model_version  : version counter for the global model
    _current_round_id: the ID of the most recently started round
    _site_statuses  : dict of site_id → SiteStatus for dashboard display
    _aggregator     : the FedProxAggregator instance
    _settings       : server configuration (min_sites, timeout, etc.)
    """

    def __init__(self) -> None:
        self._rounds:           Dict[int, FederationRound] = {}
        self._updates:          Dict[int, List[ModelUpdate]] = {}
        self._current_global:   Dict[str, List[float]] = {}      # empty until first round
        self._model_version:    int = 0
        self._current_round_id: int = 0

        # Initialise site statuses for all 5 sites — all IDLE at startup
        self._site_statuses: Dict[str, SiteStatus] = {
            f"site_{i}": SiteStatus.IDLE for i in range(1, 6)
        }

        self._aggregator = FedProxAggregator()
        self._settings   = get_settings()

    async def start_new_round(self) -> FederationRound:
        """
        Create and start a new FL round.

        Increments the round counter, creates the round data structure,
        and launches a background timeout guard.

        Returns
        -------
        FederationRound — the newly created round (in COLLECTING state)
        """
        self._current_round_id += 1
        rid = self._current_round_id

        # Initialise an empty list to accumulate updates for this round
        self._updates[rid] = []

        round_ = FederationRound(
            round_id=rid,
            status=RoundStatus.COLLECTING,
            started_at=datetime.now(timezone.utc),
        )
        self._rounds[rid] = round_

        # Launch the timeout guard as a background asyncio task.
        # This task sleeps for `round_timeout_seconds`, then triggers aggregation
        # if the round is still COLLECTING (i.e., hasn't finished yet).
        asyncio.create_task(self._timeout_guard(rid))

        log.info("round_started", round_id=rid)
        return round_

    async def receive_update(self, update: ModelUpdate) -> None:
        """
        Accept and buffer a model update from a manufacturing site.

        After buffering, checks if enough sites have responded to trigger
        aggregation. This implements the "collect until quorum" logic.

        Parameters
        ----------
        update : ModelUpdate — contains site_id, round_id, delta_W, n_samples, metrics
        """
        rid = update.round_id

        # Reject updates for unknown rounds (shouldn't normally happen)
        if rid not in self._rounds:
            log.warning("unknown_round", round_id=rid, site=update.site_id)
            return

        # Buffer the update
        self._updates[rid].append(update)
        self._rounds[rid].participating_sites.append(update.site_id)

        # Mark this site as DONE (dashboard will show it in green)
        self._site_statuses[update.site_id] = SiteStatus.DONE

        log.info(
            "update_received",
            site=update.site_id,
            round_id=rid,
            n_updates=len(self._updates[rid]),
        )

        # Trigger aggregation when the quorum threshold is reached.
        # `min_sites_per_round` is configurable — default 3 out of 5 sites.
        if len(self._updates[rid]) >= self._settings.min_sites_per_round:
            await self._aggregate(rid)

    async def _timeout_guard(self, round_id: int) -> None:
        """
        Background task: aggregate the round after a timeout even if not all sites responded.

        This prevents a round from hanging forever if one or more sites crash,
        lose network connectivity, or take too long. The server aggregates with
        whatever updates have arrived by the deadline.

        The guard checks whether the round is still COLLECTING before acting —
        if aggregation already completed (quorum reached), this is a no-op.
        """
        # Sleep for the configured timeout period
        await asyncio.sleep(self._settings.round_timeout_seconds)

        if self._rounds[round_id].status == RoundStatus.COLLECTING:
            log.info("round_timeout", round_id=round_id)
            await self._aggregate(round_id)

    async def _aggregate(self, round_id: int) -> None:
        """
        Perform FedProx aggregation for the given round.

        This method is idempotent — if called while the round is not in
        COLLECTING state (e.g., if both the timeout guard and the quorum
        trigger fire simultaneously), it returns immediately.

        The status transition ensures only one aggregation runs:
            COLLECTING → AGGREGATING → COMPLETE (or FAILED)
        """
        r = self._rounds[round_id]

        # Guard against double-aggregation (only run once per round)
        if r.status != RoundStatus.COLLECTING:
            return

        r.status = RoundStatus.AGGREGATING   # mark as in-progress
        updates = self._updates[round_id]

        if not updates:
            # No sites responded — cannot aggregate empty data
            r.status = RoundStatus.FAILED
            return

        try:
            # Run the FedProx aggregation algorithm
            gm = self._aggregator.aggregate(
                updates,
                self._current_global,    # current weights before this round
                round_id,
                self._model_version,
            )

            # Update the in-memory global model
            self._current_global = gm.weights
            self._model_version  = gm.version

            r.status = RoundStatus.COMPLETE
            r.completed_at = datetime.now(timezone.utc)
            r.global_model_version = gm.version

            log.info("round_complete", round_id=round_id, model_version=gm.version)

        except Exception as exc:
            r.status = RoundStatus.FAILED
            log.error("aggregation_failed", round_id=round_id, error=str(exc))

    async def get_round(self, round_id: int) -> Optional[FederationRound]:
        """
        Look up a round by ID.

        Returns None if the round_id doesn't exist (caller should return 404).
        """
        return self._rounds.get(round_id)

    async def get_site_statuses(self) -> Dict[str, str]:
        """
        Return the current status of all 5 sites as a dict of strings.

        Converts SiteStatus enum values to plain strings for JSON serialisation.
        Example: {"site_1": "done", "site_2": "idle", "site_3": "training", ...}
        """
        return {k: v.value for k, v in self._site_statuses.items()}

    @property
    def current_global_weights(self) -> Dict[str, List[float]]:
        """
        Read-only access to the current global model weights.

        Returns an empty dict before the first round completes.
        Used by the /models/global-model endpoint.

        PYTHON CONCEPT: @property
          The @property decorator makes this a "computed attribute" —
          it looks like `rm.current_global_weights` to callers, not
          `rm.current_global_weights()`. This hides the implementation
          detail that `_current_global` is a private attribute.
        """
        return self._current_global


@lru_cache(maxsize=1)
def get_round_manager() -> RoundManager:
    """
    Return the singleton RoundManager instance.

    @lru_cache(maxsize=1) ensures this function creates the RoundManager
    object exactly once and returns the same object on every call.
    This is essential — all requests must share the same state.

    If this were not cached, each request would create its own RoundManager
    with empty state, and no round data would be visible across requests.

    Usage in FastAPI endpoints:
        rm: RoundManager = Depends(get_round_manager)
    """
    return RoundManager()

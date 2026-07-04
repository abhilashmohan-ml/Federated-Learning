"""
Federation round state machine.

States
------
  PENDING  ->  COLLECTING  ->  AGGREGATING  ->  COMPLETE
                                             ->  FAILED
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
    def __init__(self) -> None:
        self._rounds: Dict[int, FederationRound] = {}
        self._updates: Dict[int, List[ModelUpdate]] = {}
        self._current_global: Dict[str, List[float]] = {}
        self._model_version: int = 0
        self._current_round_id: int = 0
        self._site_statuses: Dict[str, SiteStatus] = {
            f"site_{i}": SiteStatus.IDLE for i in range(1, 6)
        }
        self._aggregator = FedProxAggregator()
        self._settings = get_settings()

    async def start_new_round(self) -> FederationRound:
        self._current_round_id += 1
        rid = self._current_round_id
        self._updates[rid] = []
        round_ = FederationRound(
            round_id=rid,
            status=RoundStatus.COLLECTING,
            started_at=datetime.now(timezone.utc),
        )
        self._rounds[rid] = round_
        asyncio.create_task(self._timeout_guard(rid))
        log.info("round_started", round_id=rid)
        return round_

    async def receive_update(self, update: ModelUpdate) -> None:
        rid = update.round_id
        if rid not in self._rounds:
            log.warning("unknown_round", round_id=rid, site=update.site_id)
            return
        self._updates[rid].append(update)
        self._rounds[rid].participating_sites.append(update.site_id)
        self._site_statuses[update.site_id] = SiteStatus.DONE
        log.info("update_received", site=update.site_id, round_id=rid,
                 n_updates=len(self._updates[rid]))
        if len(self._updates[rid]) >= self._settings.min_sites_per_round:
            await self._aggregate(rid)

    async def _timeout_guard(self, round_id: int) -> None:
        await asyncio.sleep(self._settings.round_timeout_seconds)
        if self._rounds[round_id].status == RoundStatus.COLLECTING:
            log.info("round_timeout", round_id=round_id)
            await self._aggregate(round_id)

    async def _aggregate(self, round_id: int) -> None:
        r = self._rounds[round_id]
        if r.status != RoundStatus.COLLECTING:
            return
        r.status = RoundStatus.AGGREGATING
        updates = self._updates[round_id]
        if not updates:
            r.status = RoundStatus.FAILED
            return
        try:
            gm = self._aggregator.aggregate(
                updates, self._current_global, round_id, self._model_version,
            )
            self._current_global = gm.weights
            self._model_version = gm.version
            r.status = RoundStatus.COMPLETE
            r.completed_at = datetime.now(timezone.utc)
            r.global_model_version = gm.version
            log.info("round_complete", round_id=round_id, model_version=gm.version)
        except Exception as exc:
            r.status = RoundStatus.FAILED
            log.error("aggregation_failed", round_id=round_id, error=str(exc))

    async def get_round(self, round_id: int) -> Optional[FederationRound]:
        return self._rounds.get(round_id)

    async def get_site_statuses(self) -> Dict[str, str]:
        return {k: v.value for k, v in self._site_statuses.items()}

    @property
    def current_global_weights(self) -> Dict[str, List[float]]:
        return self._current_global


@lru_cache(maxsize=1)
def get_round_manager() -> RoundManager:
    return RoundManager()

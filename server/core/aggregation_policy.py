"""Pluggable aggregation trigger policies for the FL round manager."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class AggregationPolicy(Protocol):
    """Decides whether to trigger FedProx aggregation given current round state."""

    def should_aggregate(
        self,
        updates_since_last: int,
        sites_contributed: set[str],
        elapsed_seconds: float,
    ) -> bool:
        ...  # pragma: no cover


@dataclass
class QuorumPolicy:
    """Aggregate when at least `min_sites` distinct sites have contributed."""

    min_sites: int = 3

    def should_aggregate(
        self,
        updates_since_last: int,
        sites_contributed: set[str],
        elapsed_seconds: float,
    ) -> bool:
        return len(sites_contributed) >= self.min_sites


@dataclass
class TimeWindowPolicy:
    """Aggregate when `window_seconds` have elapsed since the round started AND ≥1 update arrived."""

    window_seconds: int = 1800

    def should_aggregate(
        self,
        updates_since_last: int,
        sites_contributed: set[str],
        elapsed_seconds: float,
    ) -> bool:
        return updates_since_last >= 1 and elapsed_seconds >= self.window_seconds

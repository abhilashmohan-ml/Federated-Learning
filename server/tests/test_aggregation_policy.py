"""Tests for aggregation policy implementations."""
from __future__ import annotations

import pytest

from server.core.aggregation_policy import QuorumPolicy, TimeWindowPolicy


# ── QuorumPolicy ─────────────────────────────────────────────────────────────


def test_quorum_not_met_below_threshold() -> None:
    p = QuorumPolicy(min_sites=3)
    assert not p.should_aggregate(2, {"site_1", "site_2"}, 0.0)


def test_quorum_met_exactly_at_threshold() -> None:
    p = QuorumPolicy(min_sites=3)
    assert p.should_aggregate(3, {"site_1", "site_2", "site_3"}, 0.0)


def test_quorum_met_above_threshold() -> None:
    p = QuorumPolicy(min_sites=3)
    assert p.should_aggregate(5, {"s1", "s2", "s3", "s4", "s5"}, 0.0)


def test_quorum_ignores_elapsed_time() -> None:
    p = QuorumPolicy(min_sites=3)
    assert not p.should_aggregate(2, {"site_1", "site_2"}, 99999.0)


def test_quorum_uses_sites_contributed_count() -> None:
    # updates_since_last=5 but only 2 unique sites → not met
    p = QuorumPolicy(min_sites=3)
    assert not p.should_aggregate(5, {"site_1", "site_2"}, 0.0)


# ── TimeWindowPolicy ─────────────────────────────────────────────────────────


def test_timewindow_not_met_before_window() -> None:
    p = TimeWindowPolicy(window_seconds=300)
    assert not p.should_aggregate(1, {"site_1"}, 299.9)


def test_timewindow_met_at_exact_window() -> None:
    p = TimeWindowPolicy(window_seconds=300)
    assert p.should_aggregate(1, {"site_1"}, 300.0)


def test_timewindow_met_after_window() -> None:
    p = TimeWindowPolicy(window_seconds=300)
    assert p.should_aggregate(3, {"site_1", "site_2", "site_3"}, 400.0)


def test_timewindow_not_triggered_on_zero_updates() -> None:
    p = TimeWindowPolicy(window_seconds=300)
    assert not p.should_aggregate(0, set(), 999.0)


def test_timewindow_ignores_site_count() -> None:
    p = TimeWindowPolicy(window_seconds=300)
    # Only 1 site — still aggregates when window elapsed
    assert p.should_aggregate(1, {"site_1"}, 301.0)

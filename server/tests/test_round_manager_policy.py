"""Tests for pluggable AggregationPolicy in RoundManager and run-count tracking."""
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from shared.schemas.federation import ModelUpdate, RoundStatus
from server.core.round_manager import RoundManager
from server.core.aggregation_policy import QuorumPolicy, TimeWindowPolicy


def _make_update(site_id: str, round_id: int) -> ModelUpdate:
    return ModelUpdate(
        site_id=site_id,
        round_id=round_id,
        n_samples=100,
        delta_W={"hermia_params": [1.0, 0.01, 0.001]},
        dp_noise_sigma=0.01,
        hermia_best_model="combined_1a",
        local_metrics={"flux_rmse": 1.0, "flux_ratio": 0.7, "amin_m2": 0.005,
                       "best_aic": -10.0, "best_bic": -8.0},
    )


@pytest.mark.asyncio
async def test_default_policy_is_quorum() -> None:
    rm = RoundManager()
    assert isinstance(rm._policy, QuorumPolicy)


@pytest.mark.asyncio
async def test_set_policy_swaps_live() -> None:
    rm = RoundManager()
    rm.set_policy(TimeWindowPolicy(window_seconds=9999))
    assert isinstance(rm._policy, TimeWindowPolicy)


@pytest.mark.asyncio
async def test_quorum_policy_triggers_at_min_sites() -> None:
    rm = RoundManager()
    rm.set_policy(QuorumPolicy(min_sites=2))
    await rm.start_new_round()
    await rm.receive_update(_make_update("site_a", 1))
    assert rm._rounds[1].status == RoundStatus.COLLECTING   # 1 site, need 2
    await rm.receive_update(_make_update("site_b", 1))
    assert rm._rounds[1].status in (RoundStatus.COMPLETE, RoundStatus.AGGREGATING)


@pytest.mark.asyncio
async def test_run_counts_incremented_per_site() -> None:
    rm = RoundManager()
    rm.set_policy(QuorumPolicy(min_sites=5))   # prevent aggregation
    await rm.start_new_round()
    await rm.receive_update(_make_update("site_a", 1))
    await rm.receive_update(_make_update("site_a", 1))
    assert rm._site_run_counts["site_a"] == 2


@pytest.mark.asyncio
async def test_last_run_at_set_on_update() -> None:
    rm = RoundManager()
    rm.set_policy(QuorumPolicy(min_sites=5))
    await rm.start_new_round()
    await rm.receive_update(_make_update("site_c", 1))
    assert rm._site_last_run_at["site_c"] is not None


@pytest.mark.asyncio
async def test_get_or_create_reuses_collecting_round() -> None:
    rm = RoundManager()
    r1 = await rm.get_or_create_round()
    r2 = await rm.get_or_create_round()
    assert r1.round_id == r2.round_id


@pytest.mark.asyncio
async def test_get_or_create_starts_new_after_complete() -> None:
    rm = RoundManager()
    rm.set_policy(QuorumPolicy(min_sites=1))
    r1 = await rm.get_or_create_round()
    await rm.receive_update(_make_update("site_a", r1.round_id))
    # round 1 is now complete — get_or_create should make round 2
    r2 = await rm.get_or_create_round()
    assert r2.round_id == r1.round_id + 1


@pytest.mark.asyncio
async def test_sync_site_run_info_updates_if_remote_higher() -> None:
    rm = RoundManager()
    rm.sync_site_run_info("site_b", 7, datetime.now(timezone.utc))
    assert rm._site_run_counts["site_b"] == 7


@pytest.mark.asyncio
async def test_sync_site_run_info_ignores_if_remote_lower() -> None:
    rm = RoundManager()
    rm._site_run_counts["site_b"] = 10
    rm.sync_site_run_info("site_b", 5, None)
    assert rm._site_run_counts["site_b"] == 10


@pytest.mark.asyncio
async def test_status_snapshot_includes_run_info() -> None:
    rm = RoundManager()
    snap = await rm.get_status_snapshot()
    assert "run_counts" in snap
    assert "last_run_at" in snap
    assert isinstance(snap["run_counts"], dict)   # starts empty — no hardcoded sites


@pytest.mark.asyncio
async def test_sync_site_run_info_higher_count_no_timestamp() -> None:
    rm = RoundManager()
    rm.sync_site_run_info("site_z", 3, None)
    assert rm._site_run_counts["site_z"] == 3
    assert "site_z" not in rm._site_last_run_at


@pytest.mark.asyncio
async def test_mark_site_error_sets_error_status() -> None:
    from shared.schemas.federation import SiteStatus
    rm = RoundManager()
    rm.mark_site_error("site_x")
    assert rm._site_statuses["site_x"] == SiteStatus.ERROR

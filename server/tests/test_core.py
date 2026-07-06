"""Unit tests for server/core — aggregator, round_manager, model_registry. 100% coverage."""
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.schemas.federation import (
    FederationRound, GlobalModel, ModelUpdate, RoundStatus, SiteStatus,
)
from server.core.aggregator import FedProxAggregator
from server.core.model_registry import ModelRecord, ModelRegistry
from server.core.round_manager import RoundManager, get_round_manager


# ── helpers ────────────────────────────────────────────────────────────────────

def _mock_settings(min_sites: int = 2, timeout: int = 300) -> MagicMock:
    s = MagicMock()
    s.min_sites_per_round = min_sites
    s.round_timeout_seconds = timeout
    return s


def _make_update(
    site_id: str = "site_1",
    round_id: int = 1,
    n_samples: int = 100,
    layers: dict | None = None,
    metrics: dict | None = None,
) -> ModelUpdate:
    return ModelUpdate(
        site_id=site_id,
        round_id=round_id,
        n_samples=n_samples,
        delta_W=layers or {"hermia_params": [1.0, 2.0, 3.0]},
        local_metrics=metrics or {},
    )


# ── FedProxAggregator ──────────────────────────────────────────────────────────

class TestFedProxAggregatorEmpty:
    def test_empty_updates_raises_value_error(self) -> None:
        agg = FedProxAggregator()
        with pytest.raises(ValueError):
            agg.aggregate([], {}, round_id=1, model_version=0)


class TestFedProxAggregatorBasic:
    def test_returns_global_model(self) -> None:
        agg = FedProxAggregator()
        updates = [
            _make_update("site_1", layers={"p": [1.0, 0.0]}),
            _make_update("site_2", layers={"p": [0.0, 2.0]}),
        ]
        gm = agg.aggregate(updates, {"p": [0.0, 0.0]}, round_id=1, model_version=0)
        assert isinstance(gm, GlobalModel)

    def test_version_incremented(self) -> None:
        agg = FedProxAggregator()
        gm = agg.aggregate([_make_update()], {"hermia_params": [0.0, 0.0, 0.0]},
                           round_id=1, model_version=4)
        assert gm.version == 5

    def test_round_id_preserved(self) -> None:
        agg = FedProxAggregator()
        gm = agg.aggregate([_make_update()], {}, round_id=7, model_version=0)
        assert gm.round_id == 7

    def test_weighted_average_equal_weights(self) -> None:
        """Two sites, equal n_samples → average of updates."""
        agg = FedProxAggregator()
        updates = [
            _make_update("site_1", n_samples=100, layers={"p": [10.0]}),
            _make_update("site_2", n_samples=100, layers={"p": [2.0]}),
        ]
        gm = agg.aggregate(updates, {"p": [0.0]}, round_id=1, model_version=0)
        # W_new = 0.5*(0+10) + 0.5*(0+2) = 5+1 = 6
        assert gm.weights["p"][0] == pytest.approx(6.0)

    def test_weighted_average_unequal_weights(self) -> None:
        """site_1 has 3x samples → closer to site_1 update."""
        agg = FedProxAggregator()
        updates = [
            _make_update("site_1", n_samples=300, layers={"p": [10.0]}),
            _make_update("site_2", n_samples=100, layers={"p": [2.0]}),
        ]
        gm = agg.aggregate(updates, {"p": [0.0]}, round_id=1, model_version=0)
        # W_new = 0.75*(0+10) + 0.25*(0+2) = 7.5 + 0.5 = 8.0
        assert gm.weights["p"][0] == pytest.approx(8.0)

    def test_missing_layer_in_second_update_skipped(self) -> None:
        """Layer absent from one update is skipped for that update only."""
        agg = FedProxAggregator()
        updates = [
            _make_update("site_1", n_samples=100, layers={"a": [5.0], "b": [3.0]}),
            _make_update("site_2", n_samples=100, layers={"a": [1.0]}),  # no "b"
        ]
        gm = agg.aggregate(updates, {"a": [0.0], "b": [0.0]}, round_id=1, model_version=0)
        # Both layers present
        assert "a" in gm.weights
        assert "b" in gm.weights
        # "a": both sites → 0.5*(0+5) + 0.5*(0+1) = 3.0
        assert gm.weights["a"][0] == pytest.approx(3.0)
        # "b": site_2 skipped (no "b" in delta) → 0.5*(0+3) = 1.5
        assert gm.weights["b"][0] == pytest.approx(1.5)

    def test_layer_absent_in_global_defaults_zeros(self) -> None:
        """Layer not in current_global defaults to zero-vector as base."""
        agg = FedProxAggregator()
        updates = [_make_update(layers={"new_layer": [1.0, 2.0]})]
        gm = agg.aggregate(updates, current_global={}, round_id=1, model_version=0)
        assert "new_layer" in gm.weights


class TestFedProxAggregatorMetrics:
    def test_metrics_averaged(self) -> None:
        agg = FedProxAggregator()
        updates = [
            _make_update("site_1", metrics={"flux_rmse": 2.0, "lrv_rmse": 0.5}),
            _make_update("site_2", metrics={"flux_rmse": 4.0, "lrv_rmse": 1.5}),
        ]
        gm = agg.aggregate(updates, {}, round_id=1, model_version=0)
        assert gm.global_metrics["flux_rmse"] == pytest.approx(3.0)
        assert gm.global_metrics["lrv_rmse"] == pytest.approx(1.0)

    def test_metric_absent_in_some_updates_still_averaged(self) -> None:
        agg = FedProxAggregator()
        updates = [
            _make_update("site_1", metrics={"flux_rmse": 2.0}),
            _make_update("site_2", metrics={}),
        ]
        gm = agg.aggregate(updates, {}, round_id=1, model_version=0)
        assert gm.global_metrics.get("flux_rmse") == pytest.approx(2.0)

    def test_metric_absent_everywhere_not_in_output(self) -> None:
        """If no update has a metric, it must NOT appear in global_metrics."""
        agg = FedProxAggregator()
        updates = [_make_update(metrics={}), _make_update(metrics={})]
        gm = agg.aggregate(updates, {}, round_id=1, model_version=0)
        assert "flux_rmse" not in gm.global_metrics


# ── ModelRegistry ─────────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_latest_empty_none(self) -> None:
        assert ModelRegistry().latest() is None

    def test_history_empty_list(self) -> None:
        assert ModelRegistry().history() == []

    def test_register_and_latest(self) -> None:
        reg = ModelRegistry()
        gm = GlobalModel(version=1, round_id=1, weights={"l": [1.0, 2.0]},
                         global_metrics={"flux_rmse": 1.5})
        reg.register(gm)
        rec = reg.latest()
        assert rec is not None
        assert rec.version == 1
        assert rec.round_id == 1
        assert rec.weights_summary == {"l": 2}
        assert rec.global_metrics == {"flux_rmse": 1.5}

    def test_latest_returns_last(self) -> None:
        reg = ModelRegistry()
        for i in range(1, 4):
            reg.register(GlobalModel(version=i, round_id=i, weights={"l": [0.0]}))
        assert reg.latest().version == 3

    def test_history_length(self) -> None:
        reg = ModelRegistry()
        for i in range(1, 4):
            reg.register(GlobalModel(version=i, round_id=i, weights={}))
        assert len(reg.history()) == 3

    def test_history_returns_copy_not_internal(self) -> None:
        reg = ModelRegistry()
        reg.register(GlobalModel(version=1, round_id=1, weights={"l": [0.0]}))
        h = reg.history()
        h.clear()
        assert len(reg.history()) == 1  # internal list unchanged

    def test_model_record_dataclass(self) -> None:
        mr = ModelRecord(version=2, round_id=2, weights_summary={"l": 3}, global_metrics={})
        assert mr.version == 2
        assert mr.round_id == 2


# ── RoundManager ──────────────────────────────────────────────────────────────

class TestRoundManager:
    def _make_rm(self, min_sites: int = 5, timeout: int = 300) -> RoundManager:
        with patch("server.core.round_manager.get_settings",
                   return_value=_mock_settings(min_sites=min_sites, timeout=timeout)):
            return RoundManager()

    # ── start_new_round ────────────────────────────────────────────────────────

    def test_start_round_returns_federation_round(self) -> None:
        async def _run():
            rm = self._make_rm()
            with patch("asyncio.create_task", return_value=MagicMock()):
                return await rm.start_new_round()
        r = asyncio.run(_run())
        assert isinstance(r, FederationRound)

    def test_start_round_status_collecting(self) -> None:
        async def _run():
            rm = self._make_rm()
            with patch("asyncio.create_task", return_value=MagicMock()):
                return await rm.start_new_round()
        r = asyncio.run(_run())
        assert r.status == RoundStatus.COLLECTING

    def test_start_round_id_increments(self) -> None:
        async def _run():
            rm = self._make_rm()
            with patch("asyncio.create_task", return_value=MagicMock()):
                r1 = await rm.start_new_round()
                r2 = await rm.start_new_round()
            return r1.round_id, r2.round_id
        id1, id2 = asyncio.run(_run())
        assert id1 == 1
        assert id2 == 2

    # ── receive_update ─────────────────────────────────────────────────────────

    def test_receive_update_unknown_round_ignored(self) -> None:
        async def _run():
            rm = self._make_rm()
            await rm.receive_update(_make_update(round_id=99))
        asyncio.run(_run())  # no exception

    def test_receive_update_marks_site_done(self) -> None:
        async def _run():
            rm = self._make_rm(min_sites=10)
            with patch("asyncio.create_task", return_value=MagicMock()):
                r = await rm.start_new_round()
            await rm.receive_update(_make_update("site_1", round_id=r.round_id))
            return rm._site_statuses["site_1"]
        status = asyncio.run(_run())
        assert status == SiteStatus.DONE

    def test_receive_update_appends_participating_site(self) -> None:
        async def _run():
            rm = self._make_rm(min_sites=10)
            with patch("asyncio.create_task", return_value=MagicMock()):
                r = await rm.start_new_round()
            await rm.receive_update(_make_update("site_1", round_id=r.round_id))
            return rm._rounds[r.round_id].participating_sites
        sites = asyncio.run(_run())
        assert "site_1" in sites

    def test_receive_update_triggers_aggregation_at_quorum(self) -> None:
        async def _run():
            rm = self._make_rm(min_sites=2)
            with patch("asyncio.create_task", return_value=MagicMock()):
                r = await rm.start_new_round()
            await rm.receive_update(
                _make_update("site_1", round_id=r.round_id, layers={"p": [1.0]})
            )
            assert rm._rounds[r.round_id].status == RoundStatus.COLLECTING
            await rm.receive_update(
                _make_update("site_2", round_id=r.round_id, layers={"p": [2.0]})
            )
            return rm._rounds[r.round_id].status
        status = asyncio.run(_run())
        assert status == RoundStatus.COMPLETE

    # ── _aggregate ─────────────────────────────────────────────────────────────

    def test_aggregate_idempotent_non_collecting(self) -> None:
        async def _run():
            rm = self._make_rm()
            with patch("asyncio.create_task", return_value=MagicMock()):
                r = await rm.start_new_round()
            rm._rounds[r.round_id].status = RoundStatus.COMPLETE
            await rm._aggregate(r.round_id)  # should be a no-op
            return rm._rounds[r.round_id].status
        assert asyncio.run(_run()) == RoundStatus.COMPLETE

    def test_aggregate_no_updates_sets_failed(self) -> None:
        async def _run():
            rm = self._make_rm()
            with patch("asyncio.create_task", return_value=MagicMock()):
                r = await rm.start_new_round()
            # No updates buffered → should FAIL
            await rm._aggregate(r.round_id)
            return rm._rounds[r.round_id].status
        assert asyncio.run(_run()) == RoundStatus.FAILED

    def test_aggregate_exception_sets_failed(self) -> None:
        async def _run():
            rm = self._make_rm()
            with patch("asyncio.create_task", return_value=MagicMock()):
                r = await rm.start_new_round()
            rm._updates[r.round_id].append(
                _make_update("site_1", round_id=r.round_id, layers={"p": [1.0]})
            )
            with patch.object(rm._aggregator, "aggregate",
                              side_effect=RuntimeError("boom")):
                await rm._aggregate(r.round_id)
            return rm._rounds[r.round_id].status
        assert asyncio.run(_run()) == RoundStatus.FAILED

    def test_aggregate_success_sets_complete(self) -> None:
        async def _run():
            rm = self._make_rm()
            with patch("asyncio.create_task", return_value=MagicMock()):
                r = await rm.start_new_round()
            rm._updates[r.round_id].append(
                _make_update("site_1", round_id=r.round_id, layers={"p": [1.0]})
            )
            await rm._aggregate(r.round_id)
            return rm._rounds[r.round_id].status, rm.current_global_weights
        status, weights = asyncio.run(_run())
        assert status == RoundStatus.COMPLETE
        assert "p" in weights

    # ── _timeout_guard ─────────────────────────────────────────────────────────

    def test_timeout_guard_aggregates_when_collecting(self) -> None:
        async def _run():
            rm = self._make_rm()
            with patch("asyncio.create_task", return_value=MagicMock()):
                r = await rm.start_new_round()
            rm._updates[r.round_id].append(
                _make_update("site_1", round_id=r.round_id, layers={"p": [1.0]})
            )
            with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
                await rm._timeout_guard(r.round_id)
            return rm._rounds[r.round_id].status
        assert asyncio.run(_run()) == RoundStatus.COMPLETE

    def test_timeout_guard_noop_when_not_collecting(self) -> None:
        async def _run():
            rm = self._make_rm()
            with patch("asyncio.create_task", return_value=MagicMock()):
                r = await rm.start_new_round()
            rm._rounds[r.round_id].status = RoundStatus.COMPLETE
            with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
                await rm._timeout_guard(r.round_id)
            return rm._rounds[r.round_id].status
        assert asyncio.run(_run()) == RoundStatus.COMPLETE

    # ── get_round / get_site_statuses / current_global_weights ────────────────

    def test_get_round_unknown_returns_none(self) -> None:
        async def _run():
            return await self._make_rm().get_round(999)
        assert asyncio.run(_run()) is None

    def test_get_round_known_returns_round(self) -> None:
        async def _run():
            rm = self._make_rm()
            with patch("asyncio.create_task", return_value=MagicMock()):
                r = await rm.start_new_round()
            return await rm.get_round(r.round_id)
        r = asyncio.run(_run())
        assert r is not None and r.round_id == 1

    def test_get_site_statuses_all_five_sites(self) -> None:
        async def _run():
            return await self._make_rm().get_site_statuses()
        statuses = asyncio.run(_run())
        assert set(statuses.keys()) == {f"site_{i}" for i in range(1, 6)}

    def test_get_site_statuses_all_idle(self) -> None:
        async def _run():
            return await self._make_rm().get_site_statuses()
        statuses = asyncio.run(_run())
        assert all(v == "idle" for v in statuses.values())

    def test_current_global_weights_empty_initially(self) -> None:
        assert self._make_rm().current_global_weights == {}

    def test_current_global_weights_after_aggregation(self) -> None:
        async def _run():
            rm = self._make_rm(min_sites=1)
            with patch("asyncio.create_task", return_value=MagicMock()):
                r = await rm.start_new_round()
            await rm.receive_update(
                _make_update("site_1", round_id=r.round_id, layers={"q": [3.0, 4.0]})
            )
            return rm.current_global_weights
        weights = asyncio.run(_run())
        assert "q" in weights


# ── get_round_manager singleton ───────────────────────────────────────────────

class TestGetRoundManager:
    def test_returns_round_manager(self) -> None:
        get_round_manager.cache_clear()
        try:
            with patch("server.core.round_manager.get_settings",
                       return_value=_mock_settings()):
                rm = get_round_manager()
            assert isinstance(rm, RoundManager)
        finally:
            get_round_manager.cache_clear()

    def test_singleton_same_object(self) -> None:
        get_round_manager.cache_clear()
        try:
            with patch("server.core.round_manager.get_settings",
                       return_value=_mock_settings()):
                rm1 = get_round_manager()
                rm2 = get_round_manager()
            assert rm1 is rm2
        finally:
            get_round_manager.cache_clear()

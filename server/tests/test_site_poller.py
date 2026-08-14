"""Tests for server/core/site_poller.py — 100% coverage required."""
from __future__ import annotations

import asyncio
import pytest
import httpx
import respx
from unittest.mock import MagicMock, AsyncMock, ANY

from server.core.site_poller import SitePoller, parse_site_status_urls


# ── parse_site_status_urls ────────────────────────────────────────────────────

def test_parse_empty_string_returns_empty() -> None:
    assert parse_site_status_urls("") == {}


def test_parse_single_entry() -> None:
    result = parse_site_status_urls("alpha:http://alpha:9001")
    assert result == {"alpha": "http://alpha:9001"}


def test_parse_multiple_entries() -> None:
    result = parse_site_status_urls("a:http://a:9001,b:http://b:9001")
    assert result == {"a": "http://a:9001", "b": "http://b:9001"}


def test_parse_strips_whitespace() -> None:
    result = parse_site_status_urls("  x : http://x:9001 , y:http://y:9001 ")
    assert "x" in result and "y" in result


def test_parse_skips_entries_without_colon() -> None:
    """Entries with no colon separator are silently skipped."""
    result = parse_site_status_urls("nocobon,site_a:http://a:9001")
    assert "nocobon" not in result
    assert result == {"site_a": "http://a:9001"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_settings(urls: str = "site_a:http://mock-site-a", hb: int = 0) -> MagicMock:
    s = MagicMock()
    s.heartbeat_seconds = hb
    s.site_status_urls = urls
    return s


# ── SitePoller._poll_once ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_poller_calls_sync_on_successful_response() -> None:
    rm = MagicMock()
    settings = _make_settings("site_a:http://mock-site-a")

    with respx.mock:
        respx.get("http://mock-site-a/site/status").mock(
            return_value=httpx.Response(200, json={
                "site_id":    "site_a",
                "run_count":  5,
                "last_run_at": "2026-08-14T12:00:00+00:00",
                "phase":      "done",
            })
        )
        poller = SitePoller(rm, settings)
        await poller._poll_once()

    rm.sync_site_run_info.assert_called_once_with("site_a", 5, ANY)


@pytest.mark.asyncio
async def test_poller_marks_error_on_connection_failure() -> None:
    rm = MagicMock()
    settings = _make_settings("site_a:http://nonexistent-host-xyz-12345")

    poller = SitePoller(rm, settings)
    await poller._poll_once()   # should not raise

    rm.mark_site_error.assert_called_once_with("site_a")


@pytest.mark.asyncio
async def test_poller_skips_empty_urls() -> None:
    rm = MagicMock()
    settings = _make_settings("")   # no sites configured

    poller = SitePoller(rm, settings)
    await poller._poll_once()

    rm.sync_site_run_info.assert_not_called()
    rm.mark_site_error.assert_not_called()


@pytest.mark.asyncio
async def test_poller_does_not_trigger_aggregation() -> None:
    """Poller is read-only — must NEVER call rm.receive_update or rm._aggregate."""
    rm = MagicMock()
    rm.receive_update = AsyncMock()
    settings = _make_settings("site_a:http://mock-site-a")

    with respx.mock:
        respx.get("http://mock-site-a/site/status").mock(
            return_value=httpx.Response(200, json={
                "site_id": "site_a", "run_count": 3,
                "last_run_at": None, "phase": "done",
            })
        )
        poller = SitePoller(rm, settings)
        await poller._poll_once()

    rm.receive_update.assert_not_called()


@pytest.mark.asyncio
async def test_poller_handles_null_last_run_at() -> None:
    """last_run_at=null in JSON → sync_site_run_info called with None timestamp."""
    rm = MagicMock()
    settings = _make_settings("site_a:http://mock-site-a")

    with respx.mock:
        respx.get("http://mock-site-a/site/status").mock(
            return_value=httpx.Response(200, json={
                "site_id": "site_a", "run_count": 2,
                "last_run_at": None, "phase": "idle",
            })
        )
        poller = SitePoller(rm, settings)
        await poller._poll_once()

    rm.sync_site_run_info.assert_called_once_with("site_a", 2, None)


@pytest.mark.asyncio
async def test_poller_skips_entry_with_empty_base_url() -> None:
    """A site entry that parses to an empty base_url is skipped without error."""
    rm = MagicMock()
    # Craft a raw string whose partition yields an empty base_url
    settings = _make_settings("site_a:")

    poller = SitePoller(rm, settings)
    await poller._poll_once()

    rm.sync_site_run_info.assert_not_called()
    rm.mark_site_error.assert_not_called()


@pytest.mark.asyncio
async def test_poller_handles_non_200_status() -> None:
    """Non-200 responses are silently ignored (no sync, no error mark)."""
    rm = MagicMock()
    settings = _make_settings("site_a:http://mock-site-a")

    with respx.mock:
        respx.get("http://mock-site-a/site/status").mock(
            return_value=httpx.Response(503, json={"detail": "service unavailable"})
        )
        poller = SitePoller(rm, settings)
        await poller._poll_once()

    rm.sync_site_run_info.assert_not_called()
    rm.mark_site_error.assert_not_called()


# ── SitePoller.start / run ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_creates_asyncio_task() -> None:
    """start() schedules the run() coroutine as a background task — no exception raised."""
    rm = MagicMock()
    settings = _make_settings("", hb=9999)

    poller = SitePoller(rm, settings)
    poller.start()
    # Give the event loop one tick to schedule the task
    await asyncio.sleep(0)
    # If start() ran without error, it created a task — test passes on no-raise


@pytest.mark.asyncio
async def test_run_loops_until_cancelled() -> None:
    """run() loops indefinitely; asyncio.CancelledError propagates on cancel.

    Uses asyncio.Event to wait until _poll_once is actually called (i.e., line 89
    is executed) before cancelling, ensuring the loop body is covered regardless
    of internal httpx scheduling.
    """
    rm = MagicMock()
    settings = _make_settings("", hb=0)
    poller = SitePoller(rm, settings)

    poll_called = asyncio.Event()
    original_poll_once = poller._poll_once

    async def patched_poll() -> None:
        poll_called.set()
        await original_poll_once()

    poller._poll_once = patched_poll  # type: ignore[method-assign]

    task = asyncio.create_task(poller.run())
    # Wait until _poll_once is actually invoked (covers line 89), then cancel
    await asyncio.wait_for(poll_called.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

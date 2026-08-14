"""Asyncio heartbeat task — polls each site's /site/status endpoint periodically."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

import httpx

from shared.utils.logging_config import get_logger

if TYPE_CHECKING:
    from server.config import ServerSettings
    from server.core.round_manager import RoundManager

log = get_logger(__name__)


def parse_site_status_urls(raw: str) -> dict[str, str]:
    """
    Parse SITE_STATUS_URLS env var into a {site_id: base_url} dict.

    Format: "site_a:http://a:9001,site_b:http://b:9001"

    Each pair is split on the FIRST colon only (via ``partition``), so URLs
    that contain colons (e.g. ``http://host:9001``) are preserved intact.

    Parameters
    ----------
    raw : str
        Comma-separated ``site_id:base_url`` pairs from the environment variable.

    Returns
    -------
    dict[str, str]
        Mapping of site_id → base_url.  Empty dict for empty input.
    """
    if not raw.strip():
        return {}
    result: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        site_id, _, base_url = pair.partition(":")
        result[site_id.strip()] = base_url.strip() if base_url else ""
    return result


class SitePoller:
    """Polls each registered site on a heartbeat interval to sync run counts.

    This class is deliberately read-only with respect to FL rounds:
    it only calls ``sync_site_run_info`` and ``mark_site_error`` on the
    RoundManager — it never triggers aggregation.
    """

    def __init__(self, round_manager: "RoundManager", settings: "ServerSettings") -> None:
        self._rm = round_manager
        self._settings = settings
        self._site_urls: dict[str, str] = parse_site_status_urls(
            settings.site_status_urls
        )

    async def _poll_once(self) -> None:
        """Poll all configured sites once.  Extracted for testability."""
        headers: dict[str, str] = {}
        poll_secret: str = self._settings.site_poll_secret
        if poll_secret:
            headers["Authorization"] = f"Bearer {poll_secret}"

        async with httpx.AsyncClient(timeout=5.0) as client:
            for site_id, base_url in self._site_urls.items():
                if not base_url:
                    continue
                try:
                    r = await client.get(f"{base_url}/site/status", headers=headers)
                    if r.status_code == 200:
                        data: dict[str, object] = r.json()
                        remote_count = int(data.get("run_count", 0))  # type: ignore[arg-type]
                        raw_ts = data.get("last_run_at")
                        last_run_at: datetime | None = (
                            datetime.fromisoformat(str(raw_ts)) if raw_ts else None
                        )
                        self._rm.sync_site_run_info(site_id, remote_count, last_run_at)
                except Exception as exc:
                    log.warning("site_unreachable", site=site_id, error=str(exc))
                    self._rm.mark_site_error(site_id)

    async def run(self) -> None:
        """Main heartbeat loop — runs indefinitely until cancelled."""
        while True:
            await asyncio.sleep(self._settings.heartbeat_seconds)
            await self._poll_once()

    def start(self) -> None:
        """Schedule the heartbeat loop as an asyncio background task."""
        asyncio.create_task(self.run())

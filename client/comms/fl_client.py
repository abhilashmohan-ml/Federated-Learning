"""
HTTP FL protocol client.

Handles:
  - JWT authentication  (obtain + auto-refresh)
  - Uploading model updates to server
  - Downloading the latest global model
"""
from __future__ import annotations

import httpx

from client.config             import get_client_settings
from shared.schemas.auth       import TokenRequest, TokenResponse, RefreshRequest
from shared.schemas.federation import ModelUpdate
from shared.utils.logging_config import get_logger

log = get_logger(__name__)


class FLClient:
    def __init__(self) -> None:
        self.settings      = get_client_settings()
        self._access_token = ""
        self._refresh_token= ""

    @property
    def auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._access_token}"}

    def authenticate(self) -> None:
        """Obtain access + refresh tokens from the server."""
        resp = httpx.post(
            f"{self.settings.server_url}/auth/token",
            json=TokenRequest(
                site_id=self.settings.site_id,
                site_secret=self.settings.site_secret,
            ).model_dump(),
            timeout=30,
        )
        resp.raise_for_status()
        tokens = TokenResponse(**resp.json())
        self._access_token  = tokens.access_token
        self._refresh_token = tokens.refresh_token
        log.info("authenticated", site=self.settings.site_id)

    def upload_update(self, update: ModelUpdate) -> None:
        """POST model update to server.  Auto-refreshes token on 401."""
        url  = f"{self.settings.server_url}/federation/update"
        resp = httpx.post(url, json=update.model_dump(), headers=self.auth_headers, timeout=60)
        if resp.status_code == 401:
            self._do_refresh()
            resp = httpx.post(url, json=update.model_dump(), headers=self.auth_headers, timeout=60)
        resp.raise_for_status()
        log.info("update_uploaded", site=self.settings.site_id, round_id=update.round_id)

    def get_global_model(self) -> dict:
        """Download latest global model weights from server."""
        resp = httpx.get(
            f"{self.settings.server_url}/models/global-model",
            headers=self.auth_headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _do_refresh(self) -> None:
        resp = httpx.post(
            f"{self.settings.server_url}/auth/refresh",
            json=RefreshRequest(refresh_token=self._refresh_token).model_dump(),
            timeout=30,
        )
        resp.raise_for_status()
        tokens = TokenResponse(**resp.json())
        self._access_token  = tokens.access_token
        self._refresh_token = tokens.refresh_token

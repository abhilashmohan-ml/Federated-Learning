"""
HTTP FL protocol client.

Handles:
  - JWT authentication (obtain + auto-refresh)
  - Uploading model updates to server
  - Downloading the latest global model
  - Retry with exponential backoff for transient network errors
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from client.config import get_client_settings
from shared.schemas.auth import RefreshRequest, TokenRequest, TokenResponse
from shared.schemas.federation import ModelUpdate
from shared.utils.logging_config import get_logger

log = get_logger(__name__)

_RETRYABLE: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)


class FLClient:
    def __init__(self) -> None:
        self.settings = get_client_settings()
        self._access_token = ""
        self._refresh_token = ""
        self._http = httpx.Client(
            verify=self.settings.verify_ssl,
            timeout=httpx.Timeout(
                connect=float(self.settings.connect_timeout),
                read=float(self.settings.request_timeout),
                write=float(self.settings.request_timeout),
                pool=float(self.settings.connect_timeout),
            ),
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> FLClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Request with exponential-backoff retry on transient errors."""
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(1, self.settings.retry_attempts + 1):
            try:
                return self._http.request(method, url, **kwargs)
            except _RETRYABLE as exc:
                last_exc = exc
                log.warning(
                    "request_retry",
                    attempt=attempt,
                    of=self.settings.retry_attempts,
                    url=url,
                    error=str(exc),
                )
                if attempt < self.settings.retry_attempts:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(
            f"Request to {url} failed after {self.settings.retry_attempts} attempts"
        ) from last_exc

    def authenticate(self) -> None:
        """Obtain access + refresh tokens from the server."""
        resp = self._request(
            "POST",
            f"{self.settings.server_url}/auth/token",
            json=TokenRequest(
                site_id=self.settings.site_id,
                site_secret=self.settings.site_secret,
            ).model_dump(),
        )
        resp.raise_for_status()
        tokens = TokenResponse(**resp.json())
        self._access_token = tokens.access_token
        self._refresh_token = tokens.refresh_token
        log.info("authenticated", site=self.settings.site_id)

    def upload_update(self, update: ModelUpdate) -> None:
        """POST model update. Auto-refreshes token on 401."""
        url = f"{self.settings.server_url}/federation/update"
        resp = self._request("POST", url, json=update.model_dump(), headers=self.auth_headers)
        if resp.status_code == 401:
            self._do_refresh()
            resp = self._request("POST", url, json=update.model_dump(), headers=self.auth_headers)
        resp.raise_for_status()
        log.info("update_uploaded", site=self.settings.site_id, round_id=update.round_id)

    def get_global_model(self) -> dict[str, object]:
        """Download latest global model weights from server."""
        resp = self._request(
            "GET",
            f"{self.settings.server_url}/models/global-model",
            headers=self.auth_headers,
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def _do_refresh(self) -> None:
        resp = self._request(
            "POST",
            f"{self.settings.server_url}/auth/refresh",
            json=RefreshRequest(refresh_token=self._refresh_token).model_dump(),
        )
        resp.raise_for_status()
        tokens = TokenResponse(**resp.json())
        self._access_token = tokens.access_token
        self._refresh_token = tokens.refresh_token
        log.info("token_refreshed", site=self.settings.site_id)

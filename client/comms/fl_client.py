"""
HTTP FL protocol client — handles all communication between site and server.

RESPONSIBILITIES
-----------------
1. Authentication: exchange site_id + site_secret for JWT access/refresh tokens
2. Token management: auto-refresh the access token before it expires (on 401)
3. Model update upload: POST the delta_W and metrics to the server
4. Global model download: GET the latest aggregated weights from the server
5. Retry: retry transient network failures with exponential backoff

WHY A PERSISTENT HTTP CLIENT?
------------------------------
The `httpx.Client` object manages a connection pool internally. Instead of
opening and closing a new TCP connection for every API request (expensive),
a persistent client reuses existing connections from the pool. This is
especially important for:
  - High-frequency operations (heartbeat pings every 30 seconds)
  - Large uploads (model updates can contain hundreds of float values)
  - SSL handshakes (establishing TLS is expensive — reuse pays off)

The client is created in `__init__` and reused for the entire lifetime of
the FLClient object. It is closed via `close()` or the context manager
(`with FLClient() as fl: ...`).

PYTHON CONCEPT: context manager (__enter__ / __exit__)
  Classes that implement `__enter__` and `__exit__` can be used with the
  `with` statement:
    with FLClient() as fl:
        fl.authenticate()
        fl.upload_update(update)
    # fl.close() is called automatically here
  This guarantees the connection pool is always released, even on exceptions.

EXPONENTIAL BACKOFF RETRY EXPLAINED
--------------------------------------
When a network request fails with a transient error (connection reset,
timeout), we don't give up immediately. We retry with increasing delays:

  Attempt 1: fails immediately → wait 2 seconds
  Attempt 2: fails → wait 4 seconds (2 × 2)
  Attempt 3: fails → wait 8 seconds (4 × 2)
  Attempt 4: (if retry_attempts=4) → give up, raise RuntimeError

This strategy:
  - Handles brief network blips (server restart, packet loss)
  - Avoids hammering a struggling server with rapid retries ("thundering herd")
  - Gives up quickly for persistent failures (don't wait forever)

WHAT ARE _RETRYABLE ERRORS?
----------------------------
  httpx.ConnectError        : server unreachable (wrong address, network partition)
  httpx.TimeoutException    : server took too long to respond (overloaded)
  httpx.RemoteProtocolError : server closed connection unexpectedly (restart, crash)

Non-retryable: HTTP 4xx errors (our request was bad), HTTP 5xx (server logic error).
We raise_for_status() on the response — the caller decides how to handle 4xx/5xx.

PYTHON CONCEPT: tuple[type[Exception], ...]
  `_RETRYABLE` is a tuple of exception classes. `except _RETRYABLE` catches
  any of the listed exception types in the same `except` clause. The `...`
  (Ellipsis) in the type hint means "any number of elements."
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from client.config import get_client_settings
from shared.schemas.auth import RefreshRequest, TokenRequest, TokenResponse
from shared.schemas.federation import FederationRound, ModelUpdate
from shared.utils.logging_config import get_logger

log = get_logger(__name__)

# Tuple of exception types that represent transient network failures.
# Any exception in this tuple will trigger a retry in _request().
_RETRYABLE: tuple[type[Exception], ...] = (
    httpx.ConnectError,  # cannot reach the server
    httpx.TimeoutException,  # server too slow to respond
    httpx.RemoteProtocolError,  # server closed connection unexpectedly
)


class FLClient:
    """
    Authenticated HTTP client for the FL protocol.

    One FLClient instance is created per site and kept alive for the duration
    of the client process. It manages tokens and the underlying HTTP connection.
    """

    def __init__(self) -> None:
        self.settings = get_client_settings()

        # Tokens start empty — call authenticate() before any FL operations
        self._access_token = ""
        self._refresh_token = ""

        # Create a persistent httpx.Client with a connection pool.
        # verify=settings.verify_ssl controls SSL certificate verification.
        # httpx.Timeout configures different timeouts for different phases:
        #   connect : time to establish the TCP connection
        #   read    : time to wait for the response body (can be long for training)
        #   write   : time to send the request body (model updates can be large)
        #   pool    : time to wait for a connection from the pool
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
        """Close the connection pool and release all resources."""
        self._http.close()

    def __enter__(self) -> FLClient:
        """Support `with FLClient() as fl:` usage."""
        return self

    def __exit__(self, *_: object) -> None:
        """Automatically close on exiting a `with` block."""
        self.close()

    @property
    def auth_headers(self) -> dict[str, str]:
        """
        Return the HTTP headers needed for authenticated requests.

        The Bearer token scheme: the client sends `Authorization: Bearer <token>`
        on every request to a protected endpoint. The server's `get_current_site`
        dependency extracts and verifies this token.

        @property means you access this as `fl.auth_headers` (no parentheses),
        like an attribute rather than a method call.
        """
        return {"Authorization": f"Bearer {self._access_token}"}

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """
        Make an HTTP request with exponential-backoff retry on transient errors.

        This is the core transport method — all other methods call this.
        It wraps `self._http.request()` with retry logic.

        Parameters
        ----------
        method  : str — HTTP method: "GET", "POST", "PUT", etc.
        url     : str — the full URL to request
        **kwargs: passed through to httpx (json=..., headers=..., etc.)

        Returns
        -------
        httpx.Response — caller is responsible for checking status codes

        Raises
        ------
        RuntimeError — after retry_attempts all fail with transient errors
        Any non-retryable exception propagates immediately (e.g., ValueError)
        """
        delay = 2.0  # initial wait in seconds before first retry
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
                # Don't sleep after the last attempt — we're about to raise anyway
                if attempt < self.settings.retry_attempts:
                    time.sleep(delay)
                    delay *= 2  # double the wait time: 2s → 4s → 8s

        # All attempts exhausted
        raise RuntimeError(
            f"Request to {url} failed after {self.settings.retry_attempts} attempts"
        ) from last_exc

    def authenticate(self) -> None:
        """
        Obtain initial access + refresh tokens from the server.

        Must be called once before any other FL operations. The site presents
        its site_id and site_secret; the server returns a token pair.

        The site_secret is only sent during this call. All subsequent requests
        use the short-lived access token from the Authorization header.
        """
        resp = self._request(
            "POST",
            f"{self.settings.server_url}/auth/token",
            json=TokenRequest(
                site_id=self.settings.site_id,
                site_secret=self.settings.site_secret,
            ).model_dump(),  # .model_dump() converts the Pydantic model to a plain dict
        )
        resp.raise_for_status()  # raises httpx.HTTPStatusError on 4xx/5xx
        tokens = TokenResponse(**resp.json())
        self._access_token = tokens.access_token
        self._refresh_token = tokens.refresh_token
        log.info("authenticated", site=self.settings.site_id)

    def upload_update(self, update: ModelUpdate) -> None:
        """
        POST a model update to the server.

        Called after local training completes. The update contains the
        (noisy) model parameter changes and performance metrics.

        AUTO-REFRESH ON 401
        --------------------
        The access token expires after 15 minutes. If a round takes longer
        than 15 minutes (unlikely but possible), the first upload attempt
        receives a 401 Unauthorized. We then refresh the token and retry
        exactly once. This is transparent to the caller.

        Parameters
        ----------
        update : ModelUpdate — the complete model update payload
        """
        url = f"{self.settings.server_url}/federation/update"
        resp = self._request(
            "POST",
            url,
            json=update.model_dump(),  # convert Pydantic model to JSON-serialisable dict
            headers=self.auth_headers,
        )
        if resp.status_code == 401:
            # Token expired — refresh silently and retry ONCE
            self._do_refresh()
            resp = self._request(
                "POST",
                url,
                json=update.model_dump(),
                headers=self.auth_headers,
            )
        resp.raise_for_status()
        log.info("update_uploaded", site=self.settings.site_id, round_id=update.round_id)

    def get_global_model(self) -> dict[str, object]:
        """
        Download the latest global model weights from the server.

        Called at the start of each local training cycle to get the current
        global model that this round's training should start from.

        Returns
        -------
        dict — {"layer_name": [float, ...], ...}
               or {"message": "No global model available yet"} if no rounds have completed
        """
        resp = self._request(
            "GET",
            f"{self.settings.server_url}/models/global-model",
            headers=self.auth_headers,
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def start_round(self) -> FederationRound:
        """
        POST to the server to begin a new FL federation round.

        Returns the FederationRound describing the newly-started round.

        AUTO-REFRESH ON 401
        --------------------
        If the access token has expired, refresh once silently and retry exactly once.
        """
        url = f"{self.settings.server_url}/federation/round/start"
        resp = self._request("POST", url, headers=self.auth_headers)
        if resp.status_code == 401:
            self._do_refresh()
            resp = self._request("POST", url, headers=self.auth_headers)
        resp.raise_for_status()
        return FederationRound(**resp.json())

    def _do_refresh(self) -> None:
        """
        Exchange the current refresh token for a new access + refresh token pair.

        This is called automatically by upload_update() on 401 responses.
        It implements the single-use refresh token rotation pattern:
          1. Send the current refresh token
          2. Server verifies + revokes it
          3. Server returns a new token pair
          4. Store the new tokens

        After this call, the old refresh token is permanently invalidated.
        """
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

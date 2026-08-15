# Technical Specification
## Viral Filtration Federated Learning Platform

**Version:** 2.0  
**Date:** 2026-08-15  
**Status:** Implemented (v0.2.0 — data-driven FL branch)

---

## 1. Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Server runtime | Python | 3.12 |
| Web framework | FastAPI | ≥ 0.139.0 |
| ASGI server | Uvicorn (standard) | ≥ 0.50.0 |
| DB ORM | SQLAlchemy (asyncio) | ≥ 2.0.51 |
| DB migrations | Alembic | ≥ 1.18.5 |
| DB driver (prod) | asyncpg (PostgreSQL) | ≥ 0.31.0 |
| DB driver (dev) | aiosqlite (SQLite) | ≥ 0.22.1 |
| UI framework | Flet | ≥ 0.85.3 |
| ML framework | PyTorch | ≥ 2.12.1 |
| Numerics | NumPy | ≥ 2.5.0 |
| Curve fitting | SciPy | ≥ 1.18.0 |
| Data loading | Pandas | ≥ 3.0.3 |
| ML utilities | scikit-learn | ≥ 1.9.0 |
| Schema validation | Pydantic v2 | ≥ 2.13.4 |
| Settings | pydantic-settings | ≥ 2.14.2 |
| JWT | python-jose[cryptography] | ≥ 3.5.0 |
| Password hashing | passlib[bcrypt] | ≥ 1.7.4 |
| HTTP client | httpx | ≥ 0.28.1 |
| Structured logging | structlog | ≥ 26.1.0 |
| Visualisation | Matplotlib | ≥ 3.11.0 |
| Visualisation | Plotly | ≥ 6.8.0 |
| Testing | pytest + pytest-asyncio + pytest-cov | ≥ 9.1.1 |
| Formatter | black (line-length 100) | ≥ 26.5.1 |
| Linter | ruff | ≥ 0.15.20 |
| Type checker | mypy --strict | ≥ 2.1.0 |
| Containerisation | Docker + Docker Compose | 3.9 |
| Database | PostgreSQL | 16 |

---

## 2. REST API Reference

Base URL: `http(s)://<server>:<port>`  
All federation/model/health routes require `Authorization: Bearer <access_token>`.

### 2.1 Authentication — `/auth`

#### POST `/auth/token`

Obtain access and refresh tokens.

**Request body:**
```json
{
  "site_id": "site_1",
  "site_secret": "plain-text-secret"
}
```

**Response 200:**
```json
{
  "access_token": "<HS256 JWT>",
  "refresh_token": "<HS256 JWT>",
  "token_type": "bearer",
  "expires_in": 900
}
```

**Response 401:** `{"detail": "Bad credentials"}` — site not found or secret mismatch.

**Implementation:**  
DB lookup on `site_registry` WHERE `site_id = req.site_id`. `passlib.CryptContext(bcrypt).verify(req.site_secret, row.secret_hash)`. Both tokens are HS256 JWTs signed with `SERVER_SECRET_KEY`. Each carries a `jti` (UUID4 hex) claim.

---

#### POST `/auth/refresh`

Rotate token pair. Consumes the submitted refresh token.

**Request body:**
```json
{"refresh_token": "<JWT>"}
```

**Response 200:** Same shape as `/auth/token`.

**Response 401:** Token invalid, expired, or already revoked.

**Implementation:**  
Decode JWT → check `jti` against `revoked_tokens` table → INSERT revoked row → issue new AT + RT pair.

---

#### POST `/auth/revoke`

Revoke a refresh token (logout).

**Request body:**
```json
{"refresh_token": "<JWT>"}
```

**Response 200:** `{"status": "revoked"}` — idempotent.

---

### 2.2 Federation — `/federation`

All routes require `Authorization: Bearer <access_token>`.

#### POST `/federation/round/start`

Start a new federation round.

**Response 200:**
```json
{
  "round_id": 1,
  "status": "collecting",
  "started_at": "2026-07-04T10:00:00Z",
  "completed_at": null,
  "participating_sites": [],
  "global_model_version": 0
}
```

**Implementation:** `RoundManager.start_new_round()` increments `_current_round_id`, creates `FederationRound`, starts `asyncio.Task` timeout guard.

**Admin usage:** Called by `FLClient.start_round()` (Section 12.1) and by simulation scripts to force a new round. Clients in normal operation join the open round via `GET /federation/current-round` (Section 19) rather than starting one themselves.

---

#### POST `/federation/update`

Submit a local model update. Triggers aggregation if quorum reached.

**Request body:**
```json
{
  "site_id": "site_1",
  "round_id": 1,
  "n_samples": 480,
  "delta_W": {
    "hermia_params": [42.3, 0.012, 0.0003]
  },
  "dp_noise_sigma": 0.01,
  "hermia_best_model": "combined_1a",
  "local_metrics": {
    "flux_rmse": 1.23,
    "flux_ratio": 0.71,
    "amin_m2": 0.0042,
    "best_aic": -123.4,
    "best_bic": -119.8
  },
  "timestamp": "2026-07-04T10:05:32Z"
}
```

**Response 200:**
```json
{"status": "accepted", "site_id": "site_1", "round_id": 1}
```

**Response 403:** `{"detail": "site_id mismatch with token"}` — token's sub does not match update's site_id.

---

#### GET `/federation/round/{round_id}`

Get status of a specific round.

**Response 200:** `FederationRound` JSON (same schema as start response).  
**Response 404:** Round not found.

---

#### GET `/federation/sites`

List all site statuses.

**Response 200:**
```json
{
  "sites": {
    "site_alpha": "done",
    "site_boston": "training",
    "site_chicago": "idle"
  }
}
```

Site status enum: `registered | idle | training | uploading | done | error`

---

#### GET `/federation/current-round`

Return the open `COLLECTING` round, or create one if none exists. Idempotent.

**Response 200:** `FederationRound` JSON (same schema as `/federation/round/start` response).

Used by production-mode clients that push updates without waiting for a server-initiated round.

---

### 2.4 Settings — `/settings`

All routes require `Authorization: Bearer <access_token>`.

#### GET `/settings`

Return current aggregation policy configuration from `server_settings` table.

**Response 200:**
```json
{
  "aggregation_mode": "quorum",
  "quorum_min_sites": "3",
  "time_window_seconds": "1800",
  "heartbeat_seconds": "30"
}
```

---

#### PUT `/settings`

Update one or more settings keys and apply the new policy live. Requires `X-Admin-Key: <SERVER_SECRET_KEY>` header.

**Request body:**
```json
{
  "aggregation_mode": "time_window",
  "time_window_seconds": "3600",
  "heartbeat_seconds": "60"
}
```

**Response 200:**
```json
{"status": "ok", "config": {"aggregation_mode": "time_window", ...}}
```

**Response 403:** Missing or incorrect `X-Admin-Key` header.  
**Response 422:** Numeric key contains non-integer value.

---

### 2.7 Models — `/models`

#### GET `/models/global-model`

Download current global model weights.

**Response 200:**
```json
{
  "version": 3,
  "round_id": 3,
  "weights": {
    "hermia_params": [41.9, 0.011, 0.0002]
  },
  "global_metrics": {
    "flux_rmse": 1.18,
    "flux_ratio": 0.73,
    "amin_m2": 0.0040
  },
  "created_at": "2026-07-04T10:12:00Z"
}
```

**Response 503:** No global model available yet (no round completed).

---

### 2.4 Health — `/health`

#### GET `/health/`

Liveness probe. No auth required.

**Response 200:** `{"status": "ok"}`

---

## 3. Physics Equations

### 3.1 Hermia Blocking Models

All models fitted with `scipy.optimize.curve_fit`, `maxfev=5000`.

| Model | Equation | Parameters | k-bounds |
|-------|----------|------------|---------|
| Standard | `J(t) = J0 / (1 + ks·t)²` | J0, ks | ks ∈ [0, 1000] |
| Complete | `J(t) = J0 · exp(-kc·t)` | J0, kc | kc ∈ [0, 1000] |
| Intermediate | `J(t) = J0 / (1 + J0·ki·t)` | J0, ki | ki ∈ [0, 1000] |
| Cake | `J(t) = J0 / √(1 + J0²·kcf·t)` | J0, kcf | kcf ∈ [0, 1000] |
| Combined 1-A | `J(t) = J0/(1+k1·t)² · exp(-k2·t)` | J0, k1, k2 | k1,k2 ∈ [0, 100] |

Global flux bounds: J0 ∈ [0.1, 500.0] LMH.

**Information criteria:**

```
AIC = n·ln(RSS/n) + 2k
BIC = n·ln(RSS/n) + k·ln(n)

where n = number of observations, k = number of parameters, RSS = residual sum of squares
```

Best model: `argmin(AIC)` across all successfully fitted models.

### 3.2 Manabe Capture Probability

```
Pc   = 1 - exp(-λ · J / J_crit)            [single-layer capture probability]
LRV  = log₁₀(1 / (1 - Pc)) · N_layers     [log reduction value]
```

Parameter bounds: λ ∈ [0, 100], J_crit ∈ [1, 500] LMH, Pc ∈ [0, 1].

Regulatory minimum: LRV ≥ 4.0 for parvovirus, retrovirus, herpesvirus.

### 3.3 Concentration Polarisation

```
C_wall = C_feed · exp(J · δ / D_v)             [J converted: LMH → m/s = LMH/3.6×10⁶]
C_perm = C_feed · (1-R) · exp(-J · δ / D_v)

defaults: δ = 1×10⁻⁵ m  (boundary layer thickness)
          D_v = 1×10⁻¹¹ m²/s  (virus diffusion coefficient)
          R = 0.99  (true membrane rejection)

LRV_pol = log₁₀(C_feed / C_perm)
```

### 3.4 Derived Metrics

```
Flux ratio  = J_final / J_initial              [< 0.2 => filter exhausted]
A_min (m²)  = Throughput_L / (J_avg_LMH · t_h)
```

---

## 4. FedProx Aggregation Algorithm

```python
# W_new[l] = Σᵢ (nᵢ / N_total) · (W_current[l] + ΔWᵢ[l])
# where N_total = Σᵢ nᵢ

N_total = sum(update.n_samples for update in updates)
for layer in all_layers:
    base    = W_current[layer]          # current global weights
    W_new[layer] = sum(
        (u.n_samples / N_total) * (base + u.delta_W[layer])
        for u in updates
    )
```

Global metrics: simple mean of per-site `flux_rmse`, `lrv_rmse`, `flux_ratio`, `amin_m2`.

FedProx proximal term (enforced client-side in training loss):
```
L_fedprox = (μ/2) · ‖W_local - W_global‖²     μ = FEDPROX_MU (default 0.01)
```

---

## 5. PINN Technical Details

### 5.1 Input Feature Vector (dim=11)

| Index | Feature | Unit |
|-------|---------|------|
| 0 | pore_size_nm | nm |
| 1 | nmwco_kda | kDa |
| 2 | membrane_area_m2 | m² |
| 3 | tmp_bar | bar |
| 4 | feed_flux_lmh | LMH |
| 5 | pH | — |
| 6 | IS_mM | mM |
| 7 | mab_conc_g_L | g/L |
| 8 | temperature_C | °C |
| 9 | virus_size_nm | nm |
| 10 | virus_charge | — |

### 5.2 Output Parameter Vector (dim=10)

| Index | Parameter | Constraint | Activation |
|-------|----------|------------|-----------|
| 0 | J0 | > 0 | Softplus + 1e-6 |
| 1 | ks | > 0 | Softplus + 1e-6 |
| 2 | ki | > 0 | Softplus + 1e-6 |
| 3 | kc | > 0 | Softplus + 1e-6 |
| 4 | kcf | > 0 | Softplus + 1e-6 |
| 5 | k1 | > 0 | Softplus + 1e-6 |
| 6 | k2 | > 0 | Softplus + 1e-6 |
| 7 | Pc | ∈ (0,1) | Sigmoid |
| 8 | Jcrit | > 0 | Softplus + 1e-6 |
| 9 | Dv | > 0 | Softplus + 1e-6 |

### 5.3 Network Architecture

```
ParameterPredictor:
  Linear(11 → 128) → ReLU
  Linear(128 → 128) → ReLU
  Linear(128 → 64) → ReLU
  Linear(64 → 10)  → [Softplus / Sigmoid per parameter]

BlockingRegimeClassifier:
  Linear(11 → 64) → ReLU
  Linear(64 → 5)  → (logits; CrossEntropyLoss at training time)

PhysicsSolver:
  (no learnable weights)
  J_t  = J0 / (1 + k1·t)² · exp(-k2·t)      Combined 1-A
  LRV  = log₁₀(1 / (1 - clamp(Pc, 1e-7, 1-1e-7)))
```

### 5.4 Loss Function

```
L_total = L_flux + L_LRV + L_physics + L_regime + L_fedprox

L_flux    = MSE(J_pred(t), J_obs(t))
L_LRV     = MSE(LRV_pred, LRV_obs)
L_physics = λ_phys · [Σ relu(-params) + Σ relu(Pc - 1)]   λ_phys = 1.0
L_regime  = CrossEntropy(regime_logits, regime_labels)      (0 if labels absent)
L_fedprox = (μ/2) · ‖W_local - W_global‖²
```

---

## 6. Differential Privacy — Gaussian Mechanism

```python
# Per layer:
arr = np.array(layer_weights, dtype=float32)
norm = np.linalg.norm(arr)
if norm > clip_norm:          # clip_norm = 1.0
    arr *= clip_norm / norm   # gradient clipping
arr += np.random.normal(0, sigma, arr.shape)   # sigma = DP_NOISE_SIGMA
```

`sigma` is stored in `ModelUpdate.dp_noise_sigma` for server audit.

---

## 7. JWT Token Specification

| Field | Access Token | Refresh Token |
|-------|-------------|--------------|
| Algorithm | HS256 | HS256 |
| Signing key | `SERVER_SECRET_KEY` | `SERVER_SECRET_KEY` |
| `sub` | site_id | site_id |
| `role` | `"client"` | `"client"` |
| `jti` | UUID4 hex (32 chars) | UUID4 hex (32 chars) |
| `iat` | Issue time (UTC) | Issue time (UTC) |
| `exp` | now + 15 min | now + 7 days |

Token revocation is persisted in the `revoked_tokens` table. The access token is stateless (verify by signature + expiry only). The refresh token is single-use (consumed JTI inserted into `revoked_tokens` before issuing new pair).

---

## 8. Configuration Reference

### Server (`server/config.py`)

| Env Var | Python Field | Type | Default | Description |
|---------|-------------|------|---------|-------------|
| `SERVER_SECRET_KEY` | `secret_key` | str | `"CHANGE_ME"` | JWT signing key; also used as X-Admin-Key for PUT /settings |
| `SERVER_DB_URL` | `db_url` | str | sqlite+aiosqlite://... | SQLAlchemy async DSN |
| `SERVER_HOST` | `host` | str | `"0.0.0.0"` | Bind address |
| `SERVER_PORT` | `port` | int | 8000 | FastAPI port |
| `CORS_ORIGINS` | `cors_origins` | list[str] | localhost 8550–8555 | Comma-separated; empty → allow-all (no credentials) |
| `SSL_KEYFILE` | `ssl_keyfile` | str\|None | None | Path to TLS private key |
| `SSL_CERTFILE` | `ssl_certfile` | str\|None | None | Path to TLS certificate |
| `FLET_SERVER_PORT` | `flet_port` | int | 8550 | Flet dashboard port |
| `FL_ROUNDS` | `fl_rounds` | int | 50 | Total FL rounds |
| `FEDPROX_MU` | `fedprox_mu` | float | 0.01 | FedProx μ |
| `MIN_SITES_PER_ROUND` | `min_sites_per_round` | int | 3 | Default quorum (overridden by DB settings) |
| `ROUND_TIMEOUT_SECONDS` | `round_timeout_seconds` | int | 300 | Auto-aggregate timeout |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `access_token_expire_minutes` | int | 15 | JWT access token TTL |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `refresh_token_expire_days` | int | 7 | JWT refresh token TTL |
| `HEARTBEAT_SECONDS` | `heartbeat_seconds` | int | 30 | SitePoller poll interval (overridden by DB settings) |
| `SITE_STATUS_URLS` | `site_status_urls` | str | `""` | `"site_a:http://a:9001,site_b:http://b:9001"` — sites to poll |
| `SITE_POLL_SECRET` | `site_poll_secret` | str | `""` | Bearer token sent in Authorization header when polling sites |

### Client (`client/config.py`)

| Env Var | Python Field | Type | Default | Description |
|---------|-------------|------|---------|-------------|
| `SITE_ID` | `site_id` | str | `"site_1"` | Site identifier (any unique string) |
| `SERVER_URL` | `server_url` | str | `http://localhost:8000` | FL server base URL |
| `SITE_SECRET` | `site_secret` | str | `""` | Auth secret for both FL JWT and /site/status bearer auth |
| `DP_NOISE_SIGMA` | `dp_noise_sigma` | float | 0.01 | Gaussian DP noise σ |
| `LOCAL_DATA_PATH` | `local_data_path` | str | `./data/site_1/filtration.csv` | Path to filtration CSV (prod mode) |
| `FLET_CLIENT_PORT` | `flet_client_port` | int | 8551 | Flet UI port |
| `CLIENT_STATUS_PORT` | `client_status_port` | int | 9001 | Port for `/site/status` FastAPI endpoint |
| `VERIFY_SSL` | `verify_ssl` | bool | True | SSL cert verification |
| `CONNECT_TIMEOUT` | `connect_timeout` | int | 10 | TCP connect timeout (s) |
| `REQUEST_TIMEOUT` | `request_timeout` | int | 60 | Read/write timeout (s) |
| `RETRY_ATTEMPTS` | `retry_attempts` | int | 3 | Retries on transient errors |
| `LOCAL_EPOCHS` | `local_epochs` | int | 5 | PINN training epochs per round |
| `LEARNING_RATE` | `learning_rate` | float | 0.001 | Local optimiser LR |
| `FEDPROX_MU` | `fedprox_mu` | float | 0.01 | FedProx μ |
| `DEV_MODE` | `dev_mode` | bool | False | If true, use `DevDataSource` instead of CSV files |
| `DEV_JITTER_FRACTION` | `dev_jitter_fraction` | float | 0.05 | Noise fraction for synthetic flux jitter |
| `DEV_J0` | `dev_j0` | float | 150.0 | Synthetic initial flux (LMH) |
| `DEV_K1` | `dev_k1` | float | 0.015 | Combined 1-A k1 parameter |
| `DEV_K2` | `dev_k2` | float | 0.002 | Combined 1-A k2 parameter |
| `DEV_NOISE` | `dev_noise` | float | 2.0 | Additive Gaussian noise σ (LMH) |
| `DEV_TMP_BASE` | `dev_tmp_base` | float | 1.0 | Synthetic TMP base (bar) |
| `DATA_POLL_SECONDS` | `data_poll_seconds` | int | 60 | Interval between directory polls in ProdDataSource |

---

## 9. Local Filtration CSV Format

Expected columns (minimum required):

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `time` | float | minutes | Elapsed filtration time |
| `flux` | float | LMH | Permeate flux |
| `tmp` | float | bar | Transmembrane pressure |

Optional columns used by future PINN training:
`lrv`, `ph`, `conductivity`, `mab_conc`, `temperature`

---

## 10. Code Quality Standards

| Standard | Tool | Configuration |
|----------|------|--------------|
| Formatting | black | `line-length = 100` |
| Linting | ruff | project `pyproject.toml` |
| Type checking | mypy | `--strict` |
| Import sorting | isort | `profile=black` |
| Test coverage | pytest-cov | ≥ 80% on `shared/`, `server/core/`, `client/engine/` |
| Commit style | — | `feat:` `fix:` `chore:` `docs:` `test:` |
| Branch naming | — | `feature/<ticket>-desc`, `fix/<ticket>-desc` |

---

## 11. HTTP Client Retry Policy

Retryable exceptions: `httpx.ConnectError`, `httpx.TimeoutException`, `httpx.RemoteProtocolError`.

```
Attempt 1  →  fails  →  sleep 2s
Attempt 2  →  fails  →  sleep 4s
Attempt 3  →  fails  →  raise RuntimeError
```

Delay doubles on each attempt (exponential backoff, base 2s). HTTP 4xx/5xx responses are NOT retried — only transport-level exceptions are.

On HTTP 401 from `upload_update` or `start_round`, a single token refresh is attempted before re-raising (see Section 12.1 for details).

---

## 12. Client HTTP Interface (`client/comms/fl_client.py`)

### 12.1 `FLClient.start_round() -> FederationRound`

Triggers a new FL federation round by posting to `POST /federation/round/start`.

```python
def start_round(self) -> FederationRound:
    url = f"{self.settings.server_url}/federation/round/start"
    resp = self._request("POST", url, headers=self.auth_headers)
    if resp.status_code == 401:
        self._do_refresh()
        resp = self._request("POST", url, headers=self.auth_headers)
    resp.raise_for_status()
    return FederationRound(**resp.json())
```

**401-refresh-retry pattern:** If the access token has expired mid-session, the 401 response triggers a single silent token refresh via `_do_refresh()` (which rotates both access and refresh tokens), then retries the original request exactly once. Any subsequent failure propagates to the caller.

**Returns:** `FederationRound` — the Pydantic schema object with `round_id`, `status`, `started_at`, `completed_at`, `participating_sites`, and `global_model_version`.

**Raises:** `httpx.HTTPStatusError` on non-401 HTTP errors; `RuntimeError` after all transport retries exhausted.

---

## 13. Client UI (`client/ui/`)

### 13.1 `StatusPage(page: ft.Page, fl_client: FLClient)`

Operator dashboard page showing connection info, current round state, and local training metrics.

**Constructor signature (as of fix/flet-colors-icons-api):**

```python
class StatusPage:
    def __init__(self, page: ft.Page, fl_client: FLClient) -> None:
        ...
```

`fl_client` is a **required** positional-or-keyword argument. The previous signature (no `fl_client` parameter) is removed. Callers must pass an already-authenticated `FLClient` instance.

**Trigger Manual Round button:**

The `build()` method renders an `ft.Button("Trigger Manual Round", icon=ft.Icons.PLAY_ARROW)`. Its `on_click` is bound to `_handle_round_click`, which spawns a background daemon thread:

```python
def _handle_round_click(self, e: Any) -> None:
    self._round_button.disabled = True
    self.page.update()
    threading.Thread(target=self._run_round, daemon=True, name="fl-manual-round").start()

def _run_round(self) -> None:
    try:
        # Build data source based on DEV_MODE setting
        if cfg.dev_mode:
            ds = DevDataSource(physics_cfg, jitter=cfg.dev_jitter_fraction)
        else:
            ds = ProdDataSource(data_dir)
        trainer = LocalTrainer(data_source=ds)

        # Join the open collecting round — does NOT start a new one or affect other sites
        round_info = self.fl_client.get_current_round()
        self._round_text.value = f"Round  : {round_info.round_id}"
        self._phase_text.value = "Phase  : training"
        self.page.update()

        update_state(phase="training", current_round_id=round_info.round_id)
        update = trainer.train_and_prepare_update(round_info.round_id)

        update_state(phase="uploading")
        self._phase_text.value = "Phase  : uploading"
        self.page.update()

        self.fl_client.upload_update(update)

        update_state(phase="done", last_round_completed=round_info.round_id, ...)
        self._phase_text.value = "Phase  : done"
    except Exception as exc:
        update_state(phase="error")
        self._round_text.value = "Round  : ERROR"
        self._phase_text.value = f"Phase  : {str(exc)[:40]}"
    self._round_button.disabled = False
    self.page.update()
```

Key design decisions:
- Uses `get_current_round()` (not `start_round()`) so clicking the button on one site does **not** trigger or interfere with other sites — each site runs its own training independently against the shared round.
- Phase text progresses through `"training"` → `"uploading"` → `"done"` so the operator sees incremental feedback.
- `page.update()` is called three times (training start, uploading start, done/error) so the UI remains responsive.
- Button is disabled before the thread starts and re-enabled in the `finally`-equivalent path, preventing double-clicks.

### 13.2 `client/ui/app.py — main()`

`main()` follows this construction order:

1. Load `ClientSettings` via `get_client_settings()`.
2. Instantiate `FLClient()`.
3. Call `fl.authenticate()` — blocks until tokens are obtained.
4. Pass `fl` to `StatusPage(page, fl_client=fl)`.

This guarantees `StatusPage` always holds a fully-authenticated client before any button click can occur.

---

## 14. DataSource Abstraction (`client/engine/data_source.py`)

### 14.1 Protocol

```python
class DataSource(Protocol):
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...  # returns (time_min, flux_lmh, tmp_bar) arrays
```

### 14.2 `DevDataSource`

```python
PHYSICS_DEFAULTS: dict[str, float] = {
    "J0": 150.0, "k1": 0.015, "k2": 0.0020, "noise": 2.0, "tmp_base": 1.0
}

class DevDataSource:
    def __init__(self, physics_cfg: dict[str, float], jitter: float = 0.05) -> None: ...
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
```

Generates `t = np.linspace(0, 60, 61)`, flux from Combined 1-A `J(t) = J0/(1+k1·t)^2 * exp(-k2·t)` with Gaussian jitter, TMP from `tmp_base + noise`.

### 14.3 `ProdDataSource`

```python
class ProdDataSource:
    def __init__(self, data_dir: str) -> None: ...
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
```

Scans `data_dir` for `filtration_*.csv` files. Compares against `.processed.json` sidecar. Reads first unprocessed file, marks it processed. Raises `NoNewDataError` if no new files.

---

## 15. AggregationPolicy Protocol (`server/core/aggregation_policy.py`)

```python
class AggregationPolicy(Protocol):
    def should_aggregate(
        self,
        updates_since_last: int,
        sites_contributed: set[str],
        elapsed_seconds: float,
    ) -> bool: ...

@dataclass
class QuorumPolicy:
    min_sites: int = 3
    def should_aggregate(self, updates_since_last, sites_contributed, elapsed_seconds) -> bool:
        return len(sites_contributed) >= self.min_sites

@dataclass
class TimeWindowPolicy:
    window_seconds: int = 1800
    def should_aggregate(self, updates_since_last, sites_contributed, elapsed_seconds) -> bool:
        return updates_since_last >= 1 and elapsed_seconds >= self.window_seconds
```

Policy is set via `RoundManager.set_policy(policy: AggregationPolicy)`. Change takes effect for the current round immediately.

---

## 16. SettingsStore (`server/db/settings_store.py`)

```python
DEFAULTS: dict[str, str] = {
    "aggregation_mode": "quorum",
    "quorum_min_sites": "3",
    "time_window_seconds": "1800",
    "heartbeat_seconds": "30",
}

class SettingsStore:
    async def load(self, db: AsyncSession) -> dict[str, str]: ...
    async def save(self, db: AsyncSession, key: str, value: str) -> None: ...
```

`load()` inserts `DEFAULTS` rows for any missing keys on first call. `save()` uses SQLAlchemy `merge()` (upsert behaviour). All values stored as strings.

---

## 17. SitePoller (`server/core/site_poller.py`)

```python
def parse_site_status_urls(raw: str) -> dict[str, str]:
    """Parse "site_a:http://a:9001,site_b:http://b:9001" → {"site_a": "http://a:9001", ...}"""

class SitePoller:
    def __init__(self, round_manager: RoundManager, settings: ServerSettings) -> None: ...
    async def _poll_once(self) -> None: ...
    async def run(self) -> None: ...  # infinite heartbeat loop
    def start(self) -> None: ...     # asyncio.create_task(self.run())
```

**`_poll_once` logic:**
1. For each `(site_id, base_url)` in `parse_site_status_urls(settings.site_status_urls)`:
2. `GET {base_url}/site/status` with `Authorization: Bearer {settings.site_poll_secret}` (when non-empty)
3. On HTTP 200: extract `run_count`, parse `last_run_at` → call `rm.sync_site_run_info(site_id, run_count, last_run_at)` to update run counts; also extract `phase` → call `rm.sync_site_phase(site_id, phase)` to keep site status current in the dashboard
4. On any exception: call `rm.mark_site_error(site_id)` — never raises

`sync_site_phase(site_id, phase)` maps the raw phase string from `/site/status` to a `SiteStatus` enum value. It never downgrades a `DONE` site mid-round (a site that already submitted its update stays `DONE` until the round resets).

**Started at server startup** inside `_on_startup()` FastAPI event handler.

---

## 18. Client Status Server (`client/comms/status_server.py`)

FastAPI app running as a daemon thread on `CLIENT_STATUS_PORT`:

```python
GET /site/status
```

**Authentication:** When `ClientSettings.site_secret` is non-empty, requires `Authorization: Bearer <site_secret>`. Returns HTTP 401 if missing or incorrect.

**Response:**
```json
{
  "site_id": "site_1",
  "run_count": 12,
  "last_run_at": "2026-08-15T10:30:00+00:00",
  "phase": "idle"
}
```

`run_count` and `last_run_at` are read from `TrainingState` (shared state object updated by Scheduler after each successful training run). `phase` is one of `"idle"`, `"training"`, `"uploading"`, `"done"`, `"error"`.

**Started by `start_status_server(port: int)`** in `client/main.py` before `start_scheduler()`.

---

## 19. `FLClient.get_current_round()` (`client/comms/fl_client.py`)

```python
def get_current_round(self) -> FederationRound:
    url = f"{self.settings.server_url}/federation/current-round"
    resp = self._request("GET", url, headers=self.auth_headers)
    if resp.status_code == 401:
        self._do_refresh()
        resp = self._request("GET", url, headers=self.auth_headers)
    resp.raise_for_status()
    return FederationRound(**resp.json())
```

Used by `_watch_prod()` in production mode: calls `get_current_round()` to find the open round ID, then trains and uploads. This decouples prod clients from needing a server-initiated round start.

Also used by `StatusPage._run_round()` (the "Trigger Manual Round" button) — see Section 13.1.

---

## 20. RoundManager Auto-Round Continuation (`server/core/round_manager.py`)

After a successful aggregation, `_aggregate()` automatically starts the next round without requiring a new `POST /federation/round/start` API call:

```python
async def _aggregate(self, round_id: int) -> None:
    ...
    r.status = RoundStatus.COMPLETE
    log.info("round_complete", round_id=round_id, model_version=gm.version)

    # Auto-start next round if below FL_ROUNDS limit.
    # Reset DONE sites to IDLE so they can participate again.
    if self._current_round_id < self._settings.fl_rounds:
        for site_id in list(self._site_statuses.keys()):
            if self._site_statuses[site_id] == SiteStatus.DONE:
                self._site_statuses[site_id] = SiteStatus.IDLE
        task = asyncio.create_task(self.start_new_round())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
```

**Why `_background_tasks`?** `asyncio.create_task()` returns a `Task` object that the event loop holds only weakly. If the local variable goes out of scope before the coroutine completes, the GC may cancel it. Storing strong references in `_background_tasks` and using `add_done_callback(discard)` keeps tasks alive until they finish, then auto-removes them.

**Quorum fires every round**, not just the first. Each completed round immediately opens the next collecting window, so sites train continuously without manual intervention until `FL_ROUNDS` is reached.

---

## 21. TrainingState Flux Curve (`client/engine/state.py`)

`TrainingState` stores the full flux time series from each local training run:

```python
@dataclass
class TrainingState:
    ...
    flux_times: list[float] = dataclasses.field(default_factory=list)
    flux_vals:  list[float] = dataclasses.field(default_factory=list)
```

`LocalTrainer.train_and_prepare_update()` writes these after fitting the Hermia models:

```python
update_state(flux_times=list(t_arr), flux_vals=list(flux_arr))
```

The client `LocalResultsPage` reads `get_state().flux_times` / `get_state().flux_vals` and renders a J(t) line chart as a PNG via the matplotlib Agg backend (see Section 22).

---

## 22. Dashboard Chart Rendering (server + client)

Flet ≥ 0.85 has no native chart widget. Charts are rendered server-side using the **matplotlib Agg backend**, encoded as PNG bytes, and displayed via `ft.Image(src=<bytes>)`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(times, flux)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
plt.close(fig)
png_bytes = buf.getvalue()

# Flet display:
ft.Image(src=png_bytes, fit=ft.BoxFit.CONTAIN, expand=True)
```

Three chart types are implemented:

| Component | Location | Chart |
|-----------|----------|-------|
| `FluxChart(multi_site=True)` | Server Graphs page | Amin bar chart — one bar per site |
| `LRVChart(multi_site=True)` | Server Graphs page | Flux ratio bar chart — one bar per site |
| `FluxChart(multi_site=False)` | Client Local Results page | J(t) line chart — flux decline over time |

`FluxChart.update_data(site_metrics: dict[str, dict[str, float]])` accepts the same `site_metrics` dict that `RoundManager.get_status_snapshot()` returns in the `"site_metrics"` key.

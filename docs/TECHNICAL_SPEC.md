# Technical Specification
## Viral Filtration Federated Learning Platform

**Version:** 1.0  
**Date:** 2026-07-04  
**Status:** Implemented (v0.1.0)

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
    "site_1": "done",
    "site_2": "training",
    "site_3": "idle",
    "site_4": "idle",
    "site_5": "uploading"
  }
}
```

Site status enum: `registered | idle | training | uploading | done | error`

---

### 2.3 Models — `/models`

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
| `SERVER_SECRET_KEY` | `secret_key` | str | `"CHANGE_ME"` | JWT signing key |
| `SERVER_DB_URL` | `db_url` | str | sqlite+aiosqlite://... | SQLAlchemy async DSN |
| `SERVER_HOST` | `host` | str | `"0.0.0.0"` | Bind address |
| `SERVER_PORT` | `port` | int | 8000 | FastAPI port |
| `CORS_ORIGINS` | `cors_origins` | list[str] | localhost 8550–8555 | Comma-separated; empty → allow-all (no credentials) |
| `SSL_KEYFILE` | `ssl_keyfile` | str\|None | None | Path to TLS private key |
| `SSL_CERTFILE` | `ssl_certfile` | str\|None | None | Path to TLS certificate |
| `FLET_SERVER_PORT` | `flet_port` | int | 8550 | Flet dashboard port |
| `FL_ROUNDS` | `fl_rounds` | int | 50 | Total FL rounds |
| `FEDPROX_MU` | `fedprox_mu` | float | 0.01 | FedProx μ |
| `MIN_SITES_PER_ROUND` | `min_sites_per_round` | int | 3 | Quorum to trigger aggregation |
| `ROUND_TIMEOUT_SECONDS` | `round_timeout_seconds` | int | 300 | Auto-aggregate timeout |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `access_token_expire_minutes` | int | 15 | JWT access token TTL |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `refresh_token_expire_days` | int | 7 | JWT refresh token TTL |

### Client (`client/config.py`)

| Env Var | Python Field | Type | Default | Description |
|---------|-------------|------|---------|-------------|
| `SITE_ID` | `site_id` | str | `"site_1"` | Site identifier |
| `SERVER_URL` | `server_url` | str | `http://localhost:8000` | FL server base URL |
| `SITE_SECRET` | `site_secret` | str | `"secret_site_1"` | Auth secret (must match server DB) |
| `DP_NOISE_SIGMA` | `dp_noise_sigma` | float | 0.01 | Gaussian DP noise σ |
| `LOCAL_DATA_PATH` | `local_data_path` | str | `./data/site_1/filtration.csv` | Path to filtration CSV |
| `FLET_CLIENT_PORT` | `flet_client_port` | int | 8551 | Flet UI port |
| `VERIFY_SSL` | `verify_ssl` | bool | True | SSL cert verification |
| `CONNECT_TIMEOUT` | `connect_timeout` | int | 10 | TCP connect timeout (s) |
| `REQUEST_TIMEOUT` | `request_timeout` | int | 60 | Read/write timeout (s) |
| `RETRY_ATTEMPTS` | `retry_attempts` | int | 3 | Retries on transient errors |
| `LOCAL_EPOCHS` | `local_epochs` | int | 5 | PINN training epochs per round |
| `LEARNING_RATE` | `learning_rate` | float | 0.001 | Local optimiser LR |
| `FEDPROX_MU` | `fedprox_mu` | float | 0.01 | FedProx μ |

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

On HTTP 401 from `upload_update`, a single token refresh is attempted before re-raising.

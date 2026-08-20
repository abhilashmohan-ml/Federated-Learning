# Design Specification
## Viral Filtration Federated Learning Platform

**Version:** 2.1  
**Date:** 2026-08-20  
**Status:** Implemented (v0.2.1 — monitor fixes + LC design system)

---

## 1. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Central Server                                │
│                                                                       │
│   FastAPI (port 8000)               Flet Dashboard (port 8550)        │
│   ├── /auth                         ├── Round overview                │
│   ├── /federation                   ├── Site monitor (run counts)     │
│   ├── /models                       ├── Global model viewer           │
│   ├── /settings  (admin API)        └── Settings (policy controls)   │
│   └── /health                                                         │
│                                                                       │
│   Core                               Database (PostgreSQL / SQLite)   │
│   ├── RoundManager                   ├── site_registry                │
│   │   └── AggregationPolicy          ├── rounds                       │
│   ├── FedProxAggregator              ├── model_updates                │
│   ├── ModelRegistry                  ├── revoked_tokens               │
│   └── SitePoller (heartbeat)         └── server_settings              │
│       SettingsStore                                                    │
└──────────────────────────────────────────────────────────────────────┘
          │  HTTPS + JWT Bearer                      ▲ /site/status
     ┌────┘                             └────┐       │ (heartbeat poll)
     │                                        │       │
┌────▼───────────────────────┐      ┌────────▼─────────────────┐
│  Site N  (any site_id)      │  …   │  Site M  (any site_id)   │
│  FL Client                  │      │  FL Client               │
│  ├── Engine                 │      │  ├── Engine              │
│  │   ├── DataSource         │      │  │   └── DataSource      │
│  │   │   ├── DevDataSource  │      │  │       (Prod or Dev)   │
│  │   │   └── ProdDataSource │      │  ├── Comms               │
│  │   ├── LocalTrainer       │      │  │   ├── FLClient        │
│  │   └── Scheduler          │      │  │   └── Heartbeat       │
│  ├── Comms                  │      │  ├── StatusServer :900N  │
│  │   ├── FLClient (httpx)   │      │  └── Flet UI :855N       │
│  │   └── Heartbeat          │      └──────────────────────────┘
│  ├── StatusServer :900N     │
│  └── Flet UI :855N          │
└─────────────────────────────┘
```

Each site runs in a separate network-isolated environment. In Docker dev, this is a dedicated bridge network per site. In production, sites are on separate corporate networks.

---

## 2. Component Design

### 2.1 Server — FastAPI Application (`server/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| Entry point | `server/main.py` | App factory, CORS middleware, router registration, startup event (policy load + SitePoller start) |
| Configuration | `server/config.py` | Pydantic-settings: reads `.env`, exposes typed settings; includes `heartbeat_seconds`, `site_status_urls`, `site_poll_secret` |
| Auth API | `server/api/auth.py` | `/auth/token`, `/auth/refresh`, `/auth/revoke`; `get_current_site`, `require_admin_token` dependencies |
| Federation API | `server/api/federation.py` | `/federation/round/start`, `/federation/update`, `/federation/round/{id}`, `/federation/sites`, `/federation/current-round` |
| Settings API | `server/api/settings.py` | `GET /settings` (any site), `PUT /settings` (admin key required via `X-Admin-Key` header) |
| Models API | `server/api/models.py` | `GET /models/global-model` — returns current global weights from RoundManager |
| Health API | `server/api/health.py` | `GET /health/` — liveness probe |
| RoundManager | `server/core/round_manager.py` | In-memory round state machine; pluggable `AggregationPolicy`; `sync_site_run_info`, `sync_site_phase`, `mark_site_error`; auto-starts next round after aggregation via `asyncio.create_task`; `_background_tasks` set prevents GC of in-flight tasks |
| AggregationPolicy | `server/core/aggregation_policy.py` | `AggregationPolicy` Protocol; `QuorumPolicy`; `TimeWindowPolicy` |
| SitePoller | `server/core/site_poller.py` | Asyncio heartbeat task; polls each site's `/site/status`; calls `sync_site_run_info` (run counts) and `sync_site_phase` (training phase) on success; calls `mark_site_error` on failure |
| FedProxAggregator | `server/core/aggregator.py` | Weighted FedAvg aggregation |
| ModelRegistry | `server/core/model_registry.py` | Model versioning and retrieval |
| DB engine | `server/db/database.py` | Async SQLAlchemy engine; `get_db` dependency |
| ORM models | `server/db/models.py` | SiteRegistry, RoundRecord, ModelUpdateRecord, RevokedToken, ServerSetting |
| SettingsStore | `server/db/settings_store.py` | Async key-value store backed by `server_settings` table; defaults on first load |
| Server dashboard | `server/ui/app.py` | Flet multi-page dashboard; poll loop extracts `run_counts`/`last_run_at`/`site_metrics` from snapshot |
| Site card | `server/ui/components/site_card.py` | `SiteCard.set_run_info(run_count, last_run_at)` — smart date display |
| Graphs page | `server/ui/pages/graphs.py` | Comparative charts across all sites; `build()` caches result in `self._built` to prevent Flet re-attach crash on navigation |
| FluxChart | `server/ui/components/flux_chart.py` | `multi_site=True`: Amin bar chart per site (matplotlib PNG); `multi_site=False`: J(t) line chart |
| LRVChart | `server/ui/components/lrv_chart.py` | `multi_site=True`: flux ratio bar chart per site (matplotlib PNG) |
| Settings page | `server/ui/pages/settings.py` | RadioGroup Quorum/TimeWindow; heartbeat field; `PUT /settings` via httpx |
| Internal API | `server/api/internal.py` | `GET /internal/status` — no auth; read-only `RoundManager.get_status_snapshot()` consumed by co-located Flet dashboard |

### 2.2 Client — FL Client Application (`client/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| Entry point | `client/main.py` | Wire DataSource (Dev or Prod), start StatusServer, Scheduler, Heartbeat, Flet UI |
| Configuration | `client/config.py` | Pydantic-settings: site_id, server_url, SSL, timeouts, DP noise, dev_mode, dev physics vars, client_status_port |
| DataSource | `client/engine/data_source.py` | `DataSource(Protocol)`, `DevDataSource`, `ProdDataSource`, `NoNewDataError` |
| CSV loader | `client/engine/data_loader.py` | `load_filtration_csv(path)` — returns `(time, flux, tmp)` NumPy arrays; called internally by `ProdDataSource` |
| LocalTrainer | `client/engine/local_trainer.py` | `__init__(data_source: DataSource)`; Hermia fitting, DP noise, build ModelUpdate payload |
| Scheduler | `client/engine/scheduler.py` | `_watch_dev()` / `_watch_prod()` / `start_scheduler(data_source)` — drives training loop |
| TrainingState | `client/engine/state.py` | Shared state: `run_count`, `last_run_at`, `phase`, `flux_times: list[float]`, `flux_vals: list[float]` — flux curve written by LocalTrainer, read by client charts |
| StatusServer | `client/comms/status_server.py` | FastAPI `GET /site/status` (bearer auth); `start_status_server(port)` |
| FLClient | `client/comms/fl_client.py` | HTTPS transport: authenticate, upload_update, get_global_model, get_current_round, retry with backoff |
| Heartbeat | `client/comms/heartbeat.py` | Daemon thread; periodic health ping to server |
| Client UI | `client/ui/app.py` | Flet status + local results pages; `LocalResultsPage` renders J(t) flux chart via matplotlib Agg PNG |

### 2.3 Shared Physics Library (`shared/`)

| Module | File | Content |
|--------|------|---------|
| Hermia models | `shared/models/hermia.py` | 5 blocking models + AIC/BIC selection + A_min + flux ratio |
| Manabe model | `shared/models/manabe.py` | Capture probability + LRV + compliance check |
| Polarization | `shared/models/polarization.py` | Concentration polarisation at membrane wall |
| Combined 1-A | `shared/models/combined_1a.py` | Combined flux decay model |
| PINN | `shared/models/pinn.py` | ParameterPredictor, PhysicsSolver, BlockingRegimeClassifier, FiltrationPINN, filtration_loss |
| Crypto | `shared/crypto/noise.py` | Gaussian DP noise with L2 clipping |
| Secure agg | `shared/crypto/secure_agg.py` | Additive secret-sharing stub (planned) |
| Auth schemas | `shared/schemas/auth.py` | TokenRequest, TokenResponse, RefreshRequest, TokenClaims |
| Federation schemas | `shared/schemas/federation.py` | ModelUpdate, GlobalModel, FederationRound, RoundStatus, SiteStatus |
| Filtration schemas | `shared/schemas/filtration.py` | Filtration run and result types |
| Constants | `shared/utils/constants.py` | Physical parameter bounds, PARAM_IDX, LRV thresholds |
| Logging | `shared/utils/logging_config.py` | Structured logging (structlog) |
| Design tokens | `shared/utils/theme.py` | `LiquidCarbonTheme` class + `LC` alias — all Flet UI design tokens (colours, radii, chart palette); imported by every UI file |

---

## 3. Federation Protocol — State Machine

```
Server                                    Site N
  │                                          │
  │  POST /federation/round/start            │
  ├─[round_id, status=COLLECTING]──────────► │
  │                                          │
  │  GET /models/global-model                │
  │ ◄────────────────────────────────────────┤
  │                                          │
  │                          LocalTrainer.train_and_prepare_update()
  │                          ├── load CSV
  │                          ├── fit_all_models() → best Hermia
  │                          ├── compute_flux_ratio(), compute_amin()
  │                          ├── add_gaussian_noise(delta_W, sigma)
  │                          └── return ModelUpdate
  │                                          │
  │  POST /federation/update {delta_W, ...}  │
  │ ◄────────────────────────────────────────┤
  │                                          │
  ├─[if n_updates >= MIN_SITES or TIMEOUT]   │
  │  FedProxAggregator.aggregate()           │
  │  W_new[l] = Σ (n_i/N) * (W_old[l] + ΔW_i[l])
  │  status = COMPLETE                       │
  │                                          │
  │  (Sites poll GET /models/global-model    │
  │   to get new W_new for next round)       │
```

Round status transitions:

```
PENDING → COLLECTING → AGGREGATING → COMPLETE
                                  → FAILED
```

---

## 4. Authentication Flow

```
Site                          Server DB
  │                               │
  │  POST /auth/token             │
  │  {site_id, site_secret}       │
  ├──────────────────────────────►│ SELECT site_registry WHERE site_id=?
  │                               │ bcrypt.verify(secret, hash)
  │  {access_token (15min),       │
  │   refresh_token (7days)}      │
  │◄──────────────────────────────┤
  │                               │
  │  [all FL calls]               │
  │  Authorization: Bearer <AT>   │
  │  get_current_site() verifies  │
  │  JWT signature + expiry       │
  │                               │
  │  POST /auth/refresh           │
  │  {refresh_token}              │
  ├──────────────────────────────►│ SELECT revoked_tokens WHERE jti=?
  │                               │ INSERT revoked_tokens (old JTI)
  │  {new_access_token,           │ issue new AT + RT pair
  │   new_refresh_token}          │
  │◄──────────────────────────────┤
```

JWT claims structure:

```json
{
  "sub": "site_1",
  "role": "client",
  "iat": 1751234567,
  "exp": 1751235467,
  "jti": "a1b2c3d4e5f6..."
}
```

---

## 5. PINN Architecture

```
Input x (B × 11)
  └── 11 features: pore_size_nm, nmwco_kda, membrane_area_m2,
                   tmp_bar, feed_flux_lmh, pH, IS_mM,
                   mab_conc_g_L, temperature_C,
                   virus_size_nm, virus_charge

        ┌──────────────────────────────┐
        │   ParameterPredictor         │
        │   Linear(11→128) ReLU        │
        │   Linear(128→128) ReLU       │
        │   Linear(128→64) ReLU        │
        │   Linear(64→10)              │
        │   Softplus (positivity)       │
        │   Sigmoid (Pc ∈ [0,1])       │
        └───────────────┬──────────────┘
                        │ params (B × 10)
                        │ {J0, ks, ki, kc, kcf, k1, k2, Pc, Jcrit, Dv}
        ┌───────────────▼──────────────┐
        │   PhysicsSolver              │
        │   Combined 1-A:              │
        │   J(t) = J0/(1+k1·t)² · e^(-k2·t)
        │   Manabe LRV:                │
        │   LRV = log10(1/(1-Pc))      │
        └───────────────┬──────────────┘
                        │ J_t (B×T), LRV (B,)

        ┌───────────────▼──────────────┐
        │ BlockingRegimeClassifier     │
        │ Linear(11→64) ReLU           │
        │ Linear(64→5) (logits)        │
        │ 5 classes: standard,         │
        │ complete, intermediate,      │
        │ cake, combined_1a            │
        └──────────────────────────────┘

Loss = L_flux (MSE) + L_LRV (MSE) + L_physics (constraint penalties)
     + L_regime (cross-entropy) + L_fedprox (FedProx proximal term)
```

---

## 6. Differential Privacy Design

The Gaussian mechanism is applied per layer before upload:

1. Compute L2 norm of the layer gradient vector
2. If norm > `clip_norm` (default 1.0), scale down: `w = w × (clip_norm / norm)`
3. Add `N(0, σ²I)` noise where `σ = DP_NOISE_SIGMA` (default 0.01)

The `dp_noise_sigma` used is recorded in the ModelUpdate payload for server-side audit.

**Known limitation:** The full Abadi et al. DP-SGD moment accountant for (ε, δ)-DP guarantees is not yet implemented. Current implementation is the basic Gaussian mechanism.

---

## 7. Network Isolation (Docker Dev)

```
┌─────────┐   ┌──────────────────────────────────────────┐
│   db    │   │            server                        │
│ db_net  │   │  db_net  site_1..5_net                   │
└────▲────┘   └──────────────────────────────────────────┘
     │                  ▲    ▲    ▲    ▲    ▲
     │              site_1 site_2 ...   site_5
     │              _net   _net        _net
     │                │                   │
                  ┌───┴──┐           ┌────┴─┐
                  │site_1│   ...     │site_5│
                  └──────┘           └──────┘

Isolation properties:
  site_N can reach:    server (via site_N_net)
  site_N cannot reach: site_M (M≠N), db, any other internal service
  server can reach:    db (via db_net), all sites (via site_N_nets)
```

---

## 8. Error Handling Strategy

| Layer | Error | Handling |
|-------|-------|---------|
| FLClient | `ConnectError`, `TimeoutException`, `RemoteProtocolError` | Exponential backoff (2s, 4s, 8s…), up to RETRY_ATTEMPTS |
| FLClient | HTTP 401 | Auto-refresh token once, then re-attempt |
| FLClient | HTTP 4xx (other) | Propagate as HTTPStatusError — caller logs and skips round |
| FLClient | All retries exhausted | Raise RuntimeError — Scheduler logs and waits for next round |
| RoundManager | Round timeout | Trigger aggregation with whatever updates received |
| RoundManager | No updates at timeout | Mark round FAILED |
| Aggregator | Empty updates list | Raise ValueError — round transitions to FAILED |
| Hermia fitter | scipy curve_fit fails | Exception caught silently; model excluded from AIC comparison |
| Server API | JWT invalid/expired | HTTP 401 with WWW-Authenticate: Bearer |
| Server API | site_id mismatch | HTTP 403 Forbidden |
| Settings API | PUT with non-numeric value for numeric key | HTTP 422 Unprocessable Entity (pre-commit validation) |
| Settings API | PUT without X-Admin-Key | HTTP 403 Forbidden |
| StatusServer | GET /site/status without bearer token (when SITE_SECRET set) | HTTP 401 Unauthorized |
| SitePoller | Connection error to a site | `mark_site_error(site_id)` — does not raise; logged as warning |
| ProdDataSource | No new CSV files in directory | Raise `NoNewDataError` — Scheduler sleeps and retries after `data_poll_seconds` |

---

## 9. Data-Driven FL Architecture

### 9.1 DataSource Abstraction

Sites support two data modes, selected by `DEV_MODE` env var:

```
DEV_MODE=true   →  DevDataSource(physics_cfg, jitter)
DEV_MODE=false  →  ProdDataSource(data_dir)
```

`DataSource` is a `Protocol` with one method:
```python
def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...  # returns (time, flux, tmp) arrays
```

**DevDataSource** generates synthetic Combined 1-A flux data on every call using `PHYSICS_DEFAULTS` (or per-site overrides via `DEV_J0`, `DEV_K1`, `DEV_K2`, `DEV_NOISE`, `DEV_TMP_BASE`). Gaussian jitter is applied to create intra-site variance. Inter-site variance comes from different physics defaults per site in the launcher script.

**ProdDataSource** polls a directory for new `filtration_*.csv` files. Processed filenames are tracked in a `.processed.json` sidecar (written atomically via `.tmp` rename). Only unprocessed files trigger training. Raises `NoNewDataError` when no new files exist.

### 9.2 Pluggable Aggregation Policy

`AggregationPolicy` is a `Protocol`:
```python
def should_aggregate(
    self, updates_since_last: int, sites_contributed: set[str], elapsed_seconds: float
) -> bool: ...
```

Two built-in implementations:

| Policy | Trigger condition |
|--------|------------------|
| `QuorumPolicy(min_sites=3)` | `len(sites_contributed) >= min_sites` |
| `TimeWindowPolicy(window_seconds=1800)` | `updates_since_last >= 1 and elapsed_seconds >= window_seconds` |

Policy is swapped live via `RoundManager.set_policy(policy)` — no server restart needed. The `PUT /settings` API persists the choice to `server_settings` and calls `set_policy` immediately.

### 9.3 Site Heartbeat Poller

`SitePoller` runs as an asyncio background task started at server startup. It:
1. Reads `SITE_STATUS_URLS` env var (`site_a:http://a:9001,site_b:http://b:9001`)
2. GETs `/site/status` from each configured site every `heartbeat_seconds`
3. Passes `Authorization: Bearer {SITE_POLL_SECRET}` header (when configured)
4. On success: calls `RoundManager.sync_site_run_info(site_id, run_count, last_run_at)` (run tracking) AND `RoundManager.sync_site_phase(site_id, phase)` (live status in dashboard)
5. On failure: calls `RoundManager.mark_site_error(site_id)`; never raises

`sync_site_phase` maps the raw `"phase"` string from `/site/status` to a `SiteStatus` enum. It never downgrades a `DONE` site — once a site posts its update for the current round its status stays `DONE` until the round resets. This ensures all sites configured in `SITE_STATUS_URLS` appear immediately in the server dashboard (even before they have submitted any update), because the poller registers them via the first heartbeat poll.

The poller is **read-only** — it never triggers aggregation directly.

### 9.4 Client Status Server

Each site runs a lightweight FastAPI app (`client/comms/status_server.py`) on `CLIENT_STATUS_PORT` (default 9001):

```
GET /site/status
Response: {"site_id": "...", "run_count": 5, "last_run_at": "2026-08-15T10:30:00Z", "phase": "idle"}
```

When `SITE_SECRET` is non-empty, the endpoint requires `Authorization: Bearer {SITE_SECRET}`. The server's `SITE_POLL_SECRET` must match the site's `SITE_SECRET`.

### 9.5 Dynamic Site Registration

No hardcoded `site_1..site_5` exists in any production Python code. Sites are:
- **Registered** via `REGISTERED_SITES=site_a:secret_a,site_b:secret_b` in `init_db.py`
- **Polled** via `SITE_STATUS_URLS=site_a:http://a:9001` in `server/config.py`
- **Identified** by their `SITE_ID` env var in each client container

Production code uses only `str` site identifiers with no assumed format or enumeration.

### 9.6 Auto-Round Continuation

After each successful aggregation `_aggregate()` auto-starts the next round:

```
Round N complete
  ↓
_aggregate() → r.status = COMPLETE
  ↓  (if current_round_id < fl_rounds)
Reset DONE sites → IDLE
  ↓
asyncio.create_task(start_new_round())  ← stored in _background_tasks
  ↓
Round N+1 opens (status=COLLECTING)
```

This means **quorum fires on every round**, not just the first. The server dashboard shows continuous round progression without the operator needing to call `POST /federation/round/start` after round 1. The final round (round_id == FL_ROUNDS) does not auto-start a successor.

### 9.7 Dashboard Chart Rendering

Flet ≥ 0.85 has no native chart widget. All charts use the **matplotlib Agg backend** (non-interactive PNG renderer):

```python
import matplotlib
matplotlib.use("Agg")   # no display connection needed — renders to bytes
import matplotlib.pyplot as plt
import io

fig, ax = plt.subplots(figsize=(8, 3))
# ... draw chart ...
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
plt.close(fig)           # prevent memory leak — always close after savefig

ft.Image(src=buf.getvalue(), fit=ft.BoxFit.CONTAIN, expand=True)
```

`GraphsPage.build()` caches its return value in `self._built`:
```python
def build(self) -> ft.Control:
    if self._built is not None:
        return self._built
    self._built = ft.Column([...])
    return self._built
```
This prevents Flet's "control already added to page" crash when the user navigates away from and back to the Graphs page.

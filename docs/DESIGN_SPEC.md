# Design Specification
## Viral Filtration Federated Learning Platform

**Version:** 1.0  
**Date:** 2026-07-04  
**Status:** Implemented (v0.1.0)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Central Server                            │
│                                                                  │
│   FastAPI (port 8000)          Flet Dashboard (port 8550)        │
│   ├── /auth                    ├── Round overview                │
│   ├── /federation              ├── Site monitor                  │
│   ├── /models                  ├── Global model viewer           │
│   └── /health                  └── Settings                      │
│                                                                  │
│   Core                          Database (PostgreSQL / SQLite)   │
│   ├── RoundManager              ├── site_registry                │
│   ├── FedProxAggregator         ├── rounds                       │
│   └── ModelRegistry             ├── model_updates                │
│                                 └── revoked_tokens               │
└─────────────────────────────────────────────────────────────────┘
          │  HTTPS + JWT Bearer          │
     ┌────┘                             └────┐
     │                                        │
┌────▼──────────┐                   ┌────────▼──────┐
│  Site 1       │   …               │  Site 5       │
│  FL Client    │                   │  FL Client    │
│  ├── Engine   │                   │  ├── Engine   │
│  │   ├── DataLoader               │  │   ...      │
│  │   ├── LocalTrainer             │  Flet UI      │
│  │   └── Scheduler                │  (port 8555)  │
│  ├── Comms                        └───────────────┘
│  │   ├── FLClient (httpx)
│  │   └── Heartbeat
│  └── Flet UI (port 8551)
└───────────────┘
```

Each site runs in a separate network-isolated environment. In Docker dev, this is a dedicated bridge network per site. In production, sites are on separate corporate networks.

---

## 2. Component Design

### 2.1 Server — FastAPI Application (`server/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| Entry point | `server/main.py` | App factory, CORS middleware, router registration, uvicorn launcher |
| Configuration | `server/config.py` | Pydantic-settings: reads `.env`, exposes typed settings |
| Auth API | `server/api/auth.py` | `/auth/token`, `/auth/refresh`, `/auth/revoke`; `get_current_site` dependency |
| Federation API | `server/api/federation.py` | `/federation/round/start`, `/federation/update`, `/federation/round/{id}`, `/federation/sites` |
| Models API | `server/api/models.py` | `GET /models/global-model` — returns current global weights from RoundManager |
| Health API | `server/api/health.py` | `GET /health/` — liveness probe |
| RoundManager | `server/core/round_manager.py` | In-memory round state machine; triggers aggregation |
| FedProxAggregator | `server/core/aggregator.py` | Weighted FedAvg aggregation |
| ModelRegistry | `server/core/model_registry.py` | Model versioning and retrieval |
| DB engine | `server/db/database.py` | Async SQLAlchemy engine; `get_db` dependency |
| ORM models | `server/db/models.py` | SiteRegistry, RoundRecord, ModelUpdateRecord, RevokedToken |
| Server dashboard | `server/ui/app.py` | Flet multi-page dashboard |

### 2.2 Client — FL Client Application (`client/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| Entry point | `client/main.py` | Start FLClient, Scheduler, Heartbeat, Flet UI |
| Configuration | `client/config.py` | Pydantic-settings: site_id, server_url, SSL, timeouts, DP noise |
| DataLoader | `client/engine/data_loader.py` | Load and validate local filtration CSV |
| LocalTrainer | `client/engine/local_trainer.py` | Hermia fitting, DP noise, build ModelUpdate payload |
| Scheduler | `client/engine/scheduler.py` | Poll server for active rounds; trigger training |
| FLClient | `client/comms/fl_client.py` | HTTPS transport: authenticate, upload_update, get_global_model, retry with backoff |
| Heartbeat | `client/comms/heartbeat.py` | Daemon thread; periodic health ping to server |
| Client UI | `client/ui/app.py` | Flet status + local results pages |

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

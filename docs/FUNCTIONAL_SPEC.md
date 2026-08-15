# Functional Specification
## Viral Filtration Federated Learning Platform

**Version:** 2.0  
**Date:** 2026-08-15  
**Status:** Implemented (v0.2.0 — data-driven FL branch)

---

## 1. Purpose and Scope

This system is a Federated Learning (FL) engine for optimising viral filtration of monoclonal antibodies (mAbs). Five pharmaceutical manufacturing sites collaboratively train a shared predictive model without sharing raw process data. Only privacy-protected model updates leave each site.

The system provides:
- A central aggregation server and dashboard
- Per-site FL client with local training engine
- Physics-informed predictive model of membrane filtration behaviour
- Regulatory compliance checking (LRV)

---

## 2. Stakeholders

| Role | Description |
|------|-------------|
| **Server Administrator** | Initiates FL rounds, monitors global model convergence, manages site registrations |
| **Site Operator** | Monitors local training status, reviews local metrics and compliance results |
| **Regulatory / QA** | Consumes LRV compliance reports and audit logs |

---

## 3. Business Goals

| ID | Goal |
|----|------|
| BG-01 | Predict flux decline J(t) for any filter and operating condition without raw data sharing |
| BG-02 | Calculate Log Reduction Value (LRV) for each site's filtration run via the Manabe capture-probability model |
| BG-03 | Compute minimum filter area (A_min) and flux ratio to support filter sizing decisions |
| BG-04 | Classify the dominant fouling regime (standard / complete / intermediate / cake / combined 1-A) for each run |
| BG-05 | Ensure regulatory compliance: LRV ≥ 4.0 for parvovirus, retrovirus, and herpesvirus |
| BG-06 | Maintain data sovereignty: raw filtration CSVs never leave the site container |

---

## 4. Functional Requirements

### 4.1 Federated Learning Protocol

| ID | Requirement |
|----|-------------|
| FR-01 | Server SHALL initiate a new federation round on request, assigning it a sequential integer round_id |
| FR-02 | Server SHALL broadcast the current global model weights to all registered sites at round start |
| FR-03 | Each site SHALL load its local filtration CSV, fit Hermia models, and produce a model update |
| FR-04 | Each site SHALL apply Gaussian differential privacy (DP) noise to gradients before uploading |
| FR-05 | Server SHALL aggregate updates using FedProx-weighted averaging (weighted by n_samples) |
| FR-06 | Aggregation SHALL trigger when `MIN_SITES_PER_ROUND` updates are received (default 3 of 5) |
| FR-07 | Aggregation SHALL also trigger after `ROUND_TIMEOUT_SECONDS` (default 300 s) regardless of site count |
| FR-08 | Server SHALL support up to `FL_ROUNDS` rounds (default 50) per session |
| FR-09 | The system SHALL run `FL_ROUNDS` rounds before signalling convergence |
| FR-47 | After each successful aggregation, the server SHALL automatically start the next round without requiring a new `POST /federation/round/start` API call, until `FL_ROUNDS` is reached |

### 4.2 Physics Modelling

| ID | Requirement |
|----|-------------|
| FR-10 | System SHALL fit 5 Hermia blocking models to the local flux-time data: standard, complete, intermediate, cake, combined 1-A |
| FR-11 | Best model SHALL be selected by lowest Akaike Information Criterion (AIC); BIC is also computed and stored |
| FR-12 | System SHALL fit Manabe capture-probability parameters (λ, J_crit) from (flux, LRV) pairs |
| FR-13 | System SHALL compute virus concentration polarisation (C_wall, C_perm, LRV_pol) for each run |
| FR-14 | System SHALL output: flux ratio (J_final/J_initial), A_min (m²), and per-model AIC/BIC/RMSE |
| FR-15 | The PINN Level-1 parameter predictor SHALL accept an 11-dimensional input vector (filter descriptors + process conditions + virus properties) |
| FR-16 | The PINN Level-2 physics solver SHALL use the Combined 1-A flux equation and Manabe LRV equation in a differentiable form |

### 4.3 Authentication and Security

| ID | Requirement |
|----|-------------|
| FR-17 | Each site SHALL authenticate with a site_id and site_secret before any FL API access |
| FR-18 | Server SHALL return a short-lived access token (15 min) and a long-lived refresh token (7 days) |
| FR-19 | All federation API calls SHALL require a valid Bearer JWT in the Authorization header |
| FR-20 | Refresh tokens SHALL be consumed (rotated) on use; consumed JTIs SHALL be persisted to prevent replay |
| FR-21 | Site secrets SHALL be stored as bcrypt hashes in the database — never in plaintext |
| FR-22 | The server SHALL enforce that a site may only submit updates under its own site_id |
| FR-23 | TLS SHALL be supported on the server endpoint (configurable via SSL_KEYFILE/SSL_CERTFILE) |

### 4.4 Dashboards and Monitoring

| ID | Requirement |
|----|-------------|
| FR-24 | Server dashboard SHALL display: current round status, participating sites, global model version, aggregated metrics |
| FR-25 | Server dashboard SHALL display per-site status (idle / training / uploading / done / error) |
| FR-26 | Client dashboard SHALL display: current round, local training status, local metrics (flux RMSE, LRV, A_min, flux ratio) |
| FR-27 | System SHALL emit a structured audit log entry for every federation round start, update received, and aggregation event |
| FR-48 | The client "Trigger Manual Round" button SHALL call `GET /federation/current-round` to join the active collecting round; it SHALL NOT call `POST /federation/round/start`; each site runs training independently without affecting other sites |
| FR-49 | Server and client dashboards SHALL render physics charts (J(t) flux decline, Amin bar, flux ratio bar) as PNG images using the matplotlib Agg backend, displayed via `ft.Image` |

### 4.5 Network and Deployment

| ID | Requirement |
|----|-------------|
| FR-28 | In Docker dev mode, each site SHALL be on its own isolated bridge network and SHALL NOT be able to reach other sites or the database directly |
| FR-29 | In production, sites SHALL connect to the server over HTTPS from remote networks |
| FR-30 | FL clients SHALL retry failed HTTP requests with exponential backoff (configurable attempts) |
| FR-31 | FL clients SHALL support SSL verification toggle for development with self-signed certificates |

### 4.6 Data Sources

| ID | Requirement |
|----|-------------|
| FR-32 | In dev mode (`DEV_MODE=true`), the client SHALL generate synthetic Combined 1-A flux data on each FL run rather than reading a static CSV |
| FR-33 | Dev-mode physics parameters (J0, k1, k2, noise, tmp_base) SHALL be configurable per-site via environment variables to create inter-site variance |
| FR-34 | In production mode, the client SHALL monitor a data directory for new `filtration_*.csv` files and trigger training only on unprocessed files |
| FR-35 | ProdDataSource SHALL track processed filenames in a `.processed.json` sidecar written atomically — re-run after crash SHALL NOT reprocess already-trained files |
| FR-36 | The client engine SHALL accept a `DataSource` abstraction; swapping Dev/Prod mode SHALL require no changes to `LocalTrainer` or `Scheduler` |

### 4.7 Aggregation Policy and Runtime Configuration

| ID | Requirement |
|----|-------------|
| FR-37 | The server SHALL support a pluggable aggregation policy: `QuorumPolicy` (N distinct sites) or `TimeWindowPolicy` (elapsed seconds) |
| FR-38 | The active aggregation policy SHALL be changeable at runtime without restarting the server via `PUT /settings` |
| FR-39 | Settings SHALL be persisted to the `server_settings` database table and restored on server restart |
| FR-40 | `PUT /settings` SHALL require an `X-Admin-Key` header matching `SERVER_SECRET_KEY`; requests without it SHALL be rejected with HTTP 403 |
| FR-41 | Numeric settings keys SHALL be validated before storage; invalid values SHALL return HTTP 422 |

### 4.8 Site Monitoring and Run Tracking

| ID | Requirement |
|----|-------------|
| FR-42 | Each site client SHALL expose a `GET /site/status` endpoint returning `site_id`, `run_count`, `last_run_at`, `phase` |
| FR-43 | The `/site/status` endpoint SHALL require `Authorization: Bearer` when `SITE_SECRET` is non-empty |
| FR-44 | The server SHALL periodically poll each configured site's `/site/status` endpoint and update run count and last-run timestamp |
| FR-45 | The server dashboard SHALL display `run_count` and `last_run_at` for each site; `last_run_at` SHALL show `HH:MM` for today's date and `DD Mon` for earlier dates |
| FR-46 | Site registration SHALL be dynamic — any site_id string is valid; no hardcoded `site_1..site_5` enumeration in production Python code |
| FR-50 | `SitePoller` SHALL call `RoundManager.sync_site_phase(site_id, phase)` on each successful heartbeat poll, in addition to `sync_site_run_info()`, so all configured sites appear in the server dashboard with their current training phase |
| FR-51 | `TrainingState` SHALL store `flux_times: list[float]` and `flux_vals: list[float]` from each local training run, updated by `LocalTrainer` after Hermia fitting, for display in the client flux decline chart |

---

## 5. Data Privacy Requirements

| ID | Requirement |
|----|-------------|
| DP-01 | Raw filtration CSV data SHALL never be transmitted from the site to the server or to other sites |
| DP-02 | Only model gradient updates (delta_W) and aggregated metrics SHALL be transmitted |
| DP-03 | Gaussian DP noise SHALL be applied to all weight updates before transmission (sigma configurable via DP_NOISE_SIGMA) |
| DP-04 | Noise sigma SHALL be recorded in the ModelUpdate payload so the server can audit it |
| DP-05 | No `print()` statements in production code; all output via structured logging |

---

## 6. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NF-01 | API response time ≤ 500 ms for all routes except model update aggregation |
| NF-02 | Minimum 80% test coverage on `shared/`, `server/core/`, `client/engine/` |
| NF-03 | All public functions SHALL have full type hints (mypy --strict) |
| NF-04 | Database schema changes SHALL only be applied via Alembic migrations |
| NF-05 | Physical parameter bounds SHALL be enforced in `shared/utils/constants.py` |
| NF-06 | System SHALL operate with PostgreSQL in production and SQLite (aiosqlite) in development |

---

## 7. Constraints and Assumptions

- Site identifiers are arbitrary strings; no hardcoded `site_1..site_5` enumeration in production code. Sites are registered via `REGISTERED_SITES` env var at init time.
- In dev mode, each site generates synthetic data; in prod mode, each site monitors a directory for new `filtration_*.csv` files. Both paths share the same `DataSource` protocol.
- The PINN global model weights are held in memory on the server — no persistence across server restarts in the current implementation
- LRV_required defaults to 4.0 log; adjustable in Manabe model calls
- Differential privacy guarantees follow the Gaussian mechanism; the full Abadi et al. DP-SGD guarantee is not yet enforced (moment accountant not implemented)
- Aggregation policy defaults to `QuorumPolicy(min_sites=3)` but persists to DB and is restored on restart

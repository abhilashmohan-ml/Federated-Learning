# Functional Specification
## Viral Filtration Federated Learning Platform

**Version:** 1.0  
**Date:** 2026-07-04  
**Status:** Implemented (v0.1.0)

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

### 4.5 Network and Deployment

| ID | Requirement |
|----|-------------|
| FR-28 | In Docker dev mode, each site SHALL be on its own isolated bridge network and SHALL NOT be able to reach other sites or the database directly |
| FR-29 | In production, sites SHALL connect to the server over HTTPS from remote networks |
| FR-30 | FL clients SHALL retry failed HTTP requests with exponential backoff (configurable attempts) |
| FR-31 | FL clients SHALL support SSL verification toggle for development with self-signed certificates |

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

- 5 fixed manufacturing sites (site_1 … site_5); expanding beyond 5 requires code changes in the round manager initialisation
- Each site has a local filtration CSV at a configurable path; format is time/flux/TMP columns
- The PINN global model weights are held in memory on the server — no persistence across server restarts in the current implementation
- LRV_required defaults to 4.0 log; adjustable in Manabe model calls
- Differential privacy guarantees follow the Gaussian mechanism; the full Abadi et al. DP-SGD guarantee is not yet enforced (moment accountant not implemented)

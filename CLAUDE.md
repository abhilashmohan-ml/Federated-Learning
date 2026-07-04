# CLAUDE.md  —  Viral Filtration Federated Learning Project
# AI coding assistant reference  (Claude Code / Cursor / Copilot)

## Project Overview
Federated Learning engine for viral filtration of monoclonal antibodies (mAbs)
across 5 manufacturing sites.  No raw process data leaves any site.
Only model gradient / parameter updates are shared with the central server.

## Business Goal
Build a consolidated, filter-agnostic Physics-Informed Neural Network (PINN) that:
  - Predicts flux decline J(t) for any filter / operating condition
  - Calculates LRV via the Manabe capture-probability model
  - Computes minimum filter area (Amin) and flux ratio
  - Classifies the dominant fouling regime (Hermia model selection)

## Repository Layout
  viral_fl_project/
  |-- CLAUDE.md
  |-- WORKPLAN.md
  |-- docker-compose.yml
  |-- .env.example
  |-- requirements/
  |   |-- base.txt
  |   |-- server.txt
  |   `-- client.txt
  |-- shared/
  |   |-- models/
  |   |   |-- hermia.py         6 Hermia blocking models + AIC/BIC
  |   |   |-- manabe.py         Manabe capture probability + LRV
  |   |   |-- polarization.py   Virus internal concentration polarization
  |   |   |-- combined_1a.py    Combined 1-A flux model
  |   |   `-- pinn.py           Physics-Informed Neural Network
  |   |-- schemas/
  |   |   |-- auth.py
  |   |   |-- federation.py
  |   |   `-- filtration.py
  |   |-- crypto/
  |   |   |-- noise.py          Gaussian DP noise
  |   |   `-- secure_agg.py
  |   `-- utils/
  |       |-- logging_config.py
  |       `-- constants.py
  |-- server/
  |   |-- Dockerfile
  |   |-- main.py
  |   |-- config.py
  |   |-- api/  auth.py  federation.py  models.py  health.py
  |   |-- core/ aggregator.py  round_manager.py  model_registry.py
  |   |-- db/   database.py  models.py  migrations/
  |   |-- ui/
  |   |   |-- app.py
  |   |   |-- pages/  dashboard.py  site_monitor.py  global_model.py
  |   |   |           graphs.py  settings.py
  |   |   `-- components/  site_card.py  metric_tile.py  flux_chart.py
  |   |                    lrv_chart.py  round_timeline.py  nav_rail.py
  |   `-- tests/
  |-- client/
  |   |-- Dockerfile
  |   |-- main.py
  |   |-- config.py
  |   |-- engine/  local_trainer.py  data_loader.py  scheduler.py
  |   |-- comms/   fl_client.py  heartbeat.py
  |   |-- ui/
  |   |   |-- app.py
  |   |   `-- pages/  status.py  local_results.py
  |   `-- tests/
  |-- scripts/
  |   |-- generate_synthetic_data.py
  |   |-- init_db.py
  |   |-- run_simulation.py
  |   `-- visualise_results.py
  `-- notebooks/
      |-- 01_hermia_model_exploration.ipynb
      |-- 02_manabe_lrv_fitting.ipynb
      |-- 03_pinn_architecture.ipynb
      `-- 04_federated_round_simulation.ipynb

## Federated Learning Protocol
  1. Server starts Round         ->  POST /federation/round/start
  2. Server broadcasts global W to all registered sites
  3. Each site:
       a. Load local flux/pressure/LRV CSV
       b. Fit Hermia models (AIC/BIC selection)
       c. Fit Manabe + polarization models
       d. Run PINN local training (FedProx loss)
       e. Add Gaussian DP noise to gradients
       f. POST /federation/update  {site_id, round_id, delta_W, n_samples}
  4. Aggregator: FedProx weighted average  ->  new global W
  5. Increment round or broadcast CONVERGED

## Authentication
  - Sites register with a site certificate (self-signed dev / CA-signed prod)
  - Server issues JWT access tokens (15 min) + refresh tokens (7 days)
  - All FL API calls require:  Authorization: Bearer <token>
  - Token claims: site_id, role (client|server), round_id
  - DP noise sigma configurable per site via env var

## Model Architecture  (shared/models/pinn.py)
  Input:  filter descriptors + process conditions
          (TMP, C_feed, pH, IS, virus_size, filter_area, pore_size, NMWCO)
  Level 1: Parameter Predictor NN -> {J0, ks, ki, kc, kcf, k1, k2, Pc, Jcrit, Dv}
  Level 2: Physics Solver (Hermia / Manabe equations)
  Loss:   L_flux + L_LRV + L_physics_constraints + L_FedProx

## Physics Reference
  Standard      1/sqrt(J) = 1/sqrt(J0) + ks*t
  Complete      J = J0 * exp(-kc*t)
  Intermediate  1/J = 1/J0 + ki*t
  Cake          1/J^2 = 1/J0^2 + kcf*t
  Combined 1-A  J(t) = J0/(1+k1*t)^2 * exp(-k2*t)
  Manabe        Pc = 1 - exp(-lambda*J/J_crit)  ;  LRV = log10(1/(1-Pc)) * N_layers
  FedProx       L = (mu/2) * ||W_local - W_global||^2

## Key Constraints
  - NEVER log or transmit raw data outside the site container
  - All secrets via environment variables, never hardcoded
  - No print() in production; use logging (shared/utils/logging_config.py)
  - All public functions must have type hints  (mypy --strict)
  - Minimum 80% test coverage on shared/, server/core/, client/engine/
  - DB schema changes only via Alembic migrations
  - Every federation round must emit a structured audit log entry
  - Physical parameter bounds enforced in shared/utils/constants.py

## Environment Variables  (copy .env.example -> .env)
  SERVER_SECRET_KEY    strong random string
  SERVER_DB_URL        postgresql+asyncpg://user:pass@db:5432/viral_fl
  SERVER_HOST          0.0.0.0
  SERVER_PORT          8000
  FLET_SERVER_PORT     8550
  SITE_ID              site_1  (site_1 ... site_5 per client)
  SERVER_URL           http://server:8000
  SITE_SECRET          per-site secret
  DP_NOISE_SIGMA       0.01
  LOCAL_DATA_PATH      /data/filtration.csv
  LOG_LEVEL            INFO
  FL_ROUNDS            50
  LOCAL_EPOCHS         5
  LEARNING_RATE        0.001
  FEDPROX_MU           0.01

## Quick Start  Docker
  cp .env.example .env
  docker compose up --build
  Server dashboard   http://localhost:8550
  Client site_1      http://localhost:8551  (up to 8555)

## Quick Start  venv
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements/base.txt
  pip install -r requirements/server.txt   # server
  pip install -r requirements/client.txt   # each client
  python scripts/init_db.py
  python scripts/generate_synthetic_data.py
  python scripts/run_simulation.py

## Coding Conventions
  Formatter   black  line-length 100
  Linter      ruff
  Types       mypy --strict
  Imports     isort profile=black
  Commits     feat: fix: chore: docs: test:
  Branches    feature/<ticket>-desc  fix/<ticket>-desc

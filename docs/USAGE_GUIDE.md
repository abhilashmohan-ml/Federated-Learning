# Usage Guide

Viral Filtration Federated Learning — step-by-step setup, configuration, and operation reference.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Installation — Docker](#2-installation--docker)
3. [Installation — virtual environment](#3-installation--virtual-environment)
4. [Configuration reference](#4-configuration-reference)
5. [Generating synthetic data](#5-generating-synthetic-data)
6. [Starting the system](#6-starting-the-system)
7. [Using the server dashboard](#7-using-the-server-dashboard)
8. [Using the site client dashboards](#8-using-the-site-client-dashboards)
9. [Running a simulation end-to-end](#9-running-a-simulation-end-to-end)
10. [Running tests](#10-running-tests)
11. [Providing your own data](#11-providing-your-own-data)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

### Docker setup (recommended for most users)

| Requirement | Minimum version |
|---|---|
| Docker Desktop | 4.x |
| Git | any |
| Python (for data generation scripts) | 3.11+ |

### Virtual environment setup (developers)

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.11+ | 3.12 recommended |
| PostgreSQL | 16 | must be running before `init_db.py` |
| Git | any | |

---

## 2. Installation — Docker

```bash
# Clone the repository
git clone <repo-url>
cd viral_fl_project

# Copy the environment template
cp .env.example .env
```

Open `.env` and set at minimum:

```
SERVER_SECRET_KEY=<64-character random string>
SITE_1_SECRET=<random string>
SITE_2_SECRET=<random string>
SITE_3_SECRET=<random string>
SITE_4_SECRET=<random string>
SITE_5_SECRET=<random string>
```

Generate secure values with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Run the above once for `SERVER_SECRET_KEY` and once for each `SITE_N_SECRET`.

Then:

```bash
# Generate synthetic filtration data for all 5 sites
python scripts/generate_synthetic_data.py

# Build and start all containers
docker compose up --build
```

First build takes 3–5 minutes. Subsequent starts are fast.

### Access points

| URL | What it is |
|---|---|
| http://localhost:8550 | Server dashboard (Federation Monitor) |
| http://localhost:8551 | Site 1 client dashboard |
| http://localhost:8552 | Site 2 client dashboard |
| http://localhost:8553 | Site 3 client dashboard |
| http://localhost:8554 | Site 4 client dashboard |
| http://localhost:8555 | Site 5 client dashboard |
| http://localhost:8000/docs | FastAPI auto-generated API docs |
| http://localhost:8000/health | Health check endpoint |

### Stopping

```bash
docker compose down          # stop containers, keep database
docker compose down -v       # stop containers and wipe database volume
```

---

## 3. Installation — virtual environment

### Step 1 — Create and activate the virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat
```

### Step 2 — Install dependencies

```bash
# Core dependencies (required for both server and clients)
pip install -r requirements/base.txt

# Server-only extras (FastAPI, SQLAlchemy, Flet, dev tools)
pip install -r requirements/server.txt

# Client-only extras (Flet, dev tools) — run on each client machine
pip install -r requirements/client.txt
```

### Step 3 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` — see the [Configuration reference](#4-configuration-reference) section for every variable.
Key values to change for local dev:

```
SERVER_DB_URL=postgresql+asyncpg://viral_fl:viral_fl_pass@localhost:5432/viral_fl
SERVER_URL=http://localhost:8000
```

### Step 4 — Create the PostgreSQL database

```bash
# Create the database and user (run once)
psql -U postgres -c "CREATE USER viral_fl WITH PASSWORD 'viral_fl_pass';"
psql -U postgres -c "CREATE DATABASE viral_fl OWNER viral_fl;"

# Run migrations and seed the site registry
python scripts/init_db.py
```

### Step 5 — Generate synthetic data

```bash
python scripts/generate_synthetic_data.py
# Creates data/site_1/filtration.csv through data/site_5/filtration.csv
```

### Step 6 — Start the server

```bash
# Terminal 1 — FastAPI server
python server/main.py

# Terminal 2 — Flet dashboard
python server/ui/app.py
```

### Step 7 — Start the clients

Open a separate terminal for each site:

```bash
# Windows (PowerShell)
$env:SITE_ID="site_1"; $env:FLET_CLIENT_PORT="8551"; python client/main.py
$env:SITE_ID="site_2"; $env:FLET_CLIENT_PORT="8552"; python client/main.py
# ... repeat for site_3 (8553), site_4 (8554), site_5 (8555)

# Linux / macOS
SITE_ID=site_1 FLET_CLIENT_PORT=8551 python client/main.py
SITE_ID=site_2 FLET_CLIENT_PORT=8552 python client/main.py
```

---

## 4. Configuration reference

All values are read from environment variables (or `.env`). Never hardcode secrets.

### Server variables

| Variable | Default | Description |
|---|---|---|
| `SERVER_SECRET_KEY` | — | JWT signing key. **Must be set.** Generate with `secrets.token_hex(32)`. |
| `SERVER_DB_URL` | `postgresql+asyncpg://...@db:5432/viral_fl` | PostgreSQL connection string. |
| `SERVER_HOST` | `0.0.0.0` | Interface to bind to. |
| `SERVER_PORT` | `8000` | FastAPI port. |
| `FLET_SERVER_PORT` | `8550` | Flet dashboard port. |
| `CORS_ORIGINS` | `[]` | JSON array of allowed origins. `[]` allows all (dev only). |
| `SSL_KEYFILE` | unset | Path to TLS private key (optional). |
| `SSL_CERTFILE` | unset | Path to TLS certificate (optional). |
| `SITE_1_SECRET` … `SITE_5_SECRET` | — | Per-site shared secrets. **Must be set.** Hashed with bcrypt on first `init_db.py` run. |

### Client variables

| Variable | Default | Description |
|---|---|---|
| `SITE_ID` | `site_1` | Which site this client represents (`site_1` … `site_5`). |
| `SERVER_URL` | `http://server:8000` | URL of the FL server. Use `http://localhost:8000` for local dev. |
| `SITE_SECRET` | — | Must match the server's `SITE_N_SECRET` for this site. |
| `LOCAL_DATA_PATH` | `/data/filtration.csv` | Path to this site's filtration CSV file. |
| `FLET_CLIENT_PORT` | `8551` | Flet client dashboard port. |
| `DP_NOISE_SIGMA` | `0.01` | Standard deviation of Gaussian DP noise added to model deltas. Higher = stronger privacy, lower accuracy. |
| `VERIFY_SSL` | `true` | Set `false` when using self-signed development certificates. |
| `CONNECT_TIMEOUT` | `10` | HTTP connection timeout in seconds. |
| `REQUEST_TIMEOUT` | `60` | HTTP request timeout in seconds. |
| `RETRY_ATTEMPTS` | `3` | Number of retries on transient network errors (exponential backoff). |

### Shared FL hyperparameters

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |
| `FL_ROUNDS` | `50` | Total number of federated learning rounds. |
| `LOCAL_EPOCHS` | `5` | Local training epochs per round (PINN). |
| `LEARNING_RATE` | `0.001` | Local optimizer learning rate. |
| `FEDPROX_MU` | `0.01` | FedProx proximal penalty coefficient. Higher = sites stay closer to global model. |
| `MIN_SITES_PER_ROUND` | `3` | Minimum sites that must submit updates before aggregation triggers. |
| `ROUND_TIMEOUT_SECONDS` | `300` | Seconds to wait for quorum before aggregating with whatever updates arrived. |

---

## 5. Generating synthetic data

The synthetic data generator creates realistic flux-decline time-series for all 5 sites,
with each site having slightly different filter characteristics and operating conditions.

```bash
python scripts/generate_synthetic_data.py
```

Output: `data/site_1/filtration.csv` through `data/site_5/filtration.csv`

### CSV format

Each file must contain these columns (additional columns are ignored):

| Column | Units | Description |
|---|---|---|
| `time_min` | minutes | Elapsed filtration time |
| `flux_lmh` | L/(m²·h) | Transmembrane flux (must be > 0) |
| `tmp_bar` | bar | Transmembrane pressure (must be > 0) |

Optional columns that enrich the PINN training (if present):

| Column | Description |
|---|---|
| `lrv_obs` | Measured log reduction value |
| `c_feed_mg_mL` | Feed concentration (mg/mL) |
| `ph` | Process stream pH |

---

## 6. Starting the system

### Recommended startup order

1. Database (automatic via Docker healthcheck, or start PostgreSQL manually)
2. Server — `python server/main.py`
3. Server dashboard — `python server/ui/app.py`
4. Clients — one per site

The server will refuse client connections until it can reach the database.
Clients will retry authentication with exponential backoff, so they can start before the server.

### Triggering a round manually

```bash
curl -X POST http://localhost:8000/federation/round/start \
     -H "Authorization: Bearer <admin-token>"
```

Or use the API docs at http://localhost:8000/docs.

---

## 7. Using the server dashboard

Open http://localhost:8550 in a browser.

### Dashboard page

Shows all 5 sites in real time with their current status:

| Status | Meaning |
|---|---|
| IDLE | Site connected, waiting for a round to start |
| TRAINING | Site is running local model fitting |
| UPLOADING | Site is sending its model update |
| DONE | Site submitted its update for the current round |
| ERROR | Site encountered a training or network error |

The round progress bar shows how many rounds have completed out of the configured total.

### Site Monitor page

Select a site from the dropdown to view:
- Live flux decline chart J(t)
- LRV vs flux scatter
- Key metrics: LRV, Amin, Flux Ratio, Best Hermia model, current round

### Global Model page

Displays the current global PINN parameter values after each aggregation round,
including version number, rounds completed, and sites participated.

### Graphs page

Comparative charts across all 5 sites — flux decline overlay, LRV distribution,
and Hermia model consensus (which blocking model each site selected most often).

### Settings page

Configure FL hyperparameters at runtime and manage the registered site list.

---

## 8. Using the site client dashboards

Open http://localhost:8551 (site_1) through http://localhost:8555 (site_5).

### Status tab

Shows:
- Server connection URL and site ID
- Current round number and phase
- Training progress bar
- Last trained model, flux RMSE, best Hermia model, DP noise sigma

### Local Results tab

Shows site-specific metrics after each local training run:
- Flux decline J(t) chart
- LRV, Amin (m²), Flux Ratio, Best Model name

---

## 9. Running a simulation end-to-end

The simulation script runs a complete federated learning experiment without
starting the full Docker stack:

```bash
python scripts/run_simulation.py
```

This:
1. Starts an in-process server
2. Simulates all 5 sites running local training
3. Runs `FL_ROUNDS` rounds of aggregation
4. Saves round metrics to `results/simulation_results.json`

Visualise results after the simulation:

```bash
python scripts/visualise_results.py
# Opens matplotlib plots: LRV convergence, flux RMSE, model parameter evolution
```

Jupyter notebooks for deeper exploration:

```bash
jupyter notebook notebooks/
```

| Notebook | Contents |
|---|---|
| `01_hermia_model_exploration.ipynb` | Fit all 6 Hermia models, compare AIC/BIC |
| `02_manabe_lrv_fitting.ipynb` | Manabe capture probability and LRV sensitivity |
| `03_pinn_architecture.ipynb` | PINN forward pass and loss function walkthrough |
| `04_federated_round_simulation.ipynb` | Full round simulation with plots |

---

## 10. Running tests

```bash
# Full test suite with coverage report
pytest --cov=shared --cov=server/core --cov=client/engine --cov-report=term-missing

# UI component and page tests
pytest server/tests/ui/ client/tests/ui/ -v

# Single test file
pytest shared/tests/test_hermia.py -v

# Single test class
pytest shared/tests/test_hermia.py::TestFitAllModels -v
```

Expected coverage: 100% for `shared/`, `server/core/`, `client/engine/`.

### Linting and type checking

```bash
black --check --line-length 100 .    # formatting
ruff check .                          # linting
mypy --strict shared/ server/ client/ # type checking
isort --check-only --profile black .  # import ordering
```

---

## 11. Providing your own data

### Single site

Replace `data/site_1/filtration.csv` with your CSV file.
The file must contain `time_min`, `flux_lmh`, and `tmp_bar` columns.
Rows with missing values in those columns are dropped automatically.

Update `LOCAL_DATA_PATH` in `.env` if you place it elsewhere.

### Multiple sites

Each site needs its own CSV. In Docker, mount it as a read-only volume:

```yaml
# docker-compose.yml (override for production)
volumes:
  - /path/to/real/site1/data:/data:ro
```

In venv mode, set `LOCAL_DATA_PATH` per site via environment variable.

### Production secrets

Generate fresh secrets before any production deployment:

```bash
# New server key
python -c "import secrets; print(secrets.token_hex(32))"

# New site secrets (run once per site)
python -c "import secrets; print(secrets.token_hex(32))"
```

Re-run `init_db.py` after changing `SITE_N_SECRET` values to re-hash them.

---

## 12. Troubleshooting

### `Column.__init__() got an unexpected keyword argument 'padding'`

**Cause:** Bug in a `ft.Column(padding=...)` call — `ft.Column` does not accept `padding`.
**Fix:** Already resolved in this codebase. If you see it in your own code, wrap the Column
in a `ft.Container(padding=..., content=ft.Column(...))`.

### `DeprecationWarning: app() is deprecated since version 0.80.0`

**Cause:** Old `ft.app(target=main, ...)` call.
**Fix:** Replace with `ft.run(main, ...)`. Already resolved in this codebase.

### `DNS config watch failed` in browser console

**Cause:** Chromium (embedded in Flet) cannot watch the system DNS registry key on Windows.
**Impact:** None — this is a cosmetic Chromium log message. Your app works normally.

### Server starts but clients cannot authenticate

Check:
1. `SERVER_URL` in client `.env` points to the correct host (`localhost:8000` for venv, `http://server:8000` for Docker)
2. `SITE_SECRET` in client `.env` matches the corresponding `SITE_N_SECRET` in server `.env`
3. `init_db.py` was run after setting the site secrets (secrets are bcrypt-hashed into the database)

### `FileNotFoundError: Data file not found`

Check `LOCAL_DATA_PATH` in `.env`. In Docker it defaults to `/data/filtration.csv` inside the container.
Verify the volume mount in `docker-compose.yml` points to a directory containing `filtration.csv`.

### PostgreSQL connection refused

In venv mode: ensure PostgreSQL is running and the `viral_fl` user and database exist.

```bash
# Check PostgreSQL status (Linux)
sudo systemctl status postgresql

# Verify connection
psql -U viral_fl -d viral_fl -c "SELECT 1;"
```

### Round never aggregates

If `MIN_SITES_PER_ROUND` is set higher than the number of running clients,
the round will wait until `ROUND_TIMEOUT_SECONDS` elapses before aggregating.
Reduce `MIN_SITES_PER_ROUND` in `.env` for development with fewer than 3 sites running.

### Model training produces very high RMSE

Check the CSV data:
- `flux_lmh` values should be positive (typically 50–200 LMH for mAb UF/DF)
- `time_min` should increase monotonically
- Ensure NaN rows are not the majority of the dataset

Run the Hermia exploration notebook to diagnose fitting quality:

```bash
jupyter notebook notebooks/01_hermia_model_exploration.ipynb
```

### Increasing log verbosity

Set `LOG_LEVEL=DEBUG` in `.env` and restart. All structured logs include component,
function, and contextual fields (site_id, round_id, etc.) in JSON format.

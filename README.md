# Viral Filtration Federated Learning

Privacy-preserving federated learning platform for optimising viral filtration of monoclonal antibodies (mAbs) across distributed pharmaceutical manufacturing sites.

Five sites collaboratively train a shared **Physics-Informed Neural Network (PINN)** without any raw process data leaving the site boundary. Only differential-privacy-noised gradient updates are transmitted to the central server.

---

## What it does

| Capability | Detail |
|---|---|
| Flux decline prediction | J(t) for any filter / operating condition |
| LRV calculation | Manabe capture-probability model; checks ≥ 4.0 log regulatory minimum |
| Filter sizing | Minimum filter area A_min and flux ratio |
| Fouling regime classification | Hermia AIC/BIC model selection across 5 blocking models |
| Privacy guarantee | Gaussian DP noise on all gradient updates; raw CSVs never leave the site |
| Live dashboards | Flet web UI for the central server and each of the 5 client sites |

---

## System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Central Server                            │
│                                                                  │
│   FastAPI  :8000               Flet Dashboard  :8550             │
│   ├── /auth                    ├── Round overview                │
│   ├── /federation              ├── Per-site monitor              │
│   ├── /models                  ├── Global model viewer           │
│   └── /health                  └── Settings                      │
│                                                                  │
│   Core                         Database (PostgreSQL / SQLite)    │
│   ├── RoundManager             ├── site_registry                 │
│   ├── FedProxAggregator        ├── rounds                        │
│   └── ModelRegistry            └── model_updates                 │
└──────────────────────────────────────────────────────────────────┘
          │  HTTPS + JWT Bearer
     ┌────┴────┬──────┬──────┬──────────────┐
     │         │      │      │              │
  site_1    site_2  site_3  site_4       site_5
  :8551     :8552   :8553   :8554        :8555

Each site:  DataLoader → LocalTrainer → DP noise → POST /federation/update
```

Sites run in isolated networks. A site can reach the server but **cannot** reach any other site or the database.

---

## Quick start — Docker (recommended)

The fastest path to a full 5-site federation experiment on one machine.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Git

### 1. Clone

```bash
git clone https://github.com/abhilashmohan-ml/Federated-Learning.git
cd Federated-Learning
```

### 2. Generate secrets and configure `.env`

```bash
cp .env.example .env
```

Run the generator **6 times** and keep all outputs:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Open `.env` and fill in the 6 values:

```ini
SERVER_SECRET_KEY=<1st secret>   # signs all JWT tokens — server won't start without this
SITE_1_SECRET=<2nd secret>
SITE_2_SECRET=<3rd secret>
SITE_3_SECRET=<4th secret>
SITE_4_SECRET=<5th secret>
SITE_5_SECRET=<6th secret>
```

Everything else in `.env.example` is already correct for Docker dev.

### 3. Build and start

```bash
docker compose up --build
```

First build: ~3–5 min. Subsequent starts: ~10 s.

On first boot the server automatically:
1. Creates all DB tables
2. Registers all 5 sites with bcrypt-hashed secrets
3. Generates synthetic filtration CSV data for each site

Healthy startup looks like:
```
db      | database system is ready to accept connections
server  | INFO:     Application startup complete.
site_1  | INFO  site_1 registered — awaiting round
```

### 4. Verify

| URL | Expected |
|---|---|
| http://localhost:8000/health/ | `{"status": "ok"}` |
| http://localhost:8000/docs | Swagger UI — all REST endpoints |
| http://localhost:8550 | Server dashboard — 5 sites listed |
| http://localhost:8551–8555 | Site client UIs |

### 5. Trigger a federation round

**Swagger UI (easiest):**

1. Go to http://localhost:8000/docs
2. `POST /auth/token` → **Try it out** → paste:
   ```json
   { "site_id": "site_1", "site_secret": "<your SITE_1_SECRET>" }
   ```
3. Copy the `access_token` → click **Authorize** (padlock) → `Bearer <token>`
4. `POST /federation/round/start` → **Try it out** → **Execute**
5. Watch http://localhost:8550 — round progress updates live

**curl:**

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"site_id":"site_1","site_secret":"YOUR_SITE_1_SECRET"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Start a round
curl -s -X POST http://localhost:8000/federation/round/start \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool

# Poll status (replace 1 with round_id from above)
curl -s http://localhost:8000/federation/round/1 \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool

# Fetch new global model after round completes
curl -s http://localhost:8000/models/global-model \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

**Full headless simulation (50 rounds, no UI needed):**

```bash
docker compose exec server python scripts/run_simulation.py
```

### Stop

```bash
docker compose down        # stop containers, keep database
docker compose down -v     # stop and wipe database (full reset)
```

---

## Quick start — virtual environment (developers)

Use when editing code — no Docker rebuild needed on each change.

### Prerequisites

- Python 3.11+ (3.12 recommended)
- No PostgreSQL needed — SQLite is used by default

### Setup

```bash
git clone https://github.com/abhilashmohan-ml/Federated-Learning.git
cd Federated-Learning

python -m venv .venv
source .venv/bin/activate           # Linux / macOS
# .venv\Scripts\activate            # Windows

pip install -r requirements/server.txt
pip install -r requirements/client.txt
pip install -e .                    # makes shared/ importable from all scripts

cp .env.example .env
```

Edit `.env` — make these 5 changes for venv dev:

```ini
SERVER_SECRET_KEY=<generated secret>
SERVER_DB_URL=sqlite+aiosqlite:///./viral_fl.db   # SQLite, no install needed
SERVER_URL=http://localhost:8000
SITE_1_SECRET=<secret>  # through SITE_5_SECRET
FL_ROUNDS=5             # optional: faster dev cycle
MIN_SITES_PER_ROUND=2   # optional: easier to trigger with fewer clients
```

```bash
python scripts/init_db.py                  # create tables + register 5 sites
python scripts/generate_synthetic_data.py  # create data/site_N/filtration.csv
```

### Run (7 terminals)

```bash
# Terminal 1
python server/main.py

# Terminal 2 (optional — live dashboard)
python server/ui/app.py

# Terminals 3–7
SITE_ID=site_1 python client/main.py
SITE_ID=site_2 python client/main.py
SITE_ID=site_3 python client/main.py
SITE_ID=site_4 python client/main.py
SITE_ID=site_5 python client/main.py

# Windows PowerShell:
$env:SITE_ID="site_1"; python client/main.py
```

Then trigger a round via Swagger or curl (same as Docker above).

**Or run all rounds headless:**

```bash
python scripts/run_simulation.py
```

**Visualise results after simulation:**

```bash
python scripts/visualise_results.py
```

---

## Repository layout

```
viral_fl_project/
├── server/               FastAPI aggregation server + Flet dashboard
│   ├── api/              auth.py  federation.py  models.py  health.py
│   ├── core/             aggregator.py  round_manager.py  model_registry.py
│   ├── db/               database.py  models.py  migrations/
│   └── ui/               app.py  pages/  components/
├── client/               Per-site FL client
│   ├── engine/           local_trainer.py  data_loader.py  scheduler.py
│   ├── comms/            fl_client.py  heartbeat.py
│   └── ui/               app.py  pages/
├── shared/               Code shared by server and all clients
│   ├── models/           hermia.py  manabe.py  polarization.py
│   │                     combined_1a.py  pinn.py
│   ├── crypto/           noise.py (Gaussian DP)  secure_agg.py
│   ├── schemas/          auth.py  federation.py  filtration.py (Pydantic v2)
│   └── utils/            constants.py  logging_config.py
├── scripts/              init_db.py  generate_synthetic_data.py
│                         run_simulation.py  visualise_results.py
├── notebooks/            01_hermia  02_manabe  03_pinn  04_federated_sim
├── data/                 site_1/ … site_5/  (generated — not committed)
├── requirements/         base.txt  server.txt  client.txt
├── docs/                 DEV_SETUP.md  PRODUCTION.md  FUNCTIONAL_SPEC.md
│                         TECHNICAL_SPEC.md  DESIGN_SPEC.md  DB_SCHEMA.md
├── docker-compose.yml
└── .env.example
```

---

## Federated learning protocol

```
Round N
  Server  →  POST /federation/round/start
  Server  →  broadcast global model weights W to all sites
  Site N  →  load local filtration CSV
          →  fit Hermia models (AIC/BIC selection)
          →  fit Manabe + polarisation models
          →  run PINN local training (FedProx loss)
          →  clip L2 norm of gradients + add Gaussian DP noise
          →  POST /federation/update  { site_id, round_id, delta_W, n_samples }
  Server  →  FedProx weighted average  →  new global W
  Server  →  increment round  or  broadcast CONVERGED
```

Aggregation triggers when `MIN_SITES_PER_ROUND` (default 3) updates arrive **or** `ROUND_TIMEOUT_SECONDS` (default 300 s) elapses — whichever is first.

Round status flow: `PENDING → COLLECTING → AGGREGATING → COMPLETE` (or `FAILED`)

---

## Physics models

| Model | Equation |
|---|---|
| Standard blocking | `J(t) = J0 / (1 + ks·t)²` |
| Complete blocking | `J(t) = J0 · exp(−kc·t)` |
| Intermediate blocking | `J(t) = J0 / (1 + J0·ki·t)` |
| Cake filtration | `J(t) = J0 / √(1 + J0²·kcf·t)` |
| Combined 1-A | `J(t) = J0/(1+k1·t)² · exp(−k2·t)` |
| Manabe LRV | `Pc = 1 − exp(−λ·J/J_crit)` ; `LRV = log₁₀(1/(1−Pc)) · N_layers` |
| FedProx proximal | `L_fedprox = (μ/2) · ‖W_local − W_global‖²` |

Best Hermia model selected by AIC. Regulatory minimum: LRV ≥ 4.0 for parvovirus, retrovirus, and herpesvirus.

---

## REST API (summary)

Base URL: `http(s)://<server>:8000`  
All federation routes require `Authorization: Bearer <access_token>`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/auth/token` | Obtain access + refresh tokens |
| `POST` | `/auth/refresh` | Rotate token pair |
| `POST` | `/auth/revoke` | Revoke refresh token (logout) |
| `POST` | `/federation/round/start` | Start a new FL round |
| `POST` | `/federation/update` | Submit a local model update |
| `GET` | `/federation/round/{id}` | Round status and metrics |
| `GET` | `/federation/sites` | All site statuses |
| `GET` | `/models/global-model` | Download current global weights |
| `GET` | `/health/` | Liveness probe (no auth required) |

Full schema with request/response bodies: http://localhost:8000/docs

---

## Technology stack

| Layer | Technology |
|---|---|
| Server framework | FastAPI + Uvicorn |
| Database ORM | SQLAlchemy 2 (asyncio) |
| Migrations | Alembic |
| Database | PostgreSQL 16 (prod) / SQLite (dev) |
| Dashboard UI | Flet |
| ML framework | PyTorch |
| Curve fitting | NumPy · SciPy · Pandas |
| Schema validation | Pydantic v2 |
| Auth | JWT HS256 (python-jose) + bcrypt (passlib) |
| HTTP client | httpx (with exponential-backoff retry) |
| Structured logging | structlog |
| Privacy | Gaussian DP — per-layer L2 clipping + noise injection |
| Containers | Docker + Docker Compose |

---

## Running tests

```bash
# Full suite with coverage report
pytest --cov=shared --cov=server/core --cov=client/engine --cov-report=term-missing

# UI component tests only
pytest server/tests/ui/ client/tests/ui/ -v

# Single module
pytest shared/tests/test_hermia.py -v
```

```bash
# Linting and type checking
black --check --line-length 100 .
ruff check .
mypy --strict shared/ server/ client/
isort --check-only --profile black .
```

Required coverage: ≥ 80% across `shared/`, `server/core/`, `client/engine/`.

---

## Production deployment

See [`docs/PRODUCTION.md`](docs/PRODUCTION.md) for the full guide covering:

- Cloud VM provisioning and firewall rules
- TLS via Let's Encrypt or nginx reverse proxy
- Per-site configuration (outbound-only — no inbound ports needed on site machines)
- Security checklist before going live

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/DEV_SETUP.md`](docs/DEV_SETUP.md) | Full developer setup, daily workflow patterns, troubleshooting |
| [`docs/PRODUCTION.md`](docs/PRODUCTION.md) | Production deployment, TLS, remote site configuration |
| [`docs/FUNCTIONAL_SPEC.md`](docs/FUNCTIONAL_SPEC.md) | Functional requirements (FR-01…FR-31, DP-01…DP-05) |
| [`docs/TECHNICAL_SPEC.md`](docs/TECHNICAL_SPEC.md) | REST API reference, physics equations, PINN architecture, config reference |
| [`docs/DESIGN_SPEC.md`](docs/DESIGN_SPEC.md) | Component design, state machines, auth flow, network isolation |
| [`docs/DB_SCHEMA.md`](docs/DB_SCHEMA.md) | Database schema, SQLAlchemy models, migration workflow |
| [`docs/USAGE_GUIDE.md`](docs/USAGE_GUIDE.md) | End-to-end usage guide, dashboards, Jupyter notebooks |

---

## Key constraints

- Raw filtration CSV data **never** leaves the site container
- All secrets via environment variables — never hardcoded
- No `print()` in production; all output via structured `structlog` logging
- All public functions require full type hints (`mypy --strict`)
- DB schema changes only via Alembic migrations
- Every federation round emits a structured audit log entry

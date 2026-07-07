# Viral Filtration Federated Learning

A federated learning (FL) system for viral filtration of monoclonal antibodies (mAbs) across
multiple manufacturing sites. Raw process data never leaves a site — only privacy-protected
model updates travel over the network.

---

## What it does

Each manufacturing site runs a local training engine that:

1. Loads its private filtration CSV (time, flux, TMP measurements)
2. Fits Hermia blocking models and selects the best by AIC
3. Computes the Manabe capture probability and LRV
4. Applies Gaussian differential privacy noise to the model delta
5. Uploads only the noisy parameter delta to the central server

The server aggregates deltas from all sites using FedProx weighted averaging and broadcasts
a new global Physics-Informed Neural Network (PINN) after each round.

After convergence the global model can predict flux decline J(t), calculate LRV, compute
minimum filter area (Amin), and classify the dominant fouling regime for any filter or
operating condition.

---

## Architecture

```
                        ┌─────────────────────────┐
                        │   Central FL Server      │
                        │  FastAPI  +  PostgreSQL   │
                        │  Flet dashboard :8550     │
                        └────────────┬────────────┘
                                     │  FedProx aggregation
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
   ┌──────────▼──────┐   ┌──────────▼──────┐   ┌──────────▼──────┐
   │   Site 1 :8551   │   │   Site 2 :8552   │   │ Sites 3–5 ...   │
   │  Local trainer   │   │  Local trainer   │   │  Local trainer  │
   │  Private data    │   │  Private data    │   │  Private data   │
   └─────────────────┘   └─────────────────┘   └─────────────────┘
```

Each site network is isolated — sites cannot communicate with each other,
only with the central server.

---

## Quick start — Docker (recommended)

**Prerequisites:** Docker Desktop 4.x+, Git

```bash
git clone <repo-url>
cd viral_fl_project

# 1. Copy and edit environment file
cp .env.example .env
# Edit .env: set SERVER_SECRET_KEY and SITE_N_SECRET values

# 2. Generate synthetic data for all 5 sites
python scripts/generate_synthetic_data.py

# 3. Start everything
docker compose up --build

# 4. Open dashboards
#    Server dashboard:  http://localhost:8550
#    Site 1 client:     http://localhost:8551
#    Site 2 client:     http://localhost:8552
#    Site 3 client:     http://localhost:8553
#    Site 4 client:     http://localhost:8554
#    Site 5 client:     http://localhost:8555
```

To stop: `docker compose down`
To wipe the database: `docker compose down -v`

---

## Quick start — virtual environment (development)

**Prerequisites:** Python 3.11+, PostgreSQL 16

```bash
# 1. Create and activate venv
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements/base.txt
pip install -r requirements/server.txt   # server node
pip install -r requirements/client.txt  # each client node

# 3. Configure environment
cp .env.example .env
# Edit .env — see docs/USAGE_GUIDE.md for every variable

# 4. Initialise the database
python scripts/init_db.py

# 5. Generate synthetic data
python scripts/generate_synthetic_data.py

# 6. Start server (terminal 1)
python server/main.py

# 7. Start server dashboard (terminal 2)
python server/ui/app.py

# 8. Start clients in separate terminals (terminals 3–7)
SITE_ID=site_1 python client/main.py
SITE_ID=site_2 python client/main.py
# ... repeat for site_3, site_4, site_5

# 9. Run a full simulation end-to-end
python scripts/run_simulation.py
```

---

## Repository layout

```
viral_fl_project/
├── shared/                     # Code shared by server and all clients
│   ├── models/
│   │   ├── hermia.py           # 6 Hermia blocking models + AIC/BIC selection
│   │   ├── manabe.py           # Manabe capture probability + LRV calculation
│   │   ├── polarization.py     # Virus concentration polarisation model
│   │   ├── combined_1a.py      # Combined 1-A flux decline model
│   │   └── pinn.py             # Physics-Informed Neural Network (PINN)
│   ├── schemas/                # Pydantic v2 request/response schemas
│   ├── crypto/
│   │   ├── noise.py            # Gaussian differential privacy (DP) noise
│   │   └── secure_agg.py       # Additive secret sharing
│   └── utils/
│       ├── constants.py        # Physical parameter bounds
│       └── logging_config.py   # Structured JSON logging (structlog)
├── server/
│   ├── api/                    # FastAPI routes (auth, federation, models, health)
│   ├── core/
│   │   ├── aggregator.py       # FedProx weighted aggregation
│   │   ├── round_manager.py    # FL round state machine
│   │   └── model_registry.py   # Global model version history
│   ├── db/                     # SQLAlchemy models + Alembic migrations
│   └── ui/                     # Flet server dashboard (port 8550)
├── client/
│   ├── engine/
│   │   ├── data_loader.py      # Reads local filtration CSV (data stays on-site)
│   │   ├── local_trainer.py    # Hermia fit + DP noise + update packaging
│   │   └── scheduler.py        # Polls server, triggers training each round
│   ├── comms/                  # HTTP client + heartbeat
│   └── ui/                     # Flet client dashboard (port 8551–8555)
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── init_db.py
│   ├── run_simulation.py
│   └── visualise_results.py
├── notebooks/                  # Jupyter exploration notebooks (01–04)
├── requirements/
│   ├── base.txt                # Shared deps (numpy, torch, pydantic, ...)
│   ├── server.txt              # Server-only (FastAPI, SQLAlchemy, Flet, ...)
│   └── client.txt              # Client-only (Flet, ...)
├── docker-compose.yml
├── .env.example
├── WORKPLAN.md
└── docs/
    ├── USAGE_GUIDE.md
    ├── FUNCTIONAL_SPEC.md
    ├── TECHNICAL_SPEC.md
    ├── DESIGN_SPEC.md
    ├── DB_SCHEMA.md
    └── SYSTEM_DIAGRAM.html
```

---

## Key technologies

| Layer | Technology |
|---|---|
| FL protocol | Custom FedProx over HTTP |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 16 + SQLAlchemy async + Alembic |
| ML | PyTorch (PINN), SciPy (curve fitting) |
| Privacy | Gaussian DP noise + L2 gradient clipping |
| Dashboards | Flet 0.85+ (web browser mode) |
| Schemas | Pydantic v2 |
| Logging | structlog (structured JSON) |
| Testing | pytest + pytest-cov (100% coverage on core modules) |
| Containers | Docker Compose |

---

## Privacy guarantees

- Raw filtration measurements (time, flux, TMP) never leave the site container
- Only model parameter deltas (delta_W) are transmitted
- All deltas have Gaussian DP noise added before transmission (configurable sigma)
- Site networks are isolated — inter-site communication is impossible
- JWT tokens expire after 15 minutes; refresh tokens after 7 days
- All secrets are environment variables — never hardcoded

---

## FL protocol

```
Round start    Server  →  POST /federation/round/start
Broadcast      Server  →  sends global weights W to all registered sites
Local train    Site    →  fit Hermia models → compute delta_W → add DP noise
Upload         Site    →  POST /federation/update  {site_id, round_id, delta_W}
Aggregate      Server  →  FedProx weighted average of all deltas → new W
Repeat until convergence or FL_ROUNDS reached
```

Minimum sites required per round is configurable (`MIN_SITES_PER_ROUND`, default 3).
If quorum is not reached within `ROUND_TIMEOUT_SECONDS` (default 300 s), the server
aggregates whatever updates it has received.

---

## Running tests

```bash
# All tests with coverage
pytest --cov=shared --cov=server/core --cov=client/engine --cov-report=term-missing

# UI component tests only
pytest server/tests/ui/ client/tests/ui/ -v

# Single module
pytest shared/tests/test_hermia.py -v
```

Core modules (`shared/`, `server/core/`, `client/engine/`) have 100% line and branch coverage.

---

## Documentation

| Document | Contents |
|---|---|
| `docs/USAGE_GUIDE.md` | Step-by-step setup, configuration reference, troubleshooting |
| `docs/FUNCTIONAL_SPEC.md` | Business requirements and user stories |
| `docs/TECHNICAL_SPEC.md` | API reference, data flows, deployment architecture |
| `docs/DESIGN_SPEC.md` | UI/UX design decisions |
| `docs/DB_SCHEMA.md` | Database schema and migration history |
| `docs/SYSTEM_DIAGRAM.html` | Interactive system architecture diagram |
| `WORKPLAN.md` | Build plan and progress tracker |

---

## Licence

Internal research project — not for distribution.

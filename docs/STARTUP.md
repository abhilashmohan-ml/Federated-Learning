# Startup Guide — Viral Filtration Federated Learning

Two ways to run: **Docker** (recommended, one command) or **venv** (local dev).

---

## Option A — Docker (Recommended)

### Prerequisites
- Docker Desktop running
- `git` installed

### Steps

```bash
# 1. Clone and enter the repo
git clone https://github.com/abhilashmohan-ml/Federated-Learning.git
cd Federated-Learning

# 2. Copy env file and set your secrets
cp .env.example .env
```

Edit `.env` — change every `CHANGE_ME` value:

| Variable | What to set |
|----------|-------------|
| `SERVER_SECRET_KEY` | Random 64-char string (use `python -c "import secrets; print(secrets.token_hex(32))"`) |
| `SITE_1_SECRET` … `SITE_5_SECRET` | One unique secret per site (same command above, 5 times) |
| `SITE_SECRET` | Match the site number you're launching (e.g. `SITE_1_SECRET` value for site_1) |

```bash
# 3. Build and start everything (server + DB + 5 clients)
docker compose up --build
```

### What starts

| Service | URL | Description |
|---------|-----|-------------|
| Server dashboard | http://localhost:8550 | Flet UI — round progress, site monitor, global model |
| Site 1 client UI | http://localhost:8551 | Local results + training status |
| Site 2 client UI | http://localhost:8552 | |
| Site 3 client UI | http://localhost:8553 | |
| Site 4 client UI | http://localhost:8554 | |
| Site 5 client UI | http://localhost:8555 | |
| FastAPI docs | http://localhost:8000/docs | REST API (Swagger UI) |
| FastAPI redoc | http://localhost:8000/redoc | |

### Stop everything

```bash
docker compose down          # stop containers, keep volumes
docker compose down -v       # stop + delete DB volume (full reset)
```

---

## Option B — Virtual Environment (Local Dev)

### Prerequisites
- Python 3.11+ (3.14 recommended)
- PostgreSQL (or use the default SQLite for dev — no setup needed)

### Steps

```bash
# 1. Clone
git clone https://github.com/abhilashmohan-ml/Federated-Learning.git
cd Federated-Learning

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
#    Server (includes base.txt):
pip install -r requirements/server.txt

#    Each client also needs:
pip install -r requirements/client.txt

# 4. Copy and configure env
cp .env.example .env
# Edit .env — set SERVER_SECRET_KEY and SITE_N_SECRET values

# 5. Initialise the database
#    (Creates tables + registers 5 sites with hashed secrets)
python scripts/init_db.py

# 6. Generate synthetic training data for all 5 sites
python scripts/generate_synthetic_data.py

# 7. Start the FL server (new terminal)
python server/main.py

# 8. Start the server Flet dashboard (new terminal)
python server/ui/app.py

# 9. Start each site client (separate terminal per site)
SITE_ID=site_1 python client/main.py
SITE_ID=site_2 python client/main.py
SITE_ID=site_3 python client/main.py
SITE_ID=site_4 python client/main.py
SITE_ID=site_5 python client/main.py

# 10. (Optional) Run a full simulation without UIs
python scripts/run_simulation.py
```

---

## Running the FL Protocol Manually (via REST API)

Once the server is up:

```bash
# 1. Get a token for site_1
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"site_id":"site_1","site_secret":"YOUR_SITE_1_SECRET"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2. Start a federation round
curl -X POST http://localhost:8000/federation/round/start \
  -H "Authorization: Bearer $TOKEN"

# 3. Check round status
curl http://localhost:8000/federation/round/1 \
  -H "Authorization: Bearer $TOKEN"

# 4. Get global model weights (after first round completes)
curl http://localhost:8000/models/global-model \
  -H "Authorization: Bearer $TOKEN"
```

---

## Running Tests

```bash
# All tests with coverage report
pytest --cov=shared --cov=server/core --cov=client/engine \
       --cov-report=term-missing -v

# Specific module
pytest server/tests/ -v
pytest shared/ -v
```

---

## Visualise Results (after a simulation run)

```bash
python scripts/visualise_results.py
```

Opens matplotlib plots showing:
- Flux decline J(t) per site
- LRV across rounds
- Hermia model selection distribution
- Global model convergence curve

---

## Environment Variable Reference

All variables live in `.env` (copied from `.env.example`).

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_SECRET_KEY` | `CHANGE_ME` | JWT signing key — must be strong random string |
| `SERVER_DB_URL` | `sqlite+aiosqlite:///./viral_fl.db` | DB connection (use PostgreSQL in prod) |
| `SERVER_HOST` | `0.0.0.0` | FastAPI bind address |
| `SERVER_PORT` | `8000` | FastAPI port |
| `FLET_SERVER_PORT` | `8550` | Server dashboard UI port |
| `SITE_1_SECRET`…`SITE_5_SECRET` | — | Per-site secrets (hashed in DB by `init_db.py`) |
| `MIN_SITES_PER_ROUND` | `3` | Minimum sites before aggregation triggers |
| `ROUND_TIMEOUT_SECONDS` | `300` | Auto-aggregate after this many seconds |
| `FL_ROUNDS` | `50` | Total rounds to run |
| `FEDPROX_MU` | `0.01` | FedProx proximal term weight |

### Client (per site)

| Variable | Default | Description |
|----------|---------|-------------|
| `SITE_ID` | `site_1` | Site identifier (site_1 … site_5) |
| `SERVER_URL` | `http://server:8000` | FL server base URL |
| `SITE_SECRET` | — | Must match the corresponding `SITE_N_SECRET` on the server |
| `DP_NOISE_SIGMA` | `0.01` | Differential privacy noise std dev |
| `LOCAL_DATA_PATH` | `/data/filtration.csv` | Path to the site's local filtration CSV |
| `LOCAL_EPOCHS` | `5` | Local training epochs per round |
| `LEARNING_RATE` | `0.001` | Local optimizer learning rate |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Bad credentials` on `/auth/token` | Site secret mismatch | Make sure `SITE_SECRET` matches `SITE_N_SECRET` used in `init_db.py` |
| `No global model available yet` | No round completed | Start a round via `/federation/round/start` |
| `401 Unauthorized` on federation routes | Token expired (15 min) | Re-request a token via `/auth/token` |
| Server crashes on startup | `CORS` or `SECRET_KEY` misconfigured | Check `.env` — `SERVER_SECRET_KEY` must not be `CHANGE_ME` in prod |
| `create_all` errors on DB init | Schema already exists | Run `alembic upgrade head` instead of `init_db.py` after first run |
| Port already in use | Another process on 8000/8550 | Change `SERVER_PORT` / `FLET_SERVER_PORT` in `.env` |

---

## Project Layout (quick reference)

```
viral_fl_project/
├── server/          FastAPI server + Flet dashboard
│   ├── api/         auth, federation, models, health endpoints
│   ├── core/        FedProxAggregator, RoundManager, ModelRegistry
│   ├── db/          SQLAlchemy models + Alembic migrations
│   └── ui/          Flet dashboard pages + components
├── client/          Per-site FL client
│   ├── engine/      LocalTrainer, DataLoader, Scheduler
│   ├── comms/       FLClient, Heartbeat
│   └── ui/          Flet client status + results pages
├── shared/          Code shared by server and all clients
│   ├── models/      Hermia, Manabe, Polarization, Combined-1A, PINN
│   ├── crypto/      DP noise, SecureAgg
│   ├── schemas/     Pydantic v2 auth/federation/filtration schemas
│   └── utils/       constants, logging
├── scripts/         generate_synthetic_data, init_db, run_simulation, visualise
├── notebooks/       Jupyter exploration notebooks
├── requirements/    base.txt, server.txt, client.txt
├── docs/            This file + future docs
├── docker-compose.yml
└── .env.example
```

Sources:
- [PyPI numpy](https://pypi.org/project/numpy/)
- [PyPI torch](https://pypi.org/project/torch/)
- [PyPI fastapi](https://pypi.org/project/fastapi/)
- [PyPI flet](https://pypi.org/project/flet/)

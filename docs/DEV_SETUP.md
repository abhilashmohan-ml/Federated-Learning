# Developer Setup Guide — Viral Filtration Federated Learning

> **This guide is for local development only.**
> Everything runs on your single machine.
> For deploying to real distributed sites, see [`PRODUCTION.md`](PRODUCTION.md).

---

## What You Are Setting Up

This project simulates **federated learning across 5 manufacturing sites**.
In production, each site would be a separate server in a different factory.
In dev, all 5 "sites" run as isolated containers (or processes) on your laptop/workstation.

Here is what the dev topology looks like:

```
Your Machine
│
├── server   (port 8000 + 8550)   ← FL aggregation server + dashboard UI
│   └── db   (internal only)      ← PostgreSQL, not exposed to you directly
│
├── site_1   (port 8551)          ← Simulated manufacturing site 1
├── site_2   (port 8552)          ← Simulated manufacturing site 2
├── site_3   (port 8553)          ← Simulated manufacturing site 3
├── site_4   (port 8554)          ← Simulated manufacturing site 4
└── site_5   (port 8555)          ← Simulated manufacturing site 5
```

Each site has its **own local CSV dataset** and **cannot see** any other site's data.
Only trained model parameter updates are sent to the server — this is the core FL privacy guarantee.

---

## Choosing Your Setup Method

| | Docker (Option A) | venv (Option B) |
|---|---|---|
| **Best for** | Testing the full federation round end-to-end | Active coding — editing and re-running individual components |
| **Prerequisites** | Docker Desktop | Python 3.11+ only |
| **Startup** | One command | ~10 terminal tabs |
| **Hot reload** | No (rebuild needed) | Yes (edit file, restart process) |
| **Recommended when** | You want to verify federation works | You are writing/debugging code |

**Tip for daily dev:** Use Option B (venv) while coding. Switch to Option A (Docker) to run a full integration test before committing.

---

## Option A — Docker (Full Federation in One Command)

### Why Docker for federation testing?

Docker Compose starts all 8 services (db + server + 5 clients) with proper network isolation between them. Each site container sits on its own bridge network and can reach the server but **cannot** reach any other site — exactly as it would be in production. This is the most realistic federation test you can do without real remote machines.

### Prerequisites

- [Docker Desktop](https://docs.docker.com/get-docker/) installed and **running** (check the tray icon)
- `git` installed

### Step 1 — Get the code

```bash
git clone https://github.com/abhilashmohan-ml/Federated-Learning.git
cd Federated-Learning
```

**Why:** You need a local copy of all source files, Dockerfiles, and the compose config.

### Step 2 — Create your environment file

```bash
cp .env.example .env
```

**Why:** `.env` is never committed to git because it contains secrets. The `.env.example` is a template with safe placeholder values. You must copy it and fill in real secrets before anything will run.

#### Step 2a — Generate secrets

Run this command **6 separate times** and keep each output — you need 6 unique values:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Each run prints a 64-character hex string like:
```
a3f8c2e1d4b7a9f0e2c5d8b1a4f7c0e3d6b9a2f5c8e1d4b7a0f3c6e9d2b5a8f1
```

Keep all 6 outputs open in a notepad — you will paste them below.

#### Step 2b — Edit `.env`: the 4 changes you must make

Open `.env` in your editor. You only need to change the lines shown below. Everything else is already correct for Docker dev.

---

**Change 1 — `SERVER_SECRET_KEY`**

This key signs all JWT authentication tokens. If left as `CHANGE_ME`, the server will refuse to start.

```ini
# Before:
SERVER_SECRET_KEY=CHANGE_ME_strong_random_64_chars

# After (paste your 1st generated secret):
SERVER_SECRET_KEY=a3f8c2e1d4b7a9f0e2c5d8b1a4f7c0e3d6b9a2f5c8e1d4b7a0f3c6e9d2b5a8f1
```

---

**Change 2 — `SITE_1_SECRET` through `SITE_5_SECRET`**

These are the passwords each site uses to prove its identity to the server. The server bcrypt-hashes them and stores the hash — the plain-text value here must be sent exactly by the matching client.

```ini
# Before:
SITE_1_SECRET=CHANGE_ME_site1_secret
SITE_2_SECRET=CHANGE_ME_site2_secret
SITE_3_SECRET=CHANGE_ME_site3_secret
SITE_4_SECRET=CHANGE_ME_site4_secret
SITE_5_SECRET=CHANGE_ME_site5_secret

# After (paste your 2nd–6th generated secrets, one per line, each unique):
SITE_1_SECRET=<2nd secret>
SITE_2_SECRET=<3rd secret>
SITE_3_SECRET=<4th secret>
SITE_4_SECRET=<5th secret>
SITE_5_SECRET=<6th secret>
```

---

**Change 3 — `SITE_SECRET`**

This is the active site secret used by the default client. Set it to the same value as `SITE_1_SECRET`. (When you start clients with `SITE_ID=site_2` etc., Docker injects the correct secret from `docker-compose.yml` automatically — this line is only the fallback.)

```ini
# Before:
SITE_SECRET=CHANGE_ME_site1_secret

# After (paste the same value you used for SITE_1_SECRET):
SITE_SECRET=<your 2nd secret — identical to SITE_1_SECRET>
```

---

**That is all for Docker.** The `SERVER_DB_URL` and `SERVER_URL` lines already have correct Docker values (`postgresql+asyncpg://...@db:5432/...` and `http://server:8000`). Do not change them.

#### Step 2c — Verify your `.env`

Run this quick check to confirm no `CHANGE_ME` values remain:

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
keys = ['SERVER_SECRET_KEY','SITE_1_SECRET','SITE_2_SECRET',
        'SITE_3_SECRET','SITE_4_SECRET','SITE_5_SECRET','SITE_SECRET']
all_ok = True
for k in keys:
    v = os.getenv(k, 'MISSING')
    ok = v and 'CHANGE_ME' not in v and v != 'MISSING'
    status = 'OK   ' if ok else 'FAIL '
    if not ok: all_ok = False
    print(f'{status} {k}={v[:30]}...')
print()
print('Ready to build!' if all_ok else 'Fix the FAIL lines above before continuing.')
"
```

All lines should print `OK`. Fix any `FAIL` lines before the next step.

### Step 3 — Build and start everything

```bash
docker compose up --build
```

**What this does:**
1. Builds the `server` Docker image from `server/Dockerfile`
2. Builds the `client` Docker image from `client/Dockerfile`
3. Starts PostgreSQL (`db` service)
4. Waits for the DB to pass its health check, then starts the server
5. Server runs `init_db.py` on first boot — creates tables and registers all 5 sites with hashed secrets
6. Server runs `generate_synthetic_data.py` — creates realistic but fake filtration CSV data for each site
7. Starts all 5 site clients, each pointing at the server over its private bridge network

**First build takes ~3–5 minutes** (downloading base images, installing Python packages).
Subsequent starts (without `--build`) take ~10 seconds.

You will see interleaved log output from all services. A healthy startup looks like:

```
db        | database system is ready to accept connections
server    | INFO:     Application startup complete.
server    | INFO:     Uvicorn running on http://0.0.0.0:8000
site_1    | INFO  site_1 registered — awaiting round
site_2    | INFO  site_2 registered — awaiting round
...
```

### Step 4 — Verify everything is running

Open these URLs in your browser:

| URL | What you should see |
|-----|---------------------|
| http://localhost:8000/docs | FastAPI Swagger UI — all REST endpoints |
| http://localhost:8000/health/ | `{"status": "ok"}` |
| http://localhost:8550 | Server Flet dashboard — 5 sites listed, round status |
| http://localhost:8551 | Site 1 client UI — "awaiting round" |
| http://localhost:8552 | Site 2 client UI |
| http://localhost:8553 | Site 3 client UI |
| http://localhost:8554 | Site 4 client UI |
| http://localhost:8555 | Site 5 client UI |

### Step 5 — Run a full federation round (the main test)

This is the core flow you want to test. A "round" is one complete cycle:
server broadcasts global model → all sites train locally → sites push updates → server aggregates → new global model.

**Option 5a — trigger via Swagger UI (easiest for dev)**

1. Go to http://localhost:8000/docs
2. Find `POST /auth/token` → click **Try it out** → fill in:
   ```json
   { "site_id": "site_1", "site_secret": "<your SITE_1_SECRET value>" }
   ```
3. Click **Execute** → copy the `access_token` from the response
4. Click **Authorize** (top right padlock) → paste `Bearer <token>`
5. Find `POST /federation/round/start` → click **Try it out** → **Execute**
6. Watch the server dashboard at http://localhost:8550 — you should see the round progress bar move

**Option 5b — trigger via curl**

```bash
# 1. Authenticate as site_1 and capture the token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"site_id":"site_1","site_secret":"YOUR_SITE_1_SECRET"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Token: $TOKEN"   # verify it printed something

# 2. Start a federation round
curl -s -X POST http://localhost:8000/federation/round/start \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool

# 3. Poll round status (replace 1 with the round_id from step 2)
curl -s http://localhost:8000/federation/round/1 \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool

# 4. After round completes, fetch the new global model
curl -s http://localhost:8000/models/global-model \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

**What to expect during a round:**
- Each site client automatically picks up the round, trains on its local CSV, and posts a gradient update
- After `MIN_SITES_PER_ROUND` (default: 3) sites upload, **or** after `ROUND_TIMEOUT_SECONDS` (default: 300s), the aggregator runs FedProx weighted averaging
- Round status transitions: `PENDING → IN_PROGRESS → AGGREGATING → COMPLETED`
- The server dashboard shows live updates

**Option 5c — run the full simulation script (50 rounds, no UI needed)**

```bash
docker compose exec server python scripts/run_simulation.py
```

**Why:** This is a headless end-to-end test. Runs all 50 FL rounds automatically and writes results to the `results/` directory.

### Step 6 — View training results

```bash
docker compose exec server python scripts/visualise_results.py
```

This opens matplotlib plots showing:
- Flux decline J(t) per site across rounds
- LRV (log reduction value) convergence
- Hermia model selection distribution
- Global model parameter convergence curve

### Stopping Docker

```bash
# Stop containers but keep the database volume (resumes where you left off)
docker compose down

# Stop AND wipe the database (full reset — use when DB state is corrupted)
docker compose down -v
```

---

## Option B — Virtual Environment (Active Coding Setup)

Use this when you are **writing code** and need to quickly edit a file and see the result without rebuilding Docker images.

### Prerequisites

- Python 3.11 or newer
- No PostgreSQL needed — the venv path uses SQLite by default (zero configuration)

### Step 1 — Clone and enter the repo

```bash
git clone https://github.com/abhilashmohan-ml/Federated-Learning.git
cd Federated-Learning
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv
```

**Why `.venv`?** This creates an isolated Python environment in a folder called `.venv`. Packages you install here do not pollute your global Python. The `.venv` folder is in `.gitignore` — it is never committed.

Activate it — **you must do this in every new terminal**:

```bash
# Windows (PowerShell or Git Bash)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

You will see `(.venv)` at your prompt. This means the venv is active.

### Step 3 — Install dependencies

```bash
# Install everything the server needs (includes the shared base packages)
pip install -r requirements/server.txt

# Install everything clients need (run once — all 5 clients share this)
pip install -r requirements/client.txt
```

**Why two files?** The server needs FastAPI, SQLAlchemy, Alembic, etc. Clients need PyTorch, httpx for API calls, Flet for the UI. Some packages overlap — that is fine, pip deduplicates them. `base.txt` is included by both and contains shared deps (numpy, pydantic, etc.).

**This step takes 2–10 minutes** — PyTorch is large. Subsequent activations are instant.

After installing deps, run this **once** to make `shared`, `server`, and `client` importable from any script:

```bash
pip install -e .
```

**Why:** Without this, running `python scripts/init_db.py` fails with `ModuleNotFoundError: No module named 'shared'` because Python does not automatically add the project root to `sys.path`. The `-e` (editable) flag installs a pointer to your project root inside the venv — so all three top-level packages are importable permanently, and any edits you make to the source are live immediately with no reinstall needed.

### Step 4 — Configure environment variables

```bash
cp .env.example .env
```

**Why:** `.env` is never committed to git. The `.env.example` is a committed template with placeholder values. You copy it once and fill in real secrets. If you are continuing from an existing repo clone that already has the code, this is your starting point.

#### Step 4a — Generate secrets

Run this command **6 separate times** and keep each output:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Each run produces a unique 64-character hex string like:
```
a3f8c2e1d4b7a9f0e2c5d8b1a4f7c0e3d6b9a2f5c8e1d4b7a0f3c6e9d2b5a8f1
```

Keep all 6 in a notepad — you will paste them below.

#### Step 4b — Edit `.env`: the exact 6 changes for venv

Open `D:/viral_fl_project/.env` in your editor. Make exactly these changes, in order:

---

**Change 1 — `SERVER_SECRET_KEY`**

Signs all JWT tokens. The server will not start if this is left as `CHANGE_ME`.

```ini
# Before:
SERVER_SECRET_KEY=CHANGE_ME_strong_random_64_chars

# After (paste your 1st generated secret):
SERVER_SECRET_KEY=a3f8c2e1d4b7a9f0e2c5d8b1a4f7c0e3d6b9a2f5c8e1d4b7a0f3c6e9d2b5a8f1
```

---

**Change 2 — `SERVER_DB_URL`**

The default points to PostgreSQL inside Docker. For venv dev you use SQLite — no database install required.

```ini
# Before:
SERVER_DB_URL=postgresql+asyncpg://viral_fl:viral_fl_pass@db:5432/viral_fl

# After (SQLite file created automatically in the project root):
SERVER_DB_URL=sqlite+aiosqlite:///./viral_fl.db
```

---

**Change 3 — `SITE_1_SECRET` through `SITE_5_SECRET`**

These are the passwords each site client uses to authenticate. The server hashes them with bcrypt during `init_db.py` and stores only the hash — never the plain text.

```ini
# Before:
SITE_1_SECRET=CHANGE_ME_site1_secret
SITE_2_SECRET=CHANGE_ME_site2_secret
SITE_3_SECRET=CHANGE_ME_site3_secret
SITE_4_SECRET=CHANGE_ME_site4_secret
SITE_5_SECRET=CHANGE_ME_site5_secret

# After (paste your 2nd–6th generated secrets, each unique):
SITE_1_SECRET=<2nd secret>
SITE_2_SECRET=<3rd secret>
SITE_3_SECRET=<4th secret>
SITE_4_SECRET=<5th secret>
SITE_5_SECRET=<6th secret>
```

---

**Change 4 — `SERVER_URL`**

In Docker, services find each other by hostname (`server`). In venv, everything is on your machine, so use `localhost`.

```ini
# Before:
SERVER_URL=http://server:8000

# After:
SERVER_URL=http://localhost:8000
```

---

**Change 5 — `SITE_SECRET`**

The default active site secret (used when no `SITE_ID`-specific override is set). Set it to the same value as `SITE_1_SECRET`.

```ini
# Before:
SITE_SECRET=CHANGE_ME_site1_secret

# After (paste the same value you used for SITE_1_SECRET):
SITE_SECRET=<your 2nd secret — identical to SITE_1_SECRET>
```

---

**Change 6 — optional dev tuning (recommended)**

These are not secrets — they are hyperparameters. Set them to smaller values so federation rounds complete quickly during dev:

```ini
FL_ROUNDS=5              # run 5 rounds instead of 50 (much faster dev cycle)
MIN_SITES_PER_ROUND=2    # only need 2 sites to trigger aggregation (easier to test)
LOG_LEVEL=DEBUG          # verbose output — see exactly what each component is doing
```

---

**Do not change anything else.** All other values in `.env.example` are already correct for venv dev.

#### Step 4c — Verify your `.env`

Run this quick check to confirm no `CHANGE_ME` values remain and the DB URL is SQLite:

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

checks = {
    'SERVER_SECRET_KEY': lambda v: v and 'CHANGE_ME' not in v,
    'SERVER_DB_URL':     lambda v: v and 'sqlite' in v,
    'SERVER_URL':        lambda v: v and 'localhost' in v,
    'SITE_1_SECRET':     lambda v: v and 'CHANGE_ME' not in v,
    'SITE_2_SECRET':     lambda v: v and 'CHANGE_ME' not in v,
    'SITE_3_SECRET':     lambda v: v and 'CHANGE_ME' not in v,
    'SITE_4_SECRET':     lambda v: v and 'CHANGE_ME' not in v,
    'SITE_5_SECRET':     lambda v: v and 'CHANGE_ME' not in v,
    'SITE_SECRET':       lambda v: v and 'CHANGE_ME' not in v,
}

all_ok = True
for key, test in checks.items():
    val = os.getenv(key, 'MISSING')
    ok = test(val)
    if not ok: all_ok = False
    print(f\"{'OK   ' if ok else 'FAIL '} {key}={val[:40]}\")

print()
print('All good — proceed to Step 5.' if all_ok else 'Fix FAIL lines before continuing.')
"
```

All lines must print `OK` before you move on.

### Step 5 — Initialise the database

```bash
python scripts/init_db.py
```

**What this does:**
1. Creates all database tables (sites, rounds, model_versions, audit_logs)
2. Reads `SITE_N_SECRET` values from `.env`
3. bcrypt-hashes each secret and inserts a row in the `site_registry` table for sites 1–5

You only need to run this **once**. If you change secrets in `.env`, delete `viral_fl.db` and re-run.

### Step 6 — Generate synthetic training data

```bash
python scripts/generate_synthetic_data.py
```

**What this does:**
Creates 5 CSV files in `data/site_N/filtration.csv`. Each CSV contains synthetic viral filtration measurements:
- Columns: `time_min`, `flux_lmh`, `tmp_bar`, `c_feed_gL`, `pH`, `ionic_strength`, `virus_size_nm`, `filter_area_cm2`, `pore_size_um`, `lrv`
- Values follow realistic Hermia blocking equations with added noise
- Each site has slightly different operating conditions — this is why federated learning is useful: the global model learns from all 5 regimes simultaneously

You only need to run this **once** per clean install.

### Step 7 — Start the FL server (Terminal 1)

```bash
python server/main.py
```

**What starts:** FastAPI on `http://localhost:8000`. This is the central FL server — it handles authentication, manages federation rounds, and runs the FedProx aggregator. Keep this terminal open and watch for errors.

Verify it works: http://localhost:8000/health/ should return `{"status": "ok"}`

### Step 8 — Start the server dashboard UI (Terminal 2)

```bash
python server/ui/app.py
```

**What starts:** Flet dashboard on `http://localhost:8550`. Shows live round progress, per-site status, and global model metrics. You can test federation without this — it is optional but useful for visual feedback.

### Step 9 — Start the site clients (Terminals 3–7, one per site)

Open **5 new terminals**, activate the venv in each (`.venv\Scripts\activate`), then run:

```bash
# Terminal 3
SITE_ID=site_1 python client/main.py

# Terminal 4
SITE_ID=site_2 python client/main.py

# Terminal 5
SITE_ID=site_3 python client/main.py

# Terminal 6
SITE_ID=site_4 python client/main.py

# Terminal 7
SITE_ID=site_5 python client/main.py
```

**On Windows** if `SITE_ID=...` syntax doesn't work in CMD, use:

```powershell
$env:SITE_ID="site_1"; python client/main.py
```

Each client will authenticate with the server and print something like:
```
INFO  site_1 authenticated — token valid 15 min
INFO  site_1 waiting for federation round
```

### Step 10 — Trigger a federation round

With all 5 clients running, trigger a round via curl or the Swagger UI (same as Docker Step 5 above — replace Docker hostnames with `localhost`).

Watch all 7 terminals simultaneously. You will see the round flow in real time:
- Server terminal: `Round 1 started, waiting for 3+ sites`
- Client terminals: `Training started — epoch 1/5 ...`
- Server terminal: `3 updates received — running FedProx aggregation`
- Server terminal: `Round 1 complete — new global model saved`

---

## Running the Full Simulation (Headless, No UIs)

If you just want to run all 50 FL rounds automatically without managing 7 terminals:

```bash
# With venv active:
python scripts/run_simulation.py
```

This script manages the round lifecycle internally — no separate terminals needed. Useful for quick regression testing after a code change.

---

## Development Workflow: Coding + Testing Federation

This section is for the daily cycle: **edit code → test it → iterate**.

### Pattern 1: Editing shared model code (hermia, manabe, pinn, etc.)

```bash
# 1. Edit the file
#    e.g. shared/models/hermia.py

# 2. Run unit tests immediately
pytest shared/tests/ -v --cov=shared --cov-report=term-missing

# 3. If tests pass, run a quick federation sim to check end-to-end
python scripts/run_simulation.py --rounds 2   # just 2 rounds
```

### Pattern 2: Editing server API code

```bash
# 1. Edit e.g. server/api/federation.py

# 2. Restart only the server (Ctrl+C in Terminal 1, then)
python server/main.py

# 3. Hit the endpoint in Swagger: http://localhost:8000/docs
# 4. Run server tests
pytest server/tests/ -v
```

### Pattern 3: Editing client engine code

```bash
# 1. Edit e.g. client/engine/local_trainer.py

# 2. Restart the affected site client (Ctrl+C in that terminal, then re-run)
SITE_ID=site_1 python client/main.py

# 3. Trigger a new round and watch that client's log
# 4. Run client tests
pytest client/tests/ -v
```

### Pattern 4: Full integration test before committing

```bash
# Run ALL tests with coverage
pytest --cov=shared --cov=server/core --cov=client/engine \
       --cov-report=term-missing -v

# Must see 80%+ coverage and 0 failures before committing
```

---

## Watching Federation in Real Time (Log Tips)

When running venv mode, watch multiple terminals. Key log lines to look for:

| Log line | What it means |
|----------|---------------|
| `Round N started` | Server began a new FL round, broadcast global weights |
| `site_N training epoch X/5` | Site is doing local gradient descent |
| `site_N posted update — delta_W norm: 0.023` | Site uploaded its DP-noised gradient |
| `Aggregating — N/5 updates received` | Server has enough updates to aggregate |
| `FedProx weighted average complete` | New global model computed |
| `Round N COMPLETED — model v2 saved` | Round done, model stored in DB |
| `Round N CONVERGED` | All rounds done, FL training complete |

---

## Environment Variables: Dev vs Production

Below is the complete reference. Dev defaults are already set in `.env.example`.

| Variable | Dev value | Prod value | Why it changes |
|----------|-----------|------------|----------------|
| `SERVER_URL` | `http://localhost:8000` (venv) / `http://server:8000` (Docker) | `https://fl-server.yourdomain.com` | DNS / TLS |
| `SERVER_DB_URL` | `sqlite+aiosqlite:///./viral_fl.db` | `postgresql+asyncpg://user:pass@db:5432/viral_fl` | SQLite is fine for dev, not prod |
| `VERIFY_SSL` | `false` or unset | `true` | No TLS cert in dev |
| `CORS_ORIGINS` | empty (allow all) | comma-separated site origins | Security |
| `RETRY_ATTEMPTS` | `3` | `5` | Internet has more transient errors |
| `REQUEST_TIMEOUT` | `60` | `90–120` | Network latency |
| `DP_NOISE_SIGMA` | `0.01` | tuned per privacy budget | Privacy requirement |
| `FL_ROUNDS` | `5–10` for dev testing | `50` | Faster dev iteration |
| `MIN_SITES_PER_ROUND` | `2` for dev (easier to trigger) | `3` | Need fewer sites to run a round |
| `LOG_LEVEL` | `DEBUG` | `INFO` or `WARNING` | Dev needs verbose output |

**Tip:** For federation testing in dev, set `FL_ROUNDS=5` and `MIN_SITES_PER_ROUND=2` in your `.env`. This makes rounds trigger faster and finish quicker.

---

## Troubleshooting

| Symptom | Most likely cause | Fix |
|---------|-------------------|-----|
| `Bad credentials` on `/auth/token` | `SITE_SECRET` in `.env` doesn't match `SITE_N_SECRET` | Re-check `.env` — both values must be identical plain text |
| `No global model available yet` | No round has completed yet | Start a round via `POST /federation/round/start` |
| `401 Unauthorized` mid-round | JWT expired (15 min lifetime) | The client auto-refreshes — if manual curl, re-run the auth step |
| `create_all` error on DB init | Tables already exist from a previous run | Delete `viral_fl.db` and re-run `init_db.py`, or run `alembic upgrade head` |
| Port `8000` already in use | Another process grabbed it | Change `SERVER_PORT` in `.env`, or `lsof -i :8000` (Mac/Linux) to find the culprit |
| Docker `health check failing` on db | Postgres still starting | Wait 30s — the server waits for the DB health check automatically |
| Site client connects but never trains | No round started | Clients wait passively — you must trigger a round manually or run `run_simulation.py` |
| `mypy --strict` errors | Missing type hints | Add type hints to all new public functions before committing |
| Tests below 80% coverage | New code without tests | Write unit tests for every branch in `shared/`, `server/core/`, `client/engine/` |

---

## Project Layout (Quick Reference)

```
viral_fl_project/
├── server/           FastAPI server + Flet dashboard
│   ├── api/          auth.py  federation.py  models.py  health.py
│   ├── core/         aggregator.py  round_manager.py  model_registry.py
│   ├── db/           database.py  models.py  migrations/
│   └── ui/           app.py + pages/ + components/
├── client/           Per-site FL client
│   ├── engine/       local_trainer.py  data_loader.py  scheduler.py
│   ├── comms/        fl_client.py  heartbeat.py
│   └── ui/           app.py + pages/
├── shared/           Code shared by server and all clients
│   ├── models/       hermia.py  manabe.py  polarization.py  combined_1a.py  pinn.py
│   ├── crypto/       noise.py (Gaussian DP)  secure_agg.py
│   ├── schemas/      auth.py  federation.py  filtration.py  (Pydantic v2)
│   └── utils/        constants.py  logging_config.py
├── scripts/          init_db.py  generate_synthetic_data.py
│                     run_simulation.py  visualise_results.py
├── notebooks/        01_hermia  02_manabe  03_pinn  04_federated_sim
├── data/             site_1/ … site_5/  (generated — not committed)
├── requirements/     base.txt  server.txt  client.txt
├── docs/             DEV_SETUP.md (this file)  PRODUCTION.md  specs/
├── docker-compose.yml
└── .env.example
```

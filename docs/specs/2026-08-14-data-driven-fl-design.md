# Data-Driven Federated Learning — Design Spec
**Date:** 2026-08-14
**Status:** Approved

## Overview

Replace static CSV-based data loading with a mode-aware DataSource abstraction.
In dev mode each FL round resimulates fresh filtration data with per-site physics
jitter. In prod mode a directory poller detects new instrument-written timestamped
CSVs. The server gains event-driven aggregation (quorum or time-window, configurable
via a new settings page persisted to DB), a heartbeat site poller, and per-site run
count + last-run timestamp in the dashboard UI.

---

## 1. Client-side — DataSource Abstraction

### 1.1 Protocol

**File:** `client/engine/data_source.py`

```
DataSource (Protocol)
  get_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]
               returns (time_min, flux_lmh, tmp_bar)
NoNewDataError (Exception sentinel — not a crash)
```

### 1.2 DevDataSource

Instantiated when `DEV_MODE=true`. Holds per-site base physics params identical to
`scripts/generate_synthetic_data.py` SITE_CONFIGS. On each `get_data()` call:

```
J0  = J0_base  * (1 + jitter * N(0,1))
k1  = k1_base  * (1 + jitter * N(0,1))
k2  = k2_base  * (1 + jitter * N(0,1))
flux = Combined-1A(time, J0, k1, k2) + Gaussian_noise(σ=noise_base)
tmp  = tmp_base + 0.004*time + N(0, 0.02)
```

`jitter` from `DEV_JITTER_FRACTION` env var (default `0.05`, i.e. ±5%).
No disk I/O. Inter-site variance is naturally large because base params differ
significantly per filter type.

### 1.3 ProdDataSource

Instantiated when `DEV_MODE=false`. Polls `data/{site_id}/` directory.

**Processed-files sidecar:** `data/{site_id}/.processed.json` — JSON list of
already-consumed filenames. Written atomically (write `.tmp`, rename) to survive
crashes.

On each `get_data()` call:
1. Glob `filtration_*.csv` in data directory
2. Filter out filenames in processed set
3. If none → raise `NoNewDataError`
4. `pd.concat` all new DataFrames on required columns
5. Add filenames to processed set, persist sidecar atomically
6. Return concatenated arrays

Poll interval: `DATA_POLL_SECONDS` env var (default `60`).

### 1.4 LocalTrainer changes

`LocalTrainer.__init__(data_source: DataSource)` — accepts injected source.
`train_and_prepare_update()` calls `self.data_source.get_data()` instead of
`load_filtration_csv()` directly. Zero conditional logic inside trainer.

### 1.5 Scheduler changes (prod mode)

In prod mode the scheduler no longer waits for server-initiated rounds. It polls
via `ProdDataSource.get_data()` on `DATA_POLL_SECONDS` interval. On new data:
train → push update to `POST /federation/update`. The server's
`POST /federation/round/start` endpoint still exists but prod clients do not call
it — the server aggregates purely on incoming updates via the configured policy.
In dev mode scheduler works as today (server-initiated rounds), trainer draws
fresh data each time.

### 1.6 Client status endpoint

New lightweight FastAPI app running in a background thread on port `CLIENT_STATUS_PORT`
(default `900N`, N = site number: site_1→9001 … site_5→9005).

```
GET /site/status
→ { site_id, run_count, last_run_at, phase }
```

Polled by the server heartbeat. `run_count` and `last_run_at` tracked in
`client/engine/state.py` alongside existing phase state.

### 1.7 New environment variables (client)

| Variable | Default | Purpose |
|---|---|---|
| `DEV_MODE` | `false` | Switch data source |
| `DEV_JITTER_FRACTION` | `0.05` | ±% param jitter per round |
| `DATA_POLL_SECONDS` | `60` | Prod directory poll interval |
| `CLIENT_STATUS_PORT` | derived | Status endpoint port: site_1→9001, site_2→9002 … site_5→9005 (same derivation logic as `flet_client_port`) |

---

## 2. Server-side — AggregationPolicy + SitePoller + Settings Store

### 2.1 AggregationPolicy protocol

**File:** `server/core/aggregation_policy.py`

```
AggregationPolicy (Protocol)
  should_aggregate(
      updates_since_last: int,
      sites_contributed: set[str],
      elapsed_seconds: float
  ) -> bool

QuorumPolicy(min_sites: int = 3)
  → True when len(sites_contributed) >= min_sites

TimeWindowPolicy(window_seconds: int = 1800)
  → True when elapsed_seconds >= window_seconds AND updates_since_last >= 1
```

Both are plain dataclasses. No asyncio. Fully unit-testable without a server.

### 2.2 RoundManager changes

- Holds `_policy: AggregationPolicy`, swappable via `set_policy(policy)` at runtime
- `receive_update()` calls `_policy.should_aggregate(...)` instead of hardcoded check
- `_policy_timer_start: float` tracks elapsed time for time-window policy (reset after each aggregation)
- New fields: `_site_run_counts: dict[str, int]`, `_site_last_run_at: dict[str, datetime | None]`
  - Incremented/updated on each `receive_update()` per site
  - Exposed via `get_status_snapshot()` for dashboard

### 2.3 SitePoller

**File:** `server/core/site_poller.py`

Asyncio background task, started once from `server/main.py`.

Every `HEARTBEAT_SECONDS` (default 30, env var):
1. For each site in `SITE_1_URL`…`SITE_5_URL`, GET `{site_url}/site/status`
2. Parse `{site_id, run_count, last_run_at, phase}`
3. If site's `run_count` > server's tracked count → update `RoundManager` counts
4. On request failure → log `site_unreachable` warning, mark site `ERROR` in status

Poller is read-only — does not trigger aggregation. Aggregation triggered only by
`receive_update()` (site push path).

### 2.4 Settings store

**File:** `server/db/settings_store.py`

Persists aggregation config to DB via a `server_settings` key-value table.

```sql
CREATE TABLE server_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

Seed values:
```
aggregation_mode      → "quorum"
quorum_min_sites      → "3"
time_window_seconds   → "1800"
heartbeat_seconds     → "30"
```

`SettingsStore.load() -> dict` and `SettingsStore.save(key, value)` methods.
`RoundManager` reads on startup; settings page writes + calls `rm.set_policy(...)` live.

### 2.5 Alembic migration

New migration: `server/db/migrations/versions/<alembic-hash>_add_server_settings_table.py`
Creates `server_settings` table and seeds default rows.

### 2.6 New API endpoints (server)

```
GET  /settings      → current aggregation config dict
PUT  /settings      → update config (writes DB, applies policy live)
```

### 2.7 Extended `/internal/status` response

Adds to existing payload:
```json
{
  "run_counts":   { "site_1": 7, "site_2": 3, ... },
  "last_run_at":  { "site_1": "2026-08-14T14:32:00Z", ... }
}
```

### 2.8 New environment variables (server)

| Variable | Default | Purpose |
|---|---|---|
| `HEARTBEAT_SECONDS` | `30` | Site poller interval |
| `SITE_1_URL`…`SITE_5_URL` | — | Base URLs for heartbeat poller |

---

## 3. Server UI

### 3.1 Site card updates

**File:** `server/ui/components/site_card.py`

Two new `ft.Text` widgets per card:
- `"Runs: --"` → `"Runs: {n}"` on each dashboard refresh
- `"Last: --"` → `"Last: {HH:MM}"` (today) or `"Last: {DD Mon}"` (older)

New method: `set_run_info(run_count: int, last_run_at: datetime | None)`
Dashboard polling loop calls both `set_status()` and `set_run_info()`.

### 3.2 New settings page

**File:** `server/ui/pages/settings.py`

**Aggregation Policy section:**
- Radio: `Quorum` (default) | `Time Window`
- Quorum selected: number field `Min sites required` (1–5, default 3)
- Time Window selected: number field `Window (minutes)` (default 30)
- `Apply` button → PUT `/settings`, shows confirmation snackbar

**Heartbeat section:**
- Number field: `Site poll interval (seconds)` (default 30)
- `Apply` button → PUT `/settings`

### 3.3 Nav rail update

**File:** `server/ui/components/nav_rail.py`

Add Settings destination (gear icon) routing to settings page.

---

## 4. Dev Launcher Script

**File:** `start_all_server_clients_dev.ps1` (existing, modified)

Changes:
- Add ports `9001–9005` to `$ports` array (freed on startup)
- Server launch: prepend `$env:DEV_MODE = 'true'`
- Each client launch: add `$env:DEV_MODE = 'true'` alongside `$env:SITE_ID`

**New file:** `start_all_server_clients.ps1` (prod launcher, no DEV_MODE set)

---

## 5. Prod file naming convention

Instruments write: `filtration_YYYYMMDD_HHMMSS.csv` into `data/{site_id}/`.

`LOCAL_DATA_PATH` env var repurposed in prod mode to point to the **directory**
(not a specific file). In dev mode it is unused (data generated in memory).

---

## 6. Testing

### Client tests (100% line+branch coverage required)

| File | What it tests |
|---|---|
| `test_dev_data_source.py` | Each call returns different arrays; jitter within bounds; arrays physically valid (flux>0, time monotonic) |
| `test_prod_data_source.py` | File globbing; multi-CSV concat; sidecar persist; `NoNewDataError` on no new files; atomic write |
| `test_local_trainer_datasource.py` | `get_data()` called once per `train_and_prepare_update()`; works with both source types |

### Server tests (100% line+branch coverage required)

| File | What it tests |
|---|---|
| `test_aggregation_policy.py` | QuorumPolicy triggers at exactly min_sites; TimeWindowPolicy triggers at elapsed≥window with ≥1 update, not on empty |
| `test_settings_store.py` | Round-trip save/load; defaults on first read |
| `test_site_poller.py` | Mock httpx: run count updates correctly; warning on unreachable; no aggregation triggered |
| `test_round_manager_policy.py` | `set_policy()` swaps live without data loss; run counts + last_run_at tracked correctly |

---

## 7. Files changed / created summary

### New files
```
client/engine/data_source.py
client/comms/status_server.py
server/core/aggregation_policy.py
server/core/site_poller.py
server/db/settings_store.py
server/db/migrations/versions/<alembic-hash>_add_server_settings_table.py
server/api/settings.py
server/ui/pages/settings.py
start_all_server_clients.ps1          (prod launcher)
client/tests/test_dev_data_source.py
client/tests/test_prod_data_source.py
client/tests/test_local_trainer_datasource.py
server/tests/test_aggregation_policy.py
server/tests/test_settings_store.py
server/tests/test_site_poller.py
server/tests/test_round_manager_policy.py
```

### Modified files
```
client/engine/local_trainer.py        (inject DataSource)
client/engine/scheduler.py            (prod: poll data dir; dev: server-initiated)
client/engine/state.py                (add run_count, last_run_at)
client/config.py                      (new env vars)
client/main.py                        (instantiate correct DataSource; start status server)
server/core/round_manager.py          (pluggable policy; run counts; last_run_at)
server/main.py                        (start SitePoller; register settings API)
server/api/federation.py              (extend /internal/status response)
server/ui/components/site_card.py     (run count + last run widgets)
server/ui/components/nav_rail.py      (settings destination)
server/ui/app.py                      (call set_run_info in poll loop)
start_all_server_clients_dev.ps1      (DEV_MODE=true; status ports)
```

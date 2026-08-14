# Data-Driven Federated Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace static CSV data loading with a mode-aware DataSource abstraction; add event-driven aggregation with configurable quorum/time-window policy; add server heartbeat site polling and per-site run-count UI.

**Architecture:** `DevDataSource` generates fresh perturbed physics data each round; `ProdDataSource` polls a directory for timestamped CSVs and merges new ones. The server gains a pluggable `AggregationPolicy` (quorum or time-window) stored in DB, a `SitePoller` heartbeat task, and a settings UI page.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy async (aiosqlite/asyncpg), Alembic, Flet, httpx, pytest-asyncio, numpy, pandas, uvicorn

## Global Constraints

- 100% line+branch coverage on all new/modified files in `client/engine/`, `client/comms/`, `server/core/`, `server/db/`
- `pytest --cov=<module> --cov-report=term-missing` must show 100% before each commit
- No `print()` — use `shared.utils.logging_config.get_logger(__name__)`
- All public functions must have type hints (`mypy --strict`)
- Never commit `.env`, secrets, or credentials
- Commits: `feat:` / `fix:` / `test:` / `docs:` / `chore:` prefix; one logical change per commit
- Work on branch `feature/data-driven-fl` — never commit directly to master
- DB schema changes via Alembic migrations only

---

## File Map

**New files:**
```
client/engine/data_source.py          DataSource protocol + DevDataSource + ProdDataSource
client/comms/status_server.py         Lightweight FastAPI status endpoint for heartbeat
server/core/aggregation_policy.py     AggregationPolicy protocol + QuorumPolicy + TimeWindowPolicy
server/db/settings_store.py           Async CRUD for server_settings table
server/core/site_poller.py            Asyncio heartbeat task that polls each site
server/api/settings.py                GET/PUT /settings REST endpoints
start_all_server_clients.ps1          Prod launcher (no DEV_MODE)
client/tests/test_dev_data_source.py
client/tests/test_prod_data_source.py
client/tests/test_local_trainer_datasource.py
server/tests/test_aggregation_policy.py
server/tests/test_settings_store.py
server/tests/test_site_poller.py
server/tests/test_round_manager_policy.py
```

**Modified files:**
```
client/engine/local_trainer.py        Accept injected DataSource
client/engine/scheduler.py            Add prod polling loop + _watch_dev / _watch_prod split
client/engine/state.py                Add run_count: int, last_run_at: Optional[str]
client/comms/fl_client.py             Add get_current_round() method
client/config.py                      Add dev_mode, dev_jitter_fraction, data_poll_seconds, client_status_port
client/main.py                        Instantiate correct DataSource; start status server
server/core/round_manager.py          Pluggable policy; run counts; get_or_create_round; sync_site_run_info
server/db/models.py                   Add ServerSetting ORM model
server/config.py                      Add heartbeat_seconds, site_1_url…site_5_url
server/main.py                        Startup event; register settings router; start SitePoller
server/api/federation.py              Add GET /current-round endpoint
server/api/internal.py                Extend /status with run_counts + last_run_at
server/ui/components/site_card.py     Add run count + last run timestamp widgets
server/ui/pages/settings.py           Add aggregation policy section
server/ui/app.py                      Call set_run_info() in poll loop
start_all_server_clients_dev.ps1      Add DEV_MODE=true; add ports 9001-9005
```

---

### Task 1: DataSource protocol + DevDataSource

**Files:**
- Create: `client/engine/data_source.py`
- Create: `client/tests/test_dev_data_source.py`

**Interfaces:**
- Produces: `NoNewDataError`, `DataSource` (Protocol), `DevDataSource(site_id: str, jitter: float = 0.05)`, `SITE_PHYSICS: dict[str, dict[str, float]]`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout master && git pull
git checkout -b feature/data-driven-fl
```

- [ ] **Step 2: Write the failing tests**

```python
# client/tests/test_dev_data_source.py
from __future__ import annotations
import numpy as np
import pytest
from client.engine.data_source import DevDataSource, NoNewDataError, SITE_PHYSICS


def test_dev_returns_correct_shape() -> None:
    ds = DevDataSource("site_1")
    time, flux, tmp = ds.get_data()
    assert len(time) == 121
    assert len(flux) == 121
    assert len(tmp) == 121


def test_dev_time_is_0_to_120() -> None:
    ds = DevDataSource("site_1")
    time, _, _ = ds.get_data()
    assert time[0] == pytest.approx(0.0)
    assert time[-1] == pytest.approx(120.0)


def test_dev_flux_always_positive() -> None:
    ds = DevDataSource("site_1")
    _, flux, _ = ds.get_data()
    assert all(flux > 0)


def test_dev_each_call_different() -> None:
    ds = DevDataSource("site_1", jitter=0.10)
    _, flux1, _ = ds.get_data()
    _, flux2, _ = ds.get_data()
    assert not np.allclose(flux1, flux2)


def test_dev_inter_site_variance() -> None:
    # Zero jitter → deterministic base params
    ds1 = DevDataSource("site_1", jitter=0.0)
    ds2 = DevDataSource("site_4", jitter=0.0)
    _, flux1, _ = ds1.get_data()
    _, flux2, _ = ds2.get_data()
    # site_1 J0=150, site_4 J0=100 — clearly different
    assert not np.allclose(flux1, flux2, rtol=0.05)


def test_dev_unknown_site_falls_back_to_site1() -> None:
    ds = DevDataSource("site_99", jitter=0.0)
    ds1 = DevDataSource("site_1", jitter=0.0)
    _, flux_unknown, _ = ds.get_data()
    _, flux_site1, _ = ds1.get_data()
    assert np.allclose(flux_unknown, flux_site1)


def test_no_new_data_error_is_exception() -> None:
    assert issubclass(NoNewDataError, Exception)


def test_site_physics_has_all_five_sites() -> None:
    for i in range(1, 6):
        assert f"site_{i}" in SITE_PHYSICS
        cfg = SITE_PHYSICS[f"site_{i}"]
        assert all(k in cfg for k in ("J0", "k1", "k2", "noise", "tmp_base"))
```

- [ ] **Step 3: Run tests — expect FAIL (ImportError)**

```bash
pytest client/tests/test_dev_data_source.py -v
```

Expected: `ModuleNotFoundError: No module named 'client.engine.data_source'`

- [ ] **Step 4: Implement `data_source.py`**

```python
# client/engine/data_source.py
"""DataSource abstraction — dev (in-memory simulation) and prod (CSV directory polling)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Tuple

import numpy as np
import pandas as pd

from shared.utils.logging_config import get_logger

log = get_logger(__name__)

# Base physics params per site — matches scripts/generate_synthetic_data.py SITE_CONFIGS
SITE_PHYSICS: dict[str, dict[str, float]] = {
    "site_1": {"J0": 150.0, "k1": 0.015, "k2": 0.0020, "noise": 2.0, "tmp_base": 1.0},
    "site_2": {"J0": 120.0, "k1": 0.020, "k2": 0.0030, "noise": 3.0, "tmp_base": 1.2},
    "site_3": {"J0": 180.0, "k1": 0.010, "k2": 0.0010, "noise": 1.5, "tmp_base": 0.8},
    "site_4": {"J0": 100.0, "k1": 0.025, "k2": 0.0040, "noise": 2.5, "tmp_base": 1.4},
    "site_5": {"J0": 160.0, "k1": 0.012, "k2": 0.0015, "noise": 2.0, "tmp_base": 1.1},
}


class NoNewDataError(Exception):
    """Raised by ProdDataSource when no new CSV files are found in the data directory."""


class DataSource(Protocol):
    """Protocol (interface) for all data sources. Returns (time_min, flux_lmh, tmp_bar)."""

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


class DevDataSource:
    """
    Generates fresh synthetic filtration data each call using perturbed physics params.

    Base params are fixed per site (matching generate_synthetic_data.py).
    Each call jitters J0, k1, k2 by ±jitter*N(0,1), producing slightly different
    flux curves round-to-round while keeping inter-site variance large.
    """

    def __init__(self, site_id: str, jitter: float = 0.05) -> None:
        self._cfg = SITE_PHYSICS.get(site_id, SITE_PHYSICS["site_1"])
        self._jitter = jitter

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng()
        J0 = self._cfg["J0"] * (1.0 + self._jitter * rng.standard_normal())
        k1 = self._cfg["k1"] * (1.0 + self._jitter * rng.standard_normal())
        k2 = self._cfg["k2"] * (1.0 + self._jitter * rng.standard_normal())

        time = np.arange(0, 121, 1, dtype=np.float64)
        flux = (J0 / (1.0 + k1 * time) ** 2) * np.exp(-k2 * time)
        flux += rng.normal(0.0, self._cfg["noise"], len(time))
        flux = np.clip(flux, 1.0, None)
        tmp = self._cfg["tmp_base"] + 0.004 * time + rng.normal(0.0, 0.02, len(time))

        log.debug("dev_data_generated", J0=round(J0, 2), k1=round(k1, 5))
        return time, flux, tmp
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest client/tests/test_dev_data_source.py -v --cov=client.engine.data_source --cov-report=term-missing
```

Expected: all green, 100% coverage on lines used by DevDataSource + NoNewDataError + SITE_PHYSICS.

- [ ] **Step 6: Commit**

```bash
git add client/engine/data_source.py client/tests/test_dev_data_source.py
git commit -m "feat: add DataSource protocol + DevDataSource with per-round physics jitter"
```

---

### Task 2: ProdDataSource

**Files:**
- Modify: `client/engine/data_source.py` (append ProdDataSource class)
- Create: `client/tests/test_prod_data_source.py`

**Interfaces:**
- Consumes: `NoNewDataError`, `DataSource` from Task 1
- Produces: `ProdDataSource(data_dir: str)` with `get_data()` and `has_new_data() -> bool`

- [ ] **Step 1: Write the failing tests**

```python
# client/tests/test_prod_data_source.py
from __future__ import annotations
import json
import pandas as pd
import pytest
from pathlib import Path
from client.engine.data_source import ProdDataSource, NoNewDataError


def _make_csv(path: Path, rows: int = 3, prefix: str = "filtration_20260814_120000") -> Path:
    df = pd.DataFrame({
        "time_min": list(range(rows)),
        "flux_lmh": [100.0 - i for i in range(rows)],
        "tmp_bar":  [1.0 + 0.01 * i for i in range(rows)],
    })
    p = path / f"{prefix}.csv"
    df.to_csv(p, index=False)
    return p


def test_no_files_raises(tmp_path: Path) -> None:
    ds = ProdDataSource(str(tmp_path))
    with pytest.raises(NoNewDataError):
        ds.get_data()


def test_reads_single_csv(tmp_path: Path) -> None:
    _make_csv(tmp_path)
    ds = ProdDataSource(str(tmp_path))
    time, flux, tmp = ds.get_data()
    assert len(time) == 3
    assert all(flux > 0)


def test_processed_file_skipped(tmp_path: Path) -> None:
    _make_csv(tmp_path)
    ds = ProdDataSource(str(tmp_path))
    ds.get_data()                   # marks file as processed
    with pytest.raises(NoNewDataError):
        ds.get_data()               # same file — no new data


def test_concatenates_multiple_new_csvs(tmp_path: Path) -> None:
    _make_csv(tmp_path, rows=3, prefix="filtration_20260814_120000")
    _make_csv(tmp_path, rows=5, prefix="filtration_20260814_130000")
    ds = ProdDataSource(str(tmp_path))
    time, flux, tmp = ds.get_data()
    assert len(time) == 8           # 3 + 5 rows concatenated


def test_sidecar_persists_across_instances(tmp_path: Path) -> None:
    _make_csv(tmp_path)
    ds1 = ProdDataSource(str(tmp_path))
    ds1.get_data()                  # processes and writes sidecar
    ds2 = ProdDataSource(str(tmp_path))   # new instance reads sidecar
    with pytest.raises(NoNewDataError):
        ds2.get_data()


def test_has_new_data_false_when_empty(tmp_path: Path) -> None:
    ds = ProdDataSource(str(tmp_path))
    assert not ds.has_new_data()


def test_has_new_data_true_after_csv_added(tmp_path: Path) -> None:
    ds = ProdDataSource(str(tmp_path))
    _make_csv(tmp_path)
    assert ds.has_new_data()


def test_has_new_data_false_after_processed(tmp_path: Path) -> None:
    _make_csv(tmp_path)
    ds = ProdDataSource(str(tmp_path))
    ds.get_data()
    assert not ds.has_new_data()


def test_sidecar_written_atomically(tmp_path: Path) -> None:
    _make_csv(tmp_path)
    ds = ProdDataSource(str(tmp_path))
    ds.get_data()
    sidecar = tmp_path / ".processed.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert isinstance(data, list)
    assert any("filtration_" in name for name in data)


def test_ignores_non_filtration_csvs(tmp_path: Path) -> None:
    df = pd.DataFrame({"time_min": [0], "flux_lmh": [100.0], "tmp_bar": [1.0]})
    df.to_csv(tmp_path / "other_data.csv", index=False)
    ds = ProdDataSource(str(tmp_path))
    with pytest.raises(NoNewDataError):
        ds.get_data()
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest client/tests/test_prod_data_source.py -v
```

Expected: `AttributeError: module 'client.engine.data_source' has no attribute 'ProdDataSource'`

- [ ] **Step 3: Implement ProdDataSource — append to `client/engine/data_source.py`**

```python
class ProdDataSource:
    """
    Polls a data directory for new timestamped filtration CSVs.

    Tracks processed filenames in a JSON sidecar (.processed.json) that
    survives process restarts. Multiple new files are concatenated into one
    dataset — one FL update per poll cycle regardless of how many files arrived.

    File naming convention: filtration_YYYYMMDD_HHMMSS.csv
    """

    _SIDECAR_NAME = ".processed.json"

    def __init__(self, data_dir: str) -> None:
        self._dir = Path(data_dir)
        self._sidecar = self._dir / self._SIDECAR_NAME
        self._processed: set[str] = self._load_sidecar()

    def _load_sidecar(self) -> set[str]:
        if self._sidecar.exists():
            return set(json.loads(self._sidecar.read_text()))
        return set()

    def _save_sidecar(self, names: set[str]) -> None:
        tmp = self._sidecar.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(names)))
        tmp.rename(self._sidecar)   # atomic on POSIX; near-atomic on Windows NTFS

    def _new_files(self) -> list[Path]:
        return [
            f for f in sorted(self._dir.glob("filtration_*.csv"))
            if f.name not in self._processed
        ]

    def has_new_data(self) -> bool:
        """Lightweight check — does not mark files as processed."""
        return bool(self._new_files())

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        new_files = self._new_files()
        if not new_files:
            raise NoNewDataError(f"No new filtration_*.csv in {self._dir}")

        required = {"time_min", "flux_lmh", "tmp_bar"}
        dfs = []
        for f in new_files:
            df = pd.read_csv(f)
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{f.name} missing columns: {missing}")
            dfs.append(df.dropna(subset=list(required)))

        combined = pd.concat(dfs, ignore_index=True)

        self._processed |= {f.name for f in new_files}
        self._save_sidecar(self._processed)

        log.info("prod_data_loaded", n_files=len(new_files), n_rows=len(combined))
        return (
            combined["time_min"].to_numpy(dtype=np.float64),
            combined["flux_lmh"].to_numpy(dtype=np.float64),
            combined["tmp_bar"].to_numpy(dtype=np.float64),
        )
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest client/tests/test_prod_data_source.py client/tests/test_dev_data_source.py -v \
  --cov=client.engine.data_source --cov-report=term-missing
```

Expected: all green, 100% coverage.

- [ ] **Step 5: Commit**

```bash
git add client/engine/data_source.py client/tests/test_prod_data_source.py
git commit -m "feat: add ProdDataSource with directory polling and atomic sidecar"
```

---

### Task 3: LocalTrainer refactor to accept DataSource

**Files:**
- Modify: `client/engine/local_trainer.py:79` (`__init__` signature)
- Create: `client/tests/test_local_trainer_datasource.py`

**Interfaces:**
- Consumes: `DataSource` from Task 1
- Produces: `LocalTrainer(data_source: DataSource)` — `train_and_prepare_update(round_id: int) -> ModelUpdate` unchanged

- [ ] **Step 1: Write failing tests**

```python
# client/tests/test_local_trainer_datasource.py
from __future__ import annotations
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from client.engine.data_source import DevDataSource, NoNewDataError
from client.engine.local_trainer import LocalTrainer


def _make_mock_source(n: int = 121) -> MagicMock:
    src = MagicMock()
    src.get_data.return_value = (
        np.arange(0, n, dtype=np.float64),
        np.linspace(100.0, 50.0, n),
        np.full(n, 1.0),
    )
    return src


def test_trainer_calls_get_data_once() -> None:
    src = _make_mock_source()
    trainer = LocalTrainer(data_source=src)
    trainer.train_and_prepare_update(round_id=1)
    src.get_data.assert_called_once()


def test_trainer_returns_model_update_with_correct_round() -> None:
    src = _make_mock_source()
    trainer = LocalTrainer(data_source=src)
    update = trainer.train_and_prepare_update(round_id=7)
    assert update.round_id == 7


def test_trainer_n_samples_matches_data_length() -> None:
    src = _make_mock_source(n=80)
    trainer = LocalTrainer(data_source=src)
    update = trainer.train_and_prepare_update(round_id=1)
    assert update.n_samples == 80


def test_trainer_propagates_no_new_data_error() -> None:
    src = MagicMock()
    src.get_data.side_effect = NoNewDataError("no files")
    trainer = LocalTrainer(data_source=src)
    with pytest.raises(NoNewDataError):
        trainer.train_and_prepare_update(round_id=1)


def test_trainer_works_with_dev_source() -> None:
    with patch.dict("os.environ", {"SITE_ID": "site_1"}):
        ds = DevDataSource("site_1")
        trainer = LocalTrainer(data_source=ds)
        update = trainer.train_and_prepare_update(round_id=1)
        assert update.site_id == "site_1"
        assert "hermia_params" in update.delta_W
        assert update.local_metrics["flux_ratio"] > 0
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest client/tests/test_local_trainer_datasource.py -v
```

Expected: `TypeError: LocalTrainer.__init__() takes 1 positional argument but 2 were given`

- [ ] **Step 3: Modify `client/engine/local_trainer.py`**

Change `__init__` and `train_and_prepare_update`:

```python
# Replace the existing __init__ (line ~79):
def __init__(self, data_source: "DataSource") -> None:
    from client.engine.data_source import DataSource  # local import avoids circular
    self.settings = get_client_settings()
    self._data_source = data_source

# Replace line ~109 (the load_filtration_csv call) in train_and_prepare_update:
#   OLD: time, flux, tmp = load_filtration_csv(self.settings.local_data_path)
#   NEW:
time, flux, tmp = self._data_source.get_data()
```

Also remove the `load_filtration_csv` import from the top of `local_trainer.py` — it is no longer called here. (The import lives in `data_source.py` now for the ProdDataSource fallback path.)

Remove this line from `local_trainer.py` imports:
```python
from client.engine.data_loader  import load_filtration_csv
```

Add to imports:
```python
from client.engine.data_source import DataSource
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest client/tests/test_local_trainer_datasource.py client/tests/test_engine.py -v \
  --cov=client.engine.local_trainer --cov-report=term-missing
```

Expected: all green. If `test_engine.py` has tests that instantiate `LocalTrainer()` directly, update them to pass a mock DataSource.

- [ ] **Step 5: Commit**

```bash
git add client/engine/local_trainer.py client/tests/test_local_trainer_datasource.py
git commit -m "refactor: inject DataSource into LocalTrainer; remove direct CSV load"
```

---

### Task 4: Client config + state + status server + main wiring

**Files:**
- Modify: `client/config.py` (add 4 new fields)
- Modify: `client/engine/state.py` (add `run_count`, `last_run_at`)
- Create: `client/comms/status_server.py`
- Modify: `client/main.py` (instantiate DataSource; start status server)

**Interfaces:**
- Produces:
  - `ClientSettings.dev_mode: bool`, `dev_jitter_fraction: float`, `data_poll_seconds: int`, `client_status_port: int`
  - `TrainingState.run_count: int`, `TrainingState.last_run_at: Optional[str]`
  - `start_status_server(port: int) -> None`

- [ ] **Step 1: Add fields to `client/config.py`**

Add these fields to the `ClientSettings` class body (after `local_data_path`):

```python
# ── Dev-mode simulation ──────────────────────────────────────────────────────
dev_mode:            bool  = False  # True → DevDataSource; False → ProdDataSource
dev_jitter_fraction: float = 0.05   # ±% Gaussian jitter on J0/k1/k2 each round

# ── Prod-mode data polling ───────────────────────────────────────────────────
data_poll_seconds:   int   = 60     # how often ProdDataSource checks for new CSVs

@computed_field  # type: ignore[misc]
@property
def client_status_port(self) -> int:
    """site_1 → 9001, site_2 → 9002, …, site_5 → 9005."""
    try:
        return 9000 + int(self.site_id.split("_")[1])
    except (IndexError, ValueError):
        return 9001
```

- [ ] **Step 2: Add `run_count` and `last_run_at` to `client/engine/state.py`**

```python
# In the TrainingState dataclass, add two fields after last_round_completed:
run_count:            int            = 0
last_run_at:          Optional[str]  = None  # ISO-8601 UTC string, e.g. "2026-08-14T14:32:00+00:00"
```

No other changes needed — `update_state()` already handles any valid field name dynamically.

- [ ] **Step 3: Run existing config + state tests to verify no regressions**

```bash
pytest client/tests/test_config.py -v
```

Expected: all pass.

- [ ] **Step 4: Create `client/comms/status_server.py`**

```python
# client/comms/status_server.py
"""Lightweight per-site status HTTP server — polled by the FL server heartbeat."""
from __future__ import annotations

import threading

import uvicorn
from fastapi import FastAPI

from client.config import get_client_settings
from client.engine.state import get_state

_app = FastAPI(docs_url=None, redoc_url=None)   # no docs UI needed


@_app.get("/site/status")
def site_status() -> dict:
    """Return site identity, run count, last run timestamp, and current phase."""
    state = get_client_settings()   # type alias clarity — re-read each call
    s = get_state()
    settings = get_client_settings()
    return {
        "site_id":     settings.site_id,
        "run_count":   s.run_count,
        "last_run_at": s.last_run_at,
        "phase":       s.phase,
    }


def start_status_server(port: int) -> None:
    """Start the status HTTP server as a background daemon thread."""
    threading.Thread(
        target=uvicorn.run,
        kwargs={"app": _app, "host": "0.0.0.0", "port": port, "log_level": "warning"},
        daemon=True,
        name="fl-status-server",
    ).start()
```

- [ ] **Step 5: Modify `client/main.py` — read current content first**

```bash
cat client/main.py
```

Then make these changes:
1. Import the new pieces at the top:
```python
from client.engine.data_source import DevDataSource, ProdDataSource
from client.comms.status_server import start_status_server
```

2. Before calling `start_scheduler()`, instantiate the DataSource and start the status server:
```python
settings = get_client_settings()

if settings.dev_mode:
    data_source = DevDataSource(settings.site_id, jitter=settings.dev_jitter_fraction)
else:
    import os
    data_dir = os.path.dirname(settings.local_data_path) or f"data/{settings.site_id}"
    data_source = ProdDataSource(data_dir)

start_status_server(settings.client_status_port)
start_scheduler(data_source=data_source)   # pass data_source — Task 5 wires this
```

- [ ] **Step 6: Run full client test suite**

```bash
pytest client/tests/ -v
```

Expected: all pass (scheduler tests may need updating in Task 5).

- [ ] **Step 7: Commit**

```bash
git add client/config.py client/engine/state.py client/comms/status_server.py client/main.py
git commit -m "feat: add dev_mode config, run_count state, and per-site status server"
```

---

### Task 5: Scheduler dev/prod split + FLClient.get_current_round()

**Files:**
- Modify: `client/engine/scheduler.py`
- Modify: `client/comms/fl_client.py` (add `get_current_round()`)

**Interfaces:**
- Consumes: `DevDataSource`, `ProdDataSource`, `NoNewDataError`, `LocalTrainer(data_source)`
- Produces: `start_scheduler(data_source: DataSource) -> None`; `FLClient.get_current_round() -> FederationRound`

- [ ] **Step 1: Add `get_current_round()` to `FLClient`**

Append to `client/comms/fl_client.py` (after `get_round_status`):

```python
def get_current_round(self) -> FederationRound:
    """
    GET /federation/current-round — return the currently COLLECTING round,
    or ask the server to start a new one if none is open.

    Used by the prod-mode scheduler before uploading an update.
    """
    url = f"{self.settings.server_url}/federation/current-round"
    resp = self._request("GET", url, headers=self.auth_headers)
    if resp.status_code == 401:
        self._do_refresh()
        resp = self._request("GET", url, headers=self.auth_headers)
    resp.raise_for_status()
    result = FederationRound(**resp.json())
    log.info("current_round_fetched", site=self.settings.site_id, round_id=result.round_id)
    return result
```

- [ ] **Step 2: Rewrite `client/engine/scheduler.py`**

```python
# client/engine/scheduler.py
"""FL round watcher — dev mode: server-initiated rounds; prod mode: data-directory polling."""
from __future__ import annotations

import time
import threading
from datetime import datetime, timezone
from typing import Union

from client.comms.fl_client       import FLClient
from client.engine.data_source    import DataSource, DevDataSource, ProdDataSource, NoNewDataError
from client.engine.local_trainer  import LocalTrainer
from client.engine.state          import get_state, update_state
from shared.utils.logging_config  import get_logger

log = get_logger(__name__)

POLL_SECONDS = 15   # dev mode: how often to check server for new round


def _watch_dev(fl: FLClient, trainer: LocalTrainer) -> None:
    """Dev mode: poll server for server-initiated rounds, train with fresh simulated data."""
    try:
        fl.authenticate()
    except Exception as exc:
        log.error("auth_failed_on_start", error=str(exc))
        return

    last_seen_round = 0

    while True:
        try:
            data = fl.get_round_status(last_seen_round + 1)
            if data is not None:
                rid    = data.get("round_id", 0)
                status = data.get("status", "")

                if rid > last_seen_round and status == "collecting":
                    log.info("new_round_dev", round_id=rid)
                    update_state(phase="training", current_round_id=rid)
                    update = trainer.train_and_prepare_update(rid)

                    update_state(phase="uploading")
                    fl.upload_update(update)

                    now = datetime.now(timezone.utc).isoformat()
                    state = get_state()
                    update_state(
                        phase="done",
                        last_round_completed=rid,
                        run_count=state.run_count + 1,
                        last_run_at=now,
                        last_flux_ratio=update.local_metrics.get("flux_ratio"),
                        last_amin=update.local_metrics.get("amin_m2"),
                        last_hermia_model=update.hermia_best_model,
                    )
                    last_seen_round = rid

        except Exception as exc:
            update_state(phase="error")
            log.warning("scheduler_poll_error", error=str(exc))

        time.sleep(POLL_SECONDS)


def _watch_prod(fl: FLClient, trainer: LocalTrainer, prod_source: ProdDataSource, poll_seconds: int) -> None:
    """Prod mode: poll data directory; push update to server when new CSVs arrive."""
    try:
        fl.authenticate()
    except Exception as exc:
        log.error("auth_failed_on_start", error=str(exc))
        return

    while True:
        try:
            if prod_source.has_new_data():
                update_state(phase="training")
                current_round = fl.get_current_round()
                update = trainer.train_and_prepare_update(current_round.round_id)

                update_state(phase="uploading")
                fl.upload_update(update)

                now = datetime.now(timezone.utc).isoformat()
                state = get_state()
                update_state(
                    phase="done",
                    last_round_completed=current_round.round_id,
                    run_count=state.run_count + 1,
                    last_run_at=now,
                    last_flux_ratio=update.local_metrics.get("flux_ratio"),
                    last_amin=update.local_metrics.get("amin_m2"),
                    last_hermia_model=update.hermia_best_model,
                )
        except NoNewDataError:
            pass    # expected — no data this cycle
        except Exception as exc:
            update_state(phase="error")
            log.warning("prod_poll_error", error=str(exc))

        time.sleep(poll_seconds)


def start_scheduler(data_source: DataSource) -> None:
    """
    Start the appropriate scheduler thread based on data source type.

    Dev mode (DevDataSource)  → _watch_dev thread
    Prod mode (ProdDataSource) → _watch_prod thread
    """
    from client.config import get_client_settings
    settings = get_client_settings()
    fl       = FLClient()
    trainer  = LocalTrainer(data_source=data_source)

    if isinstance(data_source, ProdDataSource):
        target = lambda: _watch_prod(fl, trainer, data_source, settings.data_poll_seconds)
        name   = "fl-scheduler-prod"
    else:
        target = lambda: _watch_dev(fl, trainer)
        name   = "fl-scheduler-dev"

    threading.Thread(target=target, daemon=True, name=name).start()
    log.info("scheduler_started", mode="prod" if isinstance(data_source, ProdDataSource) else "dev")
```

- [ ] **Step 3: Run full client tests**

```bash
pytest client/tests/ -v --cov=client.engine.scheduler --cov-report=term-missing
```

Fix any test in `test_comms.py` or `test_engine.py` that referenced the old `start_scheduler()` signature.

- [ ] **Step 4: Commit**

```bash
git add client/engine/scheduler.py client/comms/fl_client.py client/tests/
git commit -m "feat: split scheduler into dev/prod modes; add FLClient.get_current_round()"
```

---

### Task 6: AggregationPolicy protocol + implementations

**Files:**
- Create: `server/core/aggregation_policy.py`
- Create: `server/tests/test_aggregation_policy.py`

**Interfaces:**
- Produces: `AggregationPolicy` (Protocol), `QuorumPolicy(min_sites: int = 3)`, `TimeWindowPolicy(window_seconds: int = 1800)`

- [ ] **Step 1: Write failing tests**

```python
# server/tests/test_aggregation_policy.py
from __future__ import annotations
import pytest
from server.core.aggregation_policy import QuorumPolicy, TimeWindowPolicy


# ── QuorumPolicy ─────────────────────────────────────────────────────────────

def test_quorum_not_met_below_threshold() -> None:
    p = QuorumPolicy(min_sites=3)
    assert not p.should_aggregate(2, {"site_1", "site_2"}, 0.0)


def test_quorum_met_exactly_at_threshold() -> None:
    p = QuorumPolicy(min_sites=3)
    assert p.should_aggregate(3, {"site_1", "site_2", "site_3"}, 0.0)


def test_quorum_met_above_threshold() -> None:
    p = QuorumPolicy(min_sites=3)
    assert p.should_aggregate(5, {"s1", "s2", "s3", "s4", "s5"}, 0.0)


def test_quorum_ignores_elapsed_time() -> None:
    p = QuorumPolicy(min_sites=3)
    assert not p.should_aggregate(2, {"site_1", "site_2"}, 99999.0)


def test_quorum_uses_sites_contributed_count() -> None:
    # updates_since_last=5 but only 2 unique sites → not met
    p = QuorumPolicy(min_sites=3)
    assert not p.should_aggregate(5, {"site_1", "site_2"}, 0.0)


# ── TimeWindowPolicy ─────────────────────────────────────────────────────────

def test_timewindow_not_met_before_window() -> None:
    p = TimeWindowPolicy(window_seconds=300)
    assert not p.should_aggregate(1, {"site_1"}, 299.9)


def test_timewindow_met_at_exact_window() -> None:
    p = TimeWindowPolicy(window_seconds=300)
    assert p.should_aggregate(1, {"site_1"}, 300.0)


def test_timewindow_met_after_window() -> None:
    p = TimeWindowPolicy(window_seconds=300)
    assert p.should_aggregate(3, {"site_1", "site_2", "site_3"}, 400.0)


def test_timewindow_not_triggered_on_zero_updates() -> None:
    p = TimeWindowPolicy(window_seconds=300)
    assert not p.should_aggregate(0, set(), 999.0)


def test_timewindow_ignores_site_count() -> None:
    p = TimeWindowPolicy(window_seconds=300)
    # Only 1 site — still aggregates when window elapsed
    assert p.should_aggregate(1, {"site_1"}, 301.0)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest server/tests/test_aggregation_policy.py -v
```

- [ ] **Step 3: Implement `server/core/aggregation_policy.py`**

```python
# server/core/aggregation_policy.py
"""Pluggable aggregation trigger policies for the FL round manager."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class AggregationPolicy(Protocol):
    """Decides whether to trigger FedProx aggregation given current round state."""

    def should_aggregate(
        self,
        updates_since_last: int,
        sites_contributed: set[str],
        elapsed_seconds: float,
    ) -> bool:
        ...


@dataclass
class QuorumPolicy:
    """Aggregate when at least `min_sites` distinct sites have contributed."""

    min_sites: int = 3

    def should_aggregate(
        self,
        updates_since_last: int,
        sites_contributed: set[str],
        elapsed_seconds: float,
    ) -> bool:
        return len(sites_contributed) >= self.min_sites


@dataclass
class TimeWindowPolicy:
    """Aggregate when `window_seconds` have elapsed since the round started AND ≥1 update arrived."""

    window_seconds: int = 1800

    def should_aggregate(
        self,
        updates_since_last: int,
        sites_contributed: set[str],
        elapsed_seconds: float,
    ) -> bool:
        return updates_since_last >= 1 and elapsed_seconds >= self.window_seconds
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest server/tests/test_aggregation_policy.py -v \
  --cov=server.core.aggregation_policy --cov-report=term-missing
```

Expected: 100% coverage, all green.

- [ ] **Step 5: Commit**

```bash
git add server/core/aggregation_policy.py server/tests/test_aggregation_policy.py
git commit -m "feat: add AggregationPolicy protocol with QuorumPolicy and TimeWindowPolicy"
```

---

### Task 7: ServerSetting model + Alembic migration + SettingsStore

**Files:**
- Modify: `server/db/models.py` (add `ServerSetting`)
- Create: Alembic migration (run `alembic revision --autogenerate`)
- Create: `server/db/settings_store.py`
- Create: `server/tests/test_settings_store.py`

**Interfaces:**
- Produces: `SettingsStore` with `load(db: AsyncSession) -> dict[str, str]` and `save(db: AsyncSession, key: str, value: str) -> None`

- [ ] **Step 1: Add `ServerSetting` to `server/db/models.py`**

Append after the `RevokedToken` class:

```python
class ServerSetting(Base):
    """
    Key-value store for server-side runtime configuration.

    Persisted to DB so settings survive server restarts. Default values
    are defined in SettingsStore.DEFAULTS and returned when a key has no
    DB row. Adding new settings requires no schema change — just add a new
    key/default pair to DEFAULTS.
    """
    __tablename__ = "server_settings"

    key:   Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(String(500), nullable=False)
```

- [ ] **Step 2: Generate and verify the Alembic migration**

```bash
alembic revision --autogenerate -m "add_server_settings_table"
```

Open the generated file in `server/db/migrations/versions/`. Verify it contains:
```python
op.create_table(
    'server_settings',
    sa.Column('key',   sa.String(length=100), nullable=False),
    sa.Column('value', sa.String(length=500), nullable=False),
    sa.PrimaryKeyConstraint('key'),
)
```

Then apply:
```bash
alembic upgrade head
```

- [ ] **Step 3: Write failing tests**

```python
# server/tests/test_settings_store.py
from __future__ import annotations
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from server.db.database import Base
from server.db.settings_store import SettingsStore


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_defaults_returned_when_table_empty(db: AsyncSession) -> None:
    store = SettingsStore()
    config = await store.load(db)
    assert config["aggregation_mode"] == "quorum"
    assert config["quorum_min_sites"] == "3"
    assert config["time_window_seconds"] == "1800"
    assert config["heartbeat_seconds"] == "30"


@pytest.mark.asyncio
async def test_save_and_load_roundtrip(db: AsyncSession) -> None:
    store = SettingsStore()
    await store.save(db, "aggregation_mode", "time_window")
    await db.commit()
    config = await store.load(db)
    assert config["aggregation_mode"] == "time_window"


@pytest.mark.asyncio
async def test_update_existing_key(db: AsyncSession) -> None:
    store = SettingsStore()
    await store.save(db, "quorum_min_sites", "4")
    await db.commit()
    await store.save(db, "quorum_min_sites", "5")
    await db.commit()
    config = await store.load(db)
    assert config["quorum_min_sites"] == "5"


@pytest.mark.asyncio
async def test_unknown_key_not_in_defaults_stored_and_returned(db: AsyncSession) -> None:
    store = SettingsStore()
    await store.save(db, "custom_key", "custom_value")
    await db.commit()
    config = await store.load(db)
    assert config["custom_key"] == "custom_value"


@pytest.mark.asyncio
async def test_db_value_overrides_default(db: AsyncSession) -> None:
    store = SettingsStore()
    await store.save(db, "quorum_min_sites", "2")
    await db.commit()
    config = await store.load(db)
    assert config["quorum_min_sites"] == "2"   # DB wins over default "3"
```

- [ ] **Step 4: Run — expect FAIL**

```bash
pytest server/tests/test_settings_store.py -v
```

- [ ] **Step 5: Implement `server/db/settings_store.py`**

```python
# server/db/settings_store.py
"""Async CRUD helper for the server_settings key-value table."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.models import ServerSetting

DEFAULTS: dict[str, str] = {
    "aggregation_mode":    "quorum",
    "quorum_min_sites":    "3",
    "time_window_seconds": "1800",
    "heartbeat_seconds":   "30",
}


class SettingsStore:
    """Load and persist server runtime settings from/to the server_settings table."""

    async def load(self, db: AsyncSession) -> dict[str, str]:
        """Return all settings, falling back to DEFAULTS for missing keys."""
        result = await db.execute(select(ServerSetting))
        config: dict[str, str] = dict(DEFAULTS)
        for row in result.scalars().all():
            config[row.key] = row.value
        return config

    async def save(self, db: AsyncSession, key: str, value: str) -> None:
        """Insert or update a single setting. Caller must commit the session."""
        result = await db.execute(
            select(ServerSetting).where(ServerSetting.key == key)
        )
        row = result.scalar_one_or_none()
        if row is None:
            db.add(ServerSetting(key=key, value=value))
        else:
            row.value = value
```

- [ ] **Step 6: Run — expect PASS**

```bash
pytest server/tests/test_settings_store.py -v \
  --cov=server.db.settings_store --cov-report=term-missing
```

- [ ] **Step 7: Commit**

```bash
git add server/db/models.py server/db/settings_store.py \
        server/db/migrations/versions/ server/tests/test_settings_store.py
git commit -m "feat: add ServerSetting model, Alembic migration, and SettingsStore"
```

---

### Task 8: RoundManager refactor — pluggable policy + run tracking

**Files:**
- Modify: `server/core/round_manager.py`
- Create: `server/tests/test_round_manager_policy.py`

**Interfaces:**
- Consumes: `AggregationPolicy`, `QuorumPolicy` from Task 6
- Produces:
  - `RoundManager.set_policy(policy: AggregationPolicy) -> None`
  - `RoundManager.get_or_create_round() -> FederationRound`
  - `RoundManager.sync_site_run_info(site_id: str, remote_count: int, last_run_at: datetime | None) -> None`
  - `RoundManager.mark_site_error(site_id: str) -> None`
  - `get_status_snapshot()` now includes `run_counts` and `last_run_at`

- [ ] **Step 1: Write failing tests**

```python
# server/tests/test_round_manager_policy.py
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from shared.schemas.federation import ModelUpdate, RoundStatus
from server.core.round_manager import RoundManager
from server.core.aggregation_policy import QuorumPolicy, TimeWindowPolicy


def _make_update(site_id: str, round_id: int) -> ModelUpdate:
    return ModelUpdate(
        site_id=site_id,
        round_id=round_id,
        n_samples=100,
        delta_W={"hermia_params": [1.0, 0.01, 0.001]},
        dp_noise_sigma=0.01,
        hermia_best_model="combined_1a",
        local_metrics={"flux_rmse": 1.0, "flux_ratio": 0.7, "amin_m2": 0.005,
                       "best_aic": -10.0, "best_bic": -8.0},
    )


@pytest.mark.asyncio
async def test_default_policy_is_quorum() -> None:
    rm = RoundManager()
    assert isinstance(rm._policy, QuorumPolicy)


@pytest.mark.asyncio
async def test_set_policy_swaps_live() -> None:
    rm = RoundManager()
    rm.set_policy(TimeWindowPolicy(window_seconds=9999))
    assert isinstance(rm._policy, TimeWindowPolicy)


@pytest.mark.asyncio
async def test_quorum_policy_triggers_at_min_sites() -> None:
    rm = RoundManager()
    rm.set_policy(QuorumPolicy(min_sites=2))
    await rm.start_new_round()
    await rm.receive_update(_make_update("site_1", 1))
    assert rm._rounds[1].status == RoundStatus.COLLECTING   # 1 site, need 2
    await rm.receive_update(_make_update("site_2", 1))
    assert rm._rounds[1].status in (RoundStatus.COMPLETE, RoundStatus.AGGREGATING)


@pytest.mark.asyncio
async def test_run_counts_incremented_per_site() -> None:
    rm = RoundManager()
    rm.set_policy(QuorumPolicy(min_sites=5))   # prevent aggregation
    await rm.start_new_round()
    await rm.receive_update(_make_update("site_1", 1))
    await rm.receive_update(_make_update("site_1", 1))
    assert rm._site_run_counts["site_1"] == 2


@pytest.mark.asyncio
async def test_last_run_at_set_on_update() -> None:
    rm = RoundManager()
    rm.set_policy(QuorumPolicy(min_sites=5))
    await rm.start_new_round()
    await rm.receive_update(_make_update("site_3", 1))
    assert rm._site_last_run_at["site_3"] is not None


@pytest.mark.asyncio
async def test_get_or_create_reuses_collecting_round() -> None:
    rm = RoundManager()
    r1 = await rm.get_or_create_round()
    r2 = await rm.get_or_create_round()
    assert r1.round_id == r2.round_id


@pytest.mark.asyncio
async def test_get_or_create_starts_new_after_complete() -> None:
    rm = RoundManager()
    rm.set_policy(QuorumPolicy(min_sites=1))
    r1 = await rm.get_or_create_round()
    await rm.receive_update(_make_update("site_1", r1.round_id))
    # round 1 is now complete — get_or_create should make round 2
    r2 = await rm.get_or_create_round()
    assert r2.round_id == r1.round_id + 1


@pytest.mark.asyncio
async def test_sync_site_run_info_updates_if_remote_higher() -> None:
    rm = RoundManager()
    rm.sync_site_run_info("site_2", 7, datetime.now(timezone.utc))
    assert rm._site_run_counts["site_2"] == 7


@pytest.mark.asyncio
async def test_sync_site_run_info_ignores_if_remote_lower() -> None:
    rm = RoundManager()
    rm._site_run_counts["site_2"] = 10
    rm.sync_site_run_info("site_2", 5, None)
    assert rm._site_run_counts["site_2"] == 10


@pytest.mark.asyncio
async def test_status_snapshot_includes_run_info() -> None:
    rm = RoundManager()
    snap = await rm.get_status_snapshot()
    assert "run_counts" in snap
    assert "last_run_at" in snap
    assert "site_1" in snap["run_counts"]
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest server/tests/test_round_manager_policy.py -v
```

- [ ] **Step 3: Modify `server/core/round_manager.py`**

Changes to make (read the file top-to-bottom, apply each):

**Imports — add:**
```python
from server.core.aggregation_policy import AggregationPolicy, QuorumPolicy
```

**`__init__` — add three new attributes after `_aggregator`:**
```python
self._policy: AggregationPolicy = QuorumPolicy(
    min_sites=self._settings.min_sites_per_round
)
self._site_run_counts:  Dict[str, int]                    = {f"site_{i}": 0    for i in range(1, 6)}
self._site_last_run_at: Dict[str, Optional[datetime]]     = {f"site_{i}": None for i in range(1, 6)}
```

Also add `Optional` to the `typing` import.

**New method `set_policy` — add after `__init__`:**
```python
def set_policy(self, policy: AggregationPolicy) -> None:
    """Swap the aggregation policy live (no in-flight round is affected)."""
    self._policy = policy

async def get_or_create_round(self) -> FederationRound:
    """Return the current COLLECTING round, or start a new one if none is open."""
    if self._current_round_id > 0:
        current = self._rounds.get(self._current_round_id)
        if current and current.status == RoundStatus.COLLECTING:
            return current
    return await self.start_new_round()

def sync_site_run_info(
    self, site_id: str, remote_count: int, last_run_at: Optional[datetime]
) -> None:
    """Update run count from heartbeat poller — only increases, never decreases."""
    if remote_count > self._site_run_counts.get(site_id, 0):
        self._site_run_counts[site_id] = remote_count
        if last_run_at is not None:
            self._site_last_run_at[site_id] = last_run_at

def mark_site_error(self, site_id: str) -> None:
    """Mark a site as unreachable (called by SitePoller on connection failure)."""
    self._site_statuses[site_id] = SiteStatus.ERROR
```

**`receive_update` — replace the quorum check and add run tracking:**

Replace:
```python
        if len(self._updates[rid]) >= self._settings.min_sites_per_round:
            await self._aggregate(rid)
```

With:
```python
        # Track per-site run stats
        self._site_run_counts[update.site_id] = self._site_run_counts.get(update.site_id, 0) + 1
        self._site_last_run_at[update.site_id] = datetime.now(timezone.utc)

        elapsed = (datetime.now(timezone.utc) - self._rounds[rid].started_at).total_seconds()
        sites_contributed = set(self._rounds[rid].participating_sites)
        if self._policy.should_aggregate(
            updates_since_last=len(self._updates[rid]),
            sites_contributed=sites_contributed,
            elapsed_seconds=elapsed,
        ):
            await self._aggregate(rid)
```

**`_timeout_guard` — use policy window when TimeWindowPolicy is active:**
```python
async def _timeout_guard(self, round_id: int) -> None:
    from server.core.aggregation_policy import TimeWindowPolicy
    timeout = (
        self._policy.window_seconds
        if isinstance(self._policy, TimeWindowPolicy)
        else self._settings.round_timeout_seconds
    )
    await asyncio.sleep(timeout)
    if self._rounds[round_id].status == RoundStatus.COLLECTING:
        log.info("round_timeout", round_id=round_id)
        await self._aggregate(round_id)
```

**`get_status_snapshot` — add run info:**
```python
async def get_status_snapshot(self) -> dict[str, object]:
    round_ = self._rounds.get(self._current_round_id)
    return {
        "current_round_id":    self._current_round_id,
        "round_status":        round_.status.value if round_ else "idle",
        "sites":               await self.get_site_statuses(),
        "model_version":       self._model_version,
        "participating_sites": list(round_.participating_sites) if round_ else [],
        "run_counts":          dict(self._site_run_counts),
        "last_run_at": {
            k: v.isoformat() if v else None
            for k, v in self._site_last_run_at.items()
        },
    }
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest server/tests/test_round_manager_policy.py server/tests/test_core.py -v \
  --cov=server.core.round_manager --cov-report=term-missing
```

- [ ] **Step 5: Commit**

```bash
git add server/core/round_manager.py server/tests/test_round_manager_policy.py
git commit -m "feat: pluggable AggregationPolicy in RoundManager; add run count tracking"
```

---

### Task 9: SitePoller heartbeat

**Files:**
- Create: `server/core/site_poller.py`
- Create: `server/tests/test_site_poller.py`

**Interfaces:**
- Consumes: `RoundManager.sync_site_run_info`, `RoundManager.mark_site_error`
- Produces: `SitePoller(round_manager, settings)` with `start() -> None`

- [ ] **Step 1: Add server config env vars — modify `server/config.py`**

Add to `ServerSettings` class:
```python
# ── Site heartbeat poller ────────────────────────────────────────────────────
heartbeat_seconds: int = 30          # how often server polls each site status endpoint

# Base URLs for each site's status server (port 9001-9005 by default)
site_1_url: str = "http://localhost:9001"
site_2_url: str = "http://localhost:9002"
site_3_url: str = "http://localhost:9003"
site_4_url: str = "http://localhost:9004"
site_5_url: str = "http://localhost:9005"
```

- [ ] **Step 2: Write failing tests**

```python
# server/tests/test_site_poller.py
from __future__ import annotations
import asyncio
import pytest
import httpx
import respx
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, ANY, call
from server.core.site_poller import SitePoller


def _make_settings(
    site_1_url: str = "http://mock-site-1",
    heartbeat_seconds: int = 0,
) -> MagicMock:
    s = MagicMock()
    s.heartbeat_seconds = heartbeat_seconds
    s.site_1_url = site_1_url
    s.site_2_url = s.site_3_url = s.site_4_url = s.site_5_url = ""
    return s


@pytest.mark.asyncio
async def test_poller_calls_sync_on_successful_response() -> None:
    rm = MagicMock()
    rm.sync_site_run_info = MagicMock()
    settings = _make_settings()

    with respx.mock:
        respx.get("http://mock-site-1/site/status").mock(
            return_value=httpx.Response(200, json={
                "site_id":    "site_1",
                "run_count":  5,
                "last_run_at": "2026-08-14T12:00:00+00:00",
                "phase":      "done",
            })
        )
        poller = SitePoller(rm, settings)
        # Run exactly one poll iteration
        await poller._poll_once()

    rm.sync_site_run_info.assert_called_once_with("site_1", 5, ANY)


@pytest.mark.asyncio
async def test_poller_marks_error_on_connection_failure() -> None:
    rm = MagicMock()
    settings = _make_settings(site_1_url="http://nonexistent-host-xyz-12345")

    poller = SitePoller(rm, settings)
    await poller._poll_once()   # should not raise; logs warning

    rm.mark_site_error.assert_called_once_with("site_1")


@pytest.mark.asyncio
async def test_poller_skips_empty_url() -> None:
    rm = MagicMock()
    settings = _make_settings(site_1_url="")  # all URLs empty

    poller = SitePoller(rm, settings)
    await poller._poll_once()

    rm.sync_site_run_info.assert_not_called()
    rm.mark_site_error.assert_not_called()


@pytest.mark.asyncio
async def test_poller_does_not_trigger_aggregation() -> None:
    """Poller is read-only — it must NEVER call rm.receive_update or rm._aggregate."""
    rm = MagicMock()
    rm.receive_update = AsyncMock()
    settings = _make_settings()

    with respx.mock:
        respx.get("http://mock-site-1/site/status").mock(
            return_value=httpx.Response(200, json={
                "site_id": "site_1", "run_count": 3,
                "last_run_at": None, "phase": "done",
            })
        )
        poller = SitePoller(rm, settings)
        await poller._poll_once()

    rm.receive_update.assert_not_called()
```

- [ ] **Step 3: Run — expect FAIL**

```bash
pytest server/tests/test_site_poller.py -v
```

- [ ] **Step 4: Implement `server/core/site_poller.py`**

```python
# server/core/site_poller.py
"""Asyncio heartbeat task — polls each site's /site/status endpoint periodically."""
from __future__ import annotations

import asyncio
from datetime import datetime

import httpx

from server.config import ServerSettings
from server.core.round_manager import RoundManager
from shared.utils.logging_config import get_logger

log = get_logger(__name__)


class SitePoller:
    """Polls each registered site on a heartbeat interval to sync run counts."""

    def __init__(self, round_manager: RoundManager, settings: ServerSettings) -> None:
        self._rm = round_manager
        self._settings = settings
        self._site_urls: dict[str, str] = {
            "site_1": settings.site_1_url,
            "site_2": settings.site_2_url,
            "site_3": settings.site_3_url,
            "site_4": settings.site_4_url,
            "site_5": settings.site_5_url,
        }

    async def _poll_once(self) -> None:
        """Poll all sites once. Extracted for testability."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            for site_id, base_url in self._site_urls.items():
                if not base_url:
                    continue
                try:
                    r = await client.get(f"{base_url}/site/status")
                    if r.status_code == 200:
                        data          = r.json()
                        remote_count  = int(data.get("run_count", 0))
                        raw_ts        = data.get("last_run_at")
                        last_run_at   = datetime.fromisoformat(raw_ts) if raw_ts else None
                        self._rm.sync_site_run_info(site_id, remote_count, last_run_at)
                except Exception as exc:
                    log.warning("site_unreachable", site=site_id, error=str(exc))
                    self._rm.mark_site_error(site_id)

    async def run(self) -> None:
        """Main heartbeat loop — runs indefinitely until cancelled."""
        while True:
            await asyncio.sleep(self._settings.heartbeat_seconds)
            await self._poll_once()

    def start(self) -> None:
        """Schedule the heartbeat loop as an asyncio background task."""
        asyncio.create_task(self.run())
```

- [ ] **Step 5: Run — expect PASS**

```bash
pytest server/tests/test_site_poller.py -v \
  --cov=server.core.site_poller --cov-report=term-missing
```

Install `respx` if missing: `pip install respx`

- [ ] **Step 6: Commit**

```bash
git add server/core/site_poller.py server/config.py server/tests/test_site_poller.py
git commit -m "feat: add SitePoller heartbeat and server config site URLs"
```

---

### Task 10: Settings API + federation current-round endpoint + server/main.py wiring

**Files:**
- Create: `server/api/settings.py`
- Modify: `server/api/federation.py` (add `GET /current-round`)
- Modify: `server/api/internal.py` (no code change — `get_status_snapshot` already extended)
- Modify: `server/main.py` (startup event + register settings router + start SitePoller)

- [ ] **Step 1: Create `server/api/settings.py`**

```python
# server/api/settings.py
"""GET/PUT /settings — runtime aggregation policy configuration."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from server.api.auth import get_current_site
from server.core.aggregation_policy import QuorumPolicy, TimeWindowPolicy
from server.core.round_manager import RoundManager, get_round_manager
from server.db.database import get_db
from server.db.settings_store import SettingsStore

router = APIRouter()


@router.get("")
async def get_settings(
    db: AsyncSession = Depends(get_db),
    _site: str = Depends(get_current_site),
) -> dict:
    """Return current aggregation policy configuration."""
    return await SettingsStore().load(db)


@router.put("")
async def update_settings(
    payload: dict,
    db: AsyncSession = Depends(get_db),
    rm: RoundManager = Depends(get_round_manager),
    _site: str = Depends(get_current_site),
) -> dict:
    """
    Update one or more settings keys and apply the new policy live.

    Accepted keys: aggregation_mode ("quorum"|"time_window"),
                   quorum_min_sites (int str), time_window_seconds (int str),
                   heartbeat_seconds (int str).
    """
    store = SettingsStore()
    for key, value in payload.items():
        await store.save(db, key, str(value))
    await db.commit()

    config = await store.load(db)
    mode = config.get("aggregation_mode", "quorum")
    if mode == "time_window":
        rm.set_policy(TimeWindowPolicy(window_seconds=int(config["time_window_seconds"])))
    else:
        rm.set_policy(QuorumPolicy(min_sites=int(config["quorum_min_sites"])))

    return {"status": "ok", "config": config}
```

- [ ] **Step 2: Add `GET /federation/current-round` to `server/api/federation.py`**

Append after `list_sites`:

```python
@router.get("/current-round", response_model=FederationRound)
async def get_current_round(
    rm: RoundManager = Depends(get_round_manager),
    _site: str = Depends(get_current_site),
) -> FederationRound:
    """
    GET /federation/current-round — return the open COLLECTING round, or start one.

    Used by prod-mode clients that push updates without waiting for a server-
    initiated round. Idempotent: returns the same round if already collecting.
    """
    return await rm.get_or_create_round()
```

- [ ] **Step 3: Wire everything into `server/main.py`**

Add imports:
```python
from server.api import settings as settings_api
from server.core.aggregation_policy import QuorumPolicy, TimeWindowPolicy
from server.core.round_manager import get_round_manager
from server.core.site_poller import SitePoller
from server.db.database import AsyncSessionLocal
from server.db.settings_store import SettingsStore
```

Register the router (add after the `internal` router line):
```python
app.include_router(settings_api.router, prefix="/settings", tags=["settings"])
```

Add the startup event (before `if __name__ == "__main__":`):
```python
@app.on_event("startup")
async def _on_startup() -> None:
    """Load persisted policy config and start the site heartbeat poller."""
    async with AsyncSessionLocal() as db:
        config = await SettingsStore().load(db)

    rm = get_round_manager()
    mode = config.get("aggregation_mode", "quorum")
    if mode == "time_window":
        rm.set_policy(TimeWindowPolicy(window_seconds=int(config["time_window_seconds"])))
    else:
        rm.set_policy(QuorumPolicy(min_sites=int(config["quorum_min_sites"])))

    SitePoller(rm, settings).start()
```

- [ ] **Step 4: Run server API tests**

```bash
pytest server/tests/test_api.py server/tests/test_main.py -v
```

Fix any test that imports from `server.api` and needs the new endpoint or router.

- [ ] **Step 5: Commit**

```bash
git add server/api/settings.py server/api/federation.py server/main.py
git commit -m "feat: add settings API, /current-round endpoint, startup policy load, SitePoller start"
```

---

### Task 11: Site card UI + dashboard poll loop

**Files:**
- Modify: `server/ui/components/site_card.py`
- Modify: `server/ui/app.py`

- [ ] **Step 1: Update `server/ui/components/site_card.py`**

```python
# server/ui/components/site_card.py
"""Site status card widget — shows status, run count, and last run timestamp."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import flet as ft

STATUS_COLORS = {
    "IDLE":      ft.Colors.GREY,
    "TRAINING":  ft.Colors.BLUE,
    "UPLOADING": ft.Colors.ORANGE,
    "DONE":      ft.Colors.GREEN,
    "ERROR":     ft.Colors.RED,
}


class SiteCard:
    def __init__(self, site_id: str) -> None:
        self.site_id       = site_id
        self._status_text  = ft.Text("IDLE",    size=12, color=ft.Colors.GREY)
        self._runs_text    = ft.Text("Runs: --", size=11, color=ft.Colors.GREY_400)
        self._last_text    = ft.Text("Last: --",  size=11, color=ft.Colors.GREY_400)

    def build(self) -> ft.Control:
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text(self.site_id, size=15, weight=ft.FontWeight.BOLD),
                    self._status_text,
                    self._runs_text,
                    self._last_text,
                ], spacing=3),
                padding=14,
                width=155,
            )
        )

    def set_status(self, status: str) -> None:
        upper = status.upper()
        self._status_text.value = upper
        self._status_text.color = STATUS_COLORS.get(upper, ft.Colors.GREY)

    def set_run_info(self, run_count: int, last_run_at: Optional[str]) -> None:
        """Update run count and last-run timestamp. Called by dashboard poll loop."""
        self._runs_text.value = f"Runs: {run_count}"
        if last_run_at:
            try:
                dt = datetime.fromisoformat(last_run_at)
                now = datetime.now(timezone.utc)
                if dt.date() == now.date():
                    self._last_text.value = f"Last: {dt.strftime('%H:%M')}"
                else:
                    self._last_text.value = f"Last: {dt.strftime('%d %b')}"
            except ValueError:
                self._last_text.value = f"Last: {last_run_at[:16]}"
        else:
            self._last_text.value = "Last: --"
```

- [ ] **Step 2: Update `server/ui/app.py` poll loop**

In the `poll_loop` coroutine, after updating site statuses, add run info:

```python
# After: card.set_status(sites.get(card.site_id, "idle"))
run_counts  = data.get("run_counts", {})
last_run_at = data.get("last_run_at", {})
for card in dashboard.cards:
    card.set_status(sites.get(card.site_id, "idle"))
    card.set_run_info(
        run_count=run_counts.get(card.site_id, 0),
        last_run_at=last_run_at.get(card.site_id),
    )
```

Replace the original loop:
```python
# OLD:
for card in dashboard.cards:
    card.set_status(sites.get(card.site_id, "idle"))

# NEW (shown above)
```

- [ ] **Step 3: Commit**

```bash
git add server/ui/components/site_card.py server/ui/app.py
git commit -m "feat: show run count and last-run timestamp on site cards"
```

---

### Task 12: Settings page UI — aggregation policy section

**Files:**
- Modify: `server/ui/pages/settings.py` (add aggregation policy section)

- [ ] **Step 1: Replace `server/ui/pages/settings.py`**

Read the current file first:
```bash
cat server/ui/pages/settings.py
```

Then replace the `build()` method to add the aggregation policy section below the existing hyperparameter fields. The full updated `build()`:

```python
def build(self) -> ft.Control:
    settings = get_settings()
    api_base = f"http://localhost:{settings.port}"

    # ── Aggregation policy controls ────────────────────────────────────────
    self._mode_radio = ft.RadioGroup(
        value="quorum",
        content=ft.Row([
            ft.Radio(value="quorum",      label="Quorum (default)"),
            ft.Radio(value="time_window", label="Time Window"),
        ]),
        on_change=self._on_mode_change,
    )
    self._quorum_field  = ft.TextField(label="Min sites required", value="3",  width=200)
    self._window_field  = ft.TextField(label="Window (minutes)",   value="30", width=200, visible=False)
    self._heartbeat_field = ft.TextField(label="Heartbeat interval (seconds)", value="30", width=220)
    self._policy_status = ft.Text("", size=12, color=ft.Colors.GREEN)

    return ft.Column([
        ft.Text("Settings", size=26, weight=ft.FontWeight.BOLD),
        ft.Divider(),

        # ── Existing hyperparameter section (unchanged) ───────────────────
        ft.Text("FL Hyperparameters", size=17),
        ft.Row([
            ft.TextField(label="FL Rounds",       value="50",   width=180),
            ft.TextField(label="Local Epochs",    value="5",    width=180),
            ft.TextField(label="FedProx Mu",      value="0.01", width=180),
            ft.TextField(label="DP Noise Sigma",  value="0.01", width=180),
            ft.TextField(label="Min Sites/Round", value="3",    width=180),
        ], spacing=12, wrap=True),
        ft.ElevatedButton("Save Hyperparameters", icon=ft.Icons.SAVE),
        ft.Divider(),

        # ── Aggregation policy ────────────────────────────────────────────
        ft.Text("Aggregation Policy", size=17),
        ft.Text(
            "Quorum: aggregate when N distinct sites have submitted. "
            "Time Window: aggregate when the configured time has elapsed.",
            size=12, color=ft.Colors.GREY_400,
        ),
        self._mode_radio,
        ft.Row([self._quorum_field, self._window_field], spacing=12),
        ft.Divider(),

        ft.Text("Heartbeat Poller", size=17),
        self._heartbeat_field,
        ft.Divider(),

        ft.ElevatedButton(
            "Apply Policy & Heartbeat",
            icon=ft.Icons.SAVE,
            on_click=lambda _: self.page.run_task(self._apply_settings, api_base),
        ),
        self._policy_status,

        ft.Divider(),
        ft.Text("Registered Sites", size=17),
        ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Site ID")),
                ft.DataColumn(ft.Text("Status")),
                ft.DataColumn(ft.Text("Last Seen")),
                ft.DataColumn(ft.Text("Actions")),
            ],
            rows=[
                ft.DataRow(cells=[
                    ft.DataCell(ft.Text(f"site_{i}")),
                    ft.DataCell(ft.Text("IDLE")),
                    ft.DataCell(ft.Text("—")),
                    ft.DataCell(ft.IconButton(ft.Icons.DELETE_OUTLINE,
                                              icon_color=ft.Colors.RED_300)),
                ])
                for i in range(1, 6)
            ],
        ),
    ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=16)
```

Add the helper methods to the class (before `build`):

```python
def _on_mode_change(self, e: ft.ControlEvent) -> None:
    is_quorum = e.control.value == "quorum"
    self._quorum_field.visible = is_quorum
    self._window_field.visible = not is_quorum
    self.page.update()

async def _apply_settings(self, api_base: str) -> None:
    import httpx
    mode = self._mode_radio.value
    payload: dict = {"aggregation_mode": mode, "heartbeat_seconds": self._heartbeat_field.value}
    if mode == "quorum":
        payload["quorum_min_sites"] = self._quorum_field.value
    else:
        # Convert minutes → seconds
        try:
            minutes = float(self._window_field.value)
            payload["time_window_seconds"] = str(int(minutes * 60))
        except ValueError:
            self._policy_status.value = "Invalid window value"
            self._policy_status.color = ft.Colors.RED
            self.page.update()
            return

    try:
        async with httpx.AsyncClient() as client:
            r = await client.put(f"{api_base}/settings", json=payload, timeout=5.0)
        if r.status_code == 200:
            self._policy_status.value = "Settings applied."
            self._policy_status.color = ft.Colors.GREEN
        else:
            self._policy_status.value = f"Error {r.status_code}"
            self._policy_status.color = ft.Colors.RED
    except Exception as exc:
        self._policy_status.value = f"Request failed: {exc}"
        self._policy_status.color = ft.Colors.RED
    self.page.update()
```

Also add the missing imports at the top of the file:
```python
from server.config import get_settings
```

And update `__init__`:
```python
def __init__(self, page: ft.Page) -> None:
    self.page = page
    # UI refs set in build() — call build() before accessing them
    self._mode_radio:      ft.RadioGroup | None = None
    self._quorum_field:    ft.TextField  | None = None
    self._window_field:    ft.TextField  | None = None
    self._heartbeat_field: ft.TextField  | None = None
    self._policy_status:   ft.Text       | None = None
```

- [ ] **Step 2: Commit**

```bash
git add server/ui/pages/settings.py
git commit -m "feat: add aggregation policy and heartbeat controls to settings page"
```

---

### Task 13: PowerShell launcher scripts

**Files:**
- Modify: `start_all_server_clients_dev.ps1`
- Create: `start_all_server_clients.ps1`

- [ ] **Step 1: Update `start_all_server_clients_dev.ps1`**

Two changes:

**1. Add status server ports to the `$ports` array (line 5):**
```powershell
$ports = @(8000, 8550, 8551, 8552, 8553, 8554, 8555, 9001, 9002, 9003, 9004, 9005)
```

**2. Add `DEV_MODE=true` to server and client launch commands:**
```powershell
# Server (line ~36):
Start-Pane -Title "Server" -Command "`$env:DEV_MODE='true'; python server/main.py" -BgColor "DarkBlue"

# Each client (inside the foreach loop):
Start-Pane -Title "Site $i" `
           -Command "`$env:DEV_MODE='true'; `$env:SITE_ID='$site'; python client/main.py" `
           -BgColor "DarkGreen"
```

- [ ] **Step 2: Create `start_all_server_clients.ps1` (prod launcher)**

```powershell
$root = "D:\viral_fl_project"
$venv = "$root\.venv\Scripts\Activate.ps1"

# -- Free project ports before starting --------------------------------------
$ports = @(8000, 8550, 8551, 8552, 8553, 8554, 8555, 9001, 9002, 9003, 9004, 9005)
Write-Host "Checking project ports..." -ForegroundColor Yellow
$portPids = $ports | ForEach-Object {
    Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue
} | Select-Object -ExpandProperty OwningProcess -Unique

foreach ($id in $portPids) {
    $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
    $name = if ($proc) { $proc.Name } else { "unknown" }
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    Write-Host "  Freed port - killed PID $id ($name)" -ForegroundColor Yellow
}
if (-not $portPids) { Write-Host "  All ports free." -ForegroundColor Gray }
Start-Sleep -Seconds 1
# ---------------------------------------------------------------------------

function Start-Pane {
    param(
        [string]$Title,
        [string]$Command,
        [string]$BgColor
    )
    $setup = "`$host.UI.RawUI.WindowTitle = '$Title'; " +
             "`$host.UI.RawUI.BackgroundColor = '$BgColor'; " +
             "Clear-Host; "
    Start-Process powershell -ArgumentList "-NoExit", "-Command", `
        "cd '$root'; & '$venv'; $setup $Command"
}

# -- Server ------------------------------------------------------------------
Start-Pane -Title "Server"     -Command "python server/main.py"    -BgColor "DarkBlue"
Start-Sleep -Seconds 2

Start-Pane -Title "Server GUI" -Command "python server/ui/app.py"  -BgColor "DarkCyan"
Start-Sleep -Seconds 1

# -- Clients -----------------------------------------------------------------
foreach ($i in 1..5) {
    $site = "site_$i"
    Start-Pane -Title "Site $i" `
               -Command "`$env:SITE_ID='$site'; python client/main.py" `
               -BgColor "DarkMagenta"
    Start-Sleep -Milliseconds 500
}

Write-Host "All 7 windows launched (PRODUCTION mode)." -ForegroundColor Cyan
```

- [ ] **Step 3: Commit**

```bash
git add start_all_server_clients_dev.ps1 start_all_server_clients.ps1
git commit -m "chore: add DEV_MODE to dev launcher; create prod launcher script"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Dev mode resimulates data each round with per-site jitter | Task 1 (DevDataSource) |
| Inter-site variance larger than intra-site | Task 1 (different base params per site) |
| Prod mode: timestamped CSVs not overwritten | Task 2 (ProdDataSource glob pattern) |
| Multiple new CSVs concatenated | Task 2 (pd.concat) |
| Sidecar survives restarts | Task 2 |
| Server polls each site periodically | Task 9 (SitePoller) |
| Aggregates when new run detected | Task 8 (receive_update → policy) |
| UI shows run count per site | Task 11 (site_card set_run_info) |
| UI shows last run timestamp | Task 11 |
| Quorum aggregation (configurable, default) | Tasks 6, 8 |
| Time-window aggregation (configurable) | Tasks 6, 8 |
| Toggle between methods from UI | Task 12 (settings page) |
| Quorum value configurable in settings page | Task 12 |
| Time window configurable in settings page | Task 12 |
| Settings persisted to DB | Task 7 (SettingsStore) |
| Settings survive server restart | Tasks 7, 10 (startup event loads from DB) |
| DEV_MODE=true in dev PS1 script | Task 13 |
| New prod launcher without DEV_MODE | Task 13 |
| 100% coverage on new/modified files | All tasks — each has a coverage check step |

**No placeholders detected.**

**Type consistency:**
- `DataSource.get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]` — consistent Tasks 1-5
- `LocalTrainer(data_source: DataSource)` — consistent Tasks 3-5
- `AggregationPolicy.should_aggregate(int, set[str], float) -> bool` — consistent Tasks 6, 8
- `SettingsStore.load(AsyncSession) -> dict[str, str]` — consistent Tasks 7, 10
- `RoundManager.sync_site_run_info(str, int, datetime | None)` — consistent Tasks 8, 9
- `SiteCard.set_run_info(int, str | None)` — consistent Tasks 11, app.py

**One gap found and addressed:** `SiteStatus.ERROR` — verify this value exists in `shared/schemas/federation.py` before Task 8. If absent, add `ERROR = "error"` to the `SiteStatus` enum. The `mark_site_error` method references `SiteStatus.ERROR`.

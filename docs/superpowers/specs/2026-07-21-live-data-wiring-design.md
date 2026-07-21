# Live Data Wiring — Design Spec
**Date:** 2026-07-21  
**Branch:** fix/flet-colors-icons-api  
**Status:** Approved

---

## Problem

Both UIs (server dashboard and client status page) build their controls once on page load and never refresh. Site cards show hardcoded `IDLE`, the round timeline shows hardcoded `Round 1 / 50`, and training metrics show `--` permanently. The backend (scheduler, RoundManager) works correctly but nothing connects it to the UI.

---

## Approach: Internal Status Endpoint + HTTP Polling

The server UI (`server/ui/app.py`) and FastAPI server (`server/main.py`) run as **separate processes**. The Flet dashboard cannot access the FastAPI `RoundManager` singleton directly — HTTP is the only correct bridge.

A new unauthenticated internal endpoint is added to FastAPI. Both UIs poll it on a 5-second async loop via Flet's `page.run_task()`. Controls are mutated in-place (refs) rather than the page being rebuilt, avoiding flicker.

---

## Section 1 — New API Endpoint

**File:** `server/api/internal.py`  
**Mount:** `GET /internal/status` — no JWT required, read-only

Response shape:
```json
{
  "current_round_id": 3,
  "round_status": "collecting",
  "sites": {"site_1": "done", "site_2": "training", "site_3": "idle", "site_4": "idle", "site_5": "idle"},
  "model_version": 2,
  "participating_sites": ["site_1", "site_2"]
}
```

- Reads directly from `get_round_manager()` singleton — same instance used by the federation endpoints
- `current_round_id == 0` and `round_status == "idle"` when no round has started yet
- Registered in `server/main.py` under the `/internal` prefix

---

## Section 2 — Server Dashboard Polling Loop

**Files modified:** `server/ui/app.py`, `server/ui/pages/dashboard.py`, `server/ui/pages/global_model.py`

### Polling coroutine

`server/ui/app.py` registers a `page.run_task(poll_loop)` call on page load. The coroutine:

```
while running:
    await asyncio.sleep(5)
    fetch GET /internal/status
    update SiteCard refs
    update RoundTimeline refs
    fetch GET /models/global-model  (only if model_version > last_seen_version)
    update GlobalModelPage version/rounds/sites tiles only (see scope note below)
    page.update()
```

A `running: bool` flag is set to `False` by `page.on_disconnect` to exit cleanly when the browser tab closes.

### Control refs

`SiteCard`, `RoundTimeline`, and `GlobalModelPage` expose `ft.Ref` handles for all mutable text/progress controls. The polling loop holds these refs and mutates `.value` / `.color` directly — no page rebuild.

### Status colour mapping (server side)

| Server value | Card colour |
|---|---|
| `idle` | Grey |
| `training` | Blue |
| `uploading` | Orange |
| `done` | Green |
| `error` | Red |

---

## Section 3 — Client Status Page Polling Loop

**Files modified:** `client/ui/app.py`, `client/ui/pages/status.py`, `client/ui/pages/local_results.py`  
**New file:** `client/engine/state.py`

### Shared state

`client/engine/state.py` defines a module-level `TrainingState` dataclass instance protected by a `threading.Lock`:

```python
@dataclass
class TrainingState:
    current_round_id: int = 0
    phase: str = "idle"          # idle | training | uploading | done | error
    last_lrv: float | None = None
    last_amin: float | None = None
    last_flux_ratio: float | None = None
    last_hermia_model: str | None = None
    last_round_completed: int = 0
```

Public API: `get_state() -> TrainingState` (returns snapshot copy under lock) and `update_state(**kwargs)` (writes under lock).

### Scheduler integration

`client/engine/scheduler.py` calls `update_state(phase="training", current_round_id=rid)` before training and `update_state(phase="done", last_lrv=..., last_amin=..., ...)` after upload. On exception it calls `update_state(phase="error")`.

### Client polling coroutine

`client/ui/app.py` registers `page.run_task(poll_loop)`. Every 5 seconds:

```
state = get_state()
update StatusPage controls from state
update LocalResultsPage metric tiles from state
page.update()
```

The `FLClient` instance created in `client/ui/app.py` is also used to call `get_round_status()` to show the server-side round status (collecting / complete / failed).

> **Scope note — GlobalModelPage parameter table:** `GlobalModel.weights` stores raw NN layer tensors (e.g. `predictor.net.0.weight`), not individual physics constants (J0, k1…). Extracting physics parameters requires a forward pass through the PINN — out of scope for this wiring task. This implementation populates only the three summary tiles (Model Version, Rounds Completed, Sites Participated). The parameter table rows remain `—` until a dedicated physics-parameter extraction feature is built.

---

## Section 4 — Error Handling

| Failure | Behaviour |
|---|---|
| `GET /internal/status` unreachable | Log warning, show "Server unreachable" chip in red, retry next tick |
| `fl.get_round_status()` raises | Catch, leave displayed values unchanged, show "Disconnected" in connection tile |
| Scheduler exception mid-training | `update_state(phase="error")` — UI shows error state |
| JWT 401 in client poll | `FLClient` auto-refreshes transparently; UI loop rarely sees this |
| Browser tab closed | `page.on_disconnect` sets `running=False`; poll loop exits |

---

## Section 5 — Testing

### `client/engine/state.py` (100% coverage required)
- Write from one thread, read from another — assert values consistent under lock
- Test all `phase` transitions: idle → training → uploading → done → idle → error

### `server/api/internal.py` (unit tested)
- `GET /internal/status` with no auth → `200` + correct JSON shape
- Zero rounds started: `current_round_id == 0`, all sites `"idle"`
- After round starts: mock `RoundManager` via FastAPI dependency override, assert fields populate

### UI polling loops
- Not unit tested (Flet requires a browser); covered by manual verification

### Coverage target
- `client/engine/state.py`: 100% line + branch
- `server/api/internal.py`: 100% line + branch
- Overall project: ≥ 80%

---

## Files Changed

| File | Change |
|---|---|
| `server/api/internal.py` | **New** — internal status endpoint |
| `server/main.py` | Register `/internal` router |
| `server/ui/app.py` | Add `page.run_task(poll_loop)`, `page.on_disconnect` |
| `server/ui/pages/dashboard.py` | Expose refs, accept live data |
| `server/ui/pages/global_model.py` | Expose refs, populate from `/models/global-model` |
| `client/engine/state.py` | **New** — thread-safe shared training state |
| `client/engine/scheduler.py` | Write to `TrainingState` at phase transitions |
| `client/ui/app.py` | Add `page.run_task(poll_loop)` |
| `client/ui/pages/status.py` | Read from `TrainingState`, show live round/phase |
| `client/ui/pages/local_results.py` | Read metrics from `TrainingState` |
| `server/tests/test_internal_api.py` | **New** — endpoint tests |
| `client/tests/test_training_state.py` | **New** — state concurrency tests |

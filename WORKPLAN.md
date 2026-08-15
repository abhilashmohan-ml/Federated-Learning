# WORKPLAN.md  —  Viral Filtration FL Build Plan

## Phase 1: Foundation  (Weeks 1-2)
  - [ ] Set up virtual environments and install deps
  - [ ] Implement shared/models/hermia.py  (6 models + AIC/BIC)
  - [ ] Implement shared/models/manabe.py  (Pc, LRV)
  - [ ] Implement shared/models/polarization.py
  - [ ] Implement shared/models/combined_1a.py
  - [ ] Unit tests for all mechanistic models  (target >80% coverage)
  - [ ] Notebook 01: Hermia model exploration with synthetic data
  - [ ] Notebook 02: Manabe LRV fitting exploration

## Phase 2: PINN Architecture  (Weeks 3-4)
  - [ ] Design input feature vector (filter + process descriptors)
  - [ ] Implement shared/models/pinn.py  (param predictor + physics solver)
  - [ ] Implement shared/crypto/noise.py  (Gaussian DP)
  - [ ] Implement all Pydantic v2 schemas in shared/schemas/
  - [ ] Notebook 03: PINN architecture validation
  - [ ] Achieve >80% test coverage on shared/

## Phase 3: Client Engine  (Week 5)
  - [ ] client/engine/data_loader.py
  - [ ] client/engine/local_trainer.py  (FedProx gradient)
  - [ ] client/engine/scheduler.py
  - [ ] client/comms/fl_client.py
  - [ ] client/comms/heartbeat.py
  - [ ] scripts/generate_synthetic_data.py  (5-site datasets)
  - [ ] Test local training loop end-to-end

## Phase 4: Server Core  (Week 6)
  - [ ] server/db/  (SQLAlchemy + Alembic migration 001)
  - [ ] server/core/aggregator.py  (FedProx)
  - [ ] server/core/round_manager.py  (state machine)
  - [ ] server/core/model_registry.py
  - [ ] server/api/  (FastAPI: auth, federation, models, health)
  - [ ] scripts/init_db.py
  - [ ] Test aggregation with 5 synthetic site updates

## Phase 5: Authentication & Security  (Week 7)
  - [ ] JWT issue / refresh / revoke  (server/api/auth.py)
  - [ ] Site certificate generation  (scripts/generate_certs.sh)
  - [ ] Differential Privacy integration in client upload
  - [ ] Secure aggregation  (shared/crypto/secure_agg.py)
  - [ ] End-to-end auth test across server + 5 clients

## Phase 6: Server Flet UI  (Week 8)
  - [ ] server/ui/app.py  (nav rail + routing)
  - [ ] pages/dashboard.py  (all sites + round progress)
  - [ ] pages/site_monitor.py  (per-site J(t), LRV, Amin charts)
  - [ ] pages/global_model.py  (params + performance)
  - [ ] pages/graphs.py  (comparative charts across all sites)
  - [ ] pages/settings.py  (site management)
  - [ ] All component widgets

## Phase 7: Client Flet UI  (Week 9)
  - [x] client/ui/app.py — constructs FLClient, calls authenticate(), passes fl_client to StatusPage
  - [x] pages/status.py — Trigger Manual Round button wired to FLClient.start_round() on daemon thread
  - [ ] pages/local_results.py

## Phase 8: Docker & Integration  (Week 10)
  - [ ] server/Dockerfile
  - [ ] client/Dockerfile
  - [ ] docker-compose.yml  (server + db + 5 clients)
  - [ ] Notebook 04: full federated round simulation
  - [ ] scripts/run_simulation.py
  - [ ] scripts/visualise_results.py

## Bug Fixes
  - [x] fix(client/config): port collision + 401 auth in dev multi-site mode —
         `flet_client_port` and `site_secret` now auto-derived as `@computed_field`
         from `SITE_ID`; only `SITE_ID` env var needed to launch any site client.
         docker-compose.yml, .env.example, and docstrings updated to match.
         14 new unit tests in client/tests/test_config.py — branch fix/flet-colors-icons-api
  - [x] fix(ui): migrate all Flet UI from deprecated `ft.colors.*`/`ft.icons.*` to
         `ft.Colors.*`/`ft.Icons.*` required by Flet 0.85.3 — affects 11 files across
         server/ui/ and client/ui/
  - [x] fix(client/config): LOCAL_DATA_PATH wrong for venv dev mode — path now
         auto-derived from SITE_ID (site_1→data/site_1/filtration.csv, etc.) via
         model_validator; LOCAL_DATA_PATH commented out of .env/.env.example;
         Docker compose retains explicit per-container override; 8 new tests —
         branch fix/flet-colors-icons-api
  - [x] fix(scheduler): 401 on GET /federation/round/N after token expiry —
         added FLClient.get_round_status(round_id) with 401→_do_refresh()→retry;
         scheduler _watch() now calls fl.get_round_status() instead of raw httpx.get();
         removed unused httpx import and get_client_settings from scheduler;
         10 new tests in TestGetRoundStatus, 7 TestWatch tests updated to mock
         fl.get_round_status; 100% branch coverage maintained — branch fix/flet-colors-icons-api
  - [x] feat(client): add FLClient.start_round() with 401-refresh-retry pattern;
         StatusPage now requires fl_client: FLClient argument; Trigger Manual Round
         button calls start_round() on a background daemon thread and updates
         _round_text / _phase_text on completion; app.py constructs FLClient,
         calls authenticate(), and passes it to StatusPage — branch fix/flet-colors-icons-api

## Test Coverage
  - [x] 100% line+branch coverage achieved for shared/ (all models, schemas, crypto, utils)
        — 10 test files, 258+ test cases; commit 1fedd07
  - [x] 100% line+branch coverage achieved for server/core/ (aggregator, round_manager,
        model_registry) — async tests use asyncio.run(), singleton cache cleared in
        try/finally; commit 1fedd07
  - [x] 100% line+branch coverage achieved for client/engine/ (data_loader, local_trainer,
        scheduler) — while-True loop broken via SystemExit on time.sleep; commit 1fedd07
  - [x] Code review completed (Important findings resolved):
        * Documented IndexError when all Hermia fitters fail (Finding 1)
        * Added value assertions to missing-layer aggregation test (Finding 2)
        * Added second-poll deduplication test for scheduler (Finding 3)
        * Added complete local_metrics key assertions (Finding 6)
  - NOTE: Production bug identified — local_trainer.train_and_prepare_update raises
          IndexError if fit_all_models returns empty dict. Guard should be added in
          Phase 9 hardening.

## Phase 9: Validation & Hardening  (Weeks 11-12)
  - [ ] Validate global model vs centralised baseline
  - [ ] Audit logging for every round
  - [ ] Load testing  (concurrent site updates)
  - [ ] CI/CD pipeline  (GitHub Actions)
  - [ ] Full documentation

## Phase 10: Data-Driven FL (feature/data-driven-fl branch — 2026-08-14/15)

All 13 tasks implemented, reviewed, and merged.

  - [x] Task 1: DataSource abstraction (`client/engine/data_source.py`)
        DevDataSource(physics_cfg, jitter) — generates Combined 1-A synthetic flux each call
        ProdDataSource(data_dir) — polls directory for new filtration_*.csv files
        PHYSICS_DEFAULTS: dict[str, float] — no site-name keys
        NoNewDataError exception for empty directory
  - [x] Task 2: ClientSettings extension (`client/config.py`)
        Added: dev_mode, dev_jitter_fraction, dev_j0/k1/k2/noise/tmp_base,
        data_poll_seconds, client_status_port, site_secret (single field)
  - [x] Task 3: TrainingState extension (`client/engine/state.py`)
        Added: run_count: int, last_run_at: Optional[str] (ISO-8601 UTC)
  - [x] Task 4: LocalTrainer refactor + Scheduler rewrite
        LocalTrainer.__init__(data_source: DataSource) — replaces CSV load
        Scheduler: _watch_dev() / _watch_prod() / start_scheduler(data_source)
        Client status server: GET /site/status (bearer auth via SITE_SECRET)
  - [x] Task 5: FLClient.get_current_round() + DevDataSource completeness tests
  - [x] Task 6: AggregationPolicy Protocol (`server/core/aggregation_policy.py`)
        AggregationPolicy(Protocol), QuorumPolicy(min_sites), TimeWindowPolicy(window_seconds)
  - [x] Task 7: SettingsStore + server_settings DB table
        server/db/settings_store.py: async load/save against server_settings table
        Alembic migration: 0969c4fdf5dc_add_server_settings_table.py
        server/db/models.py: ServerSetting ORM model
  - [x] Task 8: RoundManager extensions
        set_policy(AggregationPolicy), get_or_create_round(), sync_site_run_info(),
        mark_site_error(); _site_run_counts / _site_last_run_at start empty (not hardcoded)
  - [x] Task 9: SitePoller heartbeat (`server/core/site_poller.py`)
        parse_site_status_urls(raw: str) — parses SITE_STATUS_URLS env var
        SitePoller._poll_once(), run(), start(); SITE_POLL_SECRET header auth
        server/config.py: heartbeat_seconds, site_status_urls, site_poll_secret
  - [x] Task 10: Settings API + /current-round endpoint + server/main.py wiring
        GET/PUT /settings (admin key auth via X-Admin-Key header)
        GET /federation/current-round (idempotent — returns open collecting round)
        Startup event: load persisted policy config, start SitePoller
        server/api/auth.py: require_admin_token dependency
  - [x] Task 11: Site card UI + dashboard poll loop
        SiteCard.set_run_info(run_count, last_run_at) — smart date formatting
        Dashboard poll loop: extracts run_counts/last_run_at from snapshot
  - [x] Task 12: Settings page UI — aggregation policy section
        RadioGroup Quorum/TimeWindow, quorum_field, window_field, heartbeat_field
        Registered Sites section uses informational text (no hardcoded site rows)
  - [x] Task 13: PowerShell launchers
        start_all_server_clients_dev.ps1: DEV_MODE=true, per-site physics vars,
        status ports 9001-9005
        start_all_server_clients.ps1: prod launcher (no DEV_MODE)

## Final Review Findings (feature/data-driven-fl, 2026-08-14)
  - [x] fix(settings-api): PUT /settings had no role check — any site could change policy.
        Added require_admin_token(X-Admin-Key header) dependency to PUT handler.
  - [x] fix(settings-api): PUT /settings stored values before validating numeric keys.
        Added pre-commit _NUMERIC_KEYS validation (raises 422 on bad values);
        int() conversions in _apply_settings wrapped in try-except with QuorumPolicy fallback.
  - [x] fix(status-server): GET /site/status was unauthenticated.
        Added HTTPBearer check against SITE_SECRET when non-empty;
        SitePoller passes Authorization: Bearer header using site_poll_secret.
  - [x] fix(docker-compose): missing SITE_STATUS_URLS, CLIENT_STATUS_PORT for all clients.
        Added all required env vars to server and client services.

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
  - [ ] client/ui/app.py
  - [ ] pages/status.py
  - [ ] pages/local_results.py

## Phase 8: Docker & Integration  (Week 10)
  - [ ] server/Dockerfile
  - [ ] client/Dockerfile
  - [ ] docker-compose.yml  (server + db + 5 clients)
  - [ ] Notebook 04: full federated round simulation
  - [ ] scripts/run_simulation.py
  - [ ] scripts/visualise_results.py

## Bug Fixes
  - [x] fix(ui): migrate all Flet UI from deprecated `ft.colors.*`/`ft.icons.*` to
         `ft.Colors.*`/`ft.Icons.*` required by Flet 0.85.3 — affects 11 files across
         server/ui/ and client/ui/

## Phase 9: Validation & Hardening  (Weeks 11-12)
  - [ ] Validate global model vs centralised baseline
  - [ ] Audit logging for every round
  - [ ] Load testing  (concurrent site updates)
  - [ ] CI/CD pipeline  (GitHub Actions)
  - [ ] Full documentation

"""Physical constants and parameter bounds for viral filtration models."""

# ── Flux bounds (LMH) ──────────────────────────────────────────────────────
J_MIN: float = 0.1
J_MAX: float = 500.0

# ── Hermia blocking model parameter bounds ────────────────────────────────
KS_BOUNDS  = (0.0, 1e3)
KI_BOUNDS  = (0.0, 1e3)
KC_BOUNDS  = (0.0, 1e3)
KCF_BOUNDS = (0.0, 1e3)
K1_BOUNDS  = (0.0, 1e2)
K2_BOUNDS  = (0.0, 1e2)

# ── Manabe model bounds ───────────────────────────────────────────────────
PC_BOUNDS     = (0.0,  1.0)
JCRIT_BOUNDS  = (1.0,  500.0)
LAMBDA_BOUNDS = (0.0,  100.0)

# ── LRV thresholds (regulatory minimums) ─────────────────────────────────
LRV_MIN_PARVOVIRUS  = 4.0
LRV_MIN_RETROVIRUS  = 4.0
LRV_MIN_HERPESVIRUS = 4.0

# ── Amin / flux ratio ─────────────────────────────────────────────────────
FLUX_RATIO_MIN = 0.2   # J_final/J_initial below this => filter exhausted

# ── PINN parameter index map ──────────────────────────────────────────────
PARAM_IDX: dict[str, int] = {
    "J0": 0, "ks": 1, "ki": 2, "kc": 3, "kcf": 4,
    "k1": 5, "k2": 6, "Pc": 7, "Jcrit": 8, "Dv": 9,
}
N_PARAMS: int = len(PARAM_IDX)

HERMIA_MODELS = ["standard", "complete", "intermediate", "cake", "combined_1a"]

"""
Physical constants and parameter bounds for viral filtration models.

WHY THIS FILE EXISTS
--------------------
Every mathematical model in this project uses parameters that must stay within
physically meaningful ranges. For example, flux (J) can never be negative, and
capture probability (Pc) must be between 0 and 1. We define those limits here
in ONE place so that every model uses the same values.

If we scattered these numbers throughout the code we would risk different models
using different limits, and changing a limit would require editing many files.
Keeping them here is called the "single source of truth" principle.

PYTHON CONCEPT: Module-level constants
  Variables defined at the top level of a module (not inside a function or
  class) are available to any file that imports this module.
  Writing them in UPPER_CASE is a Python convention that tells readers
  "this is a constant — do not change it at runtime."
"""


# ── Flux bounds ────────────────────────────────────────────────────────────────
#
# Flux (J) is how fast liquid passes through the filter membrane, measured in
# LMH = Litres per square Metre per Hour.
#
#  J_MIN: The slowest physically meaningful flux we accept (0.1 LMH).
#         Below this the filter is essentially blocked.
#
#  J_MAX: The fastest we would ever expect in a viral filtration process.
#         500 LMH is already very fast for a virus-retentive membrane.
#
# These bounds prevent the curve-fitting algorithm from wandering into
# nonsensical territory (negative flux, impossibly fast flux).

J_MIN: float = 0.1    # LMH — minimum physically meaningful flux
J_MAX: float = 500.0  # LMH — maximum expected flux for virus-retentive filters


# ── Hermia blocking model parameter bounds ─────────────────────────────────────
#
# Each Hermia model has a "rate constant" that describes how quickly the filter
# fouls. Rate constants must be positive (you can't have negative fouling) and
# we set generous upper bounds (1000 or 100) so the optimiser has room to find
# the best fit without straying into meaningless values.
#
# The naming convention is:
#   ks  = rate constant for Standard blocking  (pore constriction)
#   ki  = rate constant for Intermediate blocking (partial pore sealing)
#   kc  = rate constant for Complete blocking  (pore sealing)
#   kcf = rate constant for Cake Filtration   (surface cake layer)
#   k1  = pore-constriction term in Combined 1-A model
#   k2  = cake/adsorption term in Combined 1-A model
#
# Format: (lower_bound, upper_bound) — a Python "tuple" (immutable pair).

KS_BOUNDS  = (0.0, 1e3)   # standard blocking rate constant bounds
KI_BOUNDS  = (0.0, 1e3)   # intermediate blocking rate constant bounds
KC_BOUNDS  = (0.0, 1e3)   # complete blocking rate constant bounds
KCF_BOUNDS = (0.0, 1e3)   # cake filtration rate constant bounds
K1_BOUNDS  = (0.0, 1e2)   # Combined 1-A pore constriction term bounds
K2_BOUNDS  = (0.0, 1e2)   # Combined 1-A cake deposition term bounds


# ── Manabe model bounds ─────────────────────────────────────────────────────────
#
# The Manabe model describes how well the filter captures viruses.
#
#  Pc (capture probability): a probability, so it MUST be between 0 (no capture)
#     and 1 (perfect capture). We define this as a tuple for consistency.
#
#  J_crit (critical flux): the flux value at which the filter just starts to
#     capture viruses measurably. Physical range is 1–500 LMH.
#
#  lambda (membrane affinity): how strongly the membrane attracts viruses.
#     A higher lambda means more capture at the same flux.

PC_BOUNDS     = (0.0,   1.0)    # capture probability: must be in [0, 1]
JCRIT_BOUNDS  = (1.0,   500.0)  # critical flux in LMH
LAMBDA_BOUNDS = (0.0,   100.0)  # membrane affinity parameter (dimensionless)


# ── Regulatory LRV thresholds ───────────────────────────────────────────────────
#
# LRV = Log Reduction Value. It measures how many "logs" (powers of 10) of
# virus are removed. LRV = 4 means the filter reduces virus concentration by
# 10,000-fold (10^4). This is the regulatory minimum for these virus families.
#
# Source: ICH Q5A(R2), FDA guidance on viral safety of biotechnology products.

LRV_MIN_PARVOVIRUS  = 4.0  # minimum required LRV for parvovirus
LRV_MIN_RETROVIRUS  = 4.0  # minimum required LRV for retrovirus
LRV_MIN_HERPESVIRUS = 4.0  # minimum required LRV for herpesvirus


# ── Fouling severity indicator ──────────────────────────────────────────────────
#
# Flux ratio = J_final / J_initial.
# If the flux ratio falls below 0.2 (i.e. less than 20% of the initial flux
# remains), we consider the filter "exhausted" and it should be replaced.

FLUX_RATIO_MIN = 0.2   # J_final/J_initial below this => filter is exhausted


# ── PINN parameter index map ────────────────────────────────────────────────────
#
# The Physics-Informed Neural Network (PINN) predicts 10 physical parameters
# simultaneously. It outputs a single vector of length 10, and we need to know
# which position corresponds to which parameter.
#
# PARAM_IDX maps parameter name -> its index (position) in the output vector.
# For example, PARAM_IDX["J0"] = 0 means J0 is the first number in the output.
#
# PYTHON CONCEPT: dict (dictionary)
#   A dict maps keys to values, like a lookup table.
#   e.g. PARAM_IDX["ks"] returns 1.

PARAM_IDX: dict[str, int] = {
    "J0":    0,   # initial flux (LMH) — the flux at time t=0
    "ks":    1,   # standard blocking rate constant
    "ki":    2,   # intermediate blocking rate constant
    "kc":    3,   # complete blocking rate constant
    "kcf":   4,   # cake filtration rate constant
    "k1":    5,   # Combined 1-A pore constriction term
    "k2":    6,   # Combined 1-A cake/adsorption term
    "Pc":    7,   # Manabe capture probability (0–1)
    "Jcrit": 8,   # Manabe critical flux (LMH)
    "Dv":    9,   # virus diffusion coefficient (m²/s)
}

# Total number of parameters the PINN outputs — derived automatically so it
# stays in sync with PARAM_IDX without needing a magic number.
N_PARAMS: int = len(PARAM_IDX)   # = 10

# All valid Hermia model names. Used for validation and display.
HERMIA_MODELS = ["standard", "complete", "intermediate", "cake", "combined_1a"]

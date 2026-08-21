"""
Generate synthetic viral filtration data for all 5 sites.

Each site uses a DIFFERENT underlying Hermia fouling model so that AIC-based
model selection produces variety across sites (one regime per site).

  site_1 — Standard blocking     J(t) = J0 / (1 + ks*t)^2
  site_2 — Complete blocking     J(t) = J0 * exp(-kc*t)
  site_3 — Intermediate blocking J(t) = J0 / (1 + J0*ki*t)
  site_4 — Cake filtration       J(t) = J0 / sqrt(1 + J0^2*kcf*t)
  site_5 — Combined 1-A          J(t) = J0/(1+k1*t)^2 * exp(-k2*t)

Run from project root:
    python scripts/generate_synthetic_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

SITE_CONFIGS = {
    "site_1": {
        "filter":   "Planova_20N",
        "model":    "standard",
        "J0": 150.0, "ks": 0.015, "noise": 2.0,
        "tmp_base": 1.0, "lrv_mean": 4.8, "lrv_std": 0.2,
    },
    "site_2": {
        "filter":   "ViresolveNFP",
        "model":    "complete",
        "J0": 120.0, "kc": 0.020, "noise": 3.0,
        "tmp_base": 1.2, "lrv_mean": 5.1, "lrv_std": 0.3,
    },
    "site_3": {
        "filter":   "Pegasus_SV4",
        "model":    "intermediate",
        "J0": 180.0, "ki": 3.0e-4, "noise": 1.5,
        "tmp_base": 0.8, "lrv_mean": 4.6, "lrv_std": 0.2,
    },
    "site_4": {
        "filter":   "Planova_BioEX",
        "model":    "cake",
        "J0": 100.0, "kcf": 8.0e-6, "noise": 2.5,
        "tmp_base": 1.4, "lrv_mean": 5.3, "lrv_std": 0.25,
    },
    "site_5": {
        "filter":   "ViresolveNFR",
        "model":    "combined_1a",
        "J0": 160.0, "k1": 0.012, "k2": 0.0015, "noise": 2.0,
        "tmp_base": 1.1, "lrv_mean": 4.9, "lrv_std": 0.2,
    },
}


def _flux_for_model(cfg: dict, time: np.ndarray) -> np.ndarray:
    """Return noiseless flux array using the model specified in cfg['model']."""
    J0    = cfg["J0"]
    model = cfg["model"]
    if model == "standard":
        return J0 / (1.0 + cfg["ks"] * time) ** 2
    if model == "complete":
        return J0 * np.exp(-cfg["kc"] * time)
    if model == "intermediate":
        return J0 / (1.0 + J0 * cfg["ki"] * time)
    if model == "cake":
        return J0 / np.sqrt(1.0 + J0 ** 2 * cfg["kcf"] * time)
    if model == "combined_1a":
        return (J0 / (1.0 + cfg["k1"] * time) ** 2) * np.exp(-cfg["k2"] * time)
    raise ValueError(f"Unknown model: {model!r}")


def generate(site_id: str, cfg: dict) -> None:
    out = Path(f"data/{site_id}")
    out.mkdir(parents=True, exist_ok=True)

    time = np.arange(0, 121, 1, dtype=float)   # 0 .. 120 minutes

    flux = _flux_for_model(cfg, time)
    flux += np.random.normal(0.0, cfg["noise"], len(time))
    flux  = np.clip(flux, 1.0, None)

    # TMP drifts upward slightly as membrane fouls
    tmp = cfg["tmp_base"] + 0.004 * time + np.random.normal(0.0, 0.02, len(time))

    # LRV measurements (sparse — every 15 min)
    lrv_times = np.arange(0, 121, 15, dtype=float)
    lrv_vals  = np.random.normal(cfg["lrv_mean"], cfg["lrv_std"], len(lrv_times))
    lrv_vals  = np.clip(lrv_vals, 2.0, 7.0)

    pd.DataFrame({
        "time_min":    time,
        "flux_lmh":    flux,
        "tmp_bar":     tmp,
        "filter_type": cfg["filter"],
    }).to_csv(out / "filtration.csv", index=False)

    pd.DataFrame({
        "time_min": lrv_times,
        "lrv":      lrv_vals,
        "flux_lmh": np.interp(lrv_times, time, flux),
    }).to_csv(out / "lrv_measurements.csv", index=False)

    print(f"  {site_id} ({cfg['model']}): {len(time)} flux rows -> {out}/")


if __name__ == "__main__":
    print("Generating synthetic data for 5 sites...")
    for sid, cfg in SITE_CONFIGS.items():
        generate(sid, cfg)
    print("Done.")

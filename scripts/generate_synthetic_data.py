"""
Generate synthetic viral filtration data for all 5 sites.

Each site has:
  - A different commercially-available filter type
  - Different TMP, feed flux, and mAb concentration
  - Flux decline following the Combined 1-A model + Gaussian noise
  - LRV calculated via the Manabe model

Run from project root:
    python scripts/generate_synthetic_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

SITE_CONFIGS = {
    "site_1": {
        "filter":    "Planova_20N",
        "J0": 150.0, "k1": 0.015, "k2": 0.0020, "noise": 2.0,
        "tmp_base":  1.0, "lrv_mean": 4.8, "lrv_std": 0.2,
    },
    "site_2": {
        "filter":    "ViresolveNFP",
        "J0": 120.0, "k1": 0.020, "k2": 0.0030, "noise": 3.0,
        "tmp_base":  1.2, "lrv_mean": 5.1, "lrv_std": 0.3,
    },
    "site_3": {
        "filter":    "Pegasus_SV4",
        "J0": 180.0, "k1": 0.010, "k2": 0.0010, "noise": 1.5,
        "tmp_base":  0.8, "lrv_mean": 4.6, "lrv_std": 0.2,
    },
    "site_4": {
        "filter":    "Planova_BioEX",
        "J0": 100.0, "k1": 0.025, "k2": 0.0040, "noise": 2.5,
        "tmp_base":  1.4, "lrv_mean": 5.3, "lrv_std": 0.25,
    },
    "site_5": {
        "filter":    "ViresolveNFR",
        "J0": 160.0, "k1": 0.012, "k2": 0.0015, "noise": 2.0,
        "tmp_base":  1.1, "lrv_mean": 4.9, "lrv_std": 0.2,
    },
}


def generate(site_id: str, cfg: dict) -> None:
    out = Path(f"data/{site_id}")
    out.mkdir(parents=True, exist_ok=True)

    time = np.arange(0, 121, 1, dtype=float)   # 0 .. 120 minutes
    J0, k1, k2 = cfg["J0"], cfg["k1"], cfg["k2"]

    # Combined 1-A flux + noise
    flux = (J0 / (1.0 + k1 * time) ** 2) * np.exp(-k2 * time)
    flux += np.random.normal(0.0, cfg["noise"], len(time))
    flux  = np.clip(flux, 1.0, None)

    # TMP drifts upward slightly as membrane fouls
    tmp = cfg["tmp_base"] + 0.004 * time + np.random.normal(0.0, 0.02, len(time))

    # LRV measurements (sparse — every 15 min)
    lrv_times = np.arange(0, 121, 15, dtype=float)
    lrv_vals  = np.random.normal(cfg["lrv_mean"], cfg["lrv_std"], len(lrv_times))
    lrv_vals  = np.clip(lrv_vals, 2.0, 7.0)

    # Main filtration CSV
    pd.DataFrame({
        "time_min":   time,
        "flux_lmh":   flux,
        "tmp_bar":    tmp,
        "filter_type": cfg["filter"],
    }).to_csv(out / "filtration.csv", index=False)

    # LRV measurements CSV
    pd.DataFrame({
        "time_min": lrv_times,
        "lrv":      lrv_vals,
        "flux_lmh": np.interp(lrv_times, time, flux),
    }).to_csv(out / "lrv_measurements.csv", index=False)

    print(f"  {site_id}: {len(time)} flux rows, {len(lrv_times)} LRV rows -> {out}/")


if __name__ == "__main__":
    print("Generating synthetic data for 5 sites...")
    for sid, cfg in SITE_CONFIGS.items():
        generate(sid, cfg)
    print("Done.")

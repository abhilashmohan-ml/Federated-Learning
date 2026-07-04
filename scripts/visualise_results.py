"""
Standalone result visualisation.

Loads synthetic data from all 5 sites, fits Hermia models,
and plots:
  1. Flux decline overlay (all sites)
  2. LRV bar chart (all sites)
  3. Amin vs site bar chart

Run from project root:
    python scripts/visualise_results.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cycler

# Enable grid and update its appearance
plt.rcParams.update({'axes.grid': True})
plt.rcParams.update({'grid.color': 'silver'})
plt.rcParams.update({'grid.linestyle': '--'})

# Set figure resolution
plt.rcParams.update({'figure.dpi': 150})

# Hide the top and right spines
plt.rcParams.update({'axes.spines.top': False})
plt.rcParams.update({'axes.spines.right': False})

# Increase font sizes
plt.rcParams.update({'font.size': 12})  # General font size
plt.rcParams.update({'axes.titlesize': 14})  # Title font size
plt.rcParams.update({'axes.labelsize': 12})  # Axis label font size

plt.rcParams.update({'axes.prop_cycle': cycler.cycler('color', ['#0F69AF'])})

os.environ.setdefault("SITE_ID",         "viz")
os.environ.setdefault("SERVER_URL",      "http://localhost:8000")
os.environ.setdefault("SITE_SECRET",     "secret_site_1")
os.environ.setdefault("LOCAL_DATA_PATH", "data/site_1/filtration.csv")

from client.engine.data_loader   import load_filtration_csv
from shared.models.hermia        import fit_all_models, compute_flux_ratio, compute_amin

SITES  = [f"site_{i}" for i in range(1, 6)]
COLORS = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

amins      = []
lrvs_dummy = [4.8, 5.1, 4.6, 5.3, 4.9]   # placeholder until Manabe fitting added

for idx, site in enumerate(SITES):
    path = f"data/{site}/filtration.csv"
    try:
        time, flux, tmp = load_filtration_csv(path)
    except FileNotFoundError:
        print(f"  Skipping {site} — data not found. Run generate_synthetic_data.py first.")
        continue

    results = fit_all_models(time, flux)
    best    = next((r for r in results.values() if r.selected), list(results.values())[0])
    amin    = compute_amin(10.0, float(np.mean(flux)), float(time[-1]) / 60.0)
    amins.append(amin)

    axes[0].plot(time, flux, color=COLORS[idx], label=f"{site} ({best.model_name})", alpha=0.8)

axes[0].set_xlabel("Time (min)")
axes[0].set_ylabel("Flux (LMH)")
axes[0].set_title("Flux Decline — All Sites")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].bar(SITES, lrvs_dummy, color=COLORS)
axes[1].axhline(4.0, color="red", linestyle="--", linewidth=1.2, label="Min LRV = 4.0")
axes[1].set_ylabel("LRV")
axes[1].set_title("LRV — All Sites")
axes[1].legend()

if amins:
    axes[2].bar(SITES[:len(amins)], amins, color=COLORS[:len(amins)])
    axes[2].set_ylabel("Amin (m2)")
    axes[2].set_title("Minimum Filter Area — All Sites")

plt.tight_layout()
plt.savefig("results_overview.png", dpi=150)
print("Saved results_overview.png")
plt.show()

"""
Pydantic schemas for filtration run data and results.

OVERVIEW
--------
These schemas represent the physical description of a filtration experiment:
  - The filter itself (pore size, membrane area, NMWCO, etc.)
  - The process conditions (pressure, pH, mAb concentration, etc.)
  - The time-series measurements (flux and pressure over time)
  - The results after analysis (best Hermia model, LRV, A_min, compliance)

IMPORTANT PRIVACY NOTE
----------------------
`FiltrationRunData` contains raw experimental measurements. This schema exists
for internal use within the site container ONLY. It is NEVER sent to the server.
Only the distilled outputs (`FiltrationResult` metrics) are sent, and even those
go through differential privacy noise in `shared/crypto/noise.py`.

DOMAIN GLOSSARY
---------------
  Filter:           A physical membrane device that viruses cannot pass through.
  NMWCO:            Nominal Molecular Weight Cut-Off — the size of molecules the
                    filter retains, measured in kiloDaltons (kDa).
  TMP:              Trans-Membrane Pressure — the pressure driving liquid through
                    the filter (bar). Higher TMP = faster flux initially.
  LRV:              Log Reduction Value — how many 10x reductions of virus the
                    filter achieves. LRV=4 means a 10,000-fold reduction.
  mAb concentration: How much of the drug product (monoclonal antibody) is in
                    the feed liquid (g/L).
  Virus spike:      A controlled amount of virus added to the feed for testing
                    purposes (only in validation experiments, not routine runs).

PYTHON CONCEPT: Optional[...]
  `Optional[X]` means the field can be either type X or None (absent).
  Equivalent to `X | None` in Python 3.10+.

PYTHON CONCEPT: Dict[str, float]
  A dictionary mapping string keys to float values.
  For virus_spike: key = virus name (e.g. "Parvovirus"), value = feed concentration.
"""

from typing import Dict, List, Optional  # type hints for collections and optional fields
from pydantic import BaseModel           # Pydantic base class for validated schemas


class FilterDescriptor(BaseModel):
    """
    Physical description of the filter membrane being used.

    This tells the PINN what type of filter it is dealing with, which
    affects the expected fouling behaviour and virus capture performance.

    Fields
    ------
    filter_type      : commercial name, e.g. "Planova20N", "ViresolveNFP"
    pore_size_nm     : average pore diameter in nanometres
                       Typical virus-retentive filters: 15–50 nm
    nmwco_kda        : nominal molecular weight cut-off in kDa
                       Smaller NMWCO = tighter filter = more virus retention
    membrane_area_m2 : total filtration area in square metres
                       Larger area = higher throughput, slower fouling
    manufacturer     : company that made the filter (e.g. "Asahi Kasei", "Merck")
    """
    filter_type:      str    # e.g. "Planova20N"
    pore_size_nm:     float  # nanometres, e.g. 20.0
    nmwco_kda:        float  # kiloDaltons, e.g. 150.0
    membrane_area_m2: float  # square metres, e.g. 0.001 for a small test filter
    manufacturer:     str    # e.g. "Asahi Kasei Bioprocess"


class ProcessConditions(BaseModel):
    """
    The operating conditions under which the filtration was run.

    These conditions strongly affect both the flux profile (how fast filtration
    proceeds) and the LRV (how well viruses are removed).

    Fields
    ------
    tmp_bar             : transmembrane pressure (bar)
                          Typical range: 0.5–3.0 bar for virus filters
    feed_flux_lmh       : target flux at the start (LMH = L/m²/h)
                          Typical: 20–200 LMH for normal-flow viral filtration
    pH                  : pH of the feed solution; affects mAb and virus charge
                          Typical: 5.0–7.5 for mAb products
    ionic_strength_mM   : salt concentration in mM; affects electrostatic interactions
                          Higher IS generally reduces electrostatic retention
    mab_concentration_g_L: concentration of the drug product in the feed
                          High concentration increases fouling risk
    temperature_C       : process temperature in Celsius (default 25°C = room temperature)
    """
    tmp_bar:              float          # transmembrane pressure in bar
    feed_flux_lmh:        float          # initial target flux in LMH
    pH:                   float          # solution pH
    ionic_strength_mM:    float          # ionic strength in millimolar
    mab_concentration_g_L: float         # mAb concentration in g/L
    temperature_C:        float = 25.0  # temperature in Celsius (default room temperature)


class FiltrationRunData(BaseModel):
    """
    Complete dataset for one filtration run.

    This is the INPUT to the analysis pipeline. It is loaded from a local CSV
    file by `client/engine/data_loader.py` and NEVER transmitted off-site.

    The time-series lists (time_min, flux_lmh, tmp_bar_series) are parallel:
    they all have the same length, and index i in each list corresponds to the
    same measurement time point.

    Fields
    ------
    site_id          : which site ran this experiment
    run_id           : unique identifier for this specific run
    filter_descriptor: physical description of the filter used
    process_conditions: the operating conditions
    time_min         : list of time points in minutes, e.g. [0, 5, 10, 15, ...]
    flux_lmh         : flux measured at each time point in LMH
    tmp_bar_series   : pressure measured at each time point in bar
    virus_spike      : (validation runs only) virus name → initial feed concentration
    virus_permeate   : (validation runs only) virus name → permeate concentration
                       Used to calculate measured LRV for model fitting
    """
    site_id:           str
    run_id:            str
    filter_descriptor: FilterDescriptor
    process_conditions: ProcessConditions
    time_min:          List[float]                      # parallel time-series [min]
    flux_lmh:          List[float]                      # parallel flux values [LMH]
    tmp_bar_series:    List[float]                      # parallel TMP values [bar]
    virus_spike:       Optional[Dict[str, float]] = None  # only in validation runs
    virus_permeate:    Optional[Dict[str, float]] = None  # only in validation runs


class FiltrationResult(BaseModel):
    """
    Analysis results for one filtration run.

    This is the OUTPUT of the analysis pipeline. The PINN and Hermia fitting
    produce these numbers. A summary of these metrics (after DP noise) is included
    in the `ModelUpdate.local_metrics` field sent to the server.

    Fields
    ------
    site_id           : which site this result belongs to
    run_id            : the run that produced these results
    best_hermia_model : name of the Hermia model with the lowest AIC
    hermia_params     : parameter values for the best-fit model
    hermia_aic        : AIC value of the best model (lower = better fit)
    flux_ratio        : J_final / J_initial — fouling severity indicator
                        < 0.2 means the filter is exhausted
    amin_m2           : minimum filter area needed to achieve target throughput (m²)
    lrv               : log reduction value calculated by the Manabe model
    lrv_compliant     : True if lrv >= regulatory minimum (4.0)
    manabe_Pc         : capture probability from Manabe model (0–1)
    manabe_lambda     : membrane affinity parameter from Manabe fit
    manabe_J_crit     : critical flux from Manabe fit (LMH)
    """
    site_id:           str
    run_id:            str
    best_hermia_model: str              # e.g. "combined_1a"
    hermia_params:     Dict[str, float] # e.g. {"J0": 42.3, "k1": 0.012, "k2": 0.0003}
    hermia_aic:        float            # AIC of best model
    flux_ratio:        float            # J_final / J_initial
    amin_m2:           float            # minimum filter area in m²
    lrv:               float            # log reduction value
    lrv_compliant:     bool             # lrv >= LRV_MIN_PARVOVIRUS (4.0)
    manabe_Pc:         float            # capture probability [0, 1]
    manabe_lambda:     float            # membrane affinity (dimensionless)
    manabe_J_crit:     float            # critical flux [LMH]

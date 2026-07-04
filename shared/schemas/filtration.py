"""Pydantic schemas for filtration run data and results."""
from typing import Dict, List, Optional
from pydantic import BaseModel


class FilterDescriptor(BaseModel):
    filter_type: str            # e.g. "Planova20N", "ViresolveNFP"
    pore_size_nm: float
    nmwco_kda: float
    membrane_area_m2: float
    manufacturer: str


class ProcessConditions(BaseModel):
    tmp_bar: float
    feed_flux_lmh: float
    pH: float
    ionic_strength_mM: float
    mab_concentration_g_L: float
    temperature_C: float = 25.0


class FiltrationRunData(BaseModel):
    site_id: str
    run_id: str
    filter_descriptor: FilterDescriptor
    process_conditions: ProcessConditions
    time_min: List[float]
    flux_lmh: List[float]
    tmp_bar_series: List[float]
    virus_spike: Optional[Dict[str, float]] = None    # virus_name -> C_feed
    virus_permeate: Optional[Dict[str, float]] = None


class FiltrationResult(BaseModel):
    site_id: str
    run_id: str
    best_hermia_model: str
    hermia_params: Dict[str, float]
    hermia_aic: float
    flux_ratio: float
    amin_m2: float
    lrv: float
    lrv_compliant: bool
    manabe_Pc: float
    manabe_lambda: float
    manabe_J_crit: float

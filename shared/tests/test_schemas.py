"""Unit tests for shared/schemas — auth, federation, filtration. 100% coverage."""
from datetime import datetime, timezone

from shared.schemas.auth import TokenRequest, TokenResponse, RefreshRequest, TokenClaims
from shared.schemas.federation import (
    RoundStatus, SiteStatus, FederationRound, ModelUpdate, GlobalModel,
)
from shared.schemas.filtration import (
    FilterDescriptor, ProcessConditions, FiltrationRunData, FiltrationResult,
)


# ── auth schemas ──────────────────────────────────────────────────────────────

class TestTokenRequest:
    def test_fields(self) -> None:
        tr = TokenRequest(site_id="site_1", site_secret="s3cr3t")
        assert tr.site_id == "site_1"
        assert tr.site_secret == "s3cr3t"


class TestTokenResponse:
    def test_defaults(self) -> None:
        resp = TokenResponse(access_token="acc", refresh_token="ref")
        assert resp.token_type == "bearer"
        assert resp.expires_in == 900

    def test_custom_values(self) -> None:
        resp = TokenResponse(
            access_token="a", refresh_token="r",
            token_type="Bearer", expires_in=1800,
        )
        assert resp.token_type == "Bearer"
        assert resp.expires_in == 1800


class TestRefreshRequest:
    def test_field(self) -> None:
        rr = RefreshRequest(refresh_token="rt_xyz")
        assert rr.refresh_token == "rt_xyz"


class TestTokenClaims:
    def test_defaults(self) -> None:
        tc = TokenClaims(sub="site_1", role="client")
        assert tc.round_id == 0
        assert tc.exp == 0

    def test_custom(self) -> None:
        tc = TokenClaims(sub="site_3", role="server", round_id=7, exp=9_999_999)
        assert tc.sub == "site_3"
        assert tc.role == "server"
        assert tc.round_id == 7
        assert tc.exp == 9_999_999


# ── RoundStatus enum ──────────────────────────────────────────────────────────

class TestRoundStatus:
    def test_pending(self) -> None:
        assert RoundStatus.PENDING == "pending"

    def test_collecting(self) -> None:
        assert RoundStatus.COLLECTING == "collecting"

    def test_aggregating(self) -> None:
        assert RoundStatus.AGGREGATING == "aggregating"

    def test_complete(self) -> None:
        assert RoundStatus.COMPLETE == "complete"

    def test_failed(self) -> None:
        assert RoundStatus.FAILED == "failed"

    def test_five_values(self) -> None:
        assert len(RoundStatus) == 5


# ── SiteStatus enum ───────────────────────────────────────────────────────────

class TestSiteStatus:
    def test_registered(self) -> None:
        assert SiteStatus.REGISTERED == "registered"

    def test_idle(self) -> None:
        assert SiteStatus.IDLE == "idle"

    def test_training(self) -> None:
        assert SiteStatus.TRAINING == "training"

    def test_uploading(self) -> None:
        assert SiteStatus.UPLOADING == "uploading"

    def test_done(self) -> None:
        assert SiteStatus.DONE == "done"

    def test_error(self) -> None:
        assert SiteStatus.ERROR == "error"

    def test_six_values(self) -> None:
        assert len(SiteStatus) == 6


# ── FederationRound ───────────────────────────────────────────────────────────

class TestFederationRound:
    def test_basic_fields(self) -> None:
        r = FederationRound(
            round_id=3,
            status=RoundStatus.COLLECTING,
            started_at=datetime.now(timezone.utc),
        )
        assert r.round_id == 3
        assert r.status == RoundStatus.COLLECTING
        assert r.completed_at is None
        assert r.participating_sites == []
        assert r.global_model_version == 0

    def test_default_factory_isolation(self) -> None:
        r1 = FederationRound(round_id=1, status=RoundStatus.COLLECTING,
                             started_at=datetime.now(timezone.utc))
        r2 = FederationRound(round_id=2, status=RoundStatus.COLLECTING,
                             started_at=datetime.now(timezone.utc))
        r1.participating_sites.append("site_1")
        assert "site_1" not in r2.participating_sites


# ── ModelUpdate ───────────────────────────────────────────────────────────────

class TestModelUpdate:
    def test_defaults(self) -> None:
        mu = ModelUpdate(
            site_id="site_2", round_id=1,
            n_samples=150, delta_W={"hermia_params": [1.0, 2.0]},
        )
        assert mu.dp_noise_sigma == 0.0
        assert mu.hermia_best_model == "combined_1a"
        assert mu.local_metrics == {}
        assert isinstance(mu.timestamp, datetime)

    def test_custom_metrics(self) -> None:
        mu = ModelUpdate(
            site_id="site_1", round_id=2, n_samples=50,
            delta_W={}, local_metrics={"flux_rmse": 1.5, "lrv_rmse": 0.3},
        )
        assert mu.local_metrics["flux_rmse"] == 1.5
        assert mu.local_metrics["lrv_rmse"] == 0.3

    def test_default_factory_metrics_isolation(self) -> None:
        mu1 = ModelUpdate(site_id="s1", round_id=1, n_samples=10, delta_W={})
        mu2 = ModelUpdate(site_id="s2", round_id=1, n_samples=10, delta_W={})
        mu1.local_metrics["k"] = 1.0
        assert "k" not in mu2.local_metrics


# ── GlobalModel ───────────────────────────────────────────────────────────────

class TestGlobalModel:
    def test_basic(self) -> None:
        gm = GlobalModel(version=2, round_id=2, weights={"layer": [0.5, 1.5]})
        assert gm.version == 2
        assert gm.round_id == 2
        assert gm.global_metrics == {}
        assert isinstance(gm.created_at, datetime)

    def test_with_metrics(self) -> None:
        gm = GlobalModel(
            version=1, round_id=1,
            weights={},
            global_metrics={"flux_rmse": 2.3},
        )
        assert gm.global_metrics["flux_rmse"] == 2.3


# ── filtration schemas ────────────────────────────────────────────────────────

def _make_filter_descriptor() -> FilterDescriptor:
    return FilterDescriptor(
        filter_type="Planova20N",
        pore_size_nm=20.0,
        nmwco_kda=150.0,
        membrane_area_m2=0.001,
        manufacturer="Asahi Kasei",
    )


def _make_process_conditions(**kwargs) -> ProcessConditions:
    defaults = dict(
        tmp_bar=1.0, feed_flux_lmh=80.0, pH=7.0,
        ionic_strength_mM=150.0, mab_concentration_g_L=5.0,
    )
    defaults.update(kwargs)
    return ProcessConditions(**defaults)


class TestFilterDescriptor:
    def test_fields(self) -> None:
        fd = _make_filter_descriptor()
        assert fd.filter_type == "Planova20N"
        assert fd.pore_size_nm == 20.0
        assert fd.nmwco_kda == 150.0
        assert fd.membrane_area_m2 == 0.001
        assert fd.manufacturer == "Asahi Kasei"


class TestProcessConditions:
    def test_default_temperature(self) -> None:
        pc = _make_process_conditions()
        assert pc.temperature_C == 25.0

    def test_custom_temperature(self) -> None:
        pc = _make_process_conditions(temperature_C=37.0)
        assert pc.temperature_C == 37.0


class TestFiltrationRunData:
    def test_optional_fields_default_none(self) -> None:
        frd = FiltrationRunData(
            site_id="site_1", run_id="run_001",
            filter_descriptor=_make_filter_descriptor(),
            process_conditions=_make_process_conditions(),
            time_min=[0.0, 5.0, 10.0],
            flux_lmh=[100.0, 80.0, 65.0],
            tmp_bar_series=[1.0, 1.0, 1.0],
        )
        assert frd.virus_spike is None
        assert frd.virus_permeate is None

    def test_optional_fields_set(self) -> None:
        frd = FiltrationRunData(
            site_id="site_2", run_id="run_002",
            filter_descriptor=_make_filter_descriptor(),
            process_conditions=_make_process_conditions(),
            time_min=[0.0, 5.0],
            flux_lmh=[100.0, 80.0],
            tmp_bar_series=[1.0, 1.0],
            virus_spike={"Parvovirus": 1e6},
            virus_permeate={"Parvovirus": 100.0},
        )
        assert frd.virus_spike == {"Parvovirus": 1e6}
        assert frd.virus_permeate == {"Parvovirus": 100.0}


class TestFiltrationResult:
    def test_fields(self) -> None:
        fr = FiltrationResult(
            site_id="site_1", run_id="r_001",
            best_hermia_model="combined_1a",
            hermia_params={"J0": 100.0, "k1": 0.01, "k2": 0.001},
            hermia_aic=-50.0,
            flux_ratio=0.42,
            amin_m2=0.0625,
            lrv=4.8,
            lrv_compliant=True,
            manabe_Pc=0.99984,
            manabe_lambda=2.5,
            manabe_J_crit=50.0,
        )
        assert fr.lrv_compliant is True
        assert fr.lrv == 4.8
        assert fr.best_hermia_model == "combined_1a"

    def test_non_compliant(self) -> None:
        fr = FiltrationResult(
            site_id="site_3", run_id="r_002",
            best_hermia_model="standard",
            hermia_params={"J0": 50.0, "ks": 0.02},
            hermia_aic=-30.0,
            flux_ratio=0.3,
            amin_m2=0.1,
            lrv=3.5,
            lrv_compliant=False,
            manabe_Pc=0.9,
            manabe_lambda=1.0,
            manabe_J_crit=100.0,
        )
        assert fr.lrv_compliant is False

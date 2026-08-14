"""Unit tests for client/config.py — computed fields flet_client_port and site_secret."""
import pytest

from client.config import ClientSettings


# ── flet_client_port ──────────────────────────────────────────────────────────


class TestFletClientPort:
    """flet_client_port reads from FLET_CLIENT_PORT env var; default 8551."""

    def test_default_port_is_8551(self) -> None:
        s = ClientSettings(site_id="any_site_name")
        assert s.flet_client_port == 8551

    def test_explicit_port_respected(self) -> None:
        s = ClientSettings(flet_client_port=8553)
        assert s.flet_client_port == 8553

    def test_arbitrary_port_accepted(self) -> None:
        """Any valid port number works — port is not tied to site_id naming."""
        s = ClientSettings(flet_client_port=9000)
        assert s.flet_client_port == 9000


# ── site_secret ───────────────────────────────────────────────────────────────


class TestSiteSecret:
    """site_secret reads directly from SITE_SECRET env var."""

    def test_default_secret_is_empty(self) -> None:
        s = ClientSettings()
        assert s.site_secret == ""

    def test_secret_reads_from_env(self) -> None:
        s = ClientSettings(site_secret="s3cr3t_value")
        assert s.site_secret == "s3cr3t_value"

    def test_secret_is_site_id_independent(self) -> None:
        """Secret is the same regardless of site_id — no hardcoded mapping."""
        s1 = ClientSettings(site_id="basel", site_secret="mypass")
        s2 = ClientSettings(site_id="singapore", site_secret="mypass")
        assert s1.site_secret == s2.site_secret == "mypass"


# ── local_data_path ───────────────────────────────────────────────────────────


class TestLocalDataPath:
    """local_data_path is auto-derived from site_id when not explicitly set."""

    @pytest.mark.parametrize("site_id,expected_path", [
        ("site_1", "data/site_1/filtration.csv"),
        ("site_2", "data/site_2/filtration.csv"),
        ("site_3", "data/site_3/filtration.csv"),
        ("site_4", "data/site_4/filtration.csv"),
        ("site_5", "data/site_5/filtration.csv"),
    ])
    def test_path_derived_from_site_id(self, site_id: str, expected_path: str) -> None:
        s = ClientSettings(site_id=site_id)
        assert s.local_data_path == expected_path

    def test_explicit_override_respected(self) -> None:
        """When LOCAL_DATA_PATH is explicitly set, it is used as-is."""
        s = ClientSettings(site_id="site_1", local_data_path="/data/filtration.csv")
        assert s.local_data_path == "/data/filtration.csv"

    def test_explicit_override_not_site_specific(self) -> None:
        """Explicit path does not change when site_id changes — caller controls it."""
        s = ClientSettings(site_id="site_3", local_data_path="/custom/path/data.csv")
        assert s.local_data_path == "/custom/path/data.csv"

"""Unit tests for client/config.py — computed fields flet_client_port and site_secret."""
import pytest

from client.config import ClientSettings


# ── flet_client_port ──────────────────────────────────────────────────────────


class TestFletClientPort:
    """flet_client_port derives port from site_id."""

    @pytest.mark.parametrize("site_id,expected_port", [
        ("site_1", 8551),
        ("site_2", 8552),
        ("site_3", 8553),
        ("site_4", 8554),
        ("site_5", 8555),
    ])
    def test_port_derived_from_site_id(self, site_id: str, expected_port: int) -> None:
        s = ClientSettings(site_id=site_id)
        assert s.flet_client_port == expected_port

    def test_unknown_site_id_falls_back_to_8551(self) -> None:
        """Unrecognised site_id (no underscore digit) falls back to 8551."""
        s = ClientSettings(site_id="worker_abc")
        assert s.flet_client_port == 8551

    def test_site_id_without_underscore_falls_back_to_8551(self) -> None:
        s = ClientSettings(site_id="standalone")
        assert s.flet_client_port == 8551


# ── site_secret ───────────────────────────────────────────────────────────────


class TestSiteSecret:
    """site_secret selects the correct per-site secret based on site_id."""

    @pytest.mark.parametrize("site_id,field_name,secret_val", [
        ("site_1", "site_1_secret", "secret_for_1"),
        ("site_2", "site_2_secret", "secret_for_2"),
        ("site_3", "site_3_secret", "secret_for_3"),
        ("site_4", "site_4_secret", "secret_for_4"),
        ("site_5", "site_5_secret", "secret_for_5"),
    ])
    def test_correct_secret_selected(
        self, site_id: str, field_name: str, secret_val: str
    ) -> None:
        s = ClientSettings(**{"site_id": site_id, field_name: secret_val})
        assert s.site_secret == secret_val

    def test_unknown_site_id_returns_empty_string(self) -> None:
        s = ClientSettings(site_id="site_6", site_1_secret="s1", site_2_secret="s2")
        assert s.site_secret == ""

    def test_inactive_site_secrets_not_leaked(self) -> None:
        """site_secret returns only the active site's secret, not another site's."""
        s = ClientSettings(
            site_id="site_1",
            site_1_secret="correct",
            site_2_secret="wrong",
        )
        assert s.site_secret == "correct"

"""Unit tests for server/config.py and client/config.py — covers missing branches."""

import pytest

from server.config import ServerSettings, get_settings
from client.config import ClientSettings, get_client_settings


class TestServerConfig:
    """Tests for ServerSettings.parse_cors_origins and get_settings()."""

    def test_parse_cors_origins_comma_separated_string(self) -> None:
        """str with two comma-separated URLs → list of 2 items (lines 139, 140, 145)."""
        result = ServerSettings.parse_cors_origins("https://a.com,https://b.com")
        assert result == ["https://a.com", "https://b.com"]

    def test_parse_cors_origins_string_with_spaces(self) -> None:
        """str with extra whitespace around URLs → stripped list (lines 139, 140, 145)."""
        result = ServerSettings.parse_cors_origins("  https://a.com , https://b.com  ")
        assert result == ["https://a.com", "https://b.com"]

    def test_parse_cors_origins_empty_string(self) -> None:
        """Empty str '' → [] (lines 139, 140, 141, 142)."""
        result = ServerSettings.parse_cors_origins("")
        assert result == []

    def test_parse_cors_origins_whitespace_only_string(self) -> None:
        """str '  ' (whitespace only) → [] (lines 139, 140, 141, 142)."""
        result = ServerSettings.parse_cors_origins("   ")
        assert result == []

    def test_parse_cors_origins_list_passthrough(self) -> None:
        """list input → same list returned unchanged (lines 146, 147)."""
        origins = ["https://a.com"]
        result = ServerSettings.parse_cors_origins(origins)
        assert result == origins

    def test_parse_cors_origins_unexpected_type_returns_empty(self) -> None:
        """Non-str, non-list input (int 42) → [] safe default (line 148)."""
        result = ServerSettings.parse_cors_origins(42)  # type: ignore[arg-type]
        assert result == []

    def test_get_settings_returns_server_settings_instance(self) -> None:
        """get_settings() executes the cached body and returns a ServerSettings (line 162)."""
        get_settings.cache_clear()
        try:
            s = get_settings()
            assert isinstance(s, ServerSettings)
        finally:
            get_settings.cache_clear()

    def test_get_settings_returns_same_instance_on_repeat_call(self) -> None:
        """lru_cache: repeated calls return the exact same object."""
        get_settings.cache_clear()
        try:
            s1 = get_settings()
            s2 = get_settings()
            assert s1 is s2
        finally:
            get_settings.cache_clear()


class TestClientConfig:
    """Tests for get_client_settings()."""

    def test_get_client_settings_returns_instance(self) -> None:
        """get_client_settings() executes the cached body and returns a ClientSettings (line 101)."""
        get_client_settings.cache_clear()
        try:
            s = get_client_settings()
            assert isinstance(s, ClientSettings)
        finally:
            get_client_settings.cache_clear()

    def test_get_client_settings_returns_same_instance_on_repeat_call(self) -> None:
        """lru_cache: repeated calls return the exact same object."""
        get_client_settings.cache_clear()
        try:
            s1 = get_client_settings()
            s2 = get_client_settings()
            assert s1 is s2
        finally:
            get_client_settings.cache_clear()

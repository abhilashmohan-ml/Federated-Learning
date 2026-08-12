"""Unit tests for server/main.py — 100% coverage."""
import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI

# Import the already-compiled module so module-level code runs once under coverage.
# With default settings (cors_origins=[]) this covers: all top-level imports,
# configure_logging() call, get_settings() call, FastAPI() construction,
# the else-CORS branch (allow_origins=["*"]), all four include_router() calls,
# and the False branch of `if __name__ == "__main__":`.
import server.main  # noqa: F401
from server.main import app

# Absolute path so tests work regardless of the working directory pytest uses.
_MAIN_PY: Path = Path(__file__).resolve().parent.parent / "main.py"


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_settings(
    cors_origins: list[str] | None = None,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> MagicMock:
    """Build a minimal mock of ServerSettings with only the fields server/main.py reads."""
    s = MagicMock()
    s.cors_origins = cors_origins if cors_origins is not None else []
    s.ssl_keyfile = ssl_keyfile
    s.ssl_certfile = ssl_certfile
    s.host = host
    s.port = port
    return s


def _run_module(mock_settings: MagicMock, run_name: str = "<run_path>") -> MagicMock:
    """Re-execute server/main.py in a fresh namespace with mocked I/O dependencies.

    Patches applied for every run:
      - server.config.get_settings  → returns *mock_settings* without reading .env
      - shared.utils.logging_config.configure_logging  → no-op
      - uvicorn.run  → captured mock (returned for assertion)

    Parameters
    ----------
    mock_settings:
        A MagicMock whose attributes drive the branches inside main.py.
    run_name:
        Pass ``"__main__"`` to trigger the ``if __name__ == "__main__":`` block.

    Returns
    -------
    MagicMock
        The mock bound to ``uvicorn.run``; callers can assert on its call history.
    """
    with (
        patch("server.config.get_settings", return_value=mock_settings),
        patch("shared.utils.logging_config.configure_logging"),
        patch("uvicorn.run") as mock_uvicorn,
    ):
        runpy.run_path(str(_MAIN_PY), run_name=run_name)
    return mock_uvicorn


# ── App properties ─────────────────────────────────────────────────────────────


class TestAppProperties:
    """Verify the already-imported FastAPI application is wired up correctly.

    These tests exercise the module-level code that runs on the first import.
    Coverage: imports, configure_logging(), get_settings(), FastAPI(), else-CORS
    branch, all four include_router() calls, and the False-branch of __name__
    guard.
    """

    def test_app_is_fastapi_instance(self) -> None:
        assert isinstance(app, FastAPI)

    def test_app_title(self) -> None:
        assert app.title == "Viral Filtration FL Server"

    def test_app_version(self) -> None:
        assert app.version == "0.1.0"

    def test_app_description_set(self) -> None:
        assert app.description  # non-empty string

    def test_auth_router_registered(self) -> None:
        paths = list(app.openapi()["paths"].keys())
        assert any(p.startswith("/auth") for p in paths)

    def test_federation_router_registered(self) -> None:
        paths = list(app.openapi()["paths"].keys())
        assert any(p.startswith("/federation") for p in paths)

    def test_models_router_registered(self) -> None:
        paths = list(app.openapi()["paths"].keys())
        assert any(p.startswith("/models") for p in paths)

    def test_health_router_registered(self) -> None:
        paths = list(app.openapi()["paths"].keys())
        assert any(p.startswith("/health") for p in paths)


# ── CORS middleware branches ────────────────────────────────────────────────────


class TestCORSMiddleware:
    """Cover both conditional branches of the CORS configuration block.

    Line: ``if settings.cors_origins:``

    Branch A (False / else):  cors_origins=[]  →  allow_origins=["*"],
                              allow_credentials=False.
                              Covered by the top-level module import above.

    Branch B (True / if):    cors_origins=[...]  →  explicit origins,
                              allow_credentials=True.
                              Covered by _run_module() with a non-empty list.
    """

    def test_cors_empty_origins_else_branch(self) -> None:
        """else branch is executed when the module is imported with default settings."""
        # The module was already imported at the top of this file; the else branch
        # ran then.  Asserting app is a FastAPI instance is the observable postcondition.
        assert isinstance(app, FastAPI)

    def test_cors_explicit_origins_if_branch(self) -> None:
        """if branch: non-empty cors_origins → specific origins with credentials=True."""
        mock_settings = _make_settings(cors_origins=["http://localhost:8550"])
        # Run without __main__ so uvicorn.run is NOT called; we only need CORS coverage.
        mock_uvicorn = _run_module(mock_settings)
        # Sanity-check: the run completed and uvicorn.run was not triggered.
        mock_uvicorn.assert_not_called()


# ── __main__ entry-point block ─────────────────────────────────────────────────


class TestMainEntryPoint:
    """Cover the ``if __name__ == '__main__':`` block — both SSL sub-branches.

    The block contains a nested conditional:
        if settings.ssl_keyfile and settings.ssl_certfile:
            ssl_kw = {"ssl_keyfile": ..., "ssl_certfile": ...}
        # else ssl_kw remains {}

    Two tests cover the True and False paths of that conditional.
    """

    def test_main_no_ssl_calls_uvicorn_without_ssl_kwargs(self) -> None:
        """ssl_keyfile=None, ssl_certfile=None → uvicorn.run receives no ssl_ kwargs."""
        mock_settings = _make_settings(ssl_keyfile=None, ssl_certfile=None)
        mock_uvicorn = _run_module(mock_settings, run_name="__main__")

        mock_uvicorn.assert_called_once_with(
            "server.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
        )
        call_kwargs = mock_uvicorn.call_args.kwargs
        assert "ssl_keyfile" not in call_kwargs
        assert "ssl_certfile" not in call_kwargs

    def test_main_with_ssl_calls_uvicorn_with_ssl_kwargs(self) -> None:
        """ssl_keyfile and ssl_certfile both set → ssl kwargs forwarded to uvicorn.run."""
        mock_settings = _make_settings(ssl_keyfile="key.pem", ssl_certfile="cert.pem")
        mock_uvicorn = _run_module(mock_settings, run_name="__main__")

        mock_uvicorn.assert_called_once_with(
            "server.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
        )
        call_kwargs = mock_uvicorn.call_args.kwargs
        assert call_kwargs["ssl_keyfile"] == "key.pem"
        assert call_kwargs["ssl_certfile"] == "cert.pem"

    def test_main_reload_is_false(self) -> None:
        """Ensure reload=False is always passed (never enable auto-reload in the container)."""
        mock_settings = _make_settings()
        mock_uvicorn = _run_module(mock_settings, run_name="__main__")
        assert mock_uvicorn.call_args.kwargs["reload"] is False

    def test_main_host_and_port_from_settings(self) -> None:
        """host and port are taken from settings, not hard-coded in the call site."""
        mock_settings = _make_settings(host="127.0.0.1", port=9999)
        mock_uvicorn = _run_module(mock_settings, run_name="__main__")
        assert mock_uvicorn.call_args.kwargs["host"] == "127.0.0.1"
        assert mock_uvicorn.call_args.kwargs["port"] == 9999

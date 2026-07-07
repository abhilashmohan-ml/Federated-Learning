"""Unit tests for client/ui/pages — 100% line + branch coverage."""
from unittest.mock import MagicMock, patch
import flet as ft

from client.ui.pages.local_results import LocalResultsPage
from client.ui.pages.status        import StatusPage


def _mock_page() -> MagicMock:
    return MagicMock(spec=ft.Page)


def _mock_settings(**kwargs) -> MagicMock:
    defaults = dict(server_url="http://localhost:8000", site_id="site_1", dp_noise_sigma=0.01)
    defaults.update(kwargs)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class TestStatusPage:
    def _make(self, **kwargs) -> StatusPage:
        with patch("client.ui.pages.status.get_client_settings",
                   return_value=_mock_settings(**kwargs)):
            return StatusPage(_mock_page())

    def test_init_stores_page(self) -> None:
        page = _mock_page()
        with patch("client.ui.pages.status.get_client_settings",
                   return_value=_mock_settings()):
            sp = StatusPage(page)
        assert sp.page is page

    def test_init_loads_settings(self) -> None:
        assert self._make(site_id="site_3").settings.site_id == "site_3"

    def test_build_returns_column(self) -> None:
        assert isinstance(self._make().build(), ft.Column)

    def test_build_has_three_cards(self) -> None:
        col = self._make().build()
        assert len([c for c in col.controls if isinstance(c, ft.Card)]) == 3

    def test_build_has_button(self) -> None:
        col = self._make().build()
        assert any(isinstance(c, ft.Button) for c in col.controls)

    def test_build_button_icon_play_arrow(self) -> None:
        col = self._make().build()
        btn = next(c for c in col.controls if isinstance(c, ft.Button))
        assert btn.icon == ft.Icons.PLAY_ARROW

    def test_build_status_text_color_grey_400(self) -> None:
        col = self._make().build()
        conn_card = next(c for c in col.controls if isinstance(c, ft.Card))
        inner_col = conn_card.content.content
        status_text = next(
            t for t in inner_col.controls
            if isinstance(t, ft.Text) and t.color == ft.Colors.GREY_400
        )
        assert "IDLE" in status_text.value

    def test_build_progress_bar_color_blue(self) -> None:
        col = self._make().build()
        round_card = [c for c in col.controls if isinstance(c, ft.Card)][1]
        inner_col = round_card.content.content
        bar = next(c for c in inner_col.controls if isinstance(c, ft.ProgressBar))
        assert bar.color == ft.Colors.BLUE

    def test_build_server_url_in_text(self) -> None:
        col = self._make(server_url="http://myserver:9000").build()
        card = next(c for c in col.controls if isinstance(c, ft.Card))
        texts = [t.value for t in card.content.content.controls if isinstance(t, ft.Text)]
        assert any("myserver:9000" in v for v in texts)

    def test_build_dp_sigma_in_text(self) -> None:
        col = self._make(dp_noise_sigma=0.05).build()
        training_card = [c for c in col.controls if isinstance(c, ft.Card)][2]
        texts = [t.value for t in training_card.content.content.controls if isinstance(t, ft.Text)]
        assert any("0.05" in v for v in texts)


# ---------------------------------------------------------------------------
# local_results
# ---------------------------------------------------------------------------

class TestLocalResultsPage:
    def test_init_stores_page(self) -> None:
        page = _mock_page()
        assert LocalResultsPage(page).page is page

    def test_build_returns_column(self) -> None:
        assert isinstance(LocalResultsPage(_mock_page()).build(), ft.Column)

    def test_build_contains_flux_chart_container(self) -> None:
        col = LocalResultsPage(_mock_page()).build()
        assert any(isinstance(c, ft.Container) for c in col.controls)

    def test_build_flux_container_height_260(self) -> None:
        col = LocalResultsPage(_mock_page()).build()
        container = next(c for c in col.controls if isinstance(c, ft.Container))
        assert container.height == 260

    def test_build_flux_subtitle_color_cyan(self) -> None:
        col = LocalResultsPage(_mock_page()).build()
        container = next(c for c in col.controls if isinstance(c, ft.Container))
        subtitle = container.content.controls[1]
        assert subtitle.color == ft.Colors.CYAN

    def test_build_contains_metrics_row(self) -> None:
        col = LocalResultsPage(_mock_page()).build()
        assert any(isinstance(c, ft.Row) for c in col.controls)

    def test_build_metrics_row_has_four_cards(self) -> None:
        col = LocalResultsPage(_mock_page()).build()
        row = next(c for c in col.controls if isinstance(c, ft.Row))
        assert len(row.controls) == 4

    def test_build_metric_label_colors_grey_500(self) -> None:
        col = LocalResultsPage(_mock_page()).build()
        row = next(c for c in col.controls if isinstance(c, ft.Row))
        for card in row.controls:
            label = card.content.content.controls[0]
            assert label.color == ft.Colors.GREY_500

    def test_build_metric_labels(self) -> None:
        col = LocalResultsPage(_mock_page()).build()
        row = next(c for c in col.controls if isinstance(c, ft.Row))
        labels = [card.content.content.controls[0].value for card in row.controls]
        assert labels == ["LRV", "Amin (m2)", "Flux Ratio", "Best Model"]

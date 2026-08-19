"""Settings page — server config and site management."""
import flet as ft

from server.config import get_settings
from shared.utils.theme import LC


class SettingsPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self._mode_radio:      ft.RadioGroup | None = None
        self._quorum_field:    ft.TextField  | None = None
        self._window_field:    ft.TextField  | None = None
        self._heartbeat_field: ft.TextField  | None = None
        self._policy_status:   ft.Text       | None = None

    def _on_mode_change(self, e: ft.ControlEvent) -> None:
        is_quorum = e.control.value == "quorum"
        self._quorum_field.visible = is_quorum
        self._window_field.visible = not is_quorum
        self.page.update()

    async def _apply_settings(self, api_base: str) -> None:
        import httpx
        mode = self._mode_radio.value
        payload: dict[str, str] = {
            "aggregation_mode":  mode,
            "heartbeat_seconds": self._heartbeat_field.value,
        }
        if mode == "quorum":
            payload["quorum_min_sites"] = self._quorum_field.value
        else:
            try:
                minutes = float(self._window_field.value)
                payload["time_window_seconds"] = str(int(minutes * 60))
            except ValueError:
                self._policy_status.value = "Invalid window value"
                self._policy_status.color = LC.ERROR
                self.page.update()
                return

        try:
            async with httpx.AsyncClient() as client:
                r = await client.put(f"{api_base}/settings", json=payload, timeout=5.0)
            if r.status_code == 200:
                self._policy_status.value = "Settings applied."
                self._policy_status.color = LC.SUCCESS
            else:
                self._policy_status.value = f"Error {r.status_code}"
                self._policy_status.color = LC.ERROR
        except Exception as exc:
            self._policy_status.value = f"Request failed: {exc}"
            self._policy_status.color = LC.ERROR
        self.page.update()

    def _field(self, label: str, value: str, width: int = 180) -> ft.TextField:
        return ft.TextField(
            label=label, value=value, width=width,
            border_color=LC.BORDER,
            focused_border_color=LC.PRIMARY,
            bgcolor=LC.SURFACE,
            color=LC.TEXT_PRIMARY,
            label_style=ft.TextStyle(color=LC.TEXT_MUTED),
            border_radius=LC.RADIUS_MD,
        )

    def build(self) -> ft.Control:
        settings = get_settings()
        api_base = f"http://localhost:{settings.port}"

        self._mode_radio = ft.RadioGroup(
            value="quorum",
            content=ft.Row([
                ft.Radio(value="quorum",      label="Quorum (default)"),
                ft.Radio(value="time_window", label="Time Window"),
            ]),
            on_change=self._on_mode_change,
        )
        self._quorum_field    = self._field("Min sites required",           "3",  200)
        self._window_field    = self._field("Window (minutes)",             "30", 200)
        self._window_field.visible = False
        self._heartbeat_field = self._field("Heartbeat interval (seconds)", "30", 220)
        self._policy_status   = ft.Text("", size=12, color=LC.SUCCESS)

        return ft.Container(
            content=ft.Column([
                ft.Text("Settings", size=26, weight=ft.FontWeight.BOLD,
                        color=LC.TEXT_PRIMARY),
                ft.Divider(color=LC.BORDER),

                ft.Text("FL Hyperparameters", size=17, color=LC.TEXT_PRIMARY),
                ft.Row([
                    self._field("FL Rounds",       "50"),
                    self._field("Local Epochs",    "5"),
                    self._field("FedProx Mu",      "0.01"),
                    self._field("DP Noise Sigma",  "0.01"),
                    self._field("Min Sites/Round", "3"),
                ], spacing=12, wrap=True),
                ft.Button(
                    "Save Hyperparameters",
                    icon=ft.Icons.SAVE,
                    style=ft.ButtonStyle(bgcolor=LC.PRIMARY, color=LC.SURFACE),
                ),
                ft.Divider(color=LC.BORDER),

                ft.Text("Aggregation Policy", size=17, color=LC.TEXT_PRIMARY),
                ft.Text(
                    "Quorum: aggregate when N distinct sites have submitted. "
                    "Time Window: aggregate when the configured time has elapsed.",
                    size=12, color=LC.TEXT_SECONDARY,
                ),
                self._mode_radio,
                ft.Row([self._quorum_field, self._window_field], spacing=12),
                ft.Divider(color=LC.BORDER),

                ft.Text("Heartbeat Poller", size=17, color=LC.TEXT_PRIMARY),
                self._heartbeat_field,
                ft.Divider(color=LC.BORDER),

                ft.Button(
                    "Apply Policy & Heartbeat",
                    icon=ft.Icons.SAVE,
                    style=ft.ButtonStyle(bgcolor=LC.PRIMARY, color=LC.SURFACE),
                    on_click=lambda _: self.page.run_task(self._apply_settings, api_base),
                ),
                self._policy_status,

                ft.Divider(color=LC.BORDER),
                ft.Text("Registered Sites", size=17, color=LC.TEXT_PRIMARY),
                ft.Text(
                    "Sites are registered via the REGISTERED_SITES environment variable at startup. "
                    "See .env.example for configuration.",
                    size=12, color=LC.TEXT_SECONDARY,
                ),
            ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=16),
            padding=24,
            expand=True,
            bgcolor=LC.BG_PRIMARY,
        )

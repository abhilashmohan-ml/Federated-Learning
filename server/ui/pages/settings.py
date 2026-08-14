"""Settings page — server config and site management."""
import flet as ft

from server.config import get_settings


class SettingsPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        # UI refs set in build() — call build() before accessing them
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
                self._policy_status.color = ft.Colors.RED
                self.page.update()
                return

        try:
            async with httpx.AsyncClient() as client:
                r = await client.put(f"{api_base}/settings", json=payload, timeout=5.0)
            if r.status_code == 200:
                self._policy_status.value = "Settings applied."
                self._policy_status.color = ft.Colors.GREEN
            else:
                self._policy_status.value = f"Error {r.status_code}"
                self._policy_status.color = ft.Colors.RED
        except Exception as exc:
            self._policy_status.value = f"Request failed: {exc}"
            self._policy_status.color = ft.Colors.RED
        self.page.update()

    def build(self) -> ft.Control:
        settings = get_settings()
        api_base = f"http://localhost:{settings.port}"

        # ── Aggregation policy controls ────────────────────────────────────────
        self._mode_radio = ft.RadioGroup(
            value="quorum",
            content=ft.Row([
                ft.Radio(value="quorum",      label="Quorum (default)"),
                ft.Radio(value="time_window", label="Time Window"),
            ]),
            on_change=self._on_mode_change,
        )
        self._quorum_field    = ft.TextField(label="Min sites required",            value="3",  width=200)
        self._window_field    = ft.TextField(label="Window (minutes)",              value="30", width=200, visible=False)
        self._heartbeat_field = ft.TextField(label="Heartbeat interval (seconds)",  value="30", width=220)
        self._policy_status   = ft.Text("", size=12, color=ft.Colors.GREEN)

        return ft.Column([
            ft.Text("Settings", size=26, weight=ft.FontWeight.BOLD),
            ft.Divider(),

            # ── Existing hyperparameter section ───────────────────────────────
            ft.Text("FL Hyperparameters", size=17),
            ft.Row([
                ft.TextField(label="FL Rounds",       value="50",   width=180),
                ft.TextField(label="Local Epochs",    value="5",    width=180),
                ft.TextField(label="FedProx Mu",      value="0.01", width=180),
                ft.TextField(label="DP Noise Sigma",  value="0.01", width=180),
                ft.TextField(label="Min Sites/Round", value="3",    width=180),
            ], spacing=12, wrap=True),
            ft.ElevatedButton("Save Hyperparameters", icon=ft.Icons.SAVE),
            ft.Divider(),

            # ── Aggregation policy ────────────────────────────────────────────
            ft.Text("Aggregation Policy", size=17),
            ft.Text(
                "Quorum: aggregate when N distinct sites have submitted. "
                "Time Window: aggregate when the configured time has elapsed.",
                size=12, color=ft.Colors.GREY_400,
            ),
            self._mode_radio,
            ft.Row([self._quorum_field, self._window_field], spacing=12),
            ft.Divider(),

            ft.Text("Heartbeat Poller", size=17),
            self._heartbeat_field,
            ft.Divider(),

            ft.ElevatedButton(
                "Apply Policy & Heartbeat",
                icon=ft.Icons.SAVE,
                on_click=lambda _: self.page.run_task(self._apply_settings, api_base),
            ),
            self._policy_status,

            ft.Divider(),
            ft.Text("Registered Sites", size=17),
            ft.Text(
                "Sites are registered via the REGISTERED_SITES environment variable at startup. "
                "See .env.example for configuration.",
                size=12, color=ft.Colors.GREY_400,
            ),
        ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=16)

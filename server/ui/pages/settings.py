"""Settings page — server config and site management."""
import flet as ft


class SettingsPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page

    def build(self) -> ft.Control:
        return ft.Column([
            ft.Text("Settings", size=26, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Text("FL Hyperparameters", size=17),
            ft.Row([
                ft.TextField(label="FL Rounds",       value="50",   width=180),
                ft.TextField(label="Local Epochs",    value="5",    width=180),
                ft.TextField(label="FedProx Mu",      value="0.01", width=180),
                ft.TextField(label="DP Noise Sigma",  value="0.01", width=180),
                ft.TextField(label="Min Sites/Round", value="3",    width=180),
            ], spacing=12, wrap=True),
            ft.ElevatedButton("Save Hyperparameters", icon=ft.icons.SAVE),
            ft.Divider(),
            ft.Text("Registered Sites", size=17),
            ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Site ID")),
                    ft.DataColumn(ft.Text("Status")),
                    ft.DataColumn(ft.Text("Last Seen")),
                    ft.DataColumn(ft.Text("Actions")),
                ],
                rows=[
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text(f"site_{i}")),
                        ft.DataCell(ft.Text("IDLE")),
                        ft.DataCell(ft.Text("—")),
                        ft.DataCell(ft.IconButton(ft.icons.DELETE_OUTLINE,
                                                  icon_color=ft.colors.RED_300)),
                    ])
                    for i in range(1, 6)
                ],
            ),
        ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=16, padding=24)

"""Global Model page — PINN parameters + cross-round performance."""
import flet as ft


_PARAM_ROWS = [
    ("J0",     "LMH",    "Initial flux"),
    ("k1",     "1/min",  "Pore constriction rate"),
    ("k2",     "1/min",  "Adsorption/cake rate"),
    ("ks",     "1/min",  "Standard blocking constant"),
    ("ki",     "1/min",  "Intermediate blocking constant"),
    ("kc",     "1/min",  "Complete blocking constant"),
    ("kcf",    "1/min2", "Cake filtration constant"),
    ("Pc",     "—",      "Capture probability (Manabe)"),
    ("J_crit", "LMH",    "Critical flux (Manabe)"),
    ("Dv",     "m2/s",   "Virus diffusion coefficient"),
]


class GlobalModelPage:
    def __init__(self, page: ft.Page) -> None:
        self.page = page

    def build(self) -> ft.Control:
        table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Parameter")),
                ft.DataColumn(ft.Text("Global Value")),
                ft.DataColumn(ft.Text("Std Dev")),
                ft.DataColumn(ft.Text("Units")),
                ft.DataColumn(ft.Text("Description")),
            ],
            rows=[
                ft.DataRow(cells=[
                    ft.DataCell(ft.Text(p)),
                    ft.DataCell(ft.Text("—")),
                    ft.DataCell(ft.Text("—")),
                    ft.DataCell(ft.Text(u)),
                    ft.DataCell(ft.Text(d)),
                ])
                for p, u, d in _PARAM_ROWS
            ],
        )

        return ft.Column([
            ft.Text("Global Consolidated Model", size=26, weight=ft.FontWeight.BOLD),
            ft.Text("Physics-Informed Neural Network — updated after each FL round",
                    size=13, color=ft.Colors.GREY_400),
            ft.Divider(),
            ft.Row([
                ft.Card(content=ft.Container(ft.Column([
                    ft.Text("Model Version", size=12, color=ft.Colors.GREY_500),
                    ft.Text("—", size=28, weight=ft.FontWeight.BOLD),
                ], spacing=2), padding=14, width=130)),
                ft.Card(content=ft.Container(ft.Column([
                    ft.Text("Rounds Completed", size=12, color=ft.Colors.GREY_500),
                    ft.Text("—", size=28, weight=ft.FontWeight.BOLD),
                ], spacing=2), padding=14, width=160)),
                ft.Card(content=ft.Container(ft.Column([
                    ft.Text("Sites Participated", size=12, color=ft.Colors.GREY_500),
                    ft.Text("—", size=28, weight=ft.FontWeight.BOLD),
                ], spacing=2), padding=14, width=155)),
            ], spacing=12),
            ft.Text("Current Global Parameters", size=17),
            table,
        ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=16, padding=24)

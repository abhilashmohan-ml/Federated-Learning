"""Left navigation rail for the server dashboard."""
from typing import Callable
import flet as ft


def build_nav_rail(on_change: Callable) -> ft.NavigationRail:
    return ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        destinations=[
            ft.NavigationRailDestination(icon=ft.Icons.DASHBOARD,      label="Dashboard"),
            ft.NavigationRailDestination(icon=ft.Icons.MONITOR,        label="Sites"),
            ft.NavigationRailDestination(icon=ft.Icons.MODEL_TRAINING, label="Global Model"),
            ft.NavigationRailDestination(icon=ft.Icons.SHOW_CHART,     label="Graphs"),
            ft.NavigationRailDestination(icon=ft.Icons.SETTINGS,       label="Settings"),
        ],
        on_change=on_change,
    )

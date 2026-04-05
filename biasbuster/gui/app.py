"""Main application shell — tab layout and page construction."""

from __future__ import annotations

import logging

from nicegui import ui

from biasbuster.gui.settings_tab import create_settings_tab
from biasbuster.gui.evaluation_tab import create_evaluation_tab
from biasbuster.gui.training_tab import create_training_tab
from biasbuster.gui.export_tab import create_export_tab

logger = logging.getLogger(__name__)


def create_app(state: dict) -> None:
    """Build the main NiceGUI page with four tabs."""

    @ui.page("/")
    def main_page() -> None:
        ui.page_title("BiasBuster Workbench")

        plat = state["platform"]
        badge_text = plat.get("system", "Unknown")
        if plat.get("backends"):
            badge_text += f' ({", ".join(plat["backends"])})'

        with ui.header().classes("items-center justify-between q-px-md"):
            ui.label("BiasBuster Fine-Tuning Workbench").classes("text-h6")
            ui.badge(badge_text, color="blue")

        with ui.tabs().classes("w-full") as tabs:
            settings_tab = ui.tab("Settings", icon="settings")
            eval_tab = ui.tab("Evaluation", icon="assessment")
            train_tab = ui.tab("Fine-Tuning", icon="model_training")
            export_tab = ui.tab("Export", icon="publish")

        with ui.tab_panels(tabs, value=settings_tab).classes("w-full"):
            with ui.tab_panel(settings_tab):
                create_settings_tab(state)
            with ui.tab_panel(eval_tab):
                create_evaluation_tab(state)
            with ui.tab_panel(train_tab):
                create_training_tab(state)
            with ui.tab_panel(export_tab):
                create_export_tab(state)

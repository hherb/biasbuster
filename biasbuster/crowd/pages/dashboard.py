"""Dashboard page for the crowd annotation platform.

Shows the user's progress, available papers, and in-progress annotations.
"""

import logging

from nicegui import ui

from biasbuster.crowd.auth import get_current_user, require_auth
from biasbuster.crowd.db import CrowdDatabase

logger = logging.getLogger(__name__)


def register_dashboard_page(db: CrowdDatabase) -> None:
    """Register the /dashboard page route."""

    @ui.page("/dashboard")
    def dashboard_page() -> None:
        """User dashboard with progress and navigation."""
        if not require_auth():
            return

        user = get_current_user()
        if user is None:
            ui.navigate.to("/login")
            return

        user_id = user["user_id"]
        username = user["username"]
        progress = db.get_user_progress(user_id)
        global_stats = db.get_global_stats()

        with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-4"):
            # Header
            with ui.row().classes("w-full justify-between items-center"):
                ui.label(f"Welcome, {username}").classes("text-h5")
                ui.button(
                    "Logout", on_click=lambda: ui.navigate.to("/logout")
                ).props("flat color=grey")

            # Progress cards
            with ui.row().classes("w-full gap-4"):
                with ui.card().classes("flex-1"):
                    ui.label("Your Progress").classes("text-subtitle1 font-bold")
                    ui.separator()
                    with ui.column().classes("gap-1"):
                        ui.label(
                            f"Completed: {progress['completed']}"
                        ).classes("text-positive text-lg")
                        ui.label(
                            f"In progress: {progress['in_progress']}"
                        ).classes("text-warning text-lg")
                        ui.label(
                            f"Available: {progress['available']}"
                        ).classes("text-lg")

                with ui.card().classes("flex-1"):
                    ui.label("Platform Stats").classes("text-subtitle1 font-bold")
                    ui.separator()
                    with ui.column().classes("gap-1"):
                        ui.label(
                            f"Total papers: {global_stats['total_papers']}"
                        )
                        ui.label(
                            f"Active annotators: {global_stats['total_users']}"
                        )
                        ui.label(
                            f"Annotations completed: {global_stats['total_completed']}"
                        )
                        ui.label(
                            f"Papers fully annotated: {global_stats['fully_annotated']}"
                        )

            # Start annotating button
            if progress["available"] > 0:

                async def start_annotating() -> None:
                    pmid = db.get_next_paper(user_id)
                    if pmid:
                        ui.navigate.to(f"/annotate/{pmid}")
                    else:
                        ui.notify("No more papers available!", type="warning")

                ui.button(
                    "Start Annotating",
                    on_click=start_annotating,
                ).classes("w-full text-lg").props("color=primary size=lg")
            else:
                ui.label(
                    "All available papers have been annotated. Thank you!"
                ).classes("text-h6 text-positive text-center w-full")

            # In-progress annotations (revealed but not completed)
            in_progress = db.get_in_progress_annotations(user_id)
            if in_progress:
                ui.label("Continue Where You Left Off").classes(
                    "text-h6 mt-4"
                )
                with ui.card().classes("w-full"):
                    for ann in in_progress:
                        with ui.row().classes(
                            "w-full items-center justify-between py-2"
                        ):
                            title = ann.get("title", "Untitled")
                            if len(title) > 80:
                                title = title[:80] + "..."
                            ui.label(title).classes("flex-1")
                            ui.button(
                                "Continue Review",
                                on_click=lambda pmid=ann["pmid"]: (
                                    ui.navigate.to(f"/reveal/{pmid}")
                                ),
                            ).props("color=warning flat dense")
                        ui.separator()

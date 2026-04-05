"""Blind annotation page (Phase 1) for the crowd annotation platform.

Presents the paper without any AI annotation and lets the user fill in
a blank structured assessment form. A JavaScript timer tracks time spent.
On submission, saves the blind annotation and redirects to the reveal page.
"""

import html as html_mod
import logging
import time

from nicegui import ui

from biasbuster.crowd.auth import get_current_user, require_auth
from biasbuster.crowd.components.annotation_form import build_blind_form, harvest_form
from biasbuster.crowd.components.paper_display import render_paper, render_paper_compact
from biasbuster.crowd.db import CrowdDatabase
from biasbuster.prompts import REVIEWER_REFERENCE_CARD

logger = logging.getLogger(__name__)


def register_annotate_page(
    db: CrowdDatabase, max_per_hour: int = 30
) -> None:
    """Register the /annotate/{pmid} page route."""

    @ui.page("/annotate/{pmid}")
    def annotate_page(pmid: str) -> None:
        """Blind annotation page — user annotates without seeing AI output."""
        if not require_auth():
            return

        user = get_current_user()
        if user is None:
            ui.navigate.to("/login")
            return

        user_id = user["user_id"]

        # Annotation rate limiting
        recent = db.count_user_annotations_recent(user_id, minutes=60)
        if recent >= max_per_hour:
            ui.label(
                "You've reached the annotation limit for this hour. "
                "Please take a break and come back later."
            ).classes("text-warning text-h6")
            return

        # Validate PMID
        if not pmid.strip().isdigit():
            ui.label("Invalid paper ID.").classes("text-negative text-h6")
            return

        # Check if user already annotated this paper
        existing_phase = db.get_annotation_phase(user_id, pmid)
        if existing_phase == "revealed":
            ui.navigate.to(f"/reveal/{pmid}")
            return
        if existing_phase == "completed":
            ui.notify("You have already annotated this paper.", type="info")
            ui.navigate.to("/dashboard")
            return

        # Load paper
        paper = db.get_paper(pmid)
        if paper is None:
            ui.label("Paper not found.").classes("text-negative text-h6")
            return

        # Track start time (Python-side fallback)
        page_start_time = time.monotonic()

        # Page layout
        with ui.column().classes("w-full p-2 gap-2"):
            # Header
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Phase 1: Blind Assessment").classes(
                    "text-h5 text-weight-bold"
                )
                ui.button(
                    "Back to Dashboard",
                    on_click=lambda: ui.navigate.to("/dashboard"),
                ).props("flat color=grey")

            ui.label(
                "Please assess this paper for bias. "
                "You have NOT seen the AI assessment yet."
            ).classes("text-subtitle1 text-grey-8")

            # Split layout: paper on left, form on right
            with ui.splitter(value=45).classes("w-full") as splitter:
                with splitter.before:
                    with ui.tabs().classes("w-full") as tabs:
                        tab_full = ui.tab("Full Details")
                        tab_abstract = ui.tab("Abstract Only")
                        tab_guide = ui.tab("Guidelines")

                    with ui.tab_panels(tabs, value=tab_full).classes("w-full"):
                        with ui.tab_panel(tab_full):
                            full_container = ui.column().classes("w-full")
                            render_paper(paper, full_container)

                        with ui.tab_panel(tab_abstract):
                            abstract_container = ui.column().classes("w-full")
                            render_paper_compact(paper, abstract_container)

                        with ui.tab_panel(tab_guide):
                            with ui.card().classes("w-full"):
                                ui.html(
                                    f'<pre style="white-space: pre-wrap; '
                                    f'word-wrap: break-word; font-size: 0.85rem; '
                                    f'user-select: text;">'
                                    f"{html_mod.escape(REVIEWER_REFERENCE_CARD)}</pre>"
                                )

                with splitter.after:
                    with ui.card().classes("w-full"):
                        ui.label("Your Assessment").classes(
                            "text-subtitle1 font-bold"
                        )
                        ui.separator()
                        form_container = ui.column().classes("w-full")
                        form_refs = build_blind_form(form_container)

                        async def submit_blind() -> None:
                            annotation = harvest_form(form_refs)

                            # Compute time spent
                            elapsed = time.monotonic() - page_start_time

                            # Get AI annotation for agreement comparison
                            ai_ann_row = db.get_ai_annotation(pmid)
                            ai_ann = (
                                ai_ann_row["annotation"]
                                if ai_ann_row
                                else None
                            )

                            try:
                                db.save_blind_annotation(
                                    user_id=user_id,
                                    pmid=pmid,
                                    annotation=annotation,
                                    time_seconds=elapsed,
                                    ai_annotation=ai_ann,
                                )
                                ui.navigate.to(f"/reveal/{pmid}")
                            except Exception:
                                logger.exception(
                                    "Error saving blind annotation for %s", pmid
                                )
                                ui.notify(
                                    "Error saving annotation. Please try again.",
                                    type="negative",
                                )

                        ui.button(
                            "Submit Assessment & See AI Analysis",
                            on_click=submit_blind,
                        ).classes("w-full mt-4").props("color=primary size=lg")

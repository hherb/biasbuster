"""Reveal and revision page (Phase 2) for the crowd annotation platform.

After submitting the blind annotation, the AI annotation is revealed
alongside the user's assessment. The user can revise their annotation
or keep their original.
"""

import html as html_mod
import logging
import time

from nicegui import ui

from crowd.auth import get_current_user, require_auth
from crowd.components.annotation_form import build_revision_form, harvest_form
from crowd.components.paper_display import render_paper
from crowd.db import CrowdDatabase

logger = logging.getLogger(__name__)

# Severity ordering for comparison highlighting
_SEVERITY_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}

# Domains to compare
_DOMAINS = [
    ("statistical_reporting", "Statistical Reporting"),
    ("spin", "Spin"),
    ("outcome_reporting", "Outcome Reporting"),
    ("conflict_of_interest", "Conflict of Interest"),
    ("methodology", "Methodology"),
]


def _severity_badge(severity: str, label: str = "") -> ui.element:
    """Create a colored badge for a severity level."""
    colors = {
        "none": "bg-green-2 text-green-9",
        "low": "bg-lime-2 text-lime-9",
        "moderate": "bg-amber-2 text-amber-9",
        "high": "bg-orange-2 text-orange-9",
        "critical": "bg-red-2 text-red-9",
    }
    css = colors.get(severity, "bg-grey-3 text-grey-8")
    text = f"{label}: {severity}" if label else severity
    return ui.label(text).classes(f"px-2 py-1 rounded text-sm font-bold {css}")


def _build_comparison(
    user_annotation: dict,
    ai_annotation: dict,
    container: ui.element,
) -> None:
    """Build side-by-side comparison of user vs AI annotations."""
    container.clear()
    with container:
        ui.label("Comparison: Your Assessment vs AI").classes(
            "text-h6 font-bold mb-2"
        )

        # Overall severity comparison
        user_overall = user_annotation.get("overall_severity", "none")
        ai_overall = ai_annotation.get("overall_severity", "none")
        match = user_overall == ai_overall

        with ui.card().classes("w-full mb-4"):
            ui.label("Overall Severity").classes("text-subtitle1 font-bold")
            with ui.row().classes("w-full gap-8 items-center"):
                with ui.column().classes("flex-1 items-center"):
                    ui.label("You").classes("text-sm text-grey-7")
                    _severity_badge(user_overall)
                with ui.column().classes("flex-1 items-center"):
                    ui.label("AI").classes("text-sm text-grey-7")
                    _severity_badge(ai_overall)
            if match:
                ui.label("Agreement").classes(
                    "text-positive text-sm font-bold mt-1"
                )
            else:
                ui.label("Disagreement").classes(
                    "text-warning text-sm font-bold mt-1"
                )

        # Per-domain comparison
        for domain_key, domain_label in _DOMAINS:
            user_domain = user_annotation.get(domain_key, {})
            ai_domain = ai_annotation.get(domain_key, {})

            if not isinstance(user_domain, dict):
                user_domain = {}
            if not isinstance(ai_domain, dict):
                ai_domain = {}

            user_sev = user_domain.get("severity", "none")
            ai_sev = ai_domain.get("severity", "none")
            domain_match = user_sev == ai_sev

            bg_class = "" if domain_match else "bg-amber-1"

            with ui.expansion(
                f"{domain_label}",
                icon="check_circle" if domain_match else "warning",
            ).classes(f"w-full {bg_class}").props("dense"):
                with ui.row().classes("w-full gap-4"):
                    # User column
                    with ui.column().classes("flex-1"):
                        ui.label("Your Rating").classes(
                            "text-sm font-bold text-grey-7"
                        )
                        _severity_badge(user_sev, "Severity")
                        _render_domain_flags(user_domain, domain_key)

                    # AI column
                    with ui.column().classes("flex-1"):
                        ui.label("AI Rating").classes(
                            "text-sm font-bold text-grey-7"
                        )
                        _severity_badge(ai_sev, "Severity")
                        _render_domain_flags(ai_domain, domain_key)

                # AI evidence quotes
                ai_quotes = ai_domain.get("evidence_quotes", [])
                if ai_quotes and isinstance(ai_quotes, list):
                    with ui.expansion("AI Evidence Quotes").props("dense"):
                        for q in ai_quotes:
                            ui.label(f'"{q}"').classes(
                                "text-sm italic text-grey-8 ml-4"
                            )

        # AI reasoning
        ai_reasoning = ai_annotation.get("reasoning", "")
        if ai_reasoning:
            with ui.expansion("AI Reasoning", icon="psychology").classes(
                "w-full mt-2"
            ).props("dense"):
                ui.html(
                    f'<div style="white-space: pre-wrap; font-size: 0.9rem; '
                    f'user-select: text;">'
                    f"{html_mod.escape(str(ai_reasoning))}</div>"
                )

        # AI verification steps
        ai_steps = ai_annotation.get("recommended_verification_steps", [])
        if ai_steps and isinstance(ai_steps, list):
            with ui.expansion(
                "AI Recommended Verification", icon="fact_check"
            ).classes("w-full").props("dense"):
                for step in ai_steps:
                    ui.label(f"- {step}").classes("text-sm ml-4")


def _render_domain_flags(domain_data: dict, domain_key: str) -> None:
    """Render boolean flags for a domain as small labels."""
    # Known boolean fields per domain
    flag_keys = {
        "statistical_reporting": [
            "relative_only", "absolute_reported", "nnt_reported",
            "baseline_risk_reported", "selective_p_values", "subgroup_emphasis",
        ],
        "spin": [
            "conclusion_matches_results", "causal_language_from_observational",
            "focus_on_secondary_when_primary_ns", "inappropriate_extrapolation",
            "title_spin",
        ],
        "outcome_reporting": [
            "surrogate_without_validation", "composite_not_disaggregated",
        ],
        "conflict_of_interest": [
            "funding_disclosed_in_abstract", "industry_author_affiliations",
            "coi_disclosed",
        ],
        "methodology": [
            "inappropriate_comparator", "enrichment_design",
            "per_protocol_only", "premature_stopping", "short_follow_up",
        ],
    }

    keys = flag_keys.get(domain_key, [])
    for key in keys:
        val = domain_data.get(key)
        if val is True:
            ui.label(f"  {key.replace('_', ' ')}: Yes").classes(
                "text-xs text-orange-8"
            )

    # Special fields
    if "spin_level" in domain_data:
        ui.label(f"  Spin level: {domain_data['spin_level']}").classes(
            "text-xs"
        )
    if "funding_type" in domain_data:
        ui.label(f"  Funding: {domain_data['funding_type']}").classes(
            "text-xs"
        )
    if "primary_outcome_type" in domain_data:
        ui.label(
            f"  Outcome type: {domain_data['primary_outcome_type']}"
        ).classes("text-xs")


def register_reveal_page(db: CrowdDatabase) -> None:
    """Register the /reveal/{pmid} page route."""

    @ui.page("/reveal/{pmid}")
    def reveal_page(pmid: str) -> None:
        """AI reveal and revision page."""
        if not require_auth():
            return

        user = get_current_user()
        if user is None:
            ui.navigate.to("/login")
            return

        user_id = user["user_id"]

        # Validate state
        phase = db.get_annotation_phase(user_id, pmid)
        if phase is None:
            ui.notify("Please annotate this paper first.", type="warning")
            ui.navigate.to("/dashboard")
            return
        if phase == "completed":
            ui.notify("You have already completed this paper.", type="info")
            ui.navigate.to("/dashboard")
            return
        if phase != "revealed":
            ui.navigate.to(f"/annotate/{pmid}")
            return

        # Load data
        crowd_ann = db.get_crowd_annotation(user_id, pmid)
        if crowd_ann is None:
            ui.navigate.to("/dashboard")
            return

        blind_annotation = crowd_ann["blind_annotation"]
        if not isinstance(blind_annotation, dict):
            blind_annotation = {}

        ai_ann_row = db.get_ai_annotation(pmid)
        ai_annotation = (
            ai_ann_row["annotation"] if ai_ann_row else {}
        )
        if not isinstance(ai_annotation, dict):
            ai_annotation = {}

        paper = db.get_paper(pmid)

        # Track revision time
        revision_start = time.monotonic()

        with ui.column().classes("w-full p-2 gap-2"):
            # Header
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Phase 2: Review AI Assessment").classes(
                    "text-h5 text-weight-bold"
                )
                ui.button(
                    "Back to Dashboard",
                    on_click=lambda: ui.navigate.to("/dashboard"),
                ).props("flat color=grey")

            ui.label(
                "The AI assessment is now revealed. Compare it with yours, "
                "then revise your assessment if you wish."
            ).classes("text-subtitle1 text-grey-8")

            # Three-column layout: comparison | paper | revision form
            with ui.splitter(value=35).classes("w-full") as splitter:
                with splitter.before:
                    comparison_container = ui.column().classes(
                        "w-full overflow-auto"
                    ).style("max-height: 85vh")
                    _build_comparison(
                        blind_annotation, ai_annotation, comparison_container
                    )

                with splitter.after:
                    with ui.tabs().classes("w-full") as tabs:
                        tab_revise = ui.tab("Revise Your Assessment")
                        tab_paper = ui.tab("Paper")

                    with ui.tab_panels(tabs, value=tab_revise).classes(
                        "w-full"
                    ):
                        with ui.tab_panel(tab_revise):
                            with ui.card().classes("w-full"):
                                ui.label(
                                    "Edit your assessment below (pre-filled "
                                    "with your blind assessment)."
                                ).classes("text-sm text-grey-7 mb-2")

                                form_container = ui.column().classes("w-full")
                                form_refs = build_revision_form(
                                    blind_annotation, form_container
                                )

                                notes_input = ui.textarea(
                                    "Why did you change (or keep) your assessment?",
                                    value="",
                                ).classes("w-full mt-2").props(
                                    "outlined dense rows=3"
                                )

                                with ui.row().classes("w-full gap-4 mt-4"):

                                    async def submit_revision() -> None:
                                        revised = harvest_form(form_refs)
                                        elapsed = (
                                            time.monotonic() - revision_start
                                        )
                                        notes = (notes_input.value or "").strip()[
                                            :500
                                        ]

                                        try:
                                            db.save_revised_annotation(
                                                user_id=user_id,
                                                pmid=pmid,
                                                annotation=revised,
                                                time_seconds=elapsed,
                                                revision_notes=notes,
                                                ai_annotation=ai_annotation,
                                            )
                                            ui.notify(
                                                "Assessment saved!",
                                                type="positive",
                                            )
                                            ui.navigate.to("/dashboard")
                                        except Exception:
                                            logger.exception(
                                                "Error saving revision for %s",
                                                pmid,
                                            )
                                            ui.notify(
                                                "Error saving. Please try again.",
                                                type="negative",
                                            )

                                    async def keep_original() -> None:
                                        elapsed = (
                                            time.monotonic() - revision_start
                                        )
                                        notes = (notes_input.value or "").strip()[
                                            :500
                                        ]

                                        try:
                                            db.save_revised_annotation(
                                                user_id=user_id,
                                                pmid=pmid,
                                                annotation=blind_annotation,
                                                time_seconds=elapsed,
                                                revision_notes=notes
                                                or "Kept original assessment",
                                                ai_annotation=ai_annotation,
                                            )
                                            ui.notify(
                                                "Original assessment kept!",
                                                type="positive",
                                            )
                                            ui.navigate.to("/dashboard")
                                        except Exception:
                                            logger.exception(
                                                "Error saving for %s", pmid
                                            )
                                            ui.notify(
                                                "Error saving. Please try again.",
                                                type="negative",
                                            )

                                    ui.button(
                                        "Submit Revision",
                                        on_click=submit_revision,
                                    ).classes("flex-1").props(
                                        "color=primary size=lg"
                                    )
                                    ui.button(
                                        "Keep My Original",
                                        on_click=keep_original,
                                    ).classes("flex-1").props(
                                        "color=secondary size=lg outline"
                                    )

                        with ui.tab_panel(tab_paper):
                            if paper:
                                paper_container = ui.column().classes("w-full")
                                render_paper(paper, paper_container)
                            else:
                                ui.label("Paper not found.")

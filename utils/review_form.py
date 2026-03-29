"""
Structured assessment form for human review.

Renders a form matching the LLM annotation JSON schema so that human
reviewers produce structured output identical to model output.  The form
is pre-populated from the model's annotation and the reviewer corrects
as needed.

Used by review_gui.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from nicegui import ui

logger = logging.getLogger(__name__)

SEVERITY_OPTIONS = ["none", "low", "moderate", "high", "critical"]
SPIN_LEVEL_OPTIONS = ["none", "low", "moderate", "high"]
OUTCOME_TYPE_OPTIONS = ["patient_centred", "surrogate", "composite", "unclear"]
FUNDING_TYPE_OPTIONS = ["industry", "public", "mixed", "not_reported", "unclear"]
CONFIDENCE_OPTIONS = ["low", "medium", "high"]


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict-like object."""
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _checkbox_row(
    label: str, key: str, domain_data: dict, refs: dict
) -> None:
    """Create a labelled checkbox bound to refs[key]."""
    val = bool(_safe_get(domain_data, key, False))
    refs[key] = ui.checkbox(label, value=val)


def _severity_select(
    domain_data: dict, refs: dict, options: list[str] | None = None
) -> None:
    """Create a severity dropdown bound to refs['severity']."""
    options = options or SEVERITY_OPTIONS
    val = _safe_get(domain_data, "severity", options[0])
    if val not in options:
        val = options[0]
    refs["severity"] = ui.select(
        options, value=val, label="Severity"
    ).classes("min-w-[10rem]")


def _evidence_quotes(domain_data: dict, refs: dict) -> None:
    """Create an evidence quotes textarea bound to refs['evidence_quotes']."""
    quotes = _safe_get(domain_data, "evidence_quotes", [])
    if isinstance(quotes, list):
        text = "\n".join(str(q) for q in quotes)
    else:
        text = str(quotes) if quotes else ""
    refs["evidence_quotes"] = ui.textarea(
        "Evidence quotes (one per line)", value=text
    ).classes("w-full").props("outlined dense rows=2")


def _build_statistical_card(
    ann: dict, refs: dict
) -> None:
    """Build the Statistical Reporting domain card."""
    data = _safe_get(ann, "statistical_reporting", {})
    refs["statistical_reporting"] = r = {}
    with ui.expansion("1. Statistical Reporting", icon="bar_chart").classes(
        "w-full"
    ).props("dense default-opened"):
        with ui.row().classes("w-full flex-wrap gap-x-4 gap-y-1"):
            _checkbox_row("Relative only", "relative_only", data, r)
            _checkbox_row("Absolute reported", "absolute_reported", data, r)
            _checkbox_row("NNT reported", "nnt_reported", data, r)
            _checkbox_row("Baseline risk reported", "baseline_risk_reported", data, r)
            _checkbox_row("Selective p-values", "selective_p_values", data, r)
            _checkbox_row("Subgroup emphasis", "subgroup_emphasis", data, r)
        with ui.row().classes("items-end gap-4"):
            _severity_select(data, r)
        _evidence_quotes(data, r)


def _build_spin_card(ann: dict, refs: dict) -> None:
    """Build the Spin Assessment domain card."""
    data = _safe_get(ann, "spin", {})
    refs["spin"] = r = {}
    with ui.expansion("2. Spin", icon="rotate_right").classes(
        "w-full"
    ).props("dense"):
        spin_val = _safe_get(data, "spin_level", "none")
        if spin_val not in SPIN_LEVEL_OPTIONS:
            spin_val = "none"
        r["spin_level"] = ui.select(
            SPIN_LEVEL_OPTIONS, value=spin_val, label="Spin level (Boutron)"
        ).classes("min-w-[10rem]")
        with ui.row().classes("w-full flex-wrap gap-x-4 gap-y-1"):
            _checkbox_row(
                "Conclusion matches results",
                "conclusion_matches_results", data, r,
            )
            _checkbox_row(
                "Causal language from observational",
                "causal_language_from_observational", data, r,
            )
            _checkbox_row(
                "Focus on secondary when primary NS",
                "focus_on_secondary_when_primary_ns", data, r,
            )
            _checkbox_row(
                "Inappropriate extrapolation",
                "inappropriate_extrapolation", data, r,
            )
            _checkbox_row("Title spin", "title_spin", data, r)
        with ui.row().classes("items-end gap-4"):
            _severity_select(data, r)
        _evidence_quotes(data, r)


def _build_outcome_card(ann: dict, refs: dict) -> None:
    """Build the Outcome Reporting domain card."""
    data = _safe_get(ann, "outcome_reporting", {})
    refs["outcome_reporting"] = r = {}
    with ui.expansion("3. Outcome Reporting", icon="assignment").classes(
        "w-full"
    ).props("dense"):
        ot_val = _safe_get(data, "primary_outcome_type", "unclear")
        if ot_val not in OUTCOME_TYPE_OPTIONS:
            ot_val = "unclear"
        r["primary_outcome_type"] = ui.select(
            OUTCOME_TYPE_OPTIONS, value=ot_val, label="Primary outcome type"
        ).classes("min-w-[12rem]")
        with ui.row().classes("w-full flex-wrap gap-x-4 gap-y-1"):
            _checkbox_row(
                "Surrogate without validation",
                "surrogate_without_validation", data, r,
            )
            _checkbox_row(
                "Composite not disaggregated",
                "composite_not_disaggregated", data, r,
            )
        with ui.row().classes("items-end gap-4"):
            _severity_select(data, r)
        _evidence_quotes(data, r)


def _build_coi_card(ann: dict, refs: dict) -> None:
    """Build the Conflict of Interest domain card."""
    data = _safe_get(ann, "conflict_of_interest", {})
    refs["conflict_of_interest"] = r = {}
    with ui.expansion("4. Conflict of Interest", icon="monetization_on").classes(
        "w-full"
    ).props("dense"):
        ft_val = _safe_get(data, "funding_type", "unclear")
        if ft_val not in FUNDING_TYPE_OPTIONS:
            ft_val = "unclear"
        r["funding_type"] = ui.select(
            FUNDING_TYPE_OPTIONS, value=ft_val, label="Funding type"
        ).classes("min-w-[10rem]")
        with ui.row().classes("w-full flex-wrap gap-x-4 gap-y-1"):
            _checkbox_row(
                "Funding disclosed in abstract",
                "funding_disclosed_in_abstract", data, r,
            )
            _checkbox_row(
                "Industry author affiliations",
                "industry_author_affiliations", data, r,
            )
            _checkbox_row("COI disclosed", "coi_disclosed", data, r)
        with ui.row().classes("items-end gap-4"):
            _severity_select(data, r)


def _build_methodology_card(ann: dict, refs: dict) -> None:
    """Build the Methodological Red Flags domain card."""
    data = _safe_get(ann, "methodology", {})
    refs["methodology"] = r = {}
    with ui.expansion("5. Methodology", icon="science").classes(
        "w-full"
    ).props("dense"):
        with ui.row().classes("w-full flex-wrap gap-x-4 gap-y-1"):
            _checkbox_row(
                "Inappropriate comparator",
                "inappropriate_comparator", data, r,
            )
            _checkbox_row("Enrichment design", "enrichment_design", data, r)
            _checkbox_row("Per-protocol only", "per_protocol_only", data, r)
            _checkbox_row("Premature stopping", "premature_stopping", data, r)
            _checkbox_row("Short follow-up", "short_follow_up", data, r)
        with ui.row().classes("items-end gap-4"):
            _severity_select(data, r)
        _evidence_quotes(data, r)


def _build_overall_section(ann: dict, refs: dict) -> None:
    """Build the overall assessment section."""
    refs["overall"] = r = {}
    with ui.expansion("Overall Assessment", icon="summarize").classes(
        "w-full"
    ).props("dense default-opened"):
        with ui.row().classes("items-end gap-4 w-full"):
            sev_val = _safe_get(ann, "overall_severity", "none")
            if sev_val not in SEVERITY_OPTIONS:
                sev_val = "none"
            r["overall_severity"] = ui.select(
                SEVERITY_OPTIONS, value=sev_val, label="Overall severity"
            ).classes("min-w-[10rem]")

            prob_val = _safe_get(ann, "overall_bias_probability", 0.0)
            try:
                prob_val = float(prob_val)
            except (TypeError, ValueError):
                prob_val = 0.0
            r["overall_bias_probability"] = ui.slider(
                min=0.0, max=1.0, step=0.05, value=prob_val
            ).props("label-always").classes("min-w-[12rem]")
            ui.label().bind_text_from(
                r["overall_bias_probability"], "value",
                backward=lambda v: f"Bias prob: {v:.2f}",
            )

            conf_val = _safe_get(ann, "confidence", "medium")
            if conf_val not in CONFIDENCE_OPTIONS:
                conf_val = "medium"
            r["confidence"] = ui.select(
                CONFIDENCE_OPTIONS, value=conf_val, label="Confidence"
            ).classes("min-w-[8rem]")

        r["reasoning"] = ui.textarea(
            "Reasoning", value=_safe_get(ann, "reasoning", "")
        ).classes("w-full").props("outlined dense rows=3")

        vs = _safe_get(ann, "recommended_verification_steps", [])
        if isinstance(vs, list):
            vs_text = "\n".join(str(s) for s in vs)
        else:
            vs_text = str(vs) if vs else ""
        r["recommended_verification_steps"] = ui.textarea(
            "Verification steps (one per line)", value=vs_text
        ).classes("w-full").props("outlined dense rows=2")


def build_review_form(
    ann: dict,
    container: ui.element,
) -> dict:
    """Build the full structured review form inside *container*.

    Args:
        ann: The model's annotation dict (pre-populates the form).
        container: NiceGUI element to build the form inside.

    Returns:
        A *refs* dict whose keys map to NiceGUI widget references.
        Call ``collect_form_data(refs)`` to harvest the JSON.
    """
    refs: dict = {}
    container.clear()
    with container:
        _build_statistical_card(ann, refs)
        _build_spin_card(ann, refs)
        _build_outcome_card(ann, refs)
        _build_coi_card(ann, refs)
        _build_methodology_card(ann, refs)
        _build_overall_section(ann, refs)
    return refs


def _collect_domain(refs: dict, domain_key: str) -> dict:
    """Collect form values from a single domain's widget refs."""
    r = refs.get(domain_key, {})
    result = {}
    for key, widget in r.items():
        if key == "evidence_quotes":
            text = widget.value or ""
            result[key] = [q.strip() for q in text.split("\n") if q.strip()]
        elif hasattr(widget, "value"):
            result[key] = widget.value
    return result


def collect_form_data(refs: dict) -> dict:
    """Harvest all form widgets into a JSON-serializable annotation dict.

    Returns the same structure as the LLM annotation schema.
    """
    data: dict[str, Any] = {}
    for domain_key in (
        "statistical_reporting", "spin", "outcome_reporting",
        "conflict_of_interest", "methodology",
    ):
        data[domain_key] = _collect_domain(refs, domain_key)

    overall = refs.get("overall", {})
    for key in (
        "overall_severity", "confidence", "overall_bias_probability",
        "reasoning",
    ):
        widget = overall.get(key)
        if widget is not None and hasattr(widget, "value"):
            data[key] = widget.value

    vs_widget = overall.get("recommended_verification_steps")
    if vs_widget is not None:
        text = vs_widget.value or ""
        data["recommended_verification_steps"] = [
            s.strip() for s in text.split("\n") if s.strip()
        ]
    else:
        data["recommended_verification_steps"] = []

    return data

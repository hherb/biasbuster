"""Tool schemas for the v4 assessment agent.

OpenAI-style tool definitions exposed to the LLM in the assessment
agent loop. Extracted from assessment_agent.py to keep that module
under the 500-line guideline. See V4_AGENT_DESIGN.md §3 for the
full tool catalogue and the future tool list.

Each entry follows Anthropic/OpenAI's shared JSON Schema format:

    {
        "name": "<canonical tool name>",
        "description": "<what the tool does — the LLM reads this>",
        "input_schema": {<JSON Schema for arguments>},
    }

The descriptions are load-bearing — they are what the LLM reads
to decide when to call each tool. Changes here directly affect
agent behaviour, so edits require validation against the
calibration test set (scripts/validate_v4_agent.py).
"""
from __future__ import annotations

from typing import Any


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "run_mechanical_assessment",
        "description": (
            "Apply the codified BiasBuster rules to the extracted paper facts "
            "and return the draft assessment with full provenance. ALWAYS call "
            "this FIRST on every paper. The response includes per-domain "
            "severities, the flags that drove each severity, and a "
            "`domain_overridable` map telling you which severities are "
            "structural (non-overridable) vs which you may contextually override."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "check_clinicaltrials",
        "description": (
            "Look up the paper's trial registration on ClinicalTrials.gov to "
            "verify registered outcomes vs reported outcomes (detects outcome "
            "switching). Call when you suspect selective outcome reporting. "
            "Optional arguments: the NCT id and/or title. If neither is given "
            "the tool extracts an NCT id from the abstract."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nct_id": {"type": "string", "description": "NCT registration id, if known"},
                "title": {"type": "string", "description": "Paper title, for title-search fallback"},
            },
            "required": [],
        },
    },
    {
        "name": "check_open_payments",
        "description": (
            "Search CMS Open Payments for industry payment records to the "
            "paper's authors. Call when COI disclosure appears incomplete or "
            "the paper lists prominent authors with industry ties."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "authors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Author names (up to 3). Defaults to first 3 from paper metadata.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_orcid",
        "description": (
            "Fetch author affiliation histories from ORCID. Use to verify "
            "undisclosed industry ties."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "authors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Author names. Defaults to first 3 from paper metadata.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_europmc_funding",
        "description": (
            "Query Europe PMC for grant and funder metadata the paper may "
            "not have disclosed. Useful when the funding statement is ambiguous."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "check_retraction_status",
        "description": (
            "Check PubMed and Crossref for post-publication retraction notices. "
            "Call only when you have specific reason to suspect retraction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_effect_size_audit",
        "description": (
            "Re-run the effect-size auditor heuristic for a second opinion on "
            "inflated_effect_sizes or baseline_risk_reported flags. Useful when "
            "the mechanical assessment flags an effect size as inflated but you "
            "want to verify against the raw quotes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

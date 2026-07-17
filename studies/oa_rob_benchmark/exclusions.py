"""Manually-verified exclusions for the OA-first RoB benchmark.

Trials removed after human review despite passing the automated litmus test,
keyed by trial PMID with a documented reason. The ingest skips these before
admitting, so re-runs stay reproducible and the operator's exclusion decision
is never silently undone (a bare DB delete would be re-admitted on the next
ingest). Mirrors ``studies.eisele_metzger_replication.exclusions.WRONG_PAPER_RCTS``.

These are curation decisions, NOT automated litmus rules — each entry records
*why a human excluded it*, so the benchmark's inclusion set stays auditable.
"""
from __future__ import annotations

#: trial PMID → human-readable exclusion reason.
MANUAL_EXCLUSIONS: dict[str, str] = {
    # Secondary analysis of two influenza-immunization RCTs, not a primary
    # randomized trial — RoB 2 applies to the primary trial, and PubMed
    # correctly declined to tag it a trial (it surfaced only as a
    # source-asserted admit). Verified by H. Herb, 2026-07-18.
    "28984909": "secondary analysis of influenza-immunization RCT data, not a "
                "primary randomized trial",
}

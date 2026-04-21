"""Cochrane Risk of Bias 2 (RoB 2) methodology.

Registers the ``cochrane_rob2`` slug in the methodology registry. The
methodology applies to parallel-group randomized controlled trials only
(cluster-RCT and crossover RCT variants are separate tools per the
Cochrane Handbook and are not yet registered).

Orchestration shape is ``decomposed_full_text``: per-domain LLM calls
driven by :class:`assessor.CochraneRoB2Assessor`, with a deterministic
worst-wins rollup in :mod:`algorithm`. Full text is required.
"""

from __future__ import annotations

from typing import Any, Optional

from biasbuster.methodologies.base import Methodology
from biasbuster.methodologies.registry import register

from .applicability import check_applicability
from .prompts import build_system_prompt
from .schema import Assessment, RoB2Assessment

METHODOLOGY_VERSION: str = "rob2-2019"
"""RoB 2 prompt/schema version tag.

Stored on every ``cochrane_rob2`` annotation row so prompt-set iterations
(Step 8) can be distinguished without schema migration. Bump this when
the per-domain prompts or the algorithm change materially.
"""


def build_user_message(
    *,
    paper: dict,
    enrichment: Optional[dict] = None,
    sections: Optional[list] = None,
    extraction: Optional[dict] = None,
    domain: Optional[str] = None,
) -> str:
    """Return a placeholder user message for registry-level callers.

    The real per-domain user messages are built inside
    :class:`assessor.CochraneRoB2Assessor` using the Stage-1 extraction,
    not through this call. This function exists because the
    :class:`Methodology` protocol requires it; callers that have only
    the paper metadata (no extraction yet) get the placeholder, and the
    assessor's own builder takes over once extraction has run.
    """
    del enrichment, sections, extraction, domain  # unused in the placeholder
    pmid = str(paper.get("pmid", ""))
    title = str(paper.get("title", ""))
    return f"PMID: {pmid}\nTitle: {title}"


def parse_output(raw: str, stage: str) -> Optional[dict]:
    """Parse a single RoB 2 domain call's LLM response into a dict.

    Delegates to the assessor's parser for consistency. Returns ``None``
    on parse failure so the caller can retry or abort.
    """
    from .assessor import _loose_parse_json, _parse_domain_response

    # The assessor's parser needs the domain slug to validate against.
    # Stage names look like ``domain_<slug>`` so we decode back.
    if stage.startswith("domain_"):
        domain_slug = stage[len("domain_"):]
        parsed = _parse_domain_response(raw, domain_slug, pmid="")
        if parsed is None:
            return None
        from dataclasses import asdict
        return asdict(parsed)

    # Non-domain stages (extract_section, synthesize) don't have a
    # well-defined parser at the methodology level — Step 8 will fill
    # these in if/when they're needed standalone.
    return _loose_parse_json(raw)


def aggregate(domain_judgements: dict) -> dict:
    """Aggregate per-domain judgements to an outcome-level overall.

    Thin wrapper over :func:`algorithm.aggregate_outcome` that accepts
    the methodology-protocol-shaped ``dict[str, RoB2DomainJudgement]``
    input and returns a ``dict`` (not a dataclass) so it round-trips
    through JSON cleanly.
    """
    from dataclasses import asdict

    from .algorithm import aggregate_outcome
    overall = aggregate_outcome(domain_judgements)
    return {
        "overall_judgement": overall,
        "domains": {
            slug: asdict(d) for slug, d in domain_judgements.items()
        },
    }


def evaluation_mapping_to_ground_truth(paper: dict) -> Optional[dict]:
    """Return the stored Cochrane expert RoB 2 ratings for this paper, if any.

    The ``papers`` table already has hard-coded columns from Cochrane
    ingestion (``randomization_bias``, ``deviation_bias``,
    ``missing_outcome_bias``, ``measurement_bias``, ``reporting_bias``,
    ``overall_rob``). This is the authoritative ground truth for the
    Step 9 faithfulness evaluation: expert-assigned RoB 2 in the same
    vocabulary as our methodology's output.

    Returns ``None`` when any required field is missing, so callers
    know they cannot score this paper.
    """
    required_pairs = (
        ("randomization", paper.get("randomization_bias")),
        ("deviations_from_interventions", paper.get("deviation_bias")),
        ("missing_outcome_data", paper.get("missing_outcome_bias")),
        ("outcome_measurement", paper.get("measurement_bias")),
        ("selection_of_reported_result", paper.get("reporting_bias")),
    )
    if any(v is None or (isinstance(v, str) and not v.strip())
           for _, v in required_pairs):
        return None
    overall = paper.get("overall_rob")
    if not isinstance(overall, str) or not overall.strip():
        return None
    return {
        "overall": overall,
        "domains": {slug: val for slug, val in required_pairs},
    }


METHODOLOGY: Methodology = Methodology(
    name="cochrane_rob2",
    display_name="Cochrane Risk of Bias 2 (parallel-group RCTs)",
    version=METHODOLOGY_VERSION,
    applicable_study_designs=frozenset({"rct_parallel"}),
    requires_full_text=True,
    orchestration="decomposed_full_text",
    severity_rollup_levels=("low", "some_concerns", "high"),
    status="active",
    build_system_prompt=build_system_prompt,
    build_user_message=build_user_message,
    parse_output=parse_output,
    aggregate=aggregate,
    check_applicability=check_applicability,
    evaluation_mapping_to_ground_truth=evaluation_mapping_to_ground_truth,
    # Training export is biasbuster-only in v1; RoB 2 is evaluation-only.
    build_training_example=None,
    notes={
        "docs": "docs/ASSESSING_RISK_OF_BIAS.md#cochrane-rob-2",
        "handbook":
            "Cochrane Handbook for Systematic Reviews of Interventions, "
            "chapter 8 (Higgins et al., 2019+)",
    },
)


def _register_once() -> Methodology:
    """Idempotent self-registration (see biasbuster pass-through adapter)."""
    from biasbuster.methodologies.registry import REGISTRY

    if REGISTRY.get(METHODOLOGY.name) is METHODOLOGY:
        return METHODOLOGY
    return register(METHODOLOGY, replace=True)


_register_once()


# ---- Faithfulness harness integration ----------------------------------
#
# Declared here rather than as a sibling module because it's purely a
# pointer-to-functions binding — no behaviour. Kept alongside
# ``evaluation_mapping_to_ground_truth`` so future changes to slugs or
# rating vocab stay colocated with the mapping that feeds them.

_JUDGEMENT_ORDER: tuple[str, ...] = ("low", "some_concerns", "high")


def _load_prediction_view(ann: dict) -> Optional[dict]:
    """Collapse a stored RoB 2 annotation into the shared prediction shape.

    The stored JSON blob carries a per-outcome list (one judgement tree
    per outcome); the harness compares against a single per-paper
    verdict so we reduce to worst-wins across outcomes, matching
    :func:`algorithm.worst_case_across_outcomes`.
    """
    from .schema import ROB2_DOMAIN_SLUGS

    overall = ann.get("worst_across_outcomes") or ann.get("overall_severity")
    if not isinstance(overall, str) or overall not in _JUDGEMENT_ORDER:
        return None
    outcomes = ann.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        return None
    rank = {j: i for i, j in enumerate(_JUDGEMENT_ORDER)}
    by_domain: dict[str, str] = {}
    for outcome in outcomes:
        domains = outcome.get("domains") or {}
        for slug in ROB2_DOMAIN_SLUGS:
            dj = domains.get(slug)
            if not isinstance(dj, dict):
                continue
            current = dj.get("judgement")
            if current not in rank:
                continue
            prior = by_domain.get(slug)
            if prior is None or rank[current] > rank[prior]:
                by_domain[slug] = current
    if set(by_domain) != set(ROB2_DOMAIN_SLUGS):
        return None
    return {"overall": overall, "domains": by_domain}


def _build_faithfulness_spec() -> Any:
    """Construct the :class:`FaithfulnessSpec` on demand.

    Done lazily (inside a function rather than at module top-level) so
    importing ``biasbuster.methodologies.cochrane_rob2`` doesn't force
    a load of ``biasbuster.evaluation.methodology_faithfulness`` — that
    avoids a circular import and keeps the faithfulness tooling
    optional from the methodology's perspective.
    """
    from biasbuster.evaluation.methodology_faithfulness import (
        FaithfulnessSpec,
    )
    from .schema import ROB2_DOMAIN_DISPLAY, ROB2_DOMAIN_SLUGS

    return FaithfulnessSpec(
        methodology="cochrane_rob2",
        methodology_version=METHODOLOGY_VERSION,
        display_name="Cochrane RoB 2",
        judgement_order=_JUDGEMENT_ORDER,
        domain_slugs=ROB2_DOMAIN_SLUGS,
        domain_display=dict(ROB2_DOMAIN_DISPLAY),
        load_prediction_view=_load_prediction_view,
    )


def __getattr__(name: str) -> Any:
    """Lazy module-level attribute resolver (PEP 562).

    Why lazy: building :data:`FAITHFULNESS_SPEC` requires importing
    :mod:`biasbuster.evaluation.methodology_faithfulness`, which in
    turn imports ``biasbuster.database``. Doing that at methodology
    module-load time would:

    1. Force the evaluation stack to be importable before any
       methodology is even usable for annotation — unrelated
       subsystems would fail to load on minimal installs.
    2. Risk circular import with the evaluation harness, which uses
       :func:`get_spec` to load methodology specs back by slug
       (``importlib.import_module("biasbuster.methodologies.X")``).

    Python's import machinery calls module-level ``__getattr__`` for
    any name not already in the module namespace (PEP 562), including
    ``from biasbuster.methodologies.cochrane_rob2 import
    FAITHFULNESS_SPEC``. First access builds and caches the spec in
    ``globals()`` so subsequent lookups are plain attribute reads.

    Per-methodology pattern (see also :mod:`quadas_2`): declare the
    lazy builder ``_build_faithfulness_spec()`` that imports
    :class:`FaithfulnessSpec` inside the function body, then return
    the spec from this ``__getattr__``. ``FAITHFULNESS_SPEC`` belongs
    in ``__all__`` so tooling (IDE completion, Sphinx) can still see
    the name.
    """
    if name == "FAITHFULNESS_SPEC":
        spec = _build_faithfulness_spec()
        globals()["FAITHFULNESS_SPEC"] = spec
        return spec
    raise AttributeError(name)


__all__ = [
    "Assessment",
    "FAITHFULNESS_SPEC",
    "METHODOLOGY",
    "METHODOLOGY_VERSION",
    "RoB2Assessment",
    "aggregate",
    "build_system_prompt",
    "build_user_message",
    "check_applicability",
    "evaluation_mapping_to_ground_truth",
    "parse_output",
]

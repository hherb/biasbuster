"""Biasbuster default pathway exposed as a :class:`Methodology`.

This module is a thin pass-through adapter. Every callable delegates to
existing biasbuster code so behaviour is byte-identical to the
pre-methodology codebase:

- prompts live in :mod:`biasbuster.prompts` (+ ``prompts_v3`` / ``prompts_v4``
  / ``prompts_v5a``); this module only dispatches *stage names* to those
  existing constants.
- user-message assembly delegates to
  :func:`biasbuster.annotators.build_user_message`.
- JSON output parsing delegates to :func:`biasbuster.annotators.parse_llm_json`.
- domain aggregation delegates to
  :func:`biasbuster.assessment.aggregate.assess_extraction`.

The biasbuster pathway is the only methodology with
``applicable_study_designs == ANY_STUDY_DESIGN`` (it runs on any paper,
including abstract-only) and ``requires_full_text=False``. It is also the
only methodology that, at the pipeline-CLI layer, may switch orchestration
at runtime via the existing ``--single-call`` / ``--full-text`` /
``--agentic`` / ``--decomposed`` flags — the ``orchestration`` attribute
below is the *default* used when none of those flags is given. Other
methodologies honour their declared ``orchestration`` unconditionally.
"""

from __future__ import annotations

from typing import Optional

from biasbuster import prompts as _prompts_v1
from biasbuster import prompts_v3 as _prompts_v3
from biasbuster import prompts_v4 as _prompts_v4
from biasbuster import prompts_v5a as _prompts_v5a
from biasbuster.annotators import (
    build_user_message as _legacy_build_user_message,
    parse_llm_json as _legacy_parse_llm_json,
)
from biasbuster.assessment.aggregate import assess_extraction as _legacy_assess
from biasbuster.methodologies.base import ANY_STUDY_DESIGN, Methodology
from biasbuster.methodologies.registry import register

from .schema import BiasAssessment

# Stage → prompt constant. Stage names are used by the methodology-aware
# orchestration in ``BaseAnnotator`` (Step 4) to pick the right system
# prompt. All constants are kept where they are — the methodology module
# is a dispatch table, not a new home for prompt text (CLAUDE.md's
# "prompts.py as single source of truth" rule).
_STAGE_PROMPTS: dict[str, str] = {
    # v1: single-call abstract (extract + assess in one prompt)
    "single_call": _prompts_v1.ANNOTATION_SYSTEM_PROMPT,
    # v3: two-call abstract
    "extract_abstract": _prompts_v3.EXTRACTION_SYSTEM_PROMPT,
    "assess": _prompts_v3.ASSESSMENT_SYSTEM_PROMPT,
    # v3: full-text per-section extraction (shared by v3/v4/v5a full-text flows)
    "extract_section": _prompts_v3.SECTION_EXTRACTION_SYSTEM_PROMPT,
    # v4: agentic assessment with tool calls
    "assess_agentic": _prompts_v4.ASSESSMENT_AGENT_SYSTEM_PROMPT,
    # v5a: decomposed per-domain override + reasoning synthesis
    "domain_override": _prompts_v5a.DOMAIN_OVERRIDE_SYSTEM_PROMPT,
    "synthesize": _prompts_v5a.SYNTHESIS_SYSTEM_PROMPT,
    # Training export (used by export.py, not the annotators)
    "training": _prompts_v1.TRAINING_SYSTEM_PROMPT,
}

BIASBUSTER_METHODOLOGY_VERSION: str = "v5a"
"""The biasbuster default pathway's prompt/schema version tag.

Stored on every biasbuster annotation row so old and new prompt-set runs
can be distinguished without schema migrations. Bump this when prompts
or the aggregator materially change.
"""


def build_system_prompt(stage: str) -> str:
    """Return the biasbuster system prompt for the given orchestration stage.

    Raises:
        KeyError: If the stage is unknown. Fail loud — silent fallback to
            a wrong prompt would corrupt the annotation quietly.
    """
    try:
        return _STAGE_PROMPTS[stage]
    except KeyError as exc:
        raise KeyError(
            f"biasbuster methodology has no prompt for stage {stage!r}. "
            f"Known stages: {sorted(_STAGE_PROMPTS)}"
        ) from exc


def build_user_message(
    *,
    paper: dict,
    enrichment: Optional[dict] = None,
    sections: Optional[list] = None,
    extraction: Optional[dict] = None,
    domain: Optional[str] = None,
) -> str:
    """Delegate to the existing :func:`biasbuster.annotators.build_user_message`.

    The legacy helper takes ``(pmid, title, abstract, metadata)``. Its
    metadata dict already handles Cochrane RoB fields, effect-size audit,
    retraction reasons, etc. conditionally — we merge paper + enrichment
    into that metadata bag so the legacy helper sees the same inputs it
    always has.

    ``sections``, ``extraction``, ``domain`` kwargs are part of the
    Methodology protocol for symmetry with formal methodologies that
    need them; for the single-call biasbuster pathway they are unused
    (those flows invoke build_user_message at different orchestration
    stages via other paths handled in Step 4).
    """
    metadata = dict(paper)
    if enrichment:
        metadata.update(enrichment)
    return _legacy_build_user_message(
        pmid=str(paper.get("pmid", "")),
        title=str(paper.get("title", "")),
        abstract=str(paper.get("abstract", "")),
        metadata=metadata,
    )


def parse_output(raw: str, stage: str) -> Optional[dict]:
    """Parse LLM JSON output for a biasbuster stage.

    All biasbuster stages emit the same JSON shape (a biasbuster 5-domain
    annotation), so the ``stage`` argument is accepted for protocol
    symmetry but ignored here. Delegates to the shared repair+validate
    helper so behaviour is identical to the pre-methodology path.
    """
    del stage  # unused for biasbuster; kept for protocol symmetry
    return _legacy_parse_llm_json(raw)


def aggregate(domain_judgements: dict) -> dict:
    """Run the deterministic biasbuster aggregator.

    Delegates to :func:`biasbuster.assessment.aggregate.assess_extraction`.
    The input is the v3 extraction JSON blob (or the ``extraction`` key of a
    stored annotation); the output includes per-domain severities, the
    overall rollup, and a ``_provenance`` audit trail.
    """
    return _legacy_assess(domain_judgements)


def check_applicability(
    paper: dict, enrichment: dict, full_text_available: bool
) -> tuple[bool, str]:
    """Biasbuster applies to any paper, abstract-only or full-text.

    This is the deliberate design choice that makes biasbuster the safe
    default pathway — it never refuses a paper. Returning ``(True, "")``
    short-circuits the more expensive checks inside
    :func:`biasbuster.methodologies.base.check_or_raise`.
    """
    del paper, enrichment, full_text_available
    return True, ""


def evaluation_mapping_to_ground_truth(paper: dict) -> Optional[dict]:
    """Placeholder: biasbuster → Cochrane-expert translation lives elsewhere.

    The biasbuster pathway's 5-domain output does not map 1:1 onto any
    formal methodology, so a lossy translation is required. That
    translation lives in :mod:`compare_vs_cochrane` (evaluation-only;
    not a methodology protocol concern). Returning ``None`` signals
    "no built-in ground-truth mapping; use the dedicated comparison
    script for biasbuster-vs-Cochrane evaluation."
    """
    del paper
    return None


def build_training_example(annotation: dict) -> Optional[dict]:
    """Hook used by ``export.py`` to render a training example.

    In Step 13 ``export.py`` will dispatch to this callable when
    ``--methodology=biasbuster``. For now, return the annotation
    unchanged — the existing export pipeline reads the ``annotations``
    table directly and does not go through this hook yet.
    """
    return annotation


METHODOLOGY: Methodology = Methodology(
    name="biasbuster",
    display_name="Biasbuster default (5-domain heuristic)",
    version=BIASBUSTER_METHODOLOGY_VERSION,
    applicable_study_designs=frozenset({ANY_STUDY_DESIGN}),
    requires_full_text=False,
    # Default orchestration; the pipeline CLI may override via the
    # existing --single-call/--full-text/--agentic/--decomposed flags
    # (Step 5). Non-biasbuster methodologies honour this field strictly.
    orchestration="decomposed_full_text",
    severity_rollup_levels=("none", "low", "moderate", "high", "critical"),
    status="active",
    build_system_prompt=build_system_prompt,
    build_user_message=build_user_message,
    parse_output=parse_output,
    aggregate=aggregate,
    check_applicability=check_applicability,
    evaluation_mapping_to_ground_truth=evaluation_mapping_to_ground_truth,
    build_training_example=build_training_example,
    notes={
        "docs": "docs/ASSESSING_RISK_OF_BIAS.md#biasbuster",
        "prompts": (
            "biasbuster/prompts.py (v1), prompts_v3.py (two-call), "
            "prompts_v4.py (agentic), prompts_v5a.py (decomposed)"
        ),
    },
)


def _register_once() -> Methodology:
    """Register the biasbuster methodology, tolerating repeat imports.

    Scenarios this handles:

    1. First app import: registry has no ``biasbuster`` entry → register.
    2. Re-import from the module cache: Python serves the cached module
       without re-executing the body, so this function does not fire
       again. The existing REGISTRY entry stays untouched. No-op.
    3. After ``clear_registry_for_testing()``: the registry dict is
       empty but ``sys.modules`` still holds this module. Because the
       module body doesn't re-run, ``get_methodology("biasbuster")``
       will raise :class:`UnknownMethodologyError` until a test
       explicitly re-imports via ``importlib.reload`` or re-registers.
       Test files currently avoid that combination.

    The ``replace=True`` branch exists to cover pathological cases (two
    import paths producing two different ``METHODOLOGY`` instances under
    the same slug). Because biasbuster is defined in exactly one module
    there is no legitimate path that triggers replacement in production;
    treat a replacement as a warning sign, not a feature.
    """
    from biasbuster.methodologies.registry import REGISTRY

    if REGISTRY.get(METHODOLOGY.name) is METHODOLOGY:
        return METHODOLOGY
    return register(METHODOLOGY, replace=True)


_register_once()


# Re-export types for symmetry with sibling methodology subpackages.
__all__ = [
    "BIASBUSTER_METHODOLOGY_VERSION",
    "BiasAssessment",
    "METHODOLOGY",
    "aggregate",
    "build_system_prompt",
    "build_training_example",
    "build_user_message",
    "check_applicability",
    "evaluation_mapping_to_ground_truth",
    "parse_output",
]

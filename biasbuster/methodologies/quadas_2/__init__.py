"""QUADAS-2 methodology — diagnostic-accuracy studies.

Registers the ``quadas_2`` slug in the methodology registry. Applies
to primary diagnostic-accuracy studies only; review-level tools (ROBIS,
AMSTAR 2) and intervention-trial tools (Cochrane RoB 2) are separate.

Orchestration shape is ``decomposed_full_text``: per-domain LLM calls
driven by :class:`assessor.QUADAS2Assessor`, with a two-axis worst-wins
rollup (bias + applicability) in :mod:`algorithm`. Full text is required.
"""

from __future__ import annotations

from typing import Any, Optional

from biasbuster.methodologies.base import Methodology
from biasbuster.methodologies.registry import register

from .applicability import check_applicability
from .prompts import build_system_prompt
from .schema import Assessment, QUADAS2Assessment

METHODOLOGY_VERSION: str = "quadas2-2011"
"""QUADAS-2 prompt/schema version tag (Whiting et al. 2011, PMID 22007046)."""


def build_user_message(
    *,
    paper: dict,
    enrichment: Optional[dict] = None,
    sections: Optional[list] = None,
    extraction: Optional[dict] = None,
    domain: Optional[str] = None,
) -> str:
    """Placeholder user message for protocol symmetry.

    Per-domain user messages are built inside
    :class:`assessor.QUADAS2Assessor` using the Stage-1 extraction.
    """
    del enrichment, sections, extraction, domain
    pmid = str(paper.get("pmid", ""))
    title = str(paper.get("title", ""))
    return f"PMID: {pmid}\nTitle: {title}"


def parse_output(raw: str, stage: str) -> Optional[dict]:
    """Parse a single QUADAS-2 domain response into a dict.

    Stage format is ``domain_<slug>`` for domain calls. Other stages
    (extract_section) pass the raw blob through the tolerant JSON
    parser for debugging convenience.
    """
    from .assessor import _loose_parse_json, _parse_domain_response

    if stage.startswith("domain_"):
        domain_slug = stage[len("domain_"):]
        parsed = _parse_domain_response(raw, domain_slug, pmid="")
        if parsed is None:
            return None
        from dataclasses import asdict
        return asdict(parsed)
    return _loose_parse_json(raw)


def aggregate(domain_judgements: dict) -> dict:
    """Aggregate per-domain judgements to paper-level worst-bias + worst-applicability."""
    from dataclasses import asdict

    from .algorithm import worst_applicability, worst_bias
    return {
        "worst_bias": worst_bias(domain_judgements),
        "worst_applicability": worst_applicability(domain_judgements),
        "domains": {
            slug: asdict(d) for slug, d in domain_judgements.items()
        },
    }


def evaluation_mapping_to_ground_truth(paper: dict) -> Optional[dict]:
    """Return stored QUADAS-2 expert ratings for this paper, if any.

    The ``papers`` table currently only carries Cochrane RoB 2 columns
    (``overall_rob`` and the five ``*_bias`` columns) — QUADAS-2 expert
    ratings would live somewhere separate. For now this returns ``None``
    in all cases; when a QUADAS-2 gold-standard ingestion path is
    added (sidecar JSON, a dedicated ``quadas2_expert_ratings`` table,
    etc.) this function gains the code path that reads from it.

    The Step-9-style faithfulness harness handles ``None`` gracefully —
    the report's "paired" count will be zero and the Markdown output
    notes the missing ground truth, rather than crashing.
    """
    del paper
    return None


METHODOLOGY: Methodology = Methodology(
    name="quadas_2",
    display_name="QUADAS-2 (diagnostic-accuracy studies)",
    version=METHODOLOGY_VERSION,
    applicable_study_designs=frozenset({"diagnostic_accuracy"}),
    requires_full_text=True,
    orchestration="decomposed_full_text",
    severity_rollup_levels=("low", "unclear", "high"),
    status="active",
    build_system_prompt=build_system_prompt,
    build_user_message=build_user_message,
    parse_output=parse_output,
    aggregate=aggregate,
    check_applicability=check_applicability,
    evaluation_mapping_to_ground_truth=evaluation_mapping_to_ground_truth,
    build_training_example=None,
    notes={
        "docs": "docs/ASSESSING_RISK_OF_BIAS.md#quadas-2",
        "reference":
            "Whiting PF et al. QUADAS-2: a revised tool for the quality "
            "assessment of diagnostic accuracy studies. Ann Intern Med. "
            "2011;155(8):529-536. PMID 22007046.",
    },
)


def _register_once() -> Methodology:
    """Idempotent self-registration."""
    from biasbuster.methodologies.registry import REGISTRY

    if REGISTRY.get(METHODOLOGY.name) is METHODOLOGY:
        return METHODOLOGY
    return register(METHODOLOGY, replace=True)


_register_once()


# ---- Faithfulness harness integration ----------------------------------
#
# v1 scores the bias dimension only. The applicability dimension on
# domains 1-3 is stored in the annotation JSON but the currently
# ingested expert ratings (from JATS Table-2 parsing) don't carry
# applicability. When that changes the harness gets a parallel set of
# :class:`JudgementSeries`; for now ``load_prediction_view`` simply
# drops applicability.

_JUDGEMENT_ORDER: tuple[str, ...] = ("low", "unclear", "high")


def _load_prediction_view(ann: dict) -> Optional[dict]:
    """Collapse a stored QUADAS-2 annotation into the shared prediction shape.

    Unlike RoB 2, QUADAS-2 assessments are per-paper (no per-outcome
    split), so reduction is a flat read of ``worst_bias`` and each
    domain's ``bias_rating``.
    """
    from .schema import QUADAS2_DOMAIN_SLUGS

    overall = ann.get("worst_bias") or ann.get("overall_severity")
    if not isinstance(overall, str) or overall not in _JUDGEMENT_ORDER:
        return None
    domains_blob = ann.get("domains")
    if not isinstance(domains_blob, dict):
        return None
    by_domain: dict[str, str] = {}
    for slug in QUADAS2_DOMAIN_SLUGS:
        dj = domains_blob.get(slug)
        if not isinstance(dj, dict):
            return None
        bias = dj.get("bias_rating")
        if bias not in _JUDGEMENT_ORDER:
            return None
        by_domain[slug] = bias
    return {"overall": overall, "domains": by_domain}


def _build_faithfulness_spec() -> Any:
    """Lazy construction — see cochrane_rob2.__init__ for the rationale."""
    from biasbuster.evaluation.methodology_faithfulness import (
        FaithfulnessSpec,
    )
    from .schema import QUADAS2_DOMAIN_DISPLAY, QUADAS2_DOMAIN_SLUGS

    return FaithfulnessSpec(
        methodology="quadas_2",
        methodology_version=METHODOLOGY_VERSION,
        display_name="QUADAS-2",
        judgement_order=_JUDGEMENT_ORDER,
        domain_slugs=QUADAS2_DOMAIN_SLUGS,
        domain_display=dict(QUADAS2_DOMAIN_DISPLAY),
        load_prediction_view=_load_prediction_view,
    )


def __getattr__(name: str) -> Any:
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
    "QUADAS2Assessment",
    "aggregate",
    "build_system_prompt",
    "build_user_message",
    "check_applicability",
    "evaluation_mapping_to_ground_truth",
    "parse_output",
]

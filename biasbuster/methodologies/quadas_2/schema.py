"""Typed output schema for the QUADAS-2 methodology.

QUADAS-2 (Whiting et al. 2011, PMID 22007046) is the standard tool for
assessing risk of bias in primary diagnostic-accuracy studies. It has
four domains and a 2-D output shape per domain:

- Bias rating: low / high / unclear
- Applicability rating: low / high / unclear (for the first 3 domains)

The fourth domain (flow and timing) has only a bias rating — it's a
process-quality check that doesn't carry a separate applicability
dimension per the QUADAS-2 tool. We encode that asymmetry in the
dataclass invariants below.

Scale note: QUADAS-2 uses a symmetric low/high/unclear triad rather
than Cochrane RoB 2's ordinal low/some_concerns/high. "unclear" is the
explicit "no information" bucket; it's not a midpoint between low and
high. That means the faithfulness metrics in :mod:`biasbuster.evaluation`
treat it as nominal, not ordinal — a low/high swap and a low/unclear
swap both count as disagreement with no relative severity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

#: Allowed answers to a QUADAS-2 signalling question. These mirror the
#: Whiting et al. template: Yes/No/Unclear, where "Yes" is favourable
#: (low risk) or unfavourable depending on the question wording.
QUADASSignallingAnswer = Literal["yes", "no", "unclear"]
VALID_QUADAS_SIGNALLING: frozenset[str] = frozenset(
    {"yes", "no", "unclear"}
)

#: Per-domain bias rating and applicability-concern rating both use the
#: same three-valued vocabulary.
QUADASRating = Literal["low", "high", "unclear"]
VALID_QUADAS_RATINGS: frozenset[str] = frozenset({"low", "high", "unclear"})

#: Canonical slugs for the four QUADAS-2 domains, in Whiting's order.
QUADAS2_DOMAIN_SLUGS: tuple[str, ...] = (
    "patient_selection",
    "index_test",
    "reference_standard",
    "flow_and_timing",
)

#: Human-readable display names matching the QUADAS-2 reporting template.
QUADAS2_DOMAIN_DISPLAY: dict[str, str] = {
    "patient_selection": "Patient Selection",
    "index_test": "Index Test",
    "reference_standard": "Reference Standard",
    "flow_and_timing": "Flow and Timing",
}

#: Domains that carry a separate applicability-concern rating. Domain 4
#: (flow and timing) is deliberately excluded per the QUADAS-2 tool —
#: process-quality concerns about how the study was conducted don't
#: translate into applicability of the findings to the review question.
QUADAS2_APPLICABILITY_DOMAINS: frozenset[str] = frozenset({
    "patient_selection",
    "index_test",
    "reference_standard",
})


@dataclass(frozen=True)
class QUADASEvidenceQuote:
    """A verbatim text fragment supporting a signalling-question answer."""

    text: str
    section: Optional[str] = None


@dataclass
class QUADAS2DomainJudgement:
    """Per-domain QUADAS-2 judgement.

    Carries the bias rating, the signalling-question answers that led to
    it, and (for the first three domains) a separate applicability
    rating. The ``applicability`` field is ``None`` on the flow-and-
    timing domain and *must* be set on the other three — enforced in
    ``__post_init__``.
    """

    domain: str
    signalling_answers: dict[str, QUADASSignallingAnswer]
    bias_rating: QUADASRating
    applicability: Optional[QUADASRating] = None
    justification: str = ""
    evidence_quotes: list[QUADASEvidenceQuote] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.domain not in QUADAS2_DOMAIN_SLUGS:
            raise ValueError(
                f"QUADAS2DomainJudgement.domain must be one of "
                f"{QUADAS2_DOMAIN_SLUGS}, got {self.domain!r}"
            )
        if self.bias_rating not in VALID_QUADAS_RATINGS:
            raise ValueError(
                f"QUADAS2DomainJudgement.bias_rating must be one of "
                f"{sorted(VALID_QUADAS_RATINGS)}, got {self.bias_rating!r}"
            )
        # Applicability is mandatory on domains 1-3, forbidden on domain 4.
        # Enforcing both directions catches prompt-engineering mistakes
        # that would otherwise silently produce malformed reports.
        if self.domain in QUADAS2_APPLICABILITY_DOMAINS:
            if self.applicability is None:
                raise ValueError(
                    f"QUADAS-2 domain {self.domain!r} requires an "
                    "applicability rating; got None."
                )
            if self.applicability not in VALID_QUADAS_RATINGS:
                raise ValueError(
                    f"QUADAS2DomainJudgement.applicability for "
                    f"{self.domain!r} must be one of "
                    f"{sorted(VALID_QUADAS_RATINGS)}, got "
                    f"{self.applicability!r}"
                )
        else:
            if self.applicability is not None:
                raise ValueError(
                    f"QUADAS-2 domain {self.domain!r} must not carry an "
                    f"applicability rating (got {self.applicability!r}); "
                    "flow-and-timing has no applicability dimension."
                )
        bad = {
            k: v for k, v in self.signalling_answers.items()
            if v not in VALID_QUADAS_SIGNALLING
        }
        if bad:
            raise ValueError(
                f"Invalid signalling answers for domain {self.domain!r}: "
                f"{bad}. Allowed: {sorted(VALID_QUADAS_SIGNALLING)}"
            )


@dataclass
class QUADAS2Assessment:
    """Top-level QUADAS-2 result for a single diagnostic-accuracy study.

    Unlike RoB 2, QUADAS-2 doesn't have per-outcome granularity — a
    diagnostic-accuracy study reports one test characterisation, and
    the assessment is one QUADAS-2 rating per domain for the study as
    a whole. So there's no outcomes list; just a flat 4-domain dict.
    """

    pmid: str
    domains: dict[str, QUADAS2DomainJudgement]
    methodology_version: str
    #: Worst bias rating across the 4 domains (for the DB overall_severity
    #: column). Simple "any high → high; any unclear (and no high) → unclear;
    #: all low → low" rollup per common QUADAS-2 reporting practice. Not
    #: a formal Cochrane-style algorithm — QUADAS-2 deliberately doesn't
    #: mandate one, leaving the overall to expert synthesis.
    worst_bias: QUADASRating
    #: Worst applicability rating across the three domains that carry it.
    worst_applicability: QUADASRating
    notes: str = ""

    def __post_init__(self) -> None:
        missing = set(QUADAS2_DOMAIN_SLUGS) - set(self.domains)
        if missing:
            raise ValueError(
                f"QUADAS2Assessment missing domain(s): {sorted(missing)}"
            )
        if self.worst_bias not in VALID_QUADAS_RATINGS:
            raise ValueError(
                f"QUADAS2Assessment.worst_bias must be one of "
                f"{sorted(VALID_QUADAS_RATINGS)}, got {self.worst_bias!r}"
            )
        if self.worst_applicability not in VALID_QUADAS_RATINGS:
            raise ValueError(
                "QUADAS2Assessment.worst_applicability must be one of "
                f"{sorted(VALID_QUADAS_RATINGS)}, got "
                f"{self.worst_applicability!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable dict representation for JSON storage."""
        return asdict(self)


# Alias for symmetric ``<methodology>.schema.Assessment`` access.
Assessment = QUADAS2Assessment

__all__ = [
    "Assessment",
    "QUADAS2_APPLICABILITY_DOMAINS",
    "QUADAS2_DOMAIN_DISPLAY",
    "QUADAS2_DOMAIN_SLUGS",
    "QUADAS2Assessment",
    "QUADAS2DomainJudgement",
    "QUADASEvidenceQuote",
    "QUADASRating",
    "QUADASSignallingAnswer",
    "VALID_QUADAS_RATINGS",
    "VALID_QUADAS_SIGNALLING",
]

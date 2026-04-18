"""Typed output schema for the Cochrane RoB 2 methodology.

Mirrors the RoB 2 guidance from docs/ASSESSING_RISK_OF_BIAS.md:

- Five bias domains, each assessed via signalling questions answered
  Y / PY / PN / N / NI (Yes / Probably Yes / Probably No / No / No Information).
- Each domain gets a judgement in {low, some_concerns, high}.
- A trial may have multiple outcomes; RoB 2 is formally per-outcome
  per-result. We preserve the list so the "reproduces Cochrane expert
  findings" evaluation (Step 9) can inspect per-outcome agreement.
- The overall judgement is the worst across outcomes and is surfaced
  into the ``annotations.overall_severity`` column for fast aggregate
  queries.

The JSON blob stored in the ``annotations.annotation`` column serialises
the full :class:`RoB2Assessment` — per-outcome domain ratings, signalling
answers, evidence quotes. Column-level aggregates (``overall_severity``,
``overall_bias_probability`` — ``None`` for RoB 2 since no probability is
emitted) exist for query convenience.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

# ---- Value types ---------------------------------------------------------

#: Allowed answers to a RoB 2 signalling question. The "No Information"
#: answer is explicit — the Cochrane algorithm treats it as a distinct
#: input (not coerced to "Probably No") when mapping signalling answers
#: to the domain judgement.
SignallingAnswer = Literal["Y", "PY", "PN", "N", "NI"]
VALID_SIGNALLING_ANSWERS: frozenset[str] = frozenset({"Y", "PY", "PN", "N", "NI"})

#: Per-domain and per-outcome overall judgement.
RoB2Judgement = Literal["low", "some_concerns", "high"]
VALID_JUDGEMENTS: frozenset[str] = frozenset({"low", "some_concerns", "high"})

#: Canonical slugs for the five RoB 2 bias domains, in the order the
#: Cochrane Handbook specifies. Order matters for per-domain reports and
#: matches the columns on the ``papers`` table (randomization_bias,
#: deviation_bias, missing_outcome_bias, measurement_bias,
#: reporting_bias) so ground-truth joins are unambiguous.
ROB2_DOMAIN_SLUGS: tuple[str, ...] = (
    "randomization",
    "deviations_from_interventions",
    "missing_outcome_data",
    "outcome_measurement",
    "selection_of_reported_result",
)

#: Human-readable display names matching the Cochrane Handbook headings.
ROB2_DOMAIN_DISPLAY: dict[str, str] = {
    "randomization": "Bias arising from the randomization process",
    "deviations_from_interventions":
        "Bias due to deviations from intended interventions",
    "missing_outcome_data": "Bias due to missing outcome data",
    "outcome_measurement": "Bias in measurement of the outcome",
    "selection_of_reported_result": "Bias in selection of the reported result",
}


# ---- Dataclasses --------------------------------------------------------

@dataclass(frozen=True)
class EvidenceQuote:
    """A verbatim text fragment supporting a signalling-question answer."""

    text: str
    section: Optional[str] = None  # "Methods", "Table 2", etc. when known


@dataclass
class RoB2DomainJudgement:
    """Per-domain RoB 2 assessment with signalling-question answers.

    The ``signalling_answers`` dict is free-form (question key → answer)
    because the number of signalling questions differs per domain (3–7
    per the Cochrane Handbook). The ``judgement`` is the deterministic
    rollup from the signalling answers via the Cochrane algorithm — it
    is *not* an independent LLM vote; it must be reproducible from the
    signalling inputs alone.
    """

    domain: str
    signalling_answers: dict[str, SignallingAnswer]
    judgement: RoB2Judgement
    justification: str = ""
    evidence_quotes: list[EvidenceQuote] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.domain not in ROB2_DOMAIN_SLUGS:
            raise ValueError(
                f"RoB2DomainJudgement.domain must be one of "
                f"{ROB2_DOMAIN_SLUGS}, got {self.domain!r}"
            )
        if self.judgement not in VALID_JUDGEMENTS:
            raise ValueError(
                f"RoB2DomainJudgement.judgement must be one of "
                f"{sorted(VALID_JUDGEMENTS)}, got {self.judgement!r}"
            )
        bad = {
            k: v for k, v in self.signalling_answers.items()
            if v not in VALID_SIGNALLING_ANSWERS
        }
        if bad:
            raise ValueError(
                f"Invalid signalling answers for domain {self.domain!r}: "
                f"{bad}. Allowed: {sorted(VALID_SIGNALLING_ANSWERS)}"
            )


@dataclass
class RoB2OutcomeJudgement:
    """RoB 2 assessment for a single (outcome, result) combination.

    A paper may have multiple outcomes (primary, key secondary) and each
    gets its own RoB 2 judgement per the Cochrane Handbook. For the MVP
    end-to-end path (Step 7) a single synthetic "primary outcome"
    placeholder is acceptable; Step 8's prompt iteration will teach the
    LLM to actually enumerate the outcomes.
    """

    outcome_label: str
    result_label: str
    domains: dict[str, RoB2DomainJudgement]
    overall_judgement: RoB2Judgement
    overall_rationale: str = ""

    def __post_init__(self) -> None:
        if self.overall_judgement not in VALID_JUDGEMENTS:
            raise ValueError(
                "RoB2OutcomeJudgement.overall_judgement must be one of "
                f"{sorted(VALID_JUDGEMENTS)}, got {self.overall_judgement!r}"
            )
        missing = set(ROB2_DOMAIN_SLUGS) - set(self.domains)
        if missing:
            raise ValueError(
                f"RoB2OutcomeJudgement missing domain(s): "
                f"{sorted(missing)}"
            )


@dataclass
class RoB2Assessment:
    """Top-level RoB 2 result for a paper: one or more outcome judgements.

    The ``worst_across_outcomes`` rollup is stored in the DB's
    ``overall_severity`` column for query convenience; the full
    per-outcome structure is preserved in the annotation JSON blob.
    """

    pmid: str
    outcomes: list[RoB2OutcomeJudgement]
    methodology_version: str
    worst_across_outcomes: RoB2Judgement
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError(
                "RoB2Assessment requires at least one RoB2OutcomeJudgement"
            )
        if self.worst_across_outcomes not in VALID_JUDGEMENTS:
            raise ValueError(
                "RoB2Assessment.worst_across_outcomes must be one of "
                f"{sorted(VALID_JUDGEMENTS)}, got "
                f"{self.worst_across_outcomes!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable dict representation for JSON storage.

        Uses ``dataclasses.asdict`` on the nested structure so dataclass
        types (including the frozen :class:`EvidenceQuote`) serialise
        cleanly without custom encoders.
        """
        return asdict(self)


# Alias for protocol symmetry (``<methodology>.schema.Assessment``).
Assessment = RoB2Assessment

__all__ = [
    "Assessment",
    "EvidenceQuote",
    "ROB2_DOMAIN_DISPLAY",
    "ROB2_DOMAIN_SLUGS",
    "RoB2Assessment",
    "RoB2DomainJudgement",
    "RoB2Judgement",
    "RoB2OutcomeJudgement",
    "SignallingAnswer",
    "VALID_JUDGEMENTS",
    "VALID_SIGNALLING_ANSWERS",
]

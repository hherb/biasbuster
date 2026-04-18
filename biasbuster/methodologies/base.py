"""Methodology protocol + guard helpers.

A *methodology* is a risk-of-bias assessment approach: the current biasbuster
pathway, Cochrane RoB 2, QUADAS-2, etc. Each methodology is modelled as a
:class:`Methodology` dataclass whose fields describe its declarative traits
(applicable study designs, severity vocabulary, orchestration shape) and
whose callable fields provide the methodology-specific logic (prompts,
user-message assembly, output parsing, aggregation, applicability check,
evaluation mapping).

Methodologies are registered in :mod:`biasbuster.methodologies.registry`
and looked up by their ``name`` slug.

Key design points:

- The orchestration layer (`BaseAnnotator` methods) is methodology-agnostic.
  It asks the methodology for prompts and parsers; it does not hard-code any
  methodology's structure.
- Transport layers (Anthropic SDK, OpenAI-compatible HTTP, bmlib) stay
  methodology-agnostic too — they only implement ``_call_llm()``.
- Stub methodologies are registered with ``status="stub"`` and raise
  :class:`NotImplementedError` from all callable fields; this lets the
  architecture carry known-but-unimplemented methodologies (ROBINS-I,
  PROBAST, ...) without scattering conditional imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

# The set of orchestration shapes a methodology can declare it uses. These
# correspond to the existing annotator methods on ``BaseAnnotator``; the
# dispatcher in ``pipeline.stage_annotate`` routes on this value.
OrchestrationShape = Literal[
    "single_call_abstract",
    "two_call_abstract",
    "two_call_full_text",
    "agentic_full_text",
    "decomposed_full_text",
]

MethodologyStatus = Literal["active", "stub"]

# Well-known study-design slugs emitted by
# :func:`biasbuster.methodologies.study_design.detect`. Methodologies
# declare their applicability against these strings.
STUDY_DESIGN_SLUGS: frozenset[str] = frozenset({
    "rct_parallel",
    "rct_crossover",
    "rct_cluster",
    "cohort",
    "case_control",
    "case_series",
    "diagnostic_accuracy",
    "systematic_review",
    "meta_analysis",
    "narrative_review",
    "unknown",
})

# Marker used by applicable_study_designs to mean "applies to any design".
# Used by the biasbuster default pathway which intentionally runs on any paper.
ANY_STUDY_DESIGN: str = "*"


# ---- Exceptions ----------------------------------------------------------

class MethodologyError(RuntimeError):
    """Base class for methodology-related failures."""


class UnknownMethodologyError(MethodologyError, KeyError):
    """Raised when a methodology slug is not in the registry."""


class DuplicateMethodologyError(MethodologyError):
    """Raised when two modules try to register the same methodology slug."""


class ApplicabilityError(MethodologyError):
    """Raised when the chosen methodology does not apply to the paper.

    Example: the user picked ``cochrane_rob2`` but the paper is a cohort
    study. The caller (``stage_annotate`` / ``annotate_single_paper``) should
    log the refusal with the detected design and skip the paper.
    """

    def __init__(self, methodology: str, detected_design: str, reason: str) -> None:
        self.methodology = methodology
        self.detected_design = detected_design
        self.reason = reason
        super().__init__(
            f"methodology '{methodology}' is not applicable to this paper "
            f"(detected study design: {detected_design!r}): {reason}"
        )


class FullTextRequiredError(MethodologyError):
    """Raised when a methodology demands full text but only the abstract is available.

    Formal methodologies (RoB 2, QUADAS-2) cannot be faithfully applied from
    the abstract alone. Only the biasbuster-default pathway runs from abstract.
    """

    def __init__(self, methodology: str, pmid: str) -> None:
        self.methodology = methodology
        self.pmid = pmid
        super().__init__(
            f"methodology '{methodology}' requires full text but only the "
            f"abstract is available for paper {pmid}. Fetch full text or "
            "use --methodology=biasbuster for abstract-only assessment."
        )


class UnsupportedOrchestrationError(MethodologyError):
    """Raised when a methodology's orchestration conflicts with CLI flags.

    Example: user passes ``--single-call`` together with
    ``--methodology=cochrane_rob2`` but RoB 2 only supports
    ``decomposed_full_text`` orchestration.
    """


# ---- Stub callable (for methodologies not yet implemented) --------------

def _stub_callable(methodology_name: str, method: str) -> Callable[..., Any]:
    """Build a callable that raises NotImplementedError when invoked.

    Used by the ``stub`` factory so a placeholder methodology can be
    registered (appearing in listings, applicability checks, etc.) while
    actual invocations fail loud.
    """

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(
            f"methodology '{methodology_name}' is registered but "
            f"'{method}' is not implemented yet. See the "
            "multi-methodology plan for the intended spec."
        )

    _raise.__name__ = f"_stub_{methodology_name}_{method}"
    return _raise


# ---- Methodology dataclass ----------------------------------------------

@dataclass(frozen=True)
class Methodology:
    """Declarative + behavioural description of a risk-of-bias methodology.

    Instances are registered via :func:`biasbuster.methodologies.registry.register`.
    Looked up by ``name``.

    Attributes are split into two groups:

    *Declarative* (used for routing, applicability, CLI display):
        ``name``, ``display_name``, ``version``, ``applicable_study_designs``,
        ``requires_full_text``, ``orchestration``, ``severity_rollup_levels``,
        ``status``.

    *Behavioural* (callables the orchestration layer invokes):
        ``build_system_prompt``, ``build_user_message``, ``parse_output``,
        ``aggregate``, ``check_applicability``,
        ``evaluation_mapping_to_ground_truth``, ``build_training_example``.

    The behavioural callables are intentionally plain callables (not bound
    methods) so each methodology submodule can provide them as free functions
    without forcing a class hierarchy.
    """

    name: str
    display_name: str
    version: str
    applicable_study_designs: frozenset[str]
    requires_full_text: bool
    orchestration: OrchestrationShape
    severity_rollup_levels: tuple[str, ...]
    status: MethodologyStatus

    build_system_prompt: Callable[[str], str]
    build_user_message: Callable[..., str]
    parse_output: Callable[[str, str], Optional[dict]]
    aggregate: Callable[[dict], dict]
    check_applicability: Callable[[dict, dict, bool], tuple[bool, str]]
    evaluation_mapping_to_ground_truth: Callable[[dict], Optional[dict]]

    # Optional: some methodologies will not produce training data in v1
    # (only biasbuster renders the alpaca/sharegpt format). `None` means
    # "not exportable as training data".
    build_training_example: Optional[Callable[[dict], Optional[dict]]] = None

    # Optional free-form metadata (e.g. tool-spec URL, Cochrane Handbook
    # chapter reference). Not used by the orchestration layer.
    notes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Declarative invariants — fail loud at registration time rather
        # than at annotation time.
        if not self.name:
            raise ValueError("Methodology.name must be non-empty")
        if self.status not in ("active", "stub"):
            raise ValueError(
                f"Methodology.status must be 'active' or 'stub', got "
                f"{self.status!r}"
            )
        if not self.severity_rollup_levels:
            raise ValueError(
                f"Methodology {self.name!r}: severity_rollup_levels cannot be empty"
            )
        bad_designs = {
            d for d in self.applicable_study_designs
            if d != ANY_STUDY_DESIGN and d not in STUDY_DESIGN_SLUGS
        }
        if bad_designs:
            raise ValueError(
                f"Methodology {self.name!r}: unknown study-design slugs "
                f"{sorted(bad_designs)}. Valid slugs: "
                f"{sorted(STUDY_DESIGN_SLUGS)} plus {ANY_STUDY_DESIGN!r}."
            )

    def applies_to(self, detected_design: str) -> bool:
        """True iff the methodology declares itself applicable to the design.

        Biasbuster-style methodologies that declare ``ANY_STUDY_DESIGN``
        always return True. Formal methodologies (RoB 2, QUADAS-2) return
        True only for their specific design slugs.
        """
        if ANY_STUDY_DESIGN in self.applicable_study_designs:
            return True
        return detected_design in self.applicable_study_designs


# ---- Applicability + full-text guard ------------------------------------

def check_or_raise(
    methodology: Methodology,
    paper: dict,
    enrichment: dict,
    *,
    full_text_available: bool,
    detected_design: str,
) -> None:
    """Raise if the methodology refuses this paper. No-op on success.

    Called once per paper from ``stage_annotate`` and
    ``annotate_single_paper`` before invoking the annotator. Order of
    checks matters: full-text requirement is checked before applicability
    so the error message is maximally specific.

    Args:
        methodology: The methodology the user selected.
        paper: Paper record (title, abstract, mesh_terms, source, ...).
        enrichment: Enrichment record (may be empty dict if enrich stage
            hasn't run).
        full_text_available: True iff full text sections are in hand.
        detected_design: Study-design slug from
            :func:`biasbuster.methodologies.study_design.detect`.

    Raises:
        FullTextRequiredError: If the methodology needs full text but only
            the abstract is available.
        ApplicabilityError: If the methodology doesn't apply to this design.
    """
    if methodology.requires_full_text and not full_text_available:
        raise FullTextRequiredError(methodology.name, paper.get("pmid", "?"))

    # Two-stage applicability check: the declarative field
    # (``applicable_study_designs``) is a first filter; the methodology's
    # own ``check_applicability`` callable has the last word and may
    # override the declarative decision (e.g. accept a cluster-RCT after
    # inspecting the paper more carefully, or reject an apparent RCT
    # that lacks a required field). Both branches below call
    # ``check_applicability`` so the methodology always gets a voice.
    if not methodology.applies_to(detected_design):
        ok, reason = methodology.check_applicability(
            paper, enrichment, full_text_available
        )
        if not ok:
            raise ApplicabilityError(
                methodology.name, detected_design, reason,
            )
        return

    # Declarative filter said yes; give the methodology a chance to
    # refuse based on finer-grained paper state.
    ok, reason = methodology.check_applicability(
        paper, enrichment, full_text_available
    )
    if not ok:
        raise ApplicabilityError(
            methodology.name, detected_design, reason,
        )

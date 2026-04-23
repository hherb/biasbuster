"""Tests for the methodology protocol, registry, and applicability guard.

These tests exercise the plumbing that every methodology (biasbuster,
cochrane_rob2, quadas_2, ...) plugs into — not any specific methodology's
behaviour. A minimal in-test fixture methodology ("fake") is used so the
real methodology submodules do not need to be imported yet.
"""

from __future__ import annotations

import pytest

from biasbuster.methodologies import (
    ANY_STUDY_DESIGN,
    STUDY_DESIGN_SLUGS,
    ApplicabilityError,
    DuplicateMethodologyError,
    FullTextRequiredError,
    Methodology,
    UnknownMethodologyError,
    clear_registry_for_testing,
    check_or_raise,
    get_methodology,
    list_active_methodologies,
    list_methodologies,
    register,
    study_design,
)


def _make_methodology(
    name: str = "fake",
    *,
    applicable: frozenset[str] = frozenset({"rct_parallel"}),
    requires_full_text: bool = True,
    status: str = "active",
    applicable_ok: bool = True,
    applicable_reason: str = "",
    severity: tuple[str, ...] = ("low", "high"),
) -> Methodology:
    def _noop_prompt(_stage: str) -> str:
        return ""

    def _noop_user_msg(**_kwargs: object) -> str:
        return ""

    def _noop_parse(_raw: str, _stage: str) -> dict | None:
        return {}

    def _noop_aggregate(_judgements: dict) -> dict:
        return {}

    def _applicability(_p: dict, _e: dict, _f: bool) -> tuple[bool, str]:
        return applicable_ok, applicable_reason

    def _ground_truth(_p: dict) -> dict | None:
        return None

    return Methodology(
        name=name,
        display_name=f"Fake {name}",
        version="v0",
        applicable_study_designs=applicable,
        requires_full_text=requires_full_text,
        orchestration="decomposed_full_text",
        severity_rollup_levels=severity,
        status=status,  # type: ignore[arg-type]
        build_system_prompt=_noop_prompt,
        build_user_message=_noop_user_msg,
        parse_output=_noop_parse,
        aggregate=_noop_aggregate,
        check_applicability=_applicability,
        evaluation_mapping_to_ground_truth=_ground_truth,
    )


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test runs against an empty registry; restore built-ins on teardown.

    Clearing on entry keeps these tests independent of whichever built-ins
    got registered at import time. Restoring on exit ensures subsequent
    test files that rely on ``get_methodology(\"biasbuster\")`` still work
    regardless of alphabetical file ordering.
    """
    from biasbuster.methodologies import _register_builtin_methodologies

    clear_registry_for_testing()
    yield
    clear_registry_for_testing()
    _register_builtin_methodologies()


class TestMethodologyDataclass:
    """Declarative-invariant checks fail loud at construction time."""

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must be non-empty"):
            _make_methodology(name="")

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 'active' or 'stub'"):
            _make_methodology(status="experimental")

    def test_empty_severity_rejected(self) -> None:
        with pytest.raises(ValueError, match="severity_rollup_levels cannot be empty"):
            _make_methodology(severity=())

    def test_unknown_study_design_slug_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown study-design slugs"):
            _make_methodology(applicable=frozenset({"not_a_real_design"}))

    def test_any_study_design_marker_accepted(self) -> None:
        m = _make_methodology(applicable=frozenset({ANY_STUDY_DESIGN}))
        assert m.applies_to("cohort")
        assert m.applies_to("rct_parallel")
        assert m.applies_to("unknown")

    def test_applies_to_specific_design(self) -> None:
        m = _make_methodology(applicable=frozenset({"rct_parallel"}))
        assert m.applies_to("rct_parallel")
        assert not m.applies_to("cohort")
        assert not m.applies_to("diagnostic_accuracy")


class TestRegistry:
    def test_register_and_get(self) -> None:
        m = _make_methodology(name="alpha")
        register(m)
        assert get_methodology("alpha") is m

    def test_duplicate_register_rejected(self) -> None:
        register(_make_methodology(name="alpha"))
        with pytest.raises(DuplicateMethodologyError):
            register(_make_methodology(name="alpha"))

    def test_duplicate_register_with_replace(self) -> None:
        first = _make_methodology(name="alpha", severity=("low", "high"))
        second = _make_methodology(
            name="alpha", severity=("low", "high", "unclear"),
        )
        register(first)
        register(second, replace=True)
        assert get_methodology("alpha") is second

    def test_unknown_methodology_raises(self) -> None:
        with pytest.raises(UnknownMethodologyError):
            get_methodology("does_not_exist")

    def test_list_active_excludes_stubs(self) -> None:
        register(_make_methodology(name="alpha", status="active"))
        register(_make_methodology(name="beta", status="stub"))
        register(_make_methodology(name="gamma", status="active"))
        assert list_methodologies() == ["alpha", "beta", "gamma"]
        assert list_active_methodologies() == ["alpha", "gamma"]

    def test_register_builtin_methodologies_restores_biasbuster(self) -> None:
        """Built-in methodologies must be re-installable after a clear.

        Without this hook, any teardown that runs
        ``clear_registry_for_testing()`` permanently orphans the biasbuster
        methodology (its module is cached in ``sys.modules`` so the
        module-body registration side-effect does not fire again). That
        would make subsequent test files depending on the default
        registry start failing in proportion to alphabetical ordering.
        """
        from biasbuster.methodologies import _register_builtin_methodologies

        clear_registry_for_testing()
        with pytest.raises(UnknownMethodologyError):
            get_methodology("biasbuster")
        _register_builtin_methodologies()
        restored = get_methodology("biasbuster")
        assert restored.name == "biasbuster"
        assert restored.status == "active"


class TestCheckOrRaise:
    """The applicability + full-text guard is what gates annotations."""

    def test_full_text_required_raises_when_missing(self) -> None:
        m = _make_methodology(requires_full_text=True)
        paper = {"pmid": "P1"}
        with pytest.raises(FullTextRequiredError) as exc:
            check_or_raise(
                m, paper, {},
                full_text_available=False,
                detected_design="rct_parallel",
            )
        assert exc.value.pmid == "P1"
        assert exc.value.methodology == "fake"

    def test_applicability_mismatch_raises(self) -> None:
        m = _make_methodology(
            applicable=frozenset({"rct_parallel"}),
            applicable_ok=False,
            applicable_reason="paper is a cohort study",
        )
        with pytest.raises(ApplicabilityError) as exc:
            check_or_raise(
                m, {"pmid": "P1"}, {},
                full_text_available=True,
                detected_design="cohort",
            )
        assert exc.value.detected_design == "cohort"
        assert "cohort" in str(exc.value)

    def test_applicable_design_passes(self) -> None:
        m = _make_methodology(applicable=frozenset({"rct_parallel"}))
        check_or_raise(
            m, {"pmid": "P1"}, {},
            full_text_available=True,
            detected_design="rct_parallel",
        )  # no raise

    def test_any_design_methodology_passes_for_anything(self) -> None:
        m = _make_methodology(
            applicable=frozenset({ANY_STUDY_DESIGN}),
            requires_full_text=False,
        )
        for design in ("rct_parallel", "cohort", "diagnostic_accuracy", "unknown"):
            check_or_raise(
                m, {"pmid": "P1"}, {},
                full_text_available=False,
                detected_design=design,
            )

    def test_full_text_error_prioritised_over_applicability(self) -> None:
        """If both would fail, the full-text message is more actionable."""
        m = _make_methodology(
            requires_full_text=True,
            applicable=frozenset({"rct_parallel"}),
            applicable_ok=False,
            applicable_reason="wrong design",
        )
        with pytest.raises(FullTextRequiredError):
            check_or_raise(
                m, {"pmid": "P1"}, {},
                full_text_available=False,
                detected_design="cohort",
            )


class TestStudyDesignDetector:
    """Heuristic classification — strongest signal wins."""

    def test_cochrane_rob_fields_imply_rct_parallel(self) -> None:
        paper = {
            "pmid": "P1",
            "title": "Study of X",  # deliberately ambiguous
            "abstract": "We did something.",
            "randomization_bias": "low",
        }
        assert study_design.detect(paper) == "rct_parallel"

    def test_mesh_randomized_controlled_trial(self) -> None:
        paper = {
            "title": "",
            "abstract": "",
            "mesh_terms": ["Randomized Controlled Trial", "Humans"],
        }
        assert study_design.detect(paper) == "rct_parallel"

    def test_mesh_case_control(self) -> None:
        paper = {
            "title": "",
            "abstract": "",
            "mesh_terms": ["Case-Control Studies"],
        }
        assert study_design.detect(paper) == "case_control"

    def test_mesh_cohort(self) -> None:
        paper = {
            "title": "",
            "abstract": "",
            "mesh_terms": ["Cohort Studies"],
        }
        assert study_design.detect(paper) == "cohort"

    def test_mesh_systematic_review_beats_plain_review(self) -> None:
        paper = {
            "title": "",
            "abstract": "",
            "mesh_terms": ["Systematic Review", "Review"],
        }
        assert study_design.detect(paper) == "systematic_review"

    def test_mesh_meta_analysis(self) -> None:
        paper = {
            "title": "",
            "abstract": "",
            "mesh_terms": ["Meta-Analysis"],
        }
        assert study_design.detect(paper) == "meta_analysis"

    def test_mesh_meta_analysis_wins_over_systematic_review(self) -> None:
        """A paper tagged with BOTH should classify as the more specific design.

        A meta-analysis is a systematic review with statistical synthesis —
        the more specific, stronger design. Routing to ``meta_analysis``
        reflects that; the rule ordering in _MESH_DESIGN_RULES encodes it.
        """
        paper = {
            "title": "",
            "abstract": "",
            "mesh_terms": ["Systematic Review", "Meta-Analysis"],
        }
        assert study_design.detect(paper) == "meta_analysis"

    def test_mesh_cluster_analysis_does_not_imply_rct_cluster(self) -> None:
        """MeSH 'Cluster Analysis' is a statistical technique, not a trial design.

        Papers using cluster analysis for subgrouping must not be
        mis-classified as cluster-randomised trials. The keyword regex
        (``cluster-randomized``) is the canonical path for real cluster RCTs.
        """
        paper = {
            "title": "Machine learning subphenotypes of sepsis",
            "abstract": "We used k-means cluster analysis to identify subgroups.",
            "mesh_terms": ["Cluster Analysis", "Sepsis"],
        }
        # Should NOT return "rct_cluster"; likely "unknown" since nothing
        # else in this synthetic paper pins a design.
        assert study_design.detect(paper) != "rct_cluster"

    def test_mesh_diagnostic_accuracy(self) -> None:
        paper = {
            "title": "",
            "abstract": "",
            "mesh_terms": ["Sensitivity and Specificity"],
        }
        assert study_design.detect(paper) == "diagnostic_accuracy"

    def test_keyword_cluster_randomized(self) -> None:
        paper = {
            "title": "A cluster-randomized trial of...",
            "abstract": "Randomized by clinic.",
            "mesh_terms": [],
        }
        assert study_design.detect(paper) == "rct_cluster"

    def test_keyword_crossover(self) -> None:
        paper = {
            "title": "A cross-over trial of TestDrug",
            "abstract": "",
            "mesh_terms": [],
        }
        assert study_design.detect(paper) == "rct_crossover"

    def test_keyword_diagnostic_accuracy(self) -> None:
        paper = {
            "title": "Diagnostic accuracy of biomarker X",
            "abstract": "Sensitivity and specificity computed.",
            "mesh_terms": [],
        }
        assert study_design.detect(paper) == "diagnostic_accuracy"

    def test_keyword_prisma_flags_systematic_review(self) -> None:
        paper = {
            "title": "A review of X",
            "abstract": "Conducted a PRISMA-compliant search.",
            "mesh_terms": [],
        }
        assert study_design.detect(paper) == "systematic_review"

    def test_unknown_fallback(self) -> None:
        paper = {
            "title": "An exploration of something vague",
            "abstract": "We discuss trends.",
            "mesh_terms": [],
        }
        assert study_design.detect(paper) == "unknown"

    def test_cochrane_signal_beats_mesh(self) -> None:
        """Even if MeSH would say something else, Cochrane RoB fields win."""
        paper = {
            "randomization_bias": "high",
            "mesh_terms": ["Case-Control Studies"],
        }
        assert study_design.detect(paper) == "rct_parallel"

    def test_overall_rob_alone_does_not_imply_rct(self) -> None:
        """``overall_rob`` is methodology-agnostic — it can hold the
        worst rating from QUADAS-2 or ROBINS-I too. Without one of the
        five RoB 2-specific per-domain columns, ``_cochrane_signal``
        must NOT classify a paper as ``rct_parallel``.
        """
        paper = {
            "title": "",
            "abstract": "",
            "overall_rob": "low",  # could be QUADAS-2's overall, not RoB 2's
        }
        assert study_design.detect(paper) == "unknown"

    def test_empty_paper_is_unknown(self) -> None:
        assert study_design.detect({}) == "unknown"

    def test_slug_is_always_in_canonical_set(self) -> None:
        # Every possible return value is registered in STUDY_DESIGN_SLUGS.
        fixtures = [
            {"randomization_bias": "low"},
            {"mesh_terms": ["Cohort Studies"]},
            {"mesh_terms": ["Meta-Analysis"]},
            {"title": "A crossover trial"},
            {},
        ]
        for p in fixtures:
            assert study_design.detect(p) in STUDY_DESIGN_SLUGS

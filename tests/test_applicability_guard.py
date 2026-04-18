"""Step-6 tests: applicability + full-text guards wired into the pipeline.

Two call sites enforce the guard:

1. ``biasbuster.pipeline.stage_annotate`` — per-paper pre-filter (plus an
   up-front refusal when the methodology is full-text-only, because batch
   mode is abstract-only).
2. ``annotate_single_paper.annotate_paper`` — one-shot pre-flight check
   that respects the ``--full-text``/``--agentic``/``--decomposed`` CLI
   flags when deciding whether full text is available.

Tests use a fixture methodology that differs from biasbuster only in its
applicability rules, so we can drive the guard without needing a real
methodology implementation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from biasbuster.database import Database
from biasbuster.methodologies import (
    ApplicabilityError,
    FullTextRequiredError,
    Methodology,
    check_or_raise,
    get_methodology,
    register,
)


def _make_full_text_rct_methodology(name: str) -> Methodology:
    """A stand-in for cochrane_rob2: full-text required, RCT-only.

    ``check_applicability`` re-derives the study design from the paper so
    it doesn't override the declarative ``applicable_study_designs`` filter
    — a real methodology (like the upcoming cochrane_rob2) will have a
    substantive applicability check that refuses non-RCTs. Using a
    permissive ``lambda p, e, ft: (True, "")`` would bypass the filter
    via the "methodology may override design mismatch" branch of
    ``check_or_raise`` and defeat the test.
    """
    from biasbuster.methodologies import study_design as _sd
    bb = get_methodology("biasbuster")

    def _check(paper: dict, _enrichment: dict, _ft: bool) -> tuple[bool, str]:
        design = _sd.detect(paper)
        if design != "rct_parallel":
            return False, (
                f"requires a parallel-group RCT, got {design!r}"
            )
        return True, ""

    return Methodology(
        name=name,
        display_name=f"Fake {name}",
        version="v0",
        applicable_study_designs=frozenset({"rct_parallel"}),
        requires_full_text=True,
        orchestration="decomposed_full_text",
        severity_rollup_levels=("low", "some_concerns", "high"),
        status="active",
        build_system_prompt=bb.build_system_prompt,
        build_user_message=bb.build_user_message,
        parse_output=bb.parse_output,
        aggregate=bb.aggregate,
        check_applicability=_check,
        evaluation_mapping_to_ground_truth=bb.evaluation_mapping_to_ground_truth,
    )


@pytest.fixture
def seeded_db(tmp_path):
    """A DB containing one parallel-RCT paper and one cohort paper."""
    db = Database(tmp_path / "guard.db")
    db.initialize()
    db.insert_paper({
        "pmid": "RCT1",
        "title": "Trial of X vs placebo",
        "abstract": "Randomized placebo-controlled trial.",
        "source": "cochrane_rob",
        "randomization_bias": "low",  # triggers rct_parallel detection
    })
    db.insert_paper({
        "pmid": "COH1",
        "title": "Observational cohort of Y",
        "abstract": "We followed 10000 patients for 5 years.",
        "source": "test",
        "mesh_terms": '["Cohort Studies"]',
    })
    yield db
    db.close()


class TestStageAnnotateGuard:
    """Batch-mode applicability filtering in stage_annotate."""

    def test_full_text_only_methodology_skipped_at_stage_level(
        self, seeded_db, tmp_path, caplog, monkeypatch,
    ) -> None:
        """A full-text-only methodology aborts stage_annotate up front.

        Batch mode is abstract-only; the methodology's requires_full_text
        flag is detected before any per-paper loop runs.
        """
        from biasbuster import pipeline

        fake = _make_full_text_rct_methodology("fake_fulltext_rct_stage")
        register(fake)
        try:
            # We don't actually run any annotator — stage_annotate should
            # return early after logging. Build a minimal fake Config.
            cfg = MagicMock()
            cfg.annotation_max_per_source = {"cochrane_rob": 10}
            cfg.annotation_concurrency = 1
            cfg.annotation_delay = 0
            with caplog.at_level("ERROR"):
                asyncio.run(pipeline.stage_annotate(
                    cfg, seeded_db, models=["anthropic"],
                    methodology=fake.name,
                ))
            msgs = " ".join(r.getMessage() for r in caplog.records)
            assert "requires full text" in msgs
            assert fake.name in msgs
        finally:
            from biasbuster.methodologies.registry import REGISTRY
            REGISTRY.pop(fake.name, None)

    def test_biasbuster_does_not_skip_any_papers(
        self, seeded_db,
    ) -> None:
        """Biasbuster's guard is always-True; applicability never refuses.

        This is the back-compat safety net — the default pathway must
        continue to accept every paper in the DB.
        """
        rct = seeded_db.get_paper("RCT1")
        coh = seeded_db.get_paper("COH1")
        from biasbuster.methodologies import study_design
        bb = get_methodology("biasbuster")
        # Neither call raises.
        check_or_raise(
            bb, rct, {},
            full_text_available=False,
            detected_design=study_design.detect(rct),
        )
        check_or_raise(
            bb, coh, {},
            full_text_available=False,
            detected_design=study_design.detect(coh),
        )


class TestAnnotateSinglePaperGuard:
    """annotate_paper's pre-flight refuses mismatches before any LLM call."""

    def test_refuses_cohort_when_methodology_is_rct_only(
        self, seeded_db, caplog, monkeypatch,
    ) -> None:
        from annotate_single_paper import annotate_paper

        fake = _make_full_text_rct_methodology("fake_fulltext_rct_single")
        register(fake)
        try:
            cfg = MagicMock()
            # Patch create_annotator to return a sentinel — we should
            # never reach it because the guard refuses first.
            sentinel = MagicMock()
            monkeypatch.setattr(
                "annotate_single_paper.create_annotator",
                lambda c, m: sentinel,
            )

            coh = seeded_db.get_paper("COH1")
            with caplog.at_level("ERROR"):
                result = asyncio.run(annotate_paper(
                    pmid="COH1",
                    paper=coh,
                    db=seeded_db,
                    config=cfg,
                    model_name="anthropic",
                    full_text=True,  # satisfies the full-text-required flag
                    methodology=fake.name,
                ))
            assert result is None
            msgs = " ".join(r.getMessage() for r in caplog.records)
            assert "Refusing to annotate" in msgs
            assert "cohort" in msgs or "design" in msgs
            # create_annotator may or may not have been called; what
            # matters is that the annotate_paper short-circuited and no
            # LLM call was made.
            assert not sentinel.annotate_abstract.called
        finally:
            from biasbuster.methodologies.registry import REGISTRY
            REGISTRY.pop(fake.name, None)

    def test_refuses_abstract_only_when_methodology_requires_full_text(
        self, seeded_db, caplog, monkeypatch,
    ) -> None:
        """If no full-text mode flag is passed, the guard refuses."""
        from annotate_single_paper import annotate_paper

        fake = _make_full_text_rct_methodology("fake_fulltext_rct_abs")
        register(fake)
        try:
            cfg = MagicMock()
            monkeypatch.setattr(
                "annotate_single_paper.create_annotator",
                lambda c, m: MagicMock(),
            )
            rct = seeded_db.get_paper("RCT1")
            with caplog.at_level("ERROR"):
                result = asyncio.run(annotate_paper(
                    pmid="RCT1",
                    paper=rct,
                    db=seeded_db,
                    config=cfg,
                    model_name="anthropic",
                    full_text=False,  # abstract-only path
                    methodology=fake.name,
                ))
            assert result is None
            msgs = " ".join(r.getMessage() for r in caplog.records)
            assert "Refusing to annotate" in msgs
            assert "full text" in msgs
        finally:
            from biasbuster.methodologies.registry import REGISTRY
            REGISTRY.pop(fake.name, None)

    def test_biasbuster_never_refused(
        self, seeded_db, caplog, monkeypatch,
    ) -> None:
        """Biasbuster-on-any-paper must always pass the guard.

        We stub create_annotator to return None so annotate_paper bails
        out after the guard but before any LLM wiring; the key assertion
        is that no refusal log line is emitted.
        """
        from annotate_single_paper import annotate_paper

        monkeypatch.setattr(
            "annotate_single_paper.create_annotator",
            lambda c, m: None,
        )
        coh = seeded_db.get_paper("COH1")
        with caplog.at_level("ERROR"):
            result = asyncio.run(annotate_paper(
                pmid="COH1",
                paper=coh,
                db=seeded_db,
                config=MagicMock(),
                model_name="anthropic",
                methodology="biasbuster",
            ))
        assert result is None  # create_annotator returned None
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "Refusing to annotate" not in msgs

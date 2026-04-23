"""Coverage for the generic faithfulness harness.

What the tests pin:

- The spec lookup finds both registered methodologies and rejects
  unknown slugs with a clear error.
- ``load_expert_view`` reads ``expert_methodology_ratings`` for any
  methodology and validates the judgement against
  ``spec.judgement_order`` (a QUADAS-2 "unclear" must not silently
  bleed into a RoB 2 report).
- The prediction-view loaders for both methodologies reject
  malformed inputs rather than filling in defaults.
- End-to-end ``run_faithfulness`` produces a usable Markdown + JSON
  report for QUADAS-2 given synthetic annotations and expert rows.
- Confusion matrices use the correct vocabulary per methodology
  (RoB 2: some_concerns; QUADAS-2: unclear).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from biasbuster.database import Database
from biasbuster.evaluation.methodology_faithfulness import (
    FaithfulnessSpec,
    JudgementSeries,
    build_report,
    collect_paired_papers,
    get_spec,
    load_expert_view,
    render_markdown,
    run_faithfulness,
)


# ---- Spec lookup -------------------------------------------------------

class TestGetSpec:
    def test_cochrane_rob2_spec_is_loaded(self) -> None:
        spec = get_spec("cochrane_rob2")
        assert spec.methodology == "cochrane_rob2"
        assert spec.judgement_order == ("low", "some_concerns", "high")
        assert "randomization" in spec.domain_slugs

    def test_quadas_2_spec_is_loaded(self) -> None:
        spec = get_spec("quadas_2")
        assert spec.methodology == "quadas_2"
        assert spec.judgement_order == ("low", "unclear", "high")
        assert "patient_selection" in spec.domain_slugs

    def test_unknown_methodology_raises(self) -> None:
        with pytest.raises((LookupError, ModuleNotFoundError)):
            get_spec("not_a_real_methodology")


# ---- JudgementSeries with custom vocab ---------------------------------

class TestJudgementSeriesVocab:
    def test_quadas_2_scale_rejects_rob2_vocab(self) -> None:
        """A QUADAS-2 series must not silently accept 'some_concerns'."""
        s = JudgementSeries(
            judgement_order=("low", "unclear", "high"),
            expert=["low", "unclear"],
            prediction=["low", "unclear"],
        )
        # Expected path works:
        assert s.exact_match() == 1.0
        # Using the wrong vocabulary blows up rather than scoring silently.
        s_bad = JudgementSeries(
            judgement_order=("low", "unclear", "high"),
            expert=["low"],
            prediction=["some_concerns"],  # RoB 2 level, not QUADAS-2
        )
        # Either KeyError (dict lookup in weighted_kappa) or ValueError
        # (list.index in within_one / mean_abs_error) is fine — the
        # important thing is that we crash rather than silently scoring.
        with pytest.raises((KeyError, ValueError)):
            s_bad.weighted_kappa()

    def test_ordinal_distance_uses_specs_scale(self) -> None:
        """low↔unclear is distance 1 on QUADAS-2 as on RoB 2."""
        s = JudgementSeries(
            judgement_order=("low", "unclear", "high"),
            expert=["low", "low", "high"],
            prediction=["unclear", "high", "low"],
        )
        # low↔unclear=1, low↔high=2, high↔low=2 → MAE = 5/3
        assert s.mean_abs_error() == pytest.approx(5.0 / 3.0)
        assert s.within_one() == pytest.approx(1.0 / 3.0)


# ---- QUADAS-2 prediction-view loader -----------------------------------

class TestQuadas2PredictionView:
    def _well_formed_annotation(
        self, *, overall: str = "low",
    ) -> dict:
        """Mirror the shape ``QUADAS2Assessment.to_dict()`` produces."""
        return {
            "domains": {
                "patient_selection": {
                    "domain": "patient_selection",
                    "signalling_answers": {},
                    "bias_rating": "low",
                    "applicability": "low",
                },
                "index_test": {
                    "domain": "index_test",
                    "signalling_answers": {},
                    "bias_rating": "low",
                    "applicability": "low",
                },
                "reference_standard": {
                    "domain": "reference_standard",
                    "signalling_answers": {},
                    "bias_rating": "low",
                    "applicability": "low",
                },
                "flow_and_timing": {
                    "domain": "flow_and_timing",
                    "signalling_answers": {},
                    "bias_rating": "low",
                    "applicability": None,
                },
            },
            "worst_bias": overall,
            "worst_applicability": "low",
        }

    def test_well_formed_annotation_yields_view(self) -> None:
        spec = get_spec("quadas_2")
        ann = self._well_formed_annotation(overall="unclear")
        view = spec.load_prediction_view(ann)
        assert view is not None
        assert view["overall"] == "unclear"
        # Bias-only in v1 — applicability is discarded.
        assert view["domains"] == {
            "patient_selection": "low",
            "index_test": "low",
            "reference_standard": "low",
            "flow_and_timing": "low",
        }

    def test_missing_domain_returns_none(self) -> None:
        spec = get_spec("quadas_2")
        ann = self._well_formed_annotation()
        del ann["domains"]["flow_and_timing"]
        assert spec.load_prediction_view(ann) is None

    def test_off_vocabulary_overall_returns_none(self) -> None:
        """``some_concerns`` is a RoB 2 level; it must not score as QUADAS-2."""
        spec = get_spec("quadas_2")
        ann = self._well_formed_annotation(overall="some_concerns")
        assert spec.load_prediction_view(ann) is None


# ---- load_expert_view --------------------------------------------------

class TestLoadExpertView:
    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        d = Database(tmp_path / "ev.db")
        d.initialize()
        yield d
        d.close()

    def test_verified_row_preferred_over_unverified(
        self, db: Database,
    ) -> None:
        """Two ratings for the same paper → prefer the curator-verified one."""
        # Unverified row first.
        db.upsert_expert_rating(
            methodology="quadas_2", rating_source="review-A",
            study_label="S1",
            domain_ratings={
                "patient_selection": {"bias": "low"},
                "index_test": {"bias": "low"},
                "reference_standard": {"bias": "low"},
                "flow_and_timing": {"bias": "low"},
            },
            overall_rating="low", pmid="42", verified=False,
        )
        db.upsert_expert_rating(
            methodology="quadas_2", rating_source="review-B",
            study_label="S2",
            domain_ratings={
                "patient_selection": {"bias": "high"},
                "index_test": {"bias": "high"},
                "reference_standard": {"bias": "high"},
                "flow_and_timing": {"bias": "high"},
            },
            overall_rating="high", pmid="42", verified=True,
        )
        spec = get_spec("quadas_2")
        view = load_expert_view(db, spec, "42")
        assert view is not None
        assert view["overall"] == "high"
        assert view["verified"] is True

    def test_off_vocabulary_row_dropped(self, db: Database) -> None:
        """An expert row stored with a RoB 2 level must not surface as QUADAS-2."""
        db.upsert_expert_rating(
            methodology="quadas_2", rating_source="r",
            study_label="S",
            domain_ratings={
                "patient_selection": {"bias": "some_concerns"},
                "index_test": {"bias": "low"},
                "reference_standard": {"bias": "low"},
                "flow_and_timing": {"bias": "low"},
            },
            overall_rating="low", pmid="42",
        )
        spec = get_spec("quadas_2")
        assert load_expert_view(db, spec, "42") is None


# ---- End-to-end QUADAS-2 report ----------------------------------------

class TestEndToEndQuadas2:
    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        d = Database(tmp_path / "e2e.db")
        d.initialize()
        yield d
        d.close()

    def _insert_paper_and_expert(
        self, db: Database, pmid: str, overall: str = "low",
    ) -> None:
        db.insert_paper({
            "pmid": pmid, "title": f"Diag {pmid}",
            "abstract": "diag acc study", "source": "manual_import",
        })
        db.upsert_expert_rating(
            methodology="quadas_2", rating_source="jcm-15-01829",
            study_label=pmid,
            domain_ratings={
                "patient_selection": {"bias": "low"},
                "index_test": {"bias": "low"},
                "reference_standard": {"bias": "low"},
                "flow_and_timing": {"bias": "low"},
            },
            overall_rating=overall, pmid=pmid,
        )

    def _insert_prediction(
        self, db: Database, pmid: str, overall: str = "low",
    ) -> None:
        blob = {
            "domains": {
                slug: {
                    "domain": slug,
                    "signalling_answers": {},
                    "bias_rating": "low",
                    "applicability": (
                        None if slug == "flow_and_timing" else "low"
                    ),
                }
                for slug in (
                    "patient_selection", "index_test",
                    "reference_standard", "flow_and_timing",
                )
            },
            "worst_bias": overall,
            "worst_applicability": "low",
        }
        db.insert_annotation(
            pmid, "anthropic", blob,
            methodology="quadas_2",
            methodology_version="quadas2-2011",
        )

    def test_run_faithfulness_reports_agreement(
        self, db: Database, tmp_path: Path,
    ) -> None:
        """Two agreeing papers + one disagreeing → overall exact_match = 2/3."""
        self._insert_paper_and_expert(db, "P1", overall="low")
        self._insert_paper_and_expert(db, "P2", overall="unclear")
        self._insert_paper_and_expert(db, "P3", overall="high")
        self._insert_prediction(db, "P1", overall="low")
        self._insert_prediction(db, "P2", overall="unclear")
        self._insert_prediction(db, "P3", overall="low")  # disagree

        report, md_path, json_path = run_faithfulness(
            tmp_path / "e2e.db", "quadas_2", "anthropic",
            output_dir=tmp_path / "out",
        )
        assert report.n_paired == 3
        assert report.overall.exact_match() == pytest.approx(2 / 3)
        assert report.overall.n == 3
        # One discrepancy, P3 with distance=2 (high→low)
        assert len(report.discrepancies) == 1
        assert report.discrepancies[0]["pmid"] == "P3"
        assert report.discrepancies[0]["distance"] == 2

        # Report files present + parseable.
        assert md_path is not None and md_path.exists()
        assert json_path is not None and json_path.exists()
        assert "QUADAS-2 faithfulness" in md_path.read_text()
        data = json.loads(json_path.read_text())
        assert data["methodology"] == "quadas_2"
        assert data["overall"]["n"] == 3

    def test_markdown_uses_methodology_vocabulary(
        self, db: Database,
    ) -> None:
        """QUADAS-2 confusion matrix columns must be low/unclear/high."""
        self._insert_paper_and_expert(db, "P1", overall="unclear")
        self._insert_prediction(db, "P1", overall="unclear")
        spec = get_spec("quadas_2")
        paired = collect_paired_papers(db, spec, "anthropic")
        report = build_report(paired, spec, "anthropic", 1)
        md = render_markdown(report, spec)
        assert "unclear" in md
        # No RoB 2 vocabulary should appear in a QUADAS-2 report.
        assert "some_concerns" not in md


# ---- End-to-end RoB 2 via the generic harness --------------------------

class TestEndToEndRob2:
    """Symmetric coverage to ``TestEndToEndQuadas2`` — RoB 2's stored
    prediction shape (per-outcome, worst-wins) is different enough that
    the QUADAS-2 test can't prove RoB 2's path also works.
    """

    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        d = Database(tmp_path / "e2e_rob2.db")
        d.initialize()
        yield d
        d.close()

    def _insert_rob2_prediction(
        self, db: Database, pmid: str, *, overall: str,
    ) -> None:
        """Match the shape ``RoB2Assessment.to_dict()`` writes."""
        from biasbuster.methodologies.cochrane_rob2.schema import (
            ROB2_DOMAIN_SLUGS,
        )

        domains_blob = {
            slug: {
                "domain": slug,
                "signalling_answers": {"1.1": "Y"},
                "judgement": "low",
                "justification": "",
                "evidence_quotes": [],
            }
            for slug in ROB2_DOMAIN_SLUGS
        }
        db.insert_annotation(
            pmid, "anthropic",
            {
                "pmid": pmid,
                "outcomes": [{
                    "outcome_label": "primary outcome",
                    "result_label": "as reported",
                    "domains": domains_blob,
                    "overall_judgement": overall,
                    "overall_rationale": "",
                }],
                "methodology_version": "rob2-2019",
                "worst_across_outcomes": overall,
                "notes": "",
            },
            methodology="cochrane_rob2",
            methodology_version="rob2-2019",
        )

    def _seed_rob2_expert(
        self, db: Database, pmid: str, overall: str,
    ) -> None:
        db.upsert_expert_rating(
            methodology="cochrane_rob2",
            rating_source="cochrane_review_pmid:99999",
            study_label=pmid,
            domain_ratings={
                "randomization": {"bias": "low"},
                "deviations_from_interventions": {"bias": "low"},
                "missing_outcome_data": {"bias": "low"},
                "outcome_measurement": {"bias": "low"},
                "selection_of_reported_result": {"bias": "low"},
            },
            overall_rating=overall, pmid=pmid,
            methodology_version="rob2-2019",
        )

    def test_run_faithfulness_for_cochrane_rob2(
        self, db: Database, tmp_path: Path,
    ) -> None:
        """One matching + one disagreeing → overall exact_match = 0.5."""
        db.insert_paper({
            "pmid": "P1", "title": "RCT A", "abstract": "parallel RCT",
            "source": "cochrane_rob",
        })
        db.insert_paper({
            "pmid": "P2", "title": "RCT B", "abstract": "parallel RCT",
            "source": "cochrane_rob",
        })
        self._seed_rob2_expert(db, "P1", "low")
        self._seed_rob2_expert(db, "P2", "high")
        self._insert_rob2_prediction(db, "P1", overall="low")
        self._insert_rob2_prediction(db, "P2", overall="some_concerns")

        report, md_path, json_path = run_faithfulness(
            tmp_path / "e2e_rob2.db", "cochrane_rob2", "anthropic",
            output_dir=tmp_path / "out",
        )
        assert report.methodology == "cochrane_rob2"
        assert report.n_paired == 2
        assert report.overall.exact_match() == 0.5
        assert report.overall.within_one() == 1.0
        assert md_path is not None and md_path.exists()
        md = md_path.read_text()
        assert "Cochrane RoB 2 faithfulness" in md
        # Sanity: QUADAS-2 vocabulary never leaks into RoB 2 report.
        assert "unclear" not in md
        assert "some_concerns" in md


# ---- Backfill-missing warning -----------------------------------------

class TestBackfillMissingWarning:
    """The harness must tell the user why ``n_paired=0`` when
    predictions exist but no expert rows do. Silent zero was the trap
    the reviewer flagged; this test pins the remediation.
    """

    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        d = Database(tmp_path / "warn.db")
        d.initialize()
        yield d
        d.close()

    def test_warns_when_predictions_exist_but_no_expert_rows(
        self, db: Database, caplog: pytest.LogCaptureFixture,
    ) -> None:
        db.insert_paper({
            "pmid": "P1", "title": "T", "abstract": "A",
            "source": "cochrane_rob",
        })
        db.insert_annotation(
            "P1", "anthropic",
            {
                # Minimal valid RoB 2 annotation (one outcome, no domains
                # — the test exercises the missing-expert-rows warning,
                # not annotation content; the empty-domains shape is the
                # smallest payload that satisfies the schema).
                "outcomes": [
                    {"overall_judgement": "low", "domains": {}},
                ],
                "worst_across_outcomes": "low",
                "methodology_version": "rob2-2019",
            },
            methodology="cochrane_rob2",
        )
        spec = get_spec("cochrane_rob2")
        with caplog.at_level(
            logging.WARNING,
            logger="biasbuster.evaluation.methodology_faithfulness",
        ):
            paired = collect_paired_papers(db, spec, "anthropic")
        assert paired == []
        # The message must name the backfill script the user should run.
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, "expected a backfill-missing warning"
        joined = " ".join(r.getMessage() for r in warnings)
        assert "backfill_rob2_expert_ratings" in joined

    def test_no_warning_when_some_expert_rows_exist(
        self, db: Database, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Expert rows for a *different* PMID still count — we want to
        flag empty-table, not per-paper missing (which is a legitimate
        narrow-filter case)."""
        db.insert_paper({
            "pmid": "P1", "title": "T", "abstract": "A",
            "source": "cochrane_rob",
        })
        db.insert_annotation(
            "P1", "anthropic",
            {
                # Minimal valid RoB 2 annotation (one outcome, no domains
                # — the test exercises the missing-expert-rows warning,
                # not annotation content; the empty-domains shape is the
                # smallest payload that satisfies the schema).
                "outcomes": [
                    {"overall_judgement": "low", "domains": {}},
                ],
                "worst_across_outcomes": "low",
                "methodology_version": "rob2-2019",
            },
            methodology="cochrane_rob2",
        )
        # Expert row for a different paper, same methodology.
        db.upsert_expert_rating(
            methodology="cochrane_rob2",
            rating_source="cochrane_review_pmid:99999",
            study_label="P_OTHER",
            domain_ratings={
                "randomization": {"bias": "low"},
                "deviations_from_interventions": {"bias": "low"},
                "missing_outcome_data": {"bias": "low"},
                "outcome_measurement": {"bias": "low"},
                "selection_of_reported_result": {"bias": "low"},
            },
            overall_rating="low", pmid="P_OTHER",
        )
        spec = get_spec("cochrane_rob2")
        with caplog.at_level(
            logging.WARNING,
            logger="biasbuster.evaluation.methodology_faithfulness",
        ):
            collect_paired_papers(db, spec, "anthropic")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not warnings, (
            f"unexpected warnings: {[r.getMessage() for r in warnings]}"
        )

    def test_no_warning_when_no_annotations(
        self, db: Database, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A harness run against a DB with zero predictions is a
        legitimate exploratory state, not a setup bug."""
        spec = get_spec("cochrane_rob2")
        with caplog.at_level(
            logging.WARNING,
            logger="biasbuster.evaluation.methodology_faithfulness",
        ):
            collect_paired_papers(db, spec, "anthropic")
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

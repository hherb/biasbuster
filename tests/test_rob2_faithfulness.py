"""Tests for the RoB 2 faithfulness harness.

No LLM calls anywhere — the harness is pure bookkeeping:

- Paired-paper gatherer joins ``papers`` (expert ground truth) with
  ``annotations`` (model predictions).
- Metric series computes exact-match, within-one, MAE, weighted kappa,
  and a 3x3 confusion matrix on the 3-level RoB 2 scale.
- Markdown renderer emits a readable report; JSON sidecar carries the
  machine-readable version.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biasbuster.database import Database
from biasbuster.evaluation.rob2_faithfulness import (
    FaithfulnessReport,
    JudgementSeries,
    build_report,
    collect_paired_papers,
    render_markdown,
    _extract_prediction_view,
)
from biasbuster.methodologies.cochrane_rob2 import METHODOLOGY_VERSION
from biasbuster.methodologies.cochrane_rob2.schema import ROB2_DOMAIN_SLUGS


# ---- Fixtures ----------------------------------------------------------

def _make_expert_paper(
    pmid: str,
    *,
    overall: str,
    randomization: str = "low",
    deviation: str = "low",
    missing_outcome: str = "low",
    measurement: str = "low",
    reporting: str = "low",
    title: str = "Trial",
) -> dict:
    """Paper dict with the Cochrane RoB 2 expert columns populated."""
    return {
        "pmid": pmid,
        "title": title,
        "abstract": "Randomized placebo-controlled trial.",
        "source": "cochrane_rob",
        "overall_rob": overall,
        "randomization_bias": randomization,
        "deviation_bias": deviation,
        "missing_outcome_bias": missing_outcome,
        "measurement_bias": measurement,
        "reporting_bias": reporting,
    }


def _make_prediction_annotation(
    pmid: str,
    *,
    overall: str,
    per_domain: dict[str, str] | None = None,
) -> dict:
    """Build the JSON-blob shape a real cochrane_rob2 assessment would store.

    Mirrors :meth:`RoB2Assessment.to_dict` but constructed inline so tests
    don't have to go through the assessor.
    """
    per_domain = per_domain or {slug: "low" for slug in ROB2_DOMAIN_SLUGS}
    domains_blob = {
        slug: {
            "domain": slug,
            "signalling_answers": {"1.1": "Y"},
            "judgement": per_domain[slug],
            "justification": "",
            "evidence_quotes": [],
        }
        for slug in ROB2_DOMAIN_SLUGS
    }
    return {
        "pmid": pmid,
        "outcomes": [{
            "outcome_label": "primary outcome",
            "result_label": "as reported",
            "domains": domains_blob,
            "overall_judgement": overall,
            "overall_rationale": "",
        }],
        "methodology_version": METHODOLOGY_VERSION,
        "worst_across_outcomes": overall,
        "notes": "",
    }


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "fidelity.db")
    d.initialize()
    yield d
    d.close()


# ---- Unit tests --------------------------------------------------------

class TestJudgementSeries:
    def test_empty_series_returns_zeros(self) -> None:
        s = JudgementSeries()
        assert s.n == 0
        assert s.exact_match() == 0.0
        assert s.weighted_kappa() == 0.0
        # 3x3 zero matrix
        assert all(sum(row.values()) == 0 for row in s.confusion().values())

    def test_perfect_agreement(self) -> None:
        s = JudgementSeries(
            expert=["low", "some_concerns", "high", "low"],
            prediction=["low", "some_concerns", "high", "low"],
        )
        assert s.exact_match() == 1.0
        assert s.within_one() == 1.0
        assert s.mean_abs_error() == 0.0
        assert s.weighted_kappa() == 1.0

    def test_perfect_disagreement(self) -> None:
        """All low/high swaps — MAE=2 per pair, kappa strongly negative."""
        s = JudgementSeries(
            expert=["low", "low", "high", "high"],
            prediction=["high", "high", "low", "low"],
        )
        assert s.exact_match() == 0.0
        assert s.within_one() == 0.0
        assert s.mean_abs_error() == 2.0
        assert s.weighted_kappa() < 0.0

    def test_adjacent_disagreement_still_within_one(self) -> None:
        s = JudgementSeries(
            expert=["low", "some_concerns", "high"],
            prediction=["some_concerns", "low", "some_concerns"],
        )
        # Exact match 0, but every pair is adjacent on the ordinal scale.
        assert s.exact_match() == 0.0
        assert s.within_one() == 1.0
        assert s.mean_abs_error() == 1.0

    def test_confusion_matrix_layout(self) -> None:
        s = JudgementSeries(
            expert=["low", "low", "high"],
            prediction=["low", "high", "high"],
        )
        conf = s.confusion()
        assert conf["low"]["low"] == 1
        assert conf["low"]["high"] == 1
        assert conf["high"]["high"] == 1
        # Untouched cells remain zero
        assert conf["some_concerns"]["some_concerns"] == 0


class TestExtractPredictionView:
    def test_well_formed_annotation_extracts_worst_per_domain(self) -> None:
        ann = _make_prediction_annotation(
            "P1", overall="high",
            per_domain={slug: "low" for slug in ROB2_DOMAIN_SLUGS},
        )
        view = _extract_prediction_view(ann)
        assert view is not None
        assert view["overall"] == "high"
        for slug in ROB2_DOMAIN_SLUGS:
            assert view["domains"][slug] == "low"

    def test_missing_outcomes_returns_none(self) -> None:
        bad = {
            "worst_across_outcomes": "low",
            "outcomes": [],
        }
        assert _extract_prediction_view(bad) is None

    def test_invalid_overall_returns_none(self) -> None:
        bad = _make_prediction_annotation("P1", overall="high")
        bad["worst_across_outcomes"] = "invalid_level"
        assert _extract_prediction_view(bad) is None


class TestCollectPairedPapers:
    def test_returns_paper_with_both_expert_and_prediction(
        self, db: Database,
    ) -> None:
        db.insert_paper(_make_expert_paper("P1", overall="low"))
        db.insert_annotation(
            "P1", "anthropic",
            _make_prediction_annotation("P1", overall="low"),
            methodology="cochrane_rob2",
            methodology_version=METHODOLOGY_VERSION,
        )
        paired = collect_paired_papers(db, "anthropic")
        assert len(paired) == 1
        assert paired[0].pmid == "P1"
        assert paired[0].expert["overall"] == "low"
        assert paired[0].prediction["overall"] == "low"

    def test_excludes_papers_without_expert_rating(
        self, db: Database,
    ) -> None:
        # Paper has no Cochrane expert ratings
        db.insert_paper({
            "pmid": "P2", "title": "Untagged trial",
            "abstract": "A", "source": "pubmed_rct",
        })
        db.insert_annotation(
            "P2", "anthropic",
            _make_prediction_annotation("P2", overall="high"),
            methodology="cochrane_rob2",
        )
        assert collect_paired_papers(db, "anthropic") == []

    def test_excludes_papers_without_prediction(self, db: Database) -> None:
        db.insert_paper(_make_expert_paper("P3", overall="some_concerns"))
        # No annotation inserted.
        assert collect_paired_papers(db, "anthropic") == []

    def test_other_methodology_annotations_ignored(
        self, db: Database,
    ) -> None:
        """A paper annotated under biasbuster must not leak into the RoB 2 report."""
        db.insert_paper(_make_expert_paper("P4", overall="low"))
        db.insert_annotation(
            "P4", "anthropic",
            {"overall_severity": "low"},  # biasbuster-shaped, not RoB 2
            methodology="biasbuster",
        )
        assert collect_paired_papers(db, "anthropic") == []


class TestBuildReport:
    def test_report_counts_and_overall_metrics(self, db: Database) -> None:
        db.insert_paper(_make_expert_paper("P1", overall="low"))
        db.insert_paper(_make_expert_paper("P2", overall="high"))
        db.insert_annotation(
            "P1", "anthropic",
            _make_prediction_annotation("P1", overall="low"),
            methodology="cochrane_rob2",
        )
        db.insert_annotation(
            "P2", "anthropic",
            _make_prediction_annotation("P2", overall="some_concerns"),
            methodology="cochrane_rob2",
        )
        paired = collect_paired_papers(db, "anthropic")
        report = build_report(
            paired, model_name="anthropic", n_model_annotations=2,
        )
        assert report.n_paired == 2
        assert report.n_model_annotations == 2
        assert report.overall.n == 2
        # P1 matches exactly, P2 is adjacent (high vs some_concerns)
        assert report.overall.exact_match() == 0.5
        assert report.overall.within_one() == 1.0

    def test_discrepancies_sorted_worst_first(self, db: Database) -> None:
        # P_LOW_HIGH: expert=low, predicted=high → distance 2 (extreme)
        db.insert_paper(_make_expert_paper("P_LOW_HIGH", overall="low"))
        db.insert_annotation(
            "P_LOW_HIGH", "anthropic",
            _make_prediction_annotation("P_LOW_HIGH", overall="high"),
            methodology="cochrane_rob2",
        )
        # P_LOW_SOMECONCERNS: expert=low, predicted=some_concerns → distance 1
        db.insert_paper(_make_expert_paper("P_ADJ", overall="low"))
        db.insert_annotation(
            "P_ADJ", "anthropic",
            _make_prediction_annotation("P_ADJ", overall="some_concerns"),
            methodology="cochrane_rob2",
        )
        # Matching row — should not appear in discrepancies
        db.insert_paper(_make_expert_paper("P_MATCH", overall="low"))
        db.insert_annotation(
            "P_MATCH", "anthropic",
            _make_prediction_annotation("P_MATCH", overall="low"),
            methodology="cochrane_rob2",
        )
        paired = collect_paired_papers(db, "anthropic")
        report = build_report(
            paired, model_name="anthropic", n_model_annotations=3,
        )
        assert len(report.discrepancies) == 2
        assert report.discrepancies[0]["pmid"] == "P_LOW_HIGH"
        assert report.discrepancies[0]["distance"] == 2
        assert report.discrepancies[1]["pmid"] == "P_ADJ"
        assert report.discrepancies[1]["distance"] == 1

    def test_per_domain_series_populated(self, db: Database) -> None:
        db.insert_paper(_make_expert_paper(
            "P1", overall="some_concerns",
            randomization="low", deviation="some_concerns",
            missing_outcome="low", measurement="low", reporting="low",
        ))
        db.insert_annotation(
            "P1", "anthropic",
            _make_prediction_annotation(
                "P1", overall="some_concerns",
                per_domain={
                    "randomization": "low",
                    "deviations_from_interventions": "some_concerns",
                    "missing_outcome_data": "low",
                    "outcome_measurement": "low",
                    "selection_of_reported_result": "low",
                },
            ),
            methodology="cochrane_rob2",
        )
        paired = collect_paired_papers(db, "anthropic")
        report = build_report(
            paired, model_name="anthropic", n_model_annotations=1,
        )
        for slug in ROB2_DOMAIN_SLUGS:
            assert report.per_domain[slug].n == 1
            assert report.per_domain[slug].exact_match() == 1.0


class TestRenderMarkdown:
    def test_empty_report_renders_no_paired_note(self) -> None:
        report = FaithfulnessReport(
            model_name="anthropic",
            methodology_version=METHODOLOGY_VERSION,
            n_paired=0, n_model_annotations=0,
            overall=JudgementSeries(),
            per_domain={slug: JudgementSeries() for slug in ROB2_DOMAIN_SLUGS},
            discrepancies=[],
        )
        md = render_markdown(report)
        assert "Cochrane RoB 2 faithfulness" in md
        assert "No paired papers" in md

    def test_non_empty_report_includes_metrics_and_confusion(
        self, db: Database,
    ) -> None:
        db.insert_paper(_make_expert_paper("P1", overall="low"))
        db.insert_annotation(
            "P1", "anthropic",
            _make_prediction_annotation("P1", overall="low"),
            methodology="cochrane_rob2",
        )
        paired = collect_paired_papers(db, "anthropic")
        report = build_report(
            paired, model_name="anthropic", n_model_annotations=1,
        )
        md = render_markdown(report)
        assert "Overall judgement agreement" in md
        assert "exact_match=1.000" in md
        assert "weighted_kappa=" in md
        # Per-domain section present for each of the 5 RoB 2 domains
        assert md.count("###") == 5

    def test_report_json_roundtrips(self, db: Database) -> None:
        db.insert_paper(_make_expert_paper("P1", overall="high"))
        db.insert_annotation(
            "P1", "anthropic",
            _make_prediction_annotation("P1", overall="some_concerns"),
            methodology="cochrane_rob2",
        )
        paired = collect_paired_papers(db, "anthropic")
        report = build_report(
            paired, model_name="anthropic", n_model_annotations=1,
        )
        serialised = json.loads(json.dumps(report.to_dict()))
        assert serialised["model_name"] == "anthropic"
        assert serialised["n_paired"] == 1
        assert len(serialised["discrepancies"]) == 1
        assert serialised["discrepancies"][0]["distance"] == 1

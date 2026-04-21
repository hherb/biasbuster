"""Coverage for scripts/backfill_rob2_expert_ratings.py.

What matters for correctness of the faithfulness harness:

- Slug naming matches ``evaluation_mapping_to_ground_truth`` so a
  RoB 2 prediction can join to its backfilled expert row.
- ``domain_ratings`` is stored in the same ``{slug: {"bias": rating}}``
  shape QUADAS-2 uses, so the generic harness doesn't branch on
  methodology when reading the column.
- Off-vocabulary ratings are rejected rather than silently stored;
  otherwise kappa is computed over unrecognised cells and inflated.
- Rows without ``cochrane_review_pmid`` / ``doi`` are skipped — we
  refuse to conjure ground truth without clear provenance.
- Re-running is idempotent: second pass returns ``updated`` for the
  same rows (matches the upsert contract tested in
  ``test_expert_methodology_ratings``).
- Curator fields (``verified``, ``notes``) survive a re-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biasbuster.database import Database
from biasbuster.methodologies.cochrane_rob2 import (
    evaluation_mapping_to_ground_truth,
)
from scripts.backfill_rob2_expert_ratings import (
    _DOMAIN_SLUG_TO_COLUMN,
    METHODOLOGY,
    METHODOLOGY_VERSION,
    backfill,
)


def _complete_rob2_paper(
    pmid: str,
    *,
    overall: str = "low",
    randomization: str = "low",
    deviation: str = "low",
    missing_outcome: str = "low",
    measurement: str = "low",
    reporting: str = "low",
    review_pmid: str = "99999",
    review_doi: str = "10.1/cochrane-review",
    review_title: str = "Cochrane review of X",
    title: str = "Included RCT",
) -> dict:
    """Paper dict with all RoB 2 columns populated and review provenance."""
    return {
        "pmid": pmid, "title": title, "abstract": "abs",
        "source": "cochrane_rob",
        "overall_rob": overall,
        "randomization_bias": randomization,
        "deviation_bias": deviation,
        "missing_outcome_bias": missing_outcome,
        "measurement_bias": measurement,
        "reporting_bias": reporting,
        "cochrane_review_pmid": review_pmid,
        "cochrane_review_doi": review_doi,
        "cochrane_review_title": review_title,
    }


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "backfill.db")
    d.initialize()
    yield d
    d.close()


# ---- Slug alignment ----------------------------------------------------

class TestSlugAlignment:
    def test_backfill_slugs_match_evaluation_mapping(self) -> None:
        """The harness pairs expert and prediction on domain slugs; the
        backfill and :func:`evaluation_mapping_to_ground_truth` MUST
        agree on the slug set or every join silently misses.
        """
        backfill_slugs = set(_DOMAIN_SLUG_TO_COLUMN.keys())
        # Construct a dummy paper so we can read the mapping's slugs.
        paper = _complete_rob2_paper("P")
        mapping = evaluation_mapping_to_ground_truth(paper)
        assert mapping is not None
        assert set(mapping["domains"].keys()) == backfill_slugs


# ---- Happy path --------------------------------------------------------

class TestHappyPath:
    def test_backfill_inserts_expected_row(
        self, db: Database, tmp_path: Path,
    ) -> None:
        db.insert_paper(_complete_rob2_paper("P1", overall="some_concerns"))
        counts = backfill(tmp_path / "backfill.db", added_by="t")
        assert counts["eligible"] == 1
        assert counts["inserted"] == 1
        assert counts["updated"] == 0
        assert counts["skipped_invalid_rating"] == 0
        assert counts["skipped_no_source"] == 0

        [row] = db.get_expert_ratings(methodology=METHODOLOGY)
        assert row["study_label"] == "P1"
        assert row["pmid"] == "P1"
        assert row["overall_rating"] == "some_concerns"
        assert row["methodology_version"] == METHODOLOGY_VERSION
        assert row["added_by"] == "t"
        # Storage shape matches QUADAS-2: {slug: {"bias": rating}}.
        assert row["domain_ratings"] == {
            "randomization": {"bias": "low"},
            "deviations_from_interventions": {"bias": "low"},
            "missing_outcome_data": {"bias": "low"},
            "outcome_measurement": {"bias": "low"},
            "selection_of_reported_result": {"bias": "low"},
        }

    def test_rating_source_uses_review_pmid_when_present(
        self, db: Database, tmp_path: Path,
    ) -> None:
        db.insert_paper(_complete_rob2_paper("P1", review_pmid="12345"))
        backfill(tmp_path / "backfill.db")
        [row] = db.get_expert_ratings(methodology=METHODOLOGY)
        assert row["rating_source"] == "cochrane_review_pmid:12345"

    def test_rating_source_falls_back_to_review_doi(
        self, db: Database, tmp_path: Path,
    ) -> None:
        """No PMID but a DOI is still identifiable provenance."""
        paper = _complete_rob2_paper("P1", review_pmid="")
        paper["cochrane_review_doi"] = "10.1002/x"
        db.insert_paper(paper)
        backfill(tmp_path / "backfill.db")
        [row] = db.get_expert_ratings(methodology=METHODOLOGY)
        assert row["rating_source"] == "cochrane_review_doi:10.1002/x"


# ---- Invalid data is rejected -----------------------------------------

class TestInvalidData:
    def test_off_vocabulary_rating_skipped(
        self, db: Database, tmp_path: Path,
    ) -> None:
        """Legacy rows with e.g. ``overall_rob='critical'`` must be skipped."""
        bad = _complete_rob2_paper("P1")
        bad["overall_rob"] = "CRITICAL"  # not in RoB 2 vocabulary
        db.insert_paper(bad)
        counts = backfill(tmp_path / "backfill.db")
        assert counts["inserted"] == 0
        assert counts["skipped_invalid_rating"] == 1
        assert db.get_expert_ratings() == []

    def test_domain_off_vocabulary_skipped(
        self, db: Database, tmp_path: Path,
    ) -> None:
        bad = _complete_rob2_paper("P1")
        bad["randomization_bias"] = "moderate"  # 5-level not 3-level
        db.insert_paper(bad)
        counts = backfill(tmp_path / "backfill.db")
        assert counts["inserted"] == 0
        assert counts["skipped_invalid_rating"] == 1

    def test_paper_without_review_provenance_skipped(
        self, db: Database, tmp_path: Path,
    ) -> None:
        orphan = _complete_rob2_paper("P1", review_pmid="")
        orphan["cochrane_review_doi"] = ""
        db.insert_paper(orphan)
        counts = backfill(tmp_path / "backfill.db")
        assert counts["inserted"] == 0
        assert counts["skipped_no_source"] == 1

    def test_excluded_paper_not_backfilled(
        self, db: Database, tmp_path: Path,
    ) -> None:
        """Soft-deleted papers shouldn't produce expert rows."""
        db.insert_paper(_complete_rob2_paper("P1"))
        db.conn.execute(
            "UPDATE papers SET excluded=1, excluded_reason='test' "
            "WHERE pmid='P1'"
        )
        db.conn.commit()
        counts = backfill(tmp_path / "backfill.db")
        assert counts["eligible"] == 0
        assert counts["inserted"] == 0


# ---- Idempotency ------------------------------------------------------

class TestIdempotency:
    def test_second_run_reports_updated_not_inserted(
        self, db: Database, tmp_path: Path,
    ) -> None:
        db.insert_paper(_complete_rob2_paper("P1"))
        first = backfill(tmp_path / "backfill.db")
        second = backfill(tmp_path / "backfill.db")
        assert first["inserted"] == 1
        assert first["updated"] == 0
        assert second["inserted"] == 0
        assert second["updated"] == 1

    def test_curator_sign_off_survives_rerun(
        self, db: Database, tmp_path: Path,
    ) -> None:
        """``verified=1`` set by curator must not be cleared on re-backfill."""
        db.insert_paper(_complete_rob2_paper("P1"))
        backfill(tmp_path / "backfill.db")
        db.conn.execute(
            "UPDATE expert_methodology_ratings SET verified=1, notes='OK' "
            "WHERE study_label='P1'"
        )
        db.conn.commit()
        # Re-run — simulates "we discovered a new Cochrane review ingest"
        backfill(tmp_path / "backfill.db")
        [row] = db.get_expert_ratings(methodology=METHODOLOGY)
        assert row["verified"] == 1
        assert row["notes"] == "OK"


# ---- Dry-run ----------------------------------------------------------

class TestDryRun:
    def test_dry_run_writes_nothing(
        self, db: Database, tmp_path: Path,
    ) -> None:
        db.insert_paper(_complete_rob2_paper("P1"))
        counts = backfill(tmp_path / "backfill.db", dry_run=True)
        assert counts["inserted"] == 1  # would insert
        assert db.get_expert_ratings() == []

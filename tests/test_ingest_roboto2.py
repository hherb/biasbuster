"""Tests for the pure ROBoto2 record parser (Stage A seed pool 1).

Only ``parse_roboto2_record`` is unit-tested here: it is a pure function
(no network I/O), so it can be exercised offline against the synthetic
fixture. The async ``ingest_roboto2`` orchestration performs real network
calls (Europe PMC OA status, PubMed PublicationType, JATS fetch) and is
exercised end-to-end only from a terminal against the real ROBoto2 dataset
file (CLAUDE.md's >2-minute-process rule; see the module's ``__main__``).
"""
import json
from pathlib import Path

from studies.oa_rob_benchmark.ingest_roboto2 import parse_roboto2_record

RECS = json.loads(Path("tests/fixtures/oa_rob/roboto2_sample.json").read_text())


def test_parse_manual_record_extracts_pmid_and_answers():
    out = parse_roboto2_record(RECS[0])
    assert out is not None
    paper_id, domain_answers, overall = out
    assert paper_id == "111"
    assert domain_answers["randomization"] == {"1.1": "Y", "1.2": "N", "1.3": "N"}


def test_llm_assisted_only_record_is_dropped():
    assert parse_roboto2_record(RECS[1]) is None

"""Coverage for scripts/fetch_fulltext_for_expert_ratings.py.

The script itself is a thin shell over Europe PMC plus the shared
JATS cache. Tests focus on the behaviours that affect correctness
rather than mock Europe PMC's exact JSON:

- Cache skip: a PMID with a non-stub cached file is reported as
  ``cached`` and no HTTP traffic is attempted.
- Stub-size guard: the script must never write a tiny response to
  the cache as if it were valid JATS.
- Report CSV has the documented columns and one row per input PMID.
- ``_rated_pmids`` filters by methodology and drops NULL-PMID rows.
"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any

import pytest

from biasbuster.database import Database
from scripts.fetch_fulltext_for_expert_ratings import (
    MIN_JATS_BYTES,
    _already_cached,
    _rated_pmids,
    _summarise,
    _write_report,
    fetch_jats,
    probe_and_fetch,
)


# ---- Cache behaviour ---------------------------------------------------

class TestCacheBehaviour:
    def test_already_cached_true_for_full_file(
        self, tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "1.jats.xml").write_bytes(b"x" * (MIN_JATS_BYTES + 1))
        assert _already_cached(cache, "1") is True

    def test_already_cached_false_for_stub(self, tmp_path: Path) -> None:
        """A too-small file is treated as absent so probe_and_fetch re-tries."""
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "2.jats.xml").write_bytes(b"<tiny/>")
        assert _already_cached(cache, "2") is False

    def test_already_cached_false_for_missing(self, tmp_path: Path) -> None:
        assert _already_cached(tmp_path, "nope") is False


# ---- _rated_pmids ------------------------------------------------------

class TestRatedPmidsQuery:
    def test_returns_distinct_non_null_pmids(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "r.db")
        db.initialize()
        try:
            db.upsert_expert_rating(
                methodology="quadas_2", rating_source="t",
                study_label="A",
                domain_ratings={"patient_selection": {"bias": "low"}},
                overall_rating="low", pmid="100",
            )
            db.upsert_expert_rating(
                methodology="quadas_2", rating_source="t",
                study_label="B",
                domain_ratings={"patient_selection": {"bias": "high"}},
                overall_rating="high", pmid="100",  # duplicate PMID
            )
            db.upsert_expert_rating(
                methodology="quadas_2", rating_source="t",
                study_label="C",
                domain_ratings={"patient_selection": {"bias": "low"}},
                overall_rating="low", pmid=None,  # dropped
            )
            db.upsert_expert_rating(
                methodology="cochrane_rob2", rating_source="t",
                study_label="D",
                domain_ratings={"randomization": "low"},
                overall_rating="low", pmid="200",
            )
        finally:
            db.close()

        assert _rated_pmids(tmp_path / "r.db", "quadas_2") == ["100"]
        assert _rated_pmids(tmp_path / "r.db", None) == ["100", "200"]


# ---- fetch_jats stub-size guard ----------------------------------------

class _FakeResp:
    def __init__(self, content: bytes) -> None:
        self.content = content


class _FakeClient:
    """Captures request URLs and returns pre-canned responses."""

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.calls: list[str] = []

    async def get(self, url: str, **kwargs: Any) -> _FakeResp:  # noqa: D401
        """Swallow retry kwargs and return the pre-canned body."""
        self.calls.append(url)
        return _FakeResp(self._body)


class TestFetchJatsStubGuard:
    def test_tiny_response_not_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 10-byte stub must not pollute the cache as if it were JATS."""
        import scripts.fetch_fulltext_for_expert_ratings as mod

        async def fake_fetch(client, method, url, **kwargs):
            return _FakeResp(b"<nope/>")

        monkeypatch.setattr(mod, "fetch_with_retry", fake_fetch)
        client = _FakeClient(b"<nope/>")

        status, n_bytes = asyncio.run(
            fetch_jats(client, "9999", "PMC9999", tmp_path)
        )
        assert status == "too_small"
        assert n_bytes == 0
        assert not (tmp_path / "9999.jats.xml").exists()

    def test_full_response_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scripts.fetch_fulltext_for_expert_ratings as mod

        body = b"<article>" + b"x" * MIN_JATS_BYTES + b"</article>"

        async def fake_fetch(client, method, url, **kwargs):
            return _FakeResp(body)

        monkeypatch.setattr(mod, "fetch_with_retry", fake_fetch)
        client = _FakeClient(body)

        status, n_bytes = asyncio.run(
            fetch_jats(client, "9999", "PMC9999", tmp_path)
        )
        assert status == "ok"
        assert n_bytes == len(body)
        assert (tmp_path / "9999.jats.xml").read_bytes() == body


# ---- probe_and_fetch higher-level behaviour ----------------------------

class TestProbeAndFetch:
    def test_cached_pmid_skips_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cached file means no Europe PMC call is attempted."""
        import scripts.fetch_fulltext_for_expert_ratings as mod

        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "111.jats.xml").write_bytes(b"y" * (MIN_JATS_BYTES + 10))

        # Sentinel that explodes if called — proves no HTTP traffic.
        async def forbidden(*args, **kwargs):
            raise AssertionError(
                "resolve_pmcid / fetch_jats must not run for cached PMIDs"
            )

        monkeypatch.setattr(mod, "resolve_pmcid", forbidden)
        monkeypatch.setattr(mod, "fetch_jats", forbidden)

        records = asyncio.run(probe_and_fetch(
            ["111"], cache, request_delay=0,
        ))
        assert records == [{
            "pmid": "111", "pmcid": "",
            "status": "cached",
            "bytes": str((cache / "111.jats.xml").stat().st_size),
        }]


# ---- Summary + report --------------------------------------------------

class TestSummariseAndReport:
    def test_summarise_counts_by_status(self) -> None:
        records = [
            {"pmid": "1", "pmcid": "", "status": "cached", "bytes": "2000"},
            {"pmid": "2", "pmcid": "PMC2", "status": "ok", "bytes": "3000"},
            {"pmid": "3", "pmcid": "",
             "status": "not_in_pmc", "bytes": "0"},
            {"pmid": "4", "pmcid": "PMC4", "status": "ok", "bytes": "4000"},
        ]
        assert _summarise(records) == {
            "cached": 1, "ok": 2, "not_in_pmc": 1,
        }

    def test_report_has_expected_columns(self, tmp_path: Path) -> None:
        records = [
            {"pmid": "1", "pmcid": "PMC1",
             "status": "ok", "bytes": "2500"},
        ]
        out = tmp_path / "report.csv"
        _write_report(out, records)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert rows == records

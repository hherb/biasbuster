# OA-First RoB Benchmark — Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the trusted, publishable expert-labelled open-access core of the RoB benchmark (spec §6), plus the two shared collectors it needs, ending at the mandatory 20-row manual gate.

**Architecture:** Compose existing, already-implemented infrastructure — `biasbuster/collectors/rob_table_extractor.py` (structural RoB 2 table extraction), `biasbuster/utils/pubtype.py` (PublicationType classifier), `biasbuster/utils/retry.py` (backoff), `biasbuster/methodologies/cochrane_rob2/algorithms.py` (signalling-answers → domain judgement), and `scripts/fetch_fulltext_for_expert_ratings.py` (Europe PMC JATS fetch) — behind two **new** shared collectors (OA-license resolver, clean study→PMID linkage resolver) and a **new isolated study store**. Ingest ROBoto2 manual-gold + EM OA candidates into that store, then emit a manual-check manifest. No component touches `dataset/biasbuster.db` or `dataset/eisele_metzger_benchmark.db`.

**Tech Stack:** Python 3.11+, `uv`, `httpx` (async), `pytest` + `pytest-asyncio`, `sqlite3` (stdlib), `xml.etree.ElementTree` (stdlib), `difflib` (stdlib).

**Full design:** `docs/superpowers/specs/2026-07-17-oa-first-rob-benchmark-design.md`. Read it before starting; this plan implements §6 and the §4 litmus test only. Stage B (§7) is a separate follow-up plan.

## Global Constraints

Copied verbatim from CLAUDE.md and the spec — every task's requirements implicitly include these:

- **`uv` only.** Run tests with `uv run pytest`. Add deps with `uv add`, never `pip`/`venv`.
- **Docstrings + type hints mandatory** on every function/method.
- **No magic numbers** — module-level named constants (e.g. `TITLE_SIMILARITY_THRESHOLD = 0.70`).
- **Network functions retry with exponential backoff** up to `MAX_RETRIES`; reuse `biasbuster/utils/retry.py` (`fetch_with_retry`, `retry_with_backoff`). Never add a new bespoke retry loop.
- **All caught errors are logged and surfaced** — never a silent `except: pass`.
- **Incremental, resumable saves** — per-item upsert as each item is produced (an `on_result`-style write), never batch-in-memory-then-save-at-end. Rejections are logged to a rejects file, never silently skipped.
- **Never truncate full text destined for analysis** — chunk-and-map-reduce or skip-with-log; never trim.
- **Files stay < 500 lines** where feasible; split by responsibility early.
- **Processes > 2 min are never run in-session** — the plan's `__main__` runners print the command for the owner to run in their own terminal; only unit tests (fast, fixture/stub-based) run in-session.
- **Isolated store.** All new persistence goes to `dataset/oa_rob_benchmark.db` (new). Its builder must be append/upsert — **never** DROP-and-rebuild — mirroring the HANDOVER gotcha about `build_benchmark_db.py`.
- **Canonical RoB 2 rating vocabulary for this benchmark:** exactly `"low"`, `"some concerns"`, `"high"` (space form, per spec §4.2). All sources normalise to these three; anything else (`""`, `"unclear"`, `"n/a"`) is a reject. Note: `rob_table_extractor` emits `"some_concerns"` (underscore) — the tuple assembler (Task 3) maps it to `"some concerns"`.
- **Ground truth = human-expert RoB 2 only.** Never admit an automated/LLM-assisted label (drop ROBoto2's `roboto2_assessment` rows; drop the EM supplement's copied labels — re-derive from the primary review).
- **Labels stored as data + citation**, never source-review prose (spec §5).
- **Excluded RCTs:** import `WRONG_PAPER_RCTS` from `studies/eisele_metzger_replication/exclusions.py`; RCT030 never enters this benchmark.

---

## File Structure

**New shared collectors** (general-purpose, reused by Stage B later) — `biasbuster/collectors/`:
- `oa_license.py` — PMC-OA-subset + license classification for a PMID/PMCID. One responsibility: "is this trial redistributable, and under what license?"
- `study_pmid_resolver.py` — the clean study-id → trial-PMID resolver (bracket-ref, author+year+title-similarity). One responsibility: "which PMID does this table row refer to, with what confidence?" Replaces the anti-pattern resolvers in `cochrane_rob.py` (never surname-only, never first-of-many).

**New study package** — `studies/oa_rob_benchmark/`:
- `rob2_tuple.py` — assemble+validate the six-field RoB 2 tuple from either an `ExtractedStudyRow` (table path) or a dict of signalling answers (ROBoto2 path). One responsibility: "produce a complete, canonical tuple or reject."
- `store.py` — the isolated SQLite store: schema, litmus-enforcing `upsert_item`, reject-log writer. One responsibility: persistence + invariant.
- `ingest_roboto2.py` — Stage A seed pool 1 orchestration.
- `ingest_em_candidates.py` — Stage A seed pool 2 orchestration.
- `manual_gate.py` — 20-row manual-check manifest generator.
- `export.py` — redistributable artifact export (identifiers + license + tuple + provenance; CC full text only).
- `__init__.py`

**New script** — `scripts/audit_oa_rob_benchmark.py` — re-checks all four litmus rules across the finished DB.

**New tests** — `tests/`:
- `test_oa_license.py`, `test_study_pmid_resolver.py`, `test_rob2_tuple.py`,
  `test_oa_rob_store.py`, `test_ingest_roboto2.py`, `test_ingest_em_candidates.py`,
  `test_oa_rob_manual_gate.py`, `test_rob_table_extractor.py` (characterization test for the reused extractor).

**New fixtures** — `tests/fixtures/oa_rob/`: small synthetic ROBoto2 record JSON, a minimal RoB 2 JATS table, and Europe PMC/efetch response snippets. (`tests/fixtures/cochrane_reviews/jcm-15-01829.xml` already exists and is reused in Task 3.)

**Reused as-is (do not modify):** `rob_table_extractor.py`, `pubtype.py`, `retry.py`, `methodologies/cochrane_rob2/algorithms.py`, `scripts/fetch_fulltext_for_expert_ratings.py` (`resolve_pmcid`/`fetch_jats`), `exclusions.py`.

---

## Task 1: OA-license resolver

**Files:**
- Create: `biasbuster/collectors/oa_license.py`
- Test: `tests/test_oa_license.py`

**Interfaces:**
- Consumes: `httpx.AsyncClient` (injected); Europe PMC search API (`config.europmc_base`); `biasbuster/utils/retry.py::fetch_with_retry`.
- Produces:
  - `@dataclass(frozen=True) LicenseInfo(raw: str, spdx: str, redistributable: bool, non_commercial: bool, no_derivatives: bool)`
  - `classify_license(raw_license: str) -> LicenseInfo` (pure)
  - `@dataclass OAStatus(pmid: str, pmcid: str, in_oa_subset: bool, license: LicenseInfo)`
  - `async def fetch_oa_status(client: httpx.AsyncClient, pmid: str, *, base: str) -> OAStatus`

- [ ] **Step 1: Write the failing test for `classify_license`**

```python
# tests/test_oa_license.py
from biasbuster.collectors.oa_license import classify_license

def test_cc_by_is_redistributable_unrestricted():
    info = classify_license("CC BY")
    assert info.spdx == "CC-BY-4.0"
    assert info.redistributable is True
    assert info.non_commercial is False
    assert info.no_derivatives is False

def test_cc_by_nc_nd_flags_both():
    info = classify_license("CC BY-NC-ND")
    assert info.redistributable is True   # all CC permit verbatim redistribution
    assert info.non_commercial is True
    assert info.no_derivatives is True

def test_cc0_public_domain():
    info = classify_license("CC0")
    assert info.spdx == "CC0-1.0"
    assert info.redistributable is True

def test_unknown_or_all_rights_reserved_not_redistributable():
    assert classify_license("").redistributable is False
    assert classify_license("NO-CC BY").redistributable is False
    assert classify_license("copyright, all rights reserved").redistributable is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_oa_license.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'biasbuster.collectors.oa_license'`

- [ ] **Step 3: Implement `classify_license` (pure)**

```python
# biasbuster/collectors/oa_license.py
"""PMC Open-Access-subset membership and license classification.

Given a PubMed/PMC identifier, determine whether a trial is in the PMC
Open Access Subset and under which license, so the OA-first benchmark can
admit only redistributable full text (spec §4.1) and flag NC/ND per item
(spec §2 license-strictness decision).

Pure classification (`classify_license`) is separated from network I/O
(`fetch_oa_status`) so the licensing logic is unit-testable offline.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

from biasbuster.utils.retry import fetch_with_retry

logger = logging.getLogger(__name__)

#: Europe PMC returns license strings like "cc by", "cc by-nc", "cc0".
_CC_TOKEN = re.compile(r"cc[\s\-]?by", re.IGNORECASE)
_CC0_TOKEN = re.compile(r"cc0|public domain", re.IGNORECASE)


@dataclass(frozen=True)
class LicenseInfo:
    """Normalised license facts for one document."""
    raw: str
    spdx: str
    redistributable: bool
    non_commercial: bool
    no_derivatives: bool


def classify_license(raw_license: str) -> LicenseInfo:
    """Classify a raw license string into redistribution facts.

    Every Creative Commons license permits verbatim redistribution, so
    ``redistributable`` is True for any CC/CC0 license; ``non_commercial``
    and ``no_derivatives`` flag the NC and ND downstream constraints. A
    blank or all-rights-reserved string is not redistributable.
    """
    raw = (raw_license or "").strip()
    low = raw.lower()
    if _CC0_TOKEN.search(low):
        return LicenseInfo(raw, "CC0-1.0", True, False, False)
    if _CC_TOKEN.search(low) and "no-cc" not in low:
        nc = "nc" in low.replace("no-cc", "")
        nd = "nd" in low
        sa = "sa" in low
        parts = ["CC-BY"]
        if nc:
            parts.append("NC")
        if nd:
            parts.append("ND")
        elif sa:
            parts.append("SA")
        spdx = "-".join(parts) + "-4.0"
        return LicenseInfo(raw, spdx, True, nc, nd)
    return LicenseInfo(raw, "", False, False, False)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_oa_license.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Write the failing test for `fetch_oa_status` (async, fake client)**

```python
# append to tests/test_oa_license.py
import pytest
from biasbuster.collectors.oa_license import fetch_oa_status

class _FakeResponse:
    def __init__(self, payload): self._payload = payload; self.status_code = 200
    def json(self): return self._payload
    def raise_for_status(self): pass

class _FakeClient:
    def __init__(self, payload): self._payload = payload; self.calls = []
    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs)); return _FakeResponse(self._payload)

@pytest.mark.asyncio
async def test_fetch_oa_status_parses_epmc_result():
    payload = {"resultList": {"result": [
        {"pmid": "12345", "pmcid": "PMC999", "isOpenAccess": "Y",
         "inEPMC": "Y", "license": "cc by"}]}}
    client = _FakeClient(payload)
    status = await fetch_oa_status(client, "12345", base="https://epmc.test")
    assert status.pmcid == "PMC999"
    assert status.in_oa_subset is True
    assert status.license.redistributable is True
    assert status.license.non_commercial is False

@pytest.mark.asyncio
async def test_fetch_oa_status_missing_pmc_is_not_oa():
    payload = {"resultList": {"result": [
        {"pmid": "222", "isOpenAccess": "N"}]}}
    status = await fetch_oa_status(_FakeClient(payload), "222", base="https://epmc.test")
    assert status.in_oa_subset is False
    assert status.license.redistributable is False
```

- [ ] **Step 6: Run to verify it fails**

Run: `uv run pytest tests/test_oa_license.py -k fetch_oa_status -v`
Expected: FAIL with `ImportError: cannot import name 'fetch_oa_status'`

- [ ] **Step 7: Implement `fetch_oa_status`**

```python
# append to biasbuster/collectors/oa_license.py

@dataclass
class OAStatus:
    """OA-subset membership + license for one PMID."""
    pmid: str
    pmcid: str
    in_oa_subset: bool
    license: LicenseInfo


async def fetch_oa_status(
    client: httpx.AsyncClient, pmid: str, *, base: str
) -> OAStatus:
    """Query Europe PMC for OA-subset membership and license of ``pmid``.

    A trial is treated as in the OA subset only when Europe PMC reports it
    open-access, present in EPMC full text, and under a redistributable
    (CC/CC0) license. Any query or parse failure yields a non-OA status
    (fail-closed) after retries — logged, never raised to the caller.
    """
    url = f"{base}/search"
    params = {
        "query": f"EXT_ID:{pmid} AND SRC:MED",
        "resultType": "core",
        "format": "json",
    }
    try:
        resp = await fetch_with_retry(client, "GET", url, params=params)
        results = resp.json().get("resultList", {}).get("result", [])
    except Exception as exc:  # network/parse — fail closed, logged
        logger.warning("OA status fetch failed for PMID %s: %s", pmid, exc)
        return OAStatus(pmid, "", False, classify_license(""))

    if not results:
        return OAStatus(pmid, "", False, classify_license(""))
    r = results[0]
    pmcid = str(r.get("pmcid", "") or "")
    lic = classify_license(str(r.get("license", "") or ""))
    in_oa = (
        str(r.get("isOpenAccess", "")).upper() == "Y"
        and str(r.get("inEPMC", "")).upper() == "Y"
        and bool(pmcid)
        and lic.redistributable
    )
    return OAStatus(pmid, pmcid, in_oa, lic)
```

Note: confirm `fetch_with_retry`'s call signature in `biasbuster/utils/retry.py` and adapt the `await fetch_with_retry(...)` line to it (it wraps `client.request`). If its signature differs, wrap `client.get` in `retry_with_backoff` instead — do not write a new retry loop.

- [ ] **Step 8: Run to verify it passes**

Run: `uv run pytest tests/test_oa_license.py -v`
Expected: PASS (6 tests)

- [ ] **Step 9: Commit**

```bash
git add biasbuster/collectors/oa_license.py tests/test_oa_license.py
git commit -m "feat(collectors): OA-subset + license resolver for OA-first benchmark"
```

---

## Task 2: Clean study→PMID linkage resolver

**Files:**
- Create: `biasbuster/collectors/study_pmid_resolver.py`
- Test: `tests/test_study_pmid_resolver.py`

**Interfaces:**
- Consumes: nothing external — pure functions over a parsed reference list.
- Produces:
  - `@dataclass(frozen=True) Reference(ref_number: str, pmid: str, first_author: str, year: str, title: str)`
  - `@dataclass(frozen=True) Resolution(pmid: str, method: str, similarity: float)` where `method ∈ {"bracket_ref", "author_year_title", "direct_ref_pmid", "unresolved"}`
  - `resolve_study_pmid(study_id: str, ref_number: str, references: list[Reference], *, threshold: float = TITLE_SIMILARITY_THRESHOLD) -> Resolution` (pure)
  - `TITLE_SIMILARITY_THRESHOLD = 0.70`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_study_pmid_resolver.py
from biasbuster.collectors.study_pmid_resolver import (
    Reference, resolve_study_pmid, TITLE_SIMILARITY_THRESHOLD,
)

REFS = [
    Reference("28", "111", "Smith", "2020", "Effect of drug A on outcome X"),
    Reference("29", "222", "Smith", "2019", "A different Smith trial on Y"),
    Reference("30", "333", "Jones", "2021", "Jones pragmatic trial of Z"),
]

def test_bracket_ref_wins_directly():
    r = resolve_study_pmid("Smith 2020", "28", REFS)
    assert r.pmid == "111"
    assert r.method == "bracket_ref"

def test_author_year_title_disambiguates_two_smiths():
    # No bracket number; two Smiths — title similarity picks the right one.
    r = resolve_study_pmid("Smith 2020 Effect of drug A on outcome X", "", REFS)
    assert r.pmid == "111"
    assert r.method == "author_year_title"
    assert r.similarity >= TITLE_SIMILARITY_THRESHOLD

def test_surname_only_is_rejected_not_guessed():
    # Bare surname, no year/title evidence, multiple Smiths → unresolved.
    r = resolve_study_pmid("Smith", "", REFS)
    assert r.method == "unresolved"
    assert r.pmid == ""

def test_below_threshold_is_unresolved():
    r = resolve_study_pmid("Smith 2020 totally unrelated wording here", "", REFS,
                           threshold=0.95)
    assert r.method == "unresolved"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_study_pmid_resolver.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the resolver (pure)**

```python
# biasbuster/collectors/study_pmid_resolver.py
"""Clean study-id → trial-PMID resolver for RoB table rows.

Implements REBUILD_DESIGN.md §5 / FORENSICS.md §6.3: resolve a table row's
study identifier to a trial PMID using only high-confidence evidence —
a bracketed reference number, or an author+year match confirmed by title
similarity. The anti-patterns that corrupted the earlier corpus are
deliberately absent: no surname-only matching (FORENSICS §3.3), no
"first of many" PubMed search result (FORENSICS §3.4). An ambiguous row
resolves to ``unresolved`` and is dropped by the caller, never guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

#: Minimum title similarity to accept an author+year candidate.
TITLE_SIMILARITY_THRESHOLD = 0.70

_STUDY_ID_RE = re.compile(r"([A-Z][A-Za-z'\-]+)\s+((?:19|20)\d{2})\s*(.*)")


@dataclass(frozen=True)
class Reference:
    """One entry from a review's reference list."""
    ref_number: str
    pmid: str
    first_author: str
    year: str
    title: str


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving one study id to a PMID."""
    pmid: str
    method: str
    similarity: float


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def resolve_study_pmid(
    study_id: str,
    ref_number: str,
    references: list[Reference],
    *,
    threshold: float = TITLE_SIMILARITY_THRESHOLD,
) -> Resolution:
    """Resolve ``study_id`` to a trial PMID with a recorded method+confidence.

    Priority: (1) bracket reference number → direct PMID; (2) author+year
    candidates disambiguated by title similarity ≥ ``threshold``. If neither
    yields a confident match, returns an ``unresolved`` Resolution.
    """
    by_number = {r.ref_number: r for r in references if r.ref_number}
    if ref_number and ref_number in by_number and by_number[ref_number].pmid:
        return Resolution(by_number[ref_number].pmid, "bracket_ref", 1.0)

    m = _STUDY_ID_RE.match(study_id.strip())
    if not m:
        return Resolution("", "unresolved", 0.0)
    author, year, tail = m.group(1).lower(), m.group(2), m.group(3).strip()

    candidates = [
        r for r in references
        if r.first_author.lower() == author and r.year == year and r.pmid
    ]
    if not candidates:
        return Resolution("", "unresolved", 0.0)
    if len(candidates) == 1 and tail:
        # single author+year hit, but still require title evidence to accept
        sim = _similarity(tail, candidates[0].title)
        if sim >= threshold:
            return Resolution(candidates[0].pmid, "author_year_title", sim)
        return Resolution("", "unresolved", sim)

    # Multiple candidates (or no title tail): pick best title match ≥ threshold
    best, best_sim = None, 0.0
    for c in candidates:
        sim = _similarity(tail, c.title)
        if sim > best_sim:
            best, best_sim = c, sim
    if best is not None and best_sim >= threshold:
        return Resolution(best.pmid, "author_year_title", best_sim)
    return Resolution("", "unresolved", best_sim)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_study_pmid_resolver.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add biasbuster/collectors/study_pmid_resolver.py tests/test_study_pmid_resolver.py
git commit -m "feat(collectors): clean study->PMID resolver (no surname-only, no first-of-many)"
```

---

## Task 3: RoB 2 tuple assembler + extractor characterization

**Files:**
- Create: `studies/oa_rob_benchmark/rob2_tuple.py`, `studies/oa_rob_benchmark/__init__.py`
- Test: `tests/test_rob2_tuple.py`, `tests/test_rob_table_extractor.py`

**Interfaces:**
- Consumes: `biasbuster.collectors.rob_table_extractor.ExtractedStudyRow`; `biasbuster.methodologies.cochrane_rob2.algorithms.derive_domain_judgement`.
- Produces:
  - `CANONICAL_LEVELS = ("low", "some concerns", "high")`
  - `@dataclass(frozen=True) RoB2Tuple(overall, d1, d2, d3, d4, d5)` (all `str`)
  - `tuple_from_table_row(row: ExtractedStudyRow) -> RoB2Tuple | None` (None = reject / incomplete)
  - `tuple_from_signalling(domain_answers: dict[str, dict[str, str]], overall_answers: dict[str, str] | None) -> RoB2Tuple | None`
  - `normalise_level(raw: str) -> str` (maps `some_concerns`/`unclear`→`some concerns`; `""` for invalid)

- [ ] **Step 1: Write the failing tests for the tuple assembler**

```python
# tests/test_rob2_tuple.py
from studies.oa_rob_benchmark.rob2_tuple import (
    normalise_level, tuple_from_table_row, CANONICAL_LEVELS,
)
from biasbuster.collectors.rob_table_extractor import (
    ExtractedStudyRow, ExtractedRating,
)

def _rating(domain, text):
    return ExtractedRating(domain=domain, rating_text=text, rating_colour="",
                           raw_text=text, raw_style="")

def test_normalise_level_maps_underscore_and_unclear():
    assert normalise_level("some_concerns") == "some concerns"
    assert normalise_level("unclear") == "some concerns"
    assert normalise_level("Low") == "low"
    assert normalise_level("n/a") == ""

def test_complete_row_yields_tuple():
    row = ExtractedStudyRow(
        study_id="Smith 2020",
        overall=_rating("overall", "high"),
        domains=[_rating("randomization", "low"), _rating("deviations", "low"),
                 _rating("missing_outcome", "some_concerns"),
                 _rating("measurement", "low"), _rating("reporting", "high")],
        row_index=0)
    t = tuple_from_table_row(row)
    assert t is not None
    assert t.overall == "high"
    assert t.d3 == "some concerns"
    assert all(v in CANONICAL_LEVELS for v in (t.d1, t.d2, t.d3, t.d4, t.d5, t.overall))

def test_partial_row_is_rejected():
    row = ExtractedStudyRow(
        study_id="Smith 2020", overall=_rating("overall", "high"),
        domains=[_rating("randomization", "low")],  # only 1 of 5 domains
        row_index=0)
    assert tuple_from_table_row(row) is None

def test_row_with_invalid_level_is_rejected():
    row = ExtractedStudyRow(
        study_id="Smith 2020", overall=_rating("overall", ""),   # blank overall
        domains=[_rating("randomization", "low"), _rating("deviations", "low"),
                 _rating("missing_outcome", "low"),
                 _rating("measurement", "low"), _rating("reporting", "low")],
        row_index=0)
    assert tuple_from_table_row(row) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_rob2_tuple.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `rob2_tuple.py`**

```python
# studies/oa_rob_benchmark/rob2_tuple.py
"""Assemble and validate the complete six-field RoB 2 tuple (spec §4.2).

Two entry points, one output contract: a ``RoB2Tuple`` where every field is
one of ``CANONICAL_LEVELS`` — or ``None``, meaning the row is incomplete or
invalid and must be rejected (never persisted as a partial tuple).

* ``tuple_from_table_row`` — for the structural extractor path (EM candidates,
  Stage B). Normalises ``rob_table_extractor``'s ``some_concerns`` to the
  canonical ``some concerns`` and prefers cell-text over CSS colour.
* ``tuple_from_signalling`` — for ROBoto2, whose expert labels are recorded as
  signalling-question answers; domain judgements are derived deterministically
  by the RoB 2 algorithm (``derive_domain_judgement``), which faithfully
  represents the expert's assessment.
"""
from __future__ import annotations

from dataclasses import dataclass

from biasbuster.collectors.rob_table_extractor import ExtractedStudyRow
from biasbuster.methodologies.cochrane_rob2.algorithms import (
    derive_domain_judgement, synthesis_worst_wins,
)

CANONICAL_LEVELS: tuple[str, ...] = ("low", "some concerns", "high")

_ALIASES: dict[str, str] = {
    "low": "low", "low risk": "low",
    "some concerns": "some concerns", "some_concerns": "some concerns",
    "unclear": "some concerns",
    "high": "high", "high risk": "high",
}

_DOMAIN_ORDER = ("randomization", "deviations", "missing_outcome",
                 "measurement", "reporting")


@dataclass(frozen=True)
class RoB2Tuple:
    """A complete, canonical six-field RoB 2 judgement."""
    overall: str
    d1: str
    d2: str
    d3: str
    d4: str
    d5: str


def normalise_level(raw: str) -> str:
    """Map a raw rating to one of ``CANONICAL_LEVELS`` or ``""`` (invalid)."""
    t = (raw or "").strip().lower()
    return _ALIASES.get(t, "")


def tuple_from_table_row(row: ExtractedStudyRow) -> RoB2Tuple | None:
    """Build a complete tuple from an extractor row, or None if incomplete."""
    by_domain = {r.domain: (r.rating_text or r.rating_colour) for r in row.domains}
    levels = [normalise_level(by_domain.get(d, "")) for d in _DOMAIN_ORDER]
    overall = normalise_level(
        (row.overall.rating_text or row.overall.rating_colour) if row.overall else ""
    )
    if not overall or any(lv not in CANONICAL_LEVELS for lv in levels):
        return None
    return RoB2Tuple(overall, *levels)


def tuple_from_signalling(
    domain_answers: dict[str, dict[str, str]],
    overall_answers: dict[str, str] | None,
) -> RoB2Tuple | None:
    """Build a tuple from per-domain signalling answers via the RoB 2 algorithm.

    ``domain_answers`` maps a domain code/slug (accepted by
    ``derive_domain_judgement``) to that domain's signalling answers. Overall
    is taken from ``overall_answers`` if the source records it, else derived by
    worst-wins over the five domains (RoB 2's own rule).
    """
    levels: list[str] = []
    for d in _DOMAIN_ORDER:
        judged = derive_domain_judgement(d, domain_answers.get(d, {}))
        lv = normalise_level(judged or "")
        if lv not in CANONICAL_LEVELS:
            return None
        levels.append(lv)
    if overall_answers is not None:
        overall = normalise_level(overall_answers.get("overall", ""))
    else:
        overall = normalise_level(synthesis_worst_wins(levels))
    if overall not in CANONICAL_LEVELS:
        return None
    return RoB2Tuple(overall, *levels)
```

Note: confirm the exact domain-slug strings `derive_domain_judgement` accepts (see `algorithms.py:178`) and the exact return vocabulary of `synthesis_worst_wins`; adjust `_DOMAIN_ORDER` slugs and rely on `normalise_level` to canonicalise. Add a focused test if the slugs differ.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_rob2_tuple.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Write a characterization test for the reused extractor**

```python
# tests/test_rob_table_extractor.py
from pathlib import Path
from biasbuster.collectors.rob_table_extractor import extract_bias_tables, ROB2

FIXTURE = Path("tests/fixtures/cochrane_reviews/jcm-15-01829.xml")

def test_extractor_runs_on_existing_fixture_without_error():
    tables = extract_bias_tables(FIXTURE.read_bytes())
    # Characterization: pin current behaviour so Stage-A reuse is regression-safe.
    assert isinstance(tables, list)
    for t in tables:
        assert t.methodology.name in {"rob2", "quadas2", "robins_i"}

def test_malformed_xml_returns_empty_not_raises():
    assert extract_bias_tables(b"<not-valid-xml") == []
```

- [ ] **Step 6: Run both test files**

Run: `uv run pytest tests/test_rob2_tuple.py tests/test_rob_table_extractor.py -v`
Expected: PASS. If `test_extractor_runs_on_existing_fixture_without_error` reveals the fixture has no RoB 2 table, add a minimal synthetic RoB 2 JATS fixture to `tests/fixtures/oa_rob/rob2_table.xml` and assert `ROB2` is detected there instead — do not modify the extractor.

- [ ] **Step 7: Commit**

```bash
git add studies/oa_rob_benchmark/__init__.py studies/oa_rob_benchmark/rob2_tuple.py \
        tests/test_rob2_tuple.py tests/test_rob_table_extractor.py
git commit -m "feat(oa-benchmark): RoB2 tuple assembler + extractor characterization test"
```

---

## Task 4: Isolated benchmark store

**Files:**
- Create: `studies/oa_rob_benchmark/store.py`
- Test: `tests/test_oa_rob_store.py`

**Interfaces:**
- Consumes: `studies.oa_rob_benchmark.rob2_tuple.RoB2Tuple, CANONICAL_LEVELS`.
- Produces:
  - `class LitmusError(ValueError)`
  - `class BenchmarkStore` with:
    - `__init__(self, db_path: str)` — opens/creates the DB, creates schema if absent (never DROPs).
    - `upsert_item(self, item: dict) -> bool` — enforces the four-part litmus (spec §4); raises `LitmusError` on violation.
    - `log_reject(self, candidate: dict, rule: str, detail: str) -> None` — appends to `dataset/oa_rob_benchmark_rejects.jsonl`.
    - `count(self) -> int`, `all_items(self) -> list[dict]`.
  - `BENCHMARK_VERSION = "oa-stage-a-2026-07"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_oa_rob_store.py
import pytest
from studies.oa_rob_benchmark.store import BenchmarkStore, LitmusError

def _valid_item():
    return {
        "trial_pmid": "111", "trial_pmcid": "PMC1", "trial_doi": "10.1/x",
        "trial_title": "Trial A", "trial_license": "CC-BY-4.0",
        "license_redistributable": True, "non_commercial": False,
        "no_derivatives": False, "fulltext_path": "/cache/111.jats.xml",
        "rob2_overall": "high", "rob2_d1": "low", "rob2_d2": "low",
        "rob2_d3": "some concerns", "rob2_d4": "low", "rob2_d5": "high",
        "per_outcome_variant": False, "label_source": "roboto2",
        "source_review_pmid": "999", "source_review_pmcid": "PMC999",
        "table_index": 0, "row_index": 3, "resolution_method": "bracket_ref",
        "similarity_score": 1.0, "pubtype_check": "trial",
        "extraction_method": "structural_table", "manual_verified": False,
    }

def test_upsert_valid_item(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    assert store.upsert_item(_valid_item()) is True
    assert store.count() == 1

def test_reject_non_redistributable_license(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    bad = _valid_item() | {"license_redistributable": False}
    with pytest.raises(LitmusError):
        store.upsert_item(bad)

def test_reject_partial_tuple(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    bad = _valid_item() | {"rob2_d4": ""}
    with pytest.raises(LitmusError):
        store.upsert_item(bad)

def test_reject_non_trial_pubtype(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    bad = _valid_item() | {"pubtype_check": "non_trial"}
    with pytest.raises(LitmusError):
        store.upsert_item(bad)

def test_reject_missing_fulltext_or_provenance(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    with pytest.raises(LitmusError):
        store.upsert_item(_valid_item() | {"fulltext_path": ""})
    with pytest.raises(LitmusError):
        store.upsert_item(_valid_item() | {"source_review_pmid": ""})

def test_reopen_does_not_drop_rows(tmp_path):
    p = str(tmp_path / "b.db")
    BenchmarkStore(p).upsert_item(_valid_item())
    assert BenchmarkStore(p).count() == 1   # second open must not DROP
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_oa_rob_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `store.py`**

```python
# studies/oa_rob_benchmark/store.py
"""Isolated SQLite store for the OA-first RoB benchmark (spec §8).

A dedicated database (`dataset/oa_rob_benchmark.db`), never touching
`biasbuster.db` or `eisele_metzger_benchmark.db`. Schema is created with
``CREATE TABLE IF NOT EXISTS`` and is append/upsert only — it never DROPs,
mirroring the HANDOVER rule about destructive rebuilds. Every insert must
satisfy the four-part litmus test (spec §4) or raise ``LitmusError``; the
caller catches it, calls ``log_reject``, and continues.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from studies.oa_rob_benchmark.rob2_tuple import CANONICAL_LEVELS

logger = logging.getLogger(__name__)

BENCHMARK_VERSION = "oa-stage-a-2026-07"
REJECTS_PATH = "dataset/oa_rob_benchmark_rejects.jsonl"

_TUPLE_FIELDS = ("rob2_overall", "rob2_d1", "rob2_d2", "rob2_d3",
                 "rob2_d4", "rob2_d5")
_PROVENANCE_FIELDS = ("source_review_pmid", "resolution_method",
                      "extraction_method", "fulltext_path")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_item (
    trial_pmid TEXT PRIMARY KEY,
    trial_pmcid TEXT, trial_doi TEXT, trial_title TEXT,
    trial_license TEXT, license_redistributable INTEGER NOT NULL,
    non_commercial INTEGER NOT NULL, no_derivatives INTEGER NOT NULL,
    fulltext_path TEXT NOT NULL,
    rob2_overall TEXT NOT NULL, rob2_d1 TEXT NOT NULL, rob2_d2 TEXT NOT NULL,
    rob2_d3 TEXT NOT NULL, rob2_d4 TEXT NOT NULL, rob2_d5 TEXT NOT NULL,
    per_outcome_variant INTEGER NOT NULL DEFAULT 0,
    label_source TEXT NOT NULL, source_review_pmid TEXT NOT NULL,
    source_review_pmcid TEXT, table_index INTEGER, row_index INTEGER,
    resolution_method TEXT NOT NULL, similarity_score REAL,
    pubtype_check TEXT NOT NULL, extraction_method TEXT NOT NULL,
    manual_verified INTEGER NOT NULL DEFAULT 0,
    benchmark_version TEXT NOT NULL
);
"""


class LitmusError(ValueError):
    """Raised when an item fails the four-part inclusion litmus test."""


class BenchmarkStore:
    """Append/upsert store enforcing the OA-benchmark litmus invariant."""

    def __init__(self, db_path: str) -> None:
        """Open (creating if absent) the benchmark DB and ensure schema."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _check_litmus(self, item: dict) -> None:
        v: list[str] = []
        if not item.get("license_redistributable"):
            v.append("license not redistributable (litmus §4.1)")
        for f in _TUPLE_FIELDS:
            if item.get(f) not in CANONICAL_LEVELS:
                v.append(f"{f}={item.get(f)!r} not a canonical level (§4.2)")
        if item.get("pubtype_check") != "trial":
            v.append(f"pubtype_check={item.get('pubtype_check')!r} not 'trial' (§4.3)")
        for f in _PROVENANCE_FIELDS + ("trial_pmid",):
            if not str(item.get(f, "")).strip():
                v.append(f"{f} empty (§4.4)")
        if v:
            raise LitmusError("; ".join(v))

    def upsert_item(self, item: dict) -> bool:
        """Validate against the litmus test and upsert; raise on violation."""
        self._check_litmus(item)
        cols = [c for c in item if c in {r for r in _row_columns()}]
        item = {**item, "benchmark_version": BENCHMARK_VERSION}
        keys = list(item.keys())
        placeholders = ",".join("?" * len(keys))
        updates = ",".join(f"{k}=excluded.{k}" for k in keys if k != "trial_pmid")
        sql = (f"INSERT INTO benchmark_item ({','.join(keys)}) "
               f"VALUES ({placeholders}) "
               f"ON CONFLICT(trial_pmid) DO UPDATE SET {updates}")
        vals = [int(x) if isinstance(x, bool) else x for x in item.values()]
        self._conn.execute(sql, vals)
        self._conn.commit()
        return True

    def log_reject(self, candidate: dict, rule: str, detail: str) -> None:
        """Append a rejection record to the rejects JSONL (never silent)."""
        Path(REJECTS_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(REJECTS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"rule": rule, "detail": detail,
                                 "candidate": candidate}) + "\n")
        logger.info("rejected candidate (%s): %s", rule, detail)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM benchmark_item").fetchone()[0]

    def all_items(self) -> list[dict]:
        return [dict(r) for r in
                self._conn.execute("SELECT * FROM benchmark_item").fetchall()]


def _row_columns() -> list[str]:
    """Column names declared in the schema (for filtering unknown keys)."""
    import re
    return re.findall(r"^\s{4}(\w+)\s", _SCHEMA, re.MULTILINE)
```

Note: the `cols`/`_row_columns` filtering guards against callers passing extra keys; if you prefer, drop it and require callers to pass exactly the schema columns. Keep whichever is simpler once the ingest tasks are written.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_oa_rob_store.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add studies/oa_rob_benchmark/store.py tests/test_oa_rob_store.py
git commit -m "feat(oa-benchmark): isolated litmus-enforcing SQLite store"
```

---

## Task 5: Stage A — ROBoto2 manual-gold ingest

**Files:**
- Create: `studies/oa_rob_benchmark/ingest_roboto2.py`
- Test: `tests/test_ingest_roboto2.py`
- Fixture: `tests/fixtures/oa_rob/roboto2_sample.json`

**Interfaces:**
- Consumes: `store.BenchmarkStore`; `rob2_tuple.tuple_from_signalling`; `oa_license.fetch_oa_status`; `pubtype.classify`+`parse_publication_types`; `scripts/fetch_fulltext_for_expert_ratings.py::resolve_pmcid`,`fetch_jats`.
- Produces:
  - `parse_roboto2_record(rec: dict) -> tuple[str, dict, dict] | None` — pure: `(paper_id, domain_answers, overall_answers)` from a **manual** record, else None (drops LLM-assisted / incomplete).
  - `async def ingest_roboto2(dataset_path: str, store: BenchmarkStore, *, client, config) -> IngestReport`
  - `@dataclass IngestReport(seen: int, admitted: int, rejected: int)`

- [ ] **Step 1: Create the fixture (synthetic, manual + LLM-assisted rows)**

```json
// tests/fixtures/oa_rob/roboto2_sample.json
[
  {"paper_id": "PMID:111",
   "manual_assessment": [
     {"domain": "randomization", "signalling": {"1.1": "Y", "1.2": "N", "1.3": "N"}},
     {"domain": "deviations", "signalling": {"2.1": "N", "2.2": "N"}},
     {"domain": "missing_outcome", "signalling": {"3.1": "Y", "3.2": "N"}},
     {"domain": "measurement", "signalling": {"4.1": "N", "4.2": "N"}},
     {"domain": "reporting", "signalling": {"5.1": "N", "5.2": "N", "5.3": "N"}}],
   "roboto2_assessment": []},
  {"paper_id": "PMID:222",
   "manual_assessment": [],
   "roboto2_assessment": [{"domain": "randomization", "signalling": {"1.1": "Y"}}]}
]
```

- [ ] **Step 2: Write the failing test for the pure parser**

```python
# tests/test_ingest_roboto2.py
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
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_ingest_roboto2.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the parser + ingest orchestration**

```python
# studies/oa_rob_benchmark/ingest_roboto2.py
"""Stage A seed pool 1 — ingest ROBoto2 manual-gold RoB 2 assessments.

Keeps only human `manual_assessment` rows (drops LLM-assisted
`roboto2_assessment`). Each trial's signalling answers become a canonical
six-field tuple via the RoB 2 algorithm; the trial is admitted only if it is
in the PMC OA subset with a redistributable license, its PublicationType is
trial-compatible, and its JATS full text is fetched — the §4 litmus test,
enforced by the store. Rejections are logged, never silent. Incremental:
each admitted trial is upserted immediately.

ROBoto2 record shape (from github.com/larchlab/ROBoto2 dataset README):
`paper_id`, `manual_assessment` (list of per-domain signalling dicts),
`roboto2_assessment` (LLM-assisted; ignored). Reconcile the exact key names
against the real file at execution — the parser is intentionally defensive.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from studies.oa_rob_benchmark.rob2_tuple import tuple_from_signalling
from studies.oa_rob_benchmark.store import BenchmarkStore

logger = logging.getLogger(__name__)

_PMID_RE = re.compile(r"(\d{5,})")


@dataclass
class IngestReport:
    seen: int
    admitted: int
    rejected: int


def parse_roboto2_record(rec: dict) -> tuple[str, dict, dict] | None:
    """Extract (pmid, domain_answers, overall_answers) from a manual record.

    Returns None for records with no manual assessment (LLM-assisted only) or
    from which a PMID cannot be parsed.
    """
    manual = rec.get("manual_assessment") or []
    if not manual:
        return None
    m = _PMID_RE.search(str(rec.get("paper_id", "")))
    if not m:
        return None
    domain_answers = {
        str(d.get("domain", "")): dict(d.get("signalling", {}))
        for d in manual if d.get("domain")
    }
    return m.group(1), domain_answers, {}


async def ingest_roboto2(
    dataset_path: str, store: BenchmarkStore, *, client, config,
) -> IngestReport:
    """Ingest ROBoto2 manual gold into the benchmark store (incremental).

    Network steps (OA status, JATS fetch, pubtype) reuse existing modules and
    retry with backoff. This coroutine performs network I/O over the whole
    dataset — run it from a terminal, not in-session (CLAUDE.md >2 min rule).
    """
    from biasbuster.collectors.oa_license import fetch_oa_status
    from biasbuster.utils import pubtype
    from scripts.fetch_fulltext_for_expert_ratings import resolve_pmcid, fetch_jats

    records = json.loads(open(dataset_path, encoding="utf-8").read())
    seen = admitted = rejected = 0
    for rec in records:
        seen += 1
        parsed = parse_roboto2_record(rec)
        if parsed is None:
            rejected += 1
            store.log_reject(rec, "not_manual_or_no_pmid", "dropped")
            continue
        pmid, domain_answers, overall_answers = parsed
        rob2 = tuple_from_signalling(domain_answers, overall_answers or None)
        if rob2 is None:
            rejected += 1
            store.log_reject({"pmid": pmid}, "incomplete_tuple", str(domain_answers))
            continue
        oa = await fetch_oa_status(client, pmid, base=config.europmc_base)
        if not oa.in_oa_subset:
            rejected += 1
            store.log_reject({"pmid": pmid}, "not_oa_subset", oa.license.raw)
            continue
        # PublicationType check (reuse pubtype efetch + classify)
        pt = pubtype.parse_publication_types(
            await _efetch_xml(client, pmid, config))
        if pubtype.classify(pt.get(pmid, [])) != "trial":
            rejected += 1
            store.log_reject({"pmid": pmid}, "non_trial_pubtype", str(pt.get(pmid)))
            continue
        cache_dir = config_cache_dir(config)
        status, _ = await fetch_jats(client, pmid, oa.pmcid, cache_dir)
        if status != "ok":
            rejected += 1
            store.log_reject({"pmid": pmid}, "fulltext_fetch_failed", status)
            continue
        try:
            store.upsert_item(_build_item(pmid, oa, rob2, cache_dir))
            admitted += 1
        except Exception as exc:  # LitmusError or DB — logged, not fatal
            rejected += 1
            store.log_reject({"pmid": pmid}, "litmus", str(exc))
    logger.info("ROBoto2 ingest: seen=%d admitted=%d rejected=%d",
                seen, admitted, rejected)
    return IngestReport(seen, admitted, rejected)
```

Add the small helpers `_efetch_xml`, `config_cache_dir`, and `_build_item` in the same file (each < 15 lines): `_build_item` maps `(pmid, oa, rob2, cache_dir)` to the store's item dict with `label_source="roboto2"`, `extraction_method="signalling_algorithm"`, `resolution_method="roboto2_paper_id"`, `source_review_pmid="ROBoto2"`, `pubtype_check="trial"`, and the license flags from `oa.license`. Reconcile `_efetch_xml` with `pubtype`'s existing efetch helper — reuse it rather than writing a new fetch.

- [ ] **Step 5: Run the pure-parser tests to verify they pass**

Run: `uv run pytest tests/test_ingest_roboto2.py -v`
Expected: PASS (2 tests). The async `ingest_roboto2` is exercised end-to-end only against the real dataset from a terminal (Step 7); its unit surface here is the pure `parse_roboto2_record`.

- [ ] **Step 6: Add a `__main__` runner (terminal-only)**

```python
# append to studies/oa_rob_benchmark/ingest_roboto2.py
if __name__ == "__main__":
    import asyncio, httpx
    from config import Config
    async def _main() -> None:
        cfg = Config()
        store = BenchmarkStore("dataset/oa_rob_benchmark.db")
        async with httpx.AsyncClient(timeout=60) as client:
            report = await ingest_roboto2(
                "dataset/roboto2/roboto2.json", store, client=client, config=cfg)
            print(report)
    asyncio.run(_main())
```

- [ ] **Step 7: Commit**

```bash
git add studies/oa_rob_benchmark/ingest_roboto2.py tests/test_ingest_roboto2.py \
        tests/fixtures/oa_rob/roboto2_sample.json
git commit -m "feat(oa-benchmark): Stage A ROBoto2 manual-gold ingest"
```

**Do not run the full ingest in-session.** Print for the owner's terminal:
`uv run python -m studies.oa_rob_benchmark.ingest_roboto2`
(after they place the ROBoto2 dataset at `dataset/roboto2/roboto2.json` and confirm the R1 license question in the spec).

---

## Task 6: Stage A — EM OA-candidate re-derivation

**Files:**
- Create: `studies/oa_rob_benchmark/ingest_em_candidates.py`
- Test: `tests/test_ingest_em_candidates.py`
- Fixture: `tests/fixtures/oa_rob/review_with_rob2.xml` (minimal review JATS: one RoB 2 table + a reference list with PMIDs)

**Interfaces:**
- Consumes: `rob_table_extractor.extract_bias_tables`; `study_pmid_resolver.resolve_study_pmid, Reference`; `rob2_tuple.tuple_from_table_row`; `oa_license.fetch_oa_status`; `store.BenchmarkStore`; `exclusions.WRONG_PAPER_RCTS`.
- Produces:
  - `parse_reference_list(jats_xml: bytes) -> list[Reference]` (pure)
  - `derive_items_from_review(jats_xml: bytes, target_pmids: set[str]) -> list[dict]` (pure: table→tuple→resolved-PMID for PMIDs in `target_pmids`)
  - `async def ingest_em_candidates(em_pmids: list[str], store, *, client, config) -> IngestReport`

- [ ] **Step 1: Create the minimal review fixture**

Create `tests/fixtures/oa_rob/review_with_rob2.xml` — a valid JATS `<article>` containing one `<table-wrap>` whose header row names the five RoB 2 domains + Overall, one `<tbody>` row `Smith 2020` with cell texts `Low/Low/Some concerns/Low/High` and overall `High`, and a `<ref-list>` with one `<ref id="28">` carrying `<pub-id pub-id-type="pmid">111</pub-id>`, first author `Smith`, year `2020`, title `Effect of drug A on outcome X`. (Keep it under ~40 lines; model it on `tests/fixtures/cochrane_reviews/README.md` conventions.)

- [ ] **Step 2: Write the failing tests for the pure functions**

```python
# tests/test_ingest_em_candidates.py
from pathlib import Path
from studies.oa_rob_benchmark.ingest_em_candidates import (
    parse_reference_list, derive_items_from_review,
)

JATS = Path("tests/fixtures/oa_rob/review_with_rob2.xml").read_bytes()

def test_reference_list_parsed_with_pmid():
    refs = parse_reference_list(JATS)
    assert any(r.pmid == "111" and r.first_author.lower() == "smith" for r in refs)

def test_derive_items_resolves_target_trial():
    items = derive_items_from_review(JATS, target_pmids={"111"})
    assert len(items) == 1
    it = items[0]
    assert it["trial_pmid"] == "111"
    assert it["rob2_overall"] == "high"
    assert it["rob2_d3"] == "some concerns"
    assert it["resolution_method"] in {"bracket_ref", "author_year_title"}

def test_non_target_pmid_not_emitted():
    assert derive_items_from_review(JATS, target_pmids={"999"}) == []
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_ingest_em_candidates.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the module**

Implement `parse_reference_list` (walk `<ref-list>/<ref>`, pull `@id`→ref_number, `<pub-id pub-id-type="pmid">`, `<surname>` of first author, `<year>`, `<article-title>` → `Reference`), `derive_items_from_review` (run `extract_bias_tables`; for each RoB 2 table row build a tuple via `tuple_from_table_row`; resolve the row's study_id to a PMID via `resolve_study_pmid` using the parsed references; emit an item dict only when the resolved PMID ∈ `target_pmids` and the tuple is complete), and `ingest_em_candidates` (for each EM PMID **not** in `WRONG_PAPER_RCTS`: OA-filter via `fetch_oa_status`; find + fetch the trial's parent review JATS; `derive_items_from_review`; pubtype-check; upsert with `label_source="cochrane_review"`, `extraction_method="structural_table"`). Follow the same incremental/log-reject/terminal-runner structure as Task 5. Keep the file < 300 lines; if it grows, split reference parsing into `studies/oa_rob_benchmark/jats_refs.py`.

Note on review discovery: EM's benchmark records the parent Cochrane review per RCT — read that mapping from the EM study inputs (`studies/eisele_metzger_replication/`) rather than searching, to avoid re-introducing the wrong-paper failure. Confirm the exact source of the RCT→review mapping at execution; if absent, resolve the review by the trial's Europe PMC citation link and record it in provenance.

- [ ] **Step 5: Run to verify pure-function tests pass**

Run: `uv run pytest tests/test_ingest_em_candidates.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add studies/oa_rob_benchmark/ingest_em_candidates.py \
        tests/test_ingest_em_candidates.py tests/fixtures/oa_rob/review_with_rob2.xml
git commit -m "feat(oa-benchmark): Stage A EM OA-candidate re-derivation from primary reviews"
```

**Terminal-only full run** (print for owner):
`uv run python -m studies.oa_rob_benchmark.ingest_em_candidates`

---

## Task 7: Manual-gate manifest generator

**Files:**
- Create: `studies/oa_rob_benchmark/manual_gate.py`
- Test: `tests/test_oa_rob_manual_gate.py`

**Interfaces:**
- Consumes: `store.BenchmarkStore`.
- Produces: `render_manifest(items: list[dict], *, limit: int = MANUAL_SAMPLE_SIZE) -> str`; `MANUAL_SAMPLE_SIZE = 20`; a `__main__` runner writing `dataset/oa_rob_benchmark_manual_check.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oa_rob_manual_gate.py
from studies.oa_rob_benchmark.manual_gate import render_manifest, MANUAL_SAMPLE_SIZE

def _item(pmid):
    return {"trial_pmid": pmid, "trial_title": f"Trial {pmid}",
            "source_review_pmid": "999", "source_review_pmcid": "PMC999",
            "table_index": 0, "row_index": 3,
            "rob2_overall": "high", "rob2_d1": "low", "rob2_d2": "low",
            "rob2_d3": "some concerns", "rob2_d4": "low", "rob2_d5": "high",
            "resolution_method": "bracket_ref", "similarity_score": 1.0,
            "trial_license": "CC-BY-4.0", "label_source": "cochrane_review"}

def test_manifest_lists_up_to_sample_size_rows():
    md = render_manifest([_item(str(i)) for i in range(30)])
    assert md.count("Trial ") == MANUAL_SAMPLE_SIZE
    assert "some concerns" in md
    assert "PMC999" in md

def test_manifest_shows_all_six_fields_per_row():
    md = render_manifest([_item("111")])
    for field in ("overall", "D1", "D2", "D3", "D4", "D5"):
        assert field in md
```

- [ ] **Step 2: Run to verify it fails; Step 3: implement `render_manifest`** (a markdown table per row: resolved PMID + title, source review PMID/PMCID, table/row index, the six-field tuple labelled overall/D1..D5, resolution method + similarity, license). **Step 4: run to pass. Step 5: add `__main__`** that loads the store and writes `dataset/oa_rob_benchmark_manual_check.md`.

Run: `uv run pytest tests/test_oa_rob_manual_gate.py -v` → PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add studies/oa_rob_benchmark/manual_gate.py tests/test_oa_rob_manual_gate.py
git commit -m "feat(oa-benchmark): 20-row manual-gate manifest generator"
```

---

## Task 8: Litmus audit + redistributable export

**Files:**
- Create: `scripts/audit_oa_rob_benchmark.py`, `studies/oa_rob_benchmark/export.py`
- Test: extend `tests/test_oa_rob_store.py` with an audit round-trip.

**Interfaces:**
- Produces: `audit_benchmark(items: list[dict]) -> list[str]` (returns a list of violation strings, empty = clean); `export_redistributable(items: list[dict]) -> list[dict]` (drops any non-redistributable rows defensively, strips source prose, keeps identifiers+license+tuple+provenance, tags NC/ND).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_oa_rob_store.py
from scripts.audit_oa_rob_benchmark import audit_benchmark
from studies.oa_rob_benchmark.export import export_redistributable

def test_audit_flags_bad_row():
    good = _valid_item()
    bad = _valid_item() | {"trial_pmid": "222", "pubtype_check": "non_trial"}
    violations = audit_benchmark([good, bad])
    assert any("222" in v for v in violations)
    assert all("111" not in v for v in violations)

def test_export_keeps_provenance_and_flags_nc_nd():
    row = _valid_item() | {"non_commercial": True, "no_derivatives": True}
    out = export_redistributable([row])
    assert out[0]["non_commercial"] is True
    assert "source_review_prose" not in out[0]
    assert out[0]["rob2_overall"] == "high"
```

- [ ] **Step 2: Run to fail; Step 3: implement both** — `audit_benchmark` re-runs the four litmus checks (reuse `BenchmarkStore._check_litmus` logic, factored into a module-level `litmus_violations(item) -> list[str]` so both the store and the audit share one source of truth — refactor Task 4 to expose it). `export_redistributable` filters to `license_redistributable`, whitelists the shareable columns, and preserves NC/ND flags. **Step 4: run to pass.**

Run: `uv run pytest tests/test_oa_rob_store.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_oa_rob_benchmark.py studies/oa_rob_benchmark/export.py \
        studies/oa_rob_benchmark/store.py tests/test_oa_rob_store.py
git commit -m "feat(oa-benchmark): litmus audit + redistributable export"
```

---

## Manual gate (owner action — hard stop before Stage B)

After Tasks 5–6 have been run from the owner's terminal and Task 7 has generated `dataset/oa_rob_benchmark_manual_check.md`, the owner reviews 20 rows against their source reviews and signs off (≥19/20 match). Only then is Stage B (spec §7 — fresh OA-first harvest) planned and implemented as a **separate** plan reusing every module built here. This plan does **not** proceed past the gate.

---

## Self-Review

**Spec coverage** (spec §ref → task):
- §4.1 OA full text in hand → Task 1 (`fetch_oa_status`) + Task 5/6 (`fetch_jats`) + store litmus.
- §4.2 complete expert RoB 2 tuple → Task 3 (`rob2_tuple`) + store `_TUPLE_FIELDS` check.
- §4.3 verified linkage + pubtype → Task 2 (resolver) + `pubtype.classify` in Tasks 5/6 + store `pubtype_check`.
- §4.4 provenance + license recorded → Task 4 schema + `_PROVENANCE_FIELDS`.
- §5 labels-as-data (re-derive, not copy) → Task 3 `tuple_from_signalling`/`tuple_from_table_row`; Task 6 re-derives from primary review (no EM supplement read).
- §6.1 ROBoto2 manual gold, drop LLM-assisted → Task 5 `parse_roboto2_record`.
- §6.2 EM OA candidates, re-derive from primary review, exclude RCT030 → Task 6 (+ `WRONG_PAPER_RCTS`).
- §8.1 isolated store + fields + rejects log → Task 4.
- §8.2 incremental/resumable, backoff, no truncation → per-item upsert + `retry` reuse across Tasks 1/5/6.
- §7.3 20-sample manual gate → Task 7 + owner gate.
- §9 audit + export → Task 8.
- §7 Stage B harvest → deferred to a follow-up plan (explicit; gate-gated).

**Placeholder scan:** no "TBD"/"add error handling"-style steps; each code step shows real code. Orchestration tasks (5/6) unit-test their pure surface and print the network run for the terminal per CLAUDE.md — the async body is shown in full, with the small helpers named and specified rather than hand-waved. Fix on execution: reconcile ROBoto2's exact record keys and `derive_domain_judgement`'s domain slugs against the real inputs (flagged in-task).

**Type consistency:** `RoB2Tuple` fields `overall,d1..d5` used consistently; store columns `rob2_overall,rob2_d1..d5` map 1:1; `Resolution.method` strings match store `resolution_method`; canonical levels `("low","some concerns","high")` are the single vocabulary across Tasks 3/4/6/7. `fetch_oa_status`→`OAStatus.license: LicenseInfo` consumed by `_build_item` in Task 5. `litmus_violations` shared by store (Task 4) and audit (Task 8) after the Task-8 refactor.

---

## Execution Handoff

Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task with review between tasks.
2. **Inline Execution** — batch execution in this session with checkpoints.

Note: Tasks 1–4, 7, 8 complete fully in-session (fast, offline unit tests). Tasks 5–6 implement + unit-test their pure surface in-session but their **full network runs are terminal-only** (owner-run, per CLAUDE.md), and the **manual gate is an owner action**. So execution reaches "all code written, all unit tests green, manifest generator ready" in-session; the data actually populating the benchmark and the gate sign-off happen on the owner's machine.

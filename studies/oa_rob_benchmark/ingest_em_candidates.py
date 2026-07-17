"""Stage A seed pool 2 — re-derive RoB 2 labels for EM OA candidates.

The Eisele-Metzger 2025 benchmark's trial PMIDs are used **only as a
candidate list** (spec §6.2) — never their supplementary RoB 2 judgements.
For each candidate trial that survives the OA-subset + license filter
(litmus §4.1), this module locates the trial's *primary* systematic review,
extracts that review's own RoB 2 table structurally (``rob_table_extractor``,
composed via ``rob2_tuple.tuple_from_table_row``), and resolves the table
row that describes the candidate trial via a bracket-reference or
author+year+title match (``study_pmid_resolver``) — never a copy of EM's
own judgement for that trial.

``WRONG_PAPER_RCTS`` (RCT030 — the known wrong-document acquisition, see
``studies.eisele_metzger_replication.exclusions``) is excluded before any
network I/O.

``derive_items_from_review`` — a review's JATS to admitted item dicts,
restricted to rows resolving to a PMID in ``target_pmids`` — is the pure,
unit-tested core (against ``tests/fixtures/oa_rob/review_with_rob2.xml``).
Reference-list/front-matter XML parsing lives in ``jats_refs.py`` (split out
to keep this module under its line budget); ``parse_reference_list`` is
re-exported here for a stable import surface.

``ingest_em_candidates`` composes these with network I/O and is
TERMINAL-ONLY (CLAUDE.md's >2-minute-process rule) — run it from a
terminal via the module's ``__main__``, never in-session; it is not
unit-tested here. Review discovery (spec §6.2 step 3's "read that mapping
from the EM study inputs" note) reads the Eisele-Metzger CSV's ``cr_id``
column — the parent Cochrane review's identifier (e.g. ``"CD001159.PUB3"``),
an identifier crosswalk, not a judgement — from
``dataset/eisele_metzger_benchmark.db``'s ``benchmark_rct`` table (built by
``studies/eisele_metzger_replication/build_benchmark_db.py`` from EM's own
CSV; reading this identifier does not reintroduce the "copy EM's
supplement" problem this task exists to avoid). ``cr_id`` is converted to
the Cochrane Library DOI and resolved to the review's own PMID via the NCBI
ID Converter (``RetractionWatchCollector.doi_to_pmid``, already used the
same way in ``annotate_single_paper.resolve_pmid`` — no new retry loop is
written). See the task report for the execution-time reconciliation this
note flags: if ``dataset/eisele_metzger_benchmark.db`` is absent, every
candidate is rejected with ``no_review_mapping`` rather than falling back
to a citation-link search (spec §7.2's Stage B discovery method, out of
this task's scope).
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from biasbuster.collectors.rob_table_extractor import extract_bias_tables
from biasbuster.collectors.study_pmid_resolver import Resolution, resolve_study_pmid
from studies.eisele_metzger_replication.exclusions import WRONG_PAPER_RCTS
from studies.oa_rob_benchmark.jats_refs import extract_front_ids, parse_reference_list, parse_xml
from studies.oa_rob_benchmark.rob2_tuple import tuple_from_table_row
from studies.oa_rob_benchmark.store import BenchmarkStore, LitmusError

if TYPE_CHECKING:
    # ``config.py`` is gitignored (copied from config.example.py per-repo),
    # so it must not be imported at module load time — only the type
    # checker needs this; PEP 563 (``from __future__ import annotations``
    # above) makes the runtime annotation a lazy string.
    from config import Config

__all__ = [
    "IngestReport", "derive_items_from_review", "ingest_em_candidates",
    "parse_reference_list",
]

logger = logging.getLogger(__name__)

#: EM benchmark DB (Phase 3 of the EM replication study) — the source of
#: the trial-PMID → parent-review-id (``cr_id``) crosswalk (see module
#: docstring). Built by ``build_benchmark_db.py``; not checked in.
EM_BENCHMARK_DB_PATH = Path("dataset/eisele_metzger_benchmark.db")

#: Cochrane reviews' DOI namespace; a ``cr_id`` like ``"CD001159.PUB3"``
#: becomes ``"10.1002/14651858.cd001159.pub3"``.
_COCHRANE_DOI_PREFIX = "10.1002/14651858."

#: A bracketed inline reference number in a table's study-id cell, e.g. the
#: "[28]" in "Smith 2020 [28]".
_BRACKET_REF_RE = re.compile(r"\[(\d+)\]")

#: A four-digit year wrapped in parentheses, e.g. the "(2020)" in
#: "Alpha (2021)" — stripped so the resolver's ``_STUDY_ID_RE`` (which
#: expects "Author 2020" with no parentheses) has a chance to match on the
#: author+year fallback path when no bracket reference is present.
_PAREN_YEAR_RE = re.compile(r"\((\d{4})\)")

_LABEL_SOURCE = "cochrane_review"
_EXTRACTION_METHOD = "structural_table"
_PUBTYPE_TRIAL = "trial"


@dataclass
class IngestReport:
    """Summary counters for one ``ingest_em_candidates`` run."""
    seen: int
    admitted: int
    rejected: int


def _normalise_study_id(study_id: str) -> tuple[str, str]:
    """Strip a bracket ref (returning its number) and parens around a year.

    ``"Smith 2020 [28]"`` -> ``("Smith 2020", "28")``;
    ``"Alpha (2021)"`` -> ``("Alpha 2021", "")``.
    """
    m = _BRACKET_REF_RE.search(study_id)
    ref_number = m.group(1) if m else ""
    cleaned = _BRACKET_REF_RE.sub("", study_id)
    cleaned = _PAREN_YEAR_RE.sub(r"\1", cleaned)
    return " ".join(cleaned.split()), ref_number


def derive_items_from_review(jats_xml: bytes, target_pmids: set[str]) -> list[dict]:
    """Re-derive RoB 2 items for ``target_pmids`` from a review's own table.

    Extracts every RoB 2 table (``rob_table_extractor``'s QUADAS-2 and
    ROBINS-I tables are ignored — spec §6.2 step 3 is RoB 2 only), builds
    the six-field tuple for each row (``tuple_from_table_row`` — incomplete
    rows are skipped), resolves the row's study id to a trial PMID
    (``resolve_study_pmid``, using this review's own reference list), and
    emits an item dict only when that PMID is in ``target_pmids``. Never
    raises — malformed XML, tables with no RoB 2 match, or unresolvable
    rows simply contribute nothing to the result.
    """
    root = parse_xml(jats_xml)
    if root is None:
        return []

    references = parse_reference_list(jats_xml)
    review_pmid, review_pmcid = extract_front_ids(root)

    tables = extract_bias_tables(jats_xml)
    rob2_tables = [t for t in tables if t.methodology.name == "rob2"]

    items: list[dict] = []
    for table_index, table in enumerate(rob2_tables):
        for row in table.studies:
            tup = tuple_from_table_row(row)
            if tup is None:
                continue
            normalised_id, ref_number = _normalise_study_id(row.study_id)
            resolution: Resolution = resolve_study_pmid(normalised_id, ref_number, references)
            if not resolution.pmid or resolution.pmid not in target_pmids:
                continue
            items.append({
                "trial_pmid": resolution.pmid,
                "rob2_overall": tup.overall,
                "rob2_d1": tup.d1,
                "rob2_d2": tup.d2,
                "rob2_d3": tup.d3,
                "rob2_d4": tup.d4,
                "rob2_d5": tup.d5,
                "source_review_pmid": review_pmid,
                "source_review_pmcid": review_pmcid,
                "table_index": table_index,
                "row_index": row.row_index,
                "resolution_method": resolution.method,
                "similarity_score": resolution.similarity,
                "label_source": _LABEL_SOURCE,
                "extraction_method": _EXTRACTION_METHOD,
                "per_outcome_variant": False,
            })
    return items


def _load_em_benchmark_data(db_path: Path) -> tuple[dict[str, str], set[str]]:
    """Load ``({trial_pmid: cr_id}, excluded_pmids)`` from the EM benchmark DB.

    ``cr_id`` is EM's parent-Cochrane-review identifier crosswalk (e.g.
    ``"CD001159.PUB3"``) — see module docstring for why reading this
    identifier (not a judgement) does not reintroduce the "copy EM's
    supplement" problem.

    ``excluded_pmids`` resolves ``WRONG_PAPER_RCTS`` into the PMID space
    ``ingest_em_candidates`` actually filters on: that set is keyed by EM's
    ``rct_id`` (e.g. ``"RCT030"``), not PMID, so a plain
    ``pmid in WRONG_PAPER_RCTS`` check would never fire. Cross-referencing
    ``benchmark_rct``'s ``rct_id``/``pmid`` columns here recovers the actual
    (possibly wrong-document) PMID RCT030's acquisition recorded, so it can
    be excluded by the identifier the caller actually holds.

    Fails safe (returns ``({}, set())``, logs) if the DB or table is
    absent — the caller then rejects every candidate with
    ``no_review_mapping`` and the exclusion falls back to the literal
    ``rct_id`` check, rather than raising.
    """
    if not db_path.exists():
        logger.warning("EM benchmark DB not found at %s; no review mapping available", db_path)
        return {}, set()
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT pmid, cr_id, rct_id FROM benchmark_rct WHERE pmid IS NOT NULL AND pmid != ''"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning("EM benchmark DB missing expected schema: %s", exc)
        return {}, set()
    finally:
        conn.close()
    review_map = {pmid: cr_id for pmid, cr_id, _rct_id in rows if cr_id}
    excluded_pmids = {pmid for pmid, _cr_id, rct_id in rows if rct_id in WRONG_PAPER_RCTS}
    return review_map, excluded_pmids


def _cr_id_to_doi(cr_id: str) -> str:
    """Convert a Cochrane review id (e.g. "CD001159.PUB3") to its DOI."""
    return f"{_COCHRANE_DOI_PREFIX}{cr_id.strip().lower()}"


async def ingest_em_candidates(
    em_pmids: list[str], store: BenchmarkStore, *, client: httpx.AsyncClient, config: Config,
) -> IngestReport:
    """Ingest EM candidate trials, re-deriving labels from their primary review.

    For each ``em_pmids`` entry not excluded by ``WRONG_PAPER_RCTS`` (see
    ``_load_em_benchmark_data``): confirm the trial's PMC OA-subset
    membership + redistributable license (``oa_license.fetch_oa_status``);
    look up its parent review's ``cr_id``, resolve that to the review's own PMID
    via the Cochrane DOI + NCBI ID Converter, and fetch the review's PMCID
    + JATS (``fetch_jats``); re-derive this trial's RoB 2 row from the
    review's own table (``derive_items_from_review`` — never EM's
    supplement); confirm the trial's PubMed PublicationType is
    trial-compatible (``pubtype``); fetch + cache the trial's own JATS
    (litmus §4.1's full-text-in-hand requirement); then upsert. Every
    rejection path calls ``store.log_reject`` — never a silent ``continue``
    — and a ``LitmusError`` from the store's own final check is caught and
    logged rather than aborting the batch.

    Network I/O runs once per candidate across five services (OA status,
    DOI resolution, PublicationType, and two JATS fetches) — run this from
    a terminal via the module's ``__main__``, never in-session (CLAUDE.md's
    >2-minute-process rule).
    """
    from biasbuster.collectors.oa_license import fetch_oa_status
    from biasbuster.collectors.retraction_watch import RetractionWatchCollector
    from biasbuster.utils import pubtype
    from scripts.fetch_fulltext_for_expert_ratings import DEFAULT_CACHE_DIR, _cache_path, fetch_jats

    review_map, excluded_pmids = _load_em_benchmark_data(EM_BENCHMARK_DB_PATH)
    seen = admitted = rejected = 0

    for pmid in em_pmids:
        seen += 1
        if pmid in WRONG_PAPER_RCTS or pmid in excluded_pmids:
            rejected += 1
            store.log_reject({"trial_pmid": pmid}, "wrong_paper_excluded", pmid)
            continue

        oa = await fetch_oa_status(client, pmid, base=config.europmc_base)
        if not oa.in_oa_subset:
            rejected += 1
            store.log_reject({"trial_pmid": pmid}, "not_oa_subset", oa.license.raw)
            continue

        cr_id = review_map.get(pmid, "")
        if not cr_id:
            rejected += 1
            store.log_reject({"trial_pmid": pmid}, "no_review_mapping", "cr_id not found")
            continue

        review_doi = _cr_id_to_doi(cr_id)
        async with RetractionWatchCollector(
            mailto=config.crossref_mailto, ncbi_api_key=config.ncbi_api_key,
        ) as rw:
            doi_map = await rw.doi_to_pmid([review_doi])
        review_pmid = doi_map.get(review_doi, "")
        if not review_pmid:
            rejected += 1
            store.log_reject({"trial_pmid": pmid}, "review_pmid_unresolved", review_doi)
            continue

        review_oa = await fetch_oa_status(client, review_pmid, base=config.europmc_base)
        if not review_oa.pmcid:
            rejected += 1
            store.log_reject({"trial_pmid": pmid}, "review_pmcid_unresolved", review_pmid)
            continue
        review_pmcid = (
            review_oa.pmcid if review_oa.pmcid.startswith("PMC") else f"PMC{review_oa.pmcid}"
        )
        review_status, _n = await fetch_jats(client, review_pmid, review_pmcid, DEFAULT_CACHE_DIR)
        if review_status != "ok":
            rejected += 1
            store.log_reject({"trial_pmid": pmid}, "review_fulltext_fetch_failed", review_status)
            continue

        review_jats = _cache_path(DEFAULT_CACHE_DIR, review_pmid).read_bytes()
        items = derive_items_from_review(review_jats, {pmid})
        if not items:
            rejected += 1
            store.log_reject({"trial_pmid": pmid}, "no_rob2_row_resolved", review_pmid)
            continue
        item = items[0]
        # The review's own PMID/PMCID are known authoritatively here (network-
        # resolved above); prefer them over derive_items_from_review's
        # best-effort front-matter extraction, which may have found nothing.
        item["source_review_pmid"] = review_pmid
        item["source_review_pmcid"] = review_pmcid

        pt = await pubtype.fetch_publication_types(
            [pmid], client=client, ncbi_api_key=config.ncbi_api_key,
        )
        pubtype_check = pubtype.classify(pt.get(pmid, []))
        if pubtype_check != _PUBTYPE_TRIAL:
            rejected += 1
            store.log_reject({"trial_pmid": pmid}, "non_trial_pubtype", str(pt.get(pmid)))
            continue

        trial_pmcid = oa.pmcid if oa.pmcid.startswith("PMC") else f"PMC{oa.pmcid}"
        trial_status, _n = await fetch_jats(client, pmid, trial_pmcid, DEFAULT_CACHE_DIR)
        if trial_status != "ok":
            rejected += 1
            store.log_reject({"trial_pmid": pmid}, "trial_fulltext_fetch_failed", trial_status)
            continue

        lic = oa.license
        item.update({
            "trial_pmcid": oa.pmcid,
            "trial_doi": "",
            "trial_title": "",
            "trial_license": lic.spdx or lic.raw,
            "license_redistributable": lic.redistributable,
            "non_commercial": lic.non_commercial,
            "no_derivatives": lic.no_derivatives,
            "fulltext_path": str(_cache_path(DEFAULT_CACHE_DIR, pmid)),
            "pubtype_check": pubtype_check,
            "manual_verified": False,
        })

        try:
            store.upsert_item(item)
            admitted += 1
        except LitmusError as exc:
            rejected += 1
            store.log_reject({"trial_pmid": pmid}, "litmus", str(exc))

    logger.info(
        "EM candidate ingest: seen=%d admitted=%d rejected=%d", seen, admitted, rejected,
    )
    return IngestReport(seen, admitted, rejected)


if __name__ == "__main__":
    import asyncio

    from config import Config

    async def _main() -> None:
        """Run the full EM-candidate ingest against the real EM benchmark PMIDs.

        Terminal-only (CLAUDE.md >2-minute rule) — do not invoke from an
        agent session. Requires ``dataset/eisele_metzger_benchmark.db`` to
        exist (built by ``studies/eisele_metzger_replication/build_benchmark_db.py``)
        for review discovery; without it, every candidate is rejected with
        ``no_review_mapping`` (see module docstring).
        """
        cfg = Config()
        store = BenchmarkStore("dataset/oa_rob_benchmark.db")
        conn = sqlite3.connect(str(EM_BENCHMARK_DB_PATH))
        try:
            em_pmids = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT pmid FROM benchmark_rct WHERE pmid IS NOT NULL AND pmid != ''"
                ).fetchall()
            ]
        finally:
            conn.close()
        async with httpx.AsyncClient(timeout=60) as client:
            report = await ingest_em_candidates(
                em_pmids, store, client=client, config=cfg,
            )
            print(report)

    asyncio.run(_main())

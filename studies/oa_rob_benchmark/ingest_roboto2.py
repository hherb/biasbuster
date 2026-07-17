"""Stage A seed pool 1 — ingest ROBoto2 manual-gold RoB 2 assessments.

Consumes the normalized ingestion JSON produced by
``convert_roboto2_csv`` (``dataset/roboto2/roboto2.json``): one record per
human-labelled trial carrying the experts' *recorded* RoB 2 judgements
(``rob2``), the trial's identity (``title``/``abstract``/``authors``), and the
raw signalling answers (``signalling``, provenance only). See that module for
why the ground truth is the recorded labels rather than a re-derivation from
signalling.

Each trial is admitted only if the spec §4 litmus test passes (enforced by
``BenchmarkStore.upsert_item``):

1. Its **title resolves confidently to a PubMed PMID** — ROBoto2 carries no
   PMID/DOI, and ``paper_id`` is an S2ORC-internal id, so identity comes from
   a verified title search (``title_resolver.resolve_pmid_by_title``), never
   from treating ``paper_id`` as a PMID.
2. That PMID is in the PMC OA subset with a redistributable license.
3. Its PubMed PublicationType is trial-compatible.
4. Its JATS full text is fetched.

The recorded ``rob2`` labels become the canonical six-field tuple directly
(``_tuple_from_recorded`` — a domain recorded as ``none`` is not a canonical
RoB 2 level, so such a row is rejected rather than coerced). Rejections are
logged via ``store.log_reject``, never silent. Incremental: each admitted
trial is upserted immediately (CLAUDE.md's checkpoint/incremental-save rule).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from biasbuster.collectors.oa_license import OAStatus
from studies.oa_rob_benchmark.rob2_tuple import (
    CANONICAL_LEVELS, RoB2Tuple, normalise_level,
)
from studies.oa_rob_benchmark.store import BenchmarkStore, LitmusError
from studies.oa_rob_benchmark.title_resolver import (
    RESOLUTION_METHOD, TitleResolution, resolve_pmid_by_title,
)

if TYPE_CHECKING:
    # ``config.py`` is gitignored (copied from config.example.py per-repo),
    # so it must not be imported at module load time — only the type
    # checker needs this; PEP 563 (``from __future__ import annotations``
    # above) makes the runtime annotation a lazy string.
    from config import Config

__all__ = [
    "IngestReport", "parse_record", "ingest_roboto2",
]

logger = logging.getLogger(__name__)

#: Default input path — the normalized JSON emitted by ``convert_roboto2_csv``.
DEFAULT_DATASET_PATH = "dataset/roboto2/roboto2.json"

#: The five canonical RoB 2 domain names, in tuple order (d1..d5), as emitted
#: by ``convert_roboto2_csv`` under each record's ``rob2`` mapping.
_DOMAIN_ORDER = ("randomization", "deviations", "missing_outcome",
                 "measurement", "reporting")

#: Provenance/labelling constants for every ROBoto2-sourced item (spec §6.1).
_LABEL_SOURCE = "roboto2"
_SOURCE_REVIEW_PMID = "ROBoto2"  # non-empty for litmus §4.4; not a real PMID
_EXTRACTION_METHOD = "recorded_expert_label"
_PUBTYPE_CHECK = "trial"


@dataclass
class IngestReport:
    """Summary counters for one ``ingest_roboto2`` run."""
    seen: int
    admitted: int
    rejected: int


@dataclass(frozen=True)
class ParsedRecord:
    """A normalized ROBoto2 record ready for identity resolution + litmus.

    ``rob2`` is the complete canonical six-field tuple built from the
    recorded expert labels; ``title`` drives PMID resolution.
    """
    paper_id: str
    title: str
    rob2: RoB2Tuple


def _tuple_from_recorded(rob2: dict[str, Any]) -> RoB2Tuple | None:
    """Build the canonical tuple from a record's recorded ``rob2`` labels.

    Returns ``None`` if the overall or any of the five domains is missing or
    is not one of ``CANONICAL_LEVELS`` after normalisation — most notably a
    domain recorded as ``none`` (not assessed): such a row cannot yield a
    valid RoB 2 tuple and is rejected rather than silently coerced to a real
    level.
    """
    overall = normalise_level(str(rob2.get("overall", "")))
    levels = [normalise_level(str(rob2.get(name, ""))) for name in _DOMAIN_ORDER]
    if overall not in CANONICAL_LEVELS or any(lv not in CANONICAL_LEVELS for lv in levels):
        return None
    return RoB2Tuple(overall, *levels)


def parse_record(rec: dict) -> ParsedRecord | None:
    """Extract a ``ParsedRecord`` from one normalized ingestion record.

    Returns ``None`` (never raises) when the record is not a usable trial:
    a non-dict entry, a missing/blank ``title`` (identity cannot be resolved
    without one), or ``rob2`` labels that do not form a complete canonical
    tuple (e.g. a ``none`` domain — see ``_tuple_from_recorded``).
    """
    if not isinstance(rec, dict):
        return None
    title = str(rec.get("title") or "").strip()
    if not title:
        return None
    rob2_field = rec.get("rob2")
    if not isinstance(rob2_field, dict):
        return None
    tup = _tuple_from_recorded(rob2_field)
    if tup is None:
        return None
    return ParsedRecord(str(rec.get("paper_id", "")), title, tup)


def _build_item(
    resolution: TitleResolution, oa: OAStatus, rob2: RoB2Tuple,
    fulltext_path: str, title: str,
) -> dict:
    """Map a resolved + fetched trial to a ``benchmark_item`` store dict.

    Populates every NOT NULL / litmus-required column (store.py's
    ``_SCHEMA``): the resolved PMID + license facts from ``oa``, the RoB 2
    tuple's six recorded-label fields, the resolution confidence, and the
    ROBoto2-specific provenance constants defined module-level above.
    """
    lic = oa.license
    return {
        "trial_pmid": resolution.pmid,
        "trial_pmcid": oa.pmcid,
        "trial_doi": "",
        "trial_title": title,
        "trial_license": lic.spdx or lic.raw,
        "license_redistributable": lic.redistributable,
        "non_commercial": lic.non_commercial,
        "no_derivatives": lic.no_derivatives,
        "fulltext_path": fulltext_path,
        "rob2_overall": rob2.overall,
        "rob2_d1": rob2.d1,
        "rob2_d2": rob2.d2,
        "rob2_d3": rob2.d3,
        "rob2_d4": rob2.d4,
        "rob2_d5": rob2.d5,
        "per_outcome_variant": False,
        "label_source": _LABEL_SOURCE,
        "source_review_pmid": _SOURCE_REVIEW_PMID,
        "source_review_pmcid": "",
        "table_index": None,
        "row_index": None,
        "resolution_method": RESOLUTION_METHOD,
        "similarity_score": resolution.similarity,
        "pubtype_check": _PUBTYPE_CHECK,
        "extraction_method": _EXTRACTION_METHOD,
        "manual_verified": False,
    }


async def ingest_roboto2(
    dataset_path: str, store: BenchmarkStore, *, client: httpx.AsyncClient, config: Config,
) -> IngestReport:
    """Ingest ROBoto2 manual gold into the benchmark store (incremental).

    For each normalized record (``parse_record``): resolve its title to a
    PubMed PMID with a recorded confidence (``resolve_pmid_by_title`` — reject
    if unresolved), confirm PMC OA-subset membership + redistributable license
    (``oa_license.fetch_oa_status``), confirm the PublicationType is
    trial-compatible (``pubtype``), fetch + cache the JATS full text
    (``fetch_fulltext_for_expert_ratings.fetch_jats``), then upsert with the
    experts' recorded RoB 2 labels.

    Every rejection path calls ``store.log_reject`` — never a silent
    ``continue``. A ``LitmusError`` (the store's own final litmus check) or
    ``sqlite3.Error`` (a stray DB error on one row) from ``store.upsert_item``
    is caught and logged rather than aborting the batch, so one bad row never
    loses the rest of the run.

    This coroutine performs network I/O over the whole dataset (a title
    esearch + candidate efetch, OA lookup, PublicationType fetch, and JATS
    download per trial) — run it from a terminal via the module's
    ``__main__``, not in-session (CLAUDE.md's >2-minute-process rule).
    """
    from biasbuster.collectors.oa_license import fetch_oa_status
    from biasbuster.utils import pubtype
    from scripts.fetch_fulltext_for_expert_ratings import DEFAULT_CACHE_DIR, _cache_path, fetch_jats

    records = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    seen = admitted = rejected = 0
    for rec in records:
        seen += 1
        try:
            parsed = parse_record(rec)
        except Exception as exc:  # noqa: BLE001 — one bad record must not abort the run
            rejected += 1
            store.log_reject(rec, "malformed_record", str(exc))
            continue
        if parsed is None:
            rejected += 1
            store.log_reject(rec, "no_title_or_incomplete_tuple", "dropped")
            continue

        resolution = await resolve_pmid_by_title(
            client, parsed.title,
            pubmed_base=config.pubmed_base, ncbi_api_key=config.ncbi_api_key,
        )
        if not resolution.pmid:
            rejected += 1
            store.log_reject(
                {"paper_id": parsed.paper_id, "title": parsed.title},
                "title_unresolved", f"{resolution.reason} (sim={resolution.similarity:.2f})",
            )
            continue
        pmid = resolution.pmid

        oa = await fetch_oa_status(client, pmid, base=config.europmc_base)
        if not oa.in_oa_subset:
            rejected += 1
            store.log_reject({"pmid": pmid}, "not_oa_subset", oa.license.raw)
            continue

        pt = await pubtype.fetch_publication_types(
            [pmid], client=client, ncbi_api_key=config.ncbi_api_key,
        )
        if pubtype.classify(pt.get(pmid, [])) != "trial":
            rejected += 1
            store.log_reject({"pmid": pmid}, "non_trial_pubtype", str(pt.get(pmid)))
            continue

        pmcid = oa.pmcid if oa.pmcid.startswith("PMC") else f"PMC{oa.pmcid}"
        status, _n_bytes = await fetch_jats(client, pmid, pmcid, DEFAULT_CACHE_DIR)
        if status != "ok":
            rejected += 1
            store.log_reject({"pmid": pmid}, "fulltext_fetch_failed", status)
            continue

        fulltext_path = str(_cache_path(DEFAULT_CACHE_DIR, pmid))
        try:
            store.upsert_item(_build_item(resolution, oa, parsed.rob2, fulltext_path, parsed.title))
            admitted += 1
        except (LitmusError, sqlite3.Error) as exc:
            rejected += 1
            store.log_reject({"pmid": pmid}, "litmus_or_db", str(exc))

    logger.info(
        "ROBoto2 ingest: seen=%d admitted=%d rejected=%d", seen, admitted, rejected,
    )
    return IngestReport(seen, admitted, rejected)


if __name__ == "__main__":
    import asyncio

    from config import Config

    async def _main() -> None:
        """Run the full ROBoto2 ingest against the normalized dataset file.

        Terminal-only (CLAUDE.md >2-minute rule) — do not invoke from an
        agent session. Requires ``dataset/roboto2/roboto2.json`` (produced by
        ``python -m studies.oa_rob_benchmark.convert_roboto2_csv``) and the R1
        ROBoto2-reuse-terms question (spec §9) to be confirmed before
        publishing any resulting rows.
        """
        cfg = Config()
        store = BenchmarkStore("dataset/oa_rob_benchmark.db")
        async with httpx.AsyncClient(timeout=60) as client:
            report = await ingest_roboto2(
                DEFAULT_DATASET_PATH, store, client=client, config=cfg,
            )
            print(report)

    asyncio.run(_main())

"""Stage A seed pool 1 — ingest ROBoto2 manual-gold RoB 2 assessments.

Keeps only human ``manual_assessment`` rows (drops LLM-assisted
``roboto2_assessment``). Each trial's signalling answers become a canonical
six-field tuple via the RoB 2 algorithm (``tuple_from_signalling``); the
trial is admitted only if it is in the PMC OA subset with a redistributable
license, its PubMed PublicationType is trial-compatible, and its JATS full
text is fetched — the spec §4 litmus test, enforced by
``BenchmarkStore.upsert_item``. Rejections are logged via
``store.log_reject``, never silent. Incremental: each admitted trial is
upserted immediately (CLAUDE.md's checkpoint/incremental-save rule).

ROBoto2 record shape (from github.com/larchlab/ROBoto2 dataset README):
``paper_id``, ``manual_assessment`` (list of per-domain signalling dicts),
``roboto2_assessment`` (LLM-assisted; ignored). The real dataset file
(``dataset/roboto2/roboto2.json``) is not present in this repo at the time
of writing — ``parse_roboto2_record`` is intentionally defensive (tolerates
missing/malformed fields by returning None rather than raising) so it fails
safe if the real shape differs in details. See the task report for the one
assumption that most needs confirming against the real file: whether
``paper_id`` already embeds a bare PMID (as this parser assumes) or is some
other identifier requiring genuine PMID resolution (spec §6.1 step 2).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from biasbuster.collectors.oa_license import OAStatus
from studies.oa_rob_benchmark.rob2_tuple import RoB2Tuple, tuple_from_signalling
from studies.oa_rob_benchmark.store import BenchmarkStore, LitmusError

if TYPE_CHECKING:
    # ``config.py`` is gitignored (copied from config.example.py per-repo),
    # so it must not be imported at module load time — only the type
    # checker needs this; PEP 563 (``from __future__ import annotations``
    # above) makes the runtime annotation a lazy string.
    from config import Config

logger = logging.getLogger(__name__)

#: ROBoto2 ``paper_id`` values are expected to embed a bare PubMed PMID
#: (e.g. "PMID:12345678"); PMIDs are plain digit strings with no fixed
#: minimum length (current PubMed PMIDs run to 8 digits, but older ones
#: and synthetic test fixtures may be shorter), so any digit run matches.
_PMID_RE = re.compile(r"(\d+)")

#: Provenance/labelling constants for every ROBoto2-sourced item (spec §6.1).
_LABEL_SOURCE = "roboto2"
_SOURCE_REVIEW_PMID = "ROBoto2"  # non-empty for litmus §4.4; not a real PMID
_RESOLUTION_METHOD = "roboto2_paper_id"
_EXTRACTION_METHOD = "signalling_algorithm"
_PUBTYPE_CHECK = "trial"


@dataclass
class IngestReport:
    """Summary counters for one ``ingest_roboto2`` run."""
    seen: int
    admitted: int
    rejected: int


def parse_roboto2_record(rec: dict) -> tuple[str, dict, dict] | None:
    """Extract ``(pmid, domain_answers, overall_answers)`` from a manual record.

    Returns ``None`` (never raises) for records with no manual assessment
    (LLM-assisted-only rows, per spec §6.1 step 1), from which a PMID
    cannot be parsed out of ``paper_id``, or whose ``manual_assessment``
    entries are not shaped as expected (e.g. a non-dict entry, or a
    ``signalling`` value that isn't a mapping — a plausible real-world
    encoding is a flat list of ``{"question": ..., "answer": ...}`` rows
    instead of a ``{question: answer}`` dict). ``domain_answers`` maps each
    domain's canonical extractor name (``randomization`` / ``deviations`` /
    ``missing_outcome`` / ``measurement`` / ``reporting`` — see
    ``rob2_tuple._DOMAINS``) to its raw signalling-question answers dict.
    ``overall_answers`` is always ``{}`` — ROBoto2's manual records do not
    separately record an overall judgement, so ``tuple_from_signalling``
    derives it via worst-wins.
    """
    manual = rec.get("manual_assessment") or []
    if not manual:
        return None
    m = _PMID_RE.search(str(rec.get("paper_id", "")))
    if not m:
        return None
    domain_answers: dict[str, dict] = {}
    try:
        for d in manual:
            if not isinstance(d, dict):
                continue
            domain = d.get("domain")
            if not domain:
                continue
            signalling = d.get("signalling", {})
            if not isinstance(signalling, dict):
                continue
            domain_answers[str(domain)] = dict(signalling)
    except (AttributeError, TypeError, ValueError) as exc:
        # Structural surprise beyond the isinstance guards above (e.g. an
        # exotic mapping-like object that raises on dict()) — fail safe
        # rather than violate the "never raises" contract this function
        # promises its caller.
        logger.warning(
            "parse_roboto2_record: malformed manual_assessment entry: %s", exc,
        )
        return None
    return m.group(1), domain_answers, {}


def _build_item(
    pmid: str, oa: OAStatus, rob2: RoB2Tuple, fulltext_path: str,
) -> dict:
    """Map a parsed + fetched trial to a ``benchmark_item`` store dict.

    Populates every NOT NULL / litmus-required column (store.py's
    ``_SCHEMA``): identity + license facts from ``oa`` (an
    ``oa_license.OAStatus``), the RoB 2 tuple's six fields, and the
    ROBoto2-specific provenance constants module-level above.
    """
    lic = oa.license
    return {
        "trial_pmid": pmid,
        "trial_pmcid": oa.pmcid,
        "trial_doi": "",
        "trial_title": "",
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
        "resolution_method": _RESOLUTION_METHOD,
        "similarity_score": None,
        "pubtype_check": _PUBTYPE_CHECK,
        "extraction_method": _EXTRACTION_METHOD,
        "manual_verified": False,
    }


async def ingest_roboto2(
    dataset_path: str, store: BenchmarkStore, *, client: httpx.AsyncClient, config: Config,
) -> IngestReport:
    """Ingest ROBoto2 manual gold into the benchmark store (incremental).

    Network steps reuse existing modules and their own retry-with-backoff
    logic rather than a new retry loop: OA-subset + license status
    (``oa_license.fetch_oa_status``), PubMed PublicationType
    (``pubtype.fetch_publication_types`` + ``pubtype.classify``), and JATS
    full-text fetch (``fetch_fulltext_for_expert_ratings.fetch_jats``, using
    the same on-disk cache the single-paper annotator reads from).

    Every rejection path calls ``store.log_reject`` — never a silent
    ``continue``. A ``LitmusError`` from ``store.upsert_item`` (the store's
    own final litmus check) is caught and logged rather than aborting the
    batch, so one bad row never loses the rest of the run.

    This coroutine performs network I/O over the whole dataset (OA lookup +
    PublicationType fetch + JATS download per trial) — run it from a
    terminal via the module's ``__main__``, not in-session (CLAUDE.md's
    >2-minute-process rule).
    """
    from biasbuster.collectors.oa_license import fetch_oa_status
    from biasbuster.utils import pubtype
    from scripts.fetch_fulltext_for_expert_ratings import DEFAULT_CACHE_DIR, _cache_path, fetch_jats

    records = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    seen = admitted = rejected = 0
    for rec in records:
        seen += 1
        try:
            parsed = parse_roboto2_record(rec)
        except Exception as exc:  # noqa: BLE001 — one bad record must not abort the run
            rejected += 1
            store.log_reject(rec, "malformed_record", str(exc))
            continue
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
            store.upsert_item(_build_item(pmid, oa, rob2, fulltext_path))
            admitted += 1
        except LitmusError as exc:
            rejected += 1
            store.log_reject({"pmid": pmid}, "litmus", str(exc))

    logger.info(
        "ROBoto2 ingest: seen=%d admitted=%d rejected=%d", seen, admitted, rejected,
    )
    return IngestReport(seen, admitted, rejected)


if __name__ == "__main__":
    import asyncio

    from config import Config

    async def _main() -> None:
        """Run the full ROBoto2 ingest against the real dataset file.

        Terminal-only (CLAUDE.md >2-minute rule) — do not invoke from an
        agent session. Requires ``dataset/roboto2/roboto2.json`` to be
        placed by the operator, and the R1 ROBoto2-reuse-terms question
        (spec §9) to be confirmed before publishing any resulting rows.
        """
        cfg = Config()
        store = BenchmarkStore("dataset/oa_rob_benchmark.db")
        async with httpx.AsyncClient(timeout=60) as client:
            report = await ingest_roboto2(
                "dataset/roboto2/roboto2.json", store, client=client, config=cfg,
            )
            print(report)

    asyncio.run(_main())

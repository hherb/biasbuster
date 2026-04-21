"""Import a curated RoB 2 manual verification CSV into the database.

Reads a plaintext CSV seed at ``dataset/manual_verification_sets/*.csv``
and:

  1. Upserts each row into ``papers`` (source='cochrane_rob', with the
     Cochrane-authoritative RoB 2 ratings populated from the CSV).
  2. Fetches PubMed metadata (title/abstract/journal/year/authors) for
     any paper whose abstract is still empty/short.
  3. Fetches Europe PMC ``fullTextXML`` for each row's PMCID and caches
     it at ``dataset/rob2_verification_fulltexts/<PMID>.xml``.
  4. Tags each paper in the ``manually_verified`` table under the
     requested ``verification_set``.

The import is fully idempotent: re-runs refresh metadata and full text
without duplicating rows.

Usage:
    uv run python -m scripts.import_rob2_verification_set \\
        [--csv PATH] \\
        [--verification-set TAG]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from biasbuster.collectors.cochrane_rob import (
    RoBAssessment,
    rob_assessment_to_paper_dict,
)
from biasbuster.collectors.retraction_watch import RetractionWatchCollector
from biasbuster.database import Database
from biasbuster.utils.retry import fetch_with_retry
from config import Config

logger = logging.getLogger("import_rob2_verification_set")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = (
    REPO_ROOT
    / "dataset"
    / "manual_verification_sets"
    / "rob2_manual_verify_20260421.csv"
)
DEFAULT_VERIFICATION_SET = "rob2_manual_verify_20260421"
FULLTEXT_DIR = REPO_ROOT / "dataset" / "rob2_verification_fulltexts"

EUROPMC_FULLTEXT = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
)
# Matches the guard in scripts/fetch_cochrane_jats.py: responses smaller
# than this are almost always 404 stub pages, not real JATS.
MIN_JATS_BYTES = 1024
HTTP_TIMEOUT_SECONDS = 60.0
REQUEST_DELAY_SECONDS = 0.5  # ~2 req/s to Europe PMC, polite default
USER_AGENT = "biasbuster/rob2-manual-verify-import"

PMCID_RE = re.compile(r"(PMC\d+)", re.IGNORECASE)
ABSTRACT_MIN_CHARS = 100  # refetch if shorter than this


@dataclass
class RowResult:
    pmid: str
    trial_name: str
    overall_rob: str
    abstract_chars: int
    pmcid: str
    fulltext_bytes: int
    fulltext_ok: bool
    status: str  # 'ok' | 'no_abstract' | 'fulltext_missing' | 'fulltext_error'
    notes: str = ""


def parse_csv_rows(csv_path: Path) -> list[dict]:
    """Load and lightly validate the verification-set CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    required = {
        "study_id", "trial_name", "pmid", "pmc_url", "source_review",
        "randomization", "deviations", "missing_data",
        "measurement", "reporting", "overall",
    }
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing required columns: {sorted(missing)}"
            )
        rows = [dict(r) for r in reader]
    if not rows:
        raise ValueError(f"CSV {csv_path} has no data rows")
    return rows


def row_to_paper_dict(row: dict) -> dict:
    """Convert a CSV row to the ``papers`` insert dict via RoBAssessment."""
    a = RoBAssessment(
        study_id=row["trial_name"],
        pmid=row["pmid"],
        study_title=row["trial_name"],
        randomization_bias=row["randomization"],
        deviation_bias=row["deviations"],
        missing_outcome_bias=row["missing_data"],
        measurement_bias=row["measurement"],
        reporting_bias=row["reporting"],
        overall_rob=row["overall"],
        cochrane_review_title=row["source_review"],
        cochrane_review_pmid=(row.get("cochrane_review_pmid") or "").strip(),
        cochrane_review_doi=(row.get("cochrane_review_doi") or "").strip(),
    )
    return rob_assessment_to_paper_dict(a)


def extract_pmcid(pmc_url: str) -> str:
    """Extract a bare PMCID (e.g. 'PMC7383595') from a URL."""
    m = PMCID_RE.search(pmc_url)
    return m.group(1).upper() if m else ""


def fulltext_path_for(pmid: str) -> Path:
    return FULLTEXT_DIR / f"{pmid}.xml"


def fulltext_already_cached(pmid: str) -> bool:
    p = fulltext_path_for(pmid)
    return p.exists() and p.stat().st_size >= MIN_JATS_BYTES


async def fetch_fulltext(
    client: httpx.AsyncClient, pmid: str, pmcid: str
) -> tuple[int, str]:
    """Fetch Europe PMC JATS for one PMCID.

    Returns (bytes_written, status). status values:
      'ok' | 'already_cached' | 'not_found' | 'too_small' | 'error' | 'http_<N>'
    """
    if fulltext_already_cached(pmid):
        return fulltext_path_for(pmid).stat().st_size, "already_cached"
    if not pmcid:
        return 0, "no_pmcid"

    url = EUROPMC_FULLTEXT.format(pmcid=pmcid)
    try:
        resp = await fetch_with_retry(
            client, "GET", url, max_retries=3, base_delay=2.0
        )
    except Exception as exc:  # noqa: BLE001 — surface raw error
        logger.warning("PMID %s (%s): fetch error: %s", pmid, pmcid, exc)
        return 0, "error"

    if resp.status_code == 404:
        return 0, "not_found"
    if resp.status_code != 200:
        logger.warning(
            "PMID %s (%s): HTTP %d", pmid, pmcid, resp.status_code
        )
        return 0, f"http_{resp.status_code}"

    body = resp.content
    if len(body) < MIN_JATS_BYTES:
        logger.warning(
            "PMID %s (%s): response too small (%d bytes)",
            pmid, pmcid, len(body),
        )
        return len(body), "too_small"

    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    fulltext_path_for(pmid).write_bytes(body)
    return len(body), "ok"


async def fetch_all_pubmed_metadata(
    config: Config, pmids: list[str]
) -> dict[str, dict]:
    async with RetractionWatchCollector(
        mailto=config.crossref_mailto,
        ncbi_api_key=config.ncbi_api_key,
    ) as collector:
        return await collector.fetch_pubmed_abstracts_batch(pmids)


def update_paper_from_pubmed(db: Database, pmid: str, pubmed: dict) -> None:
    """Patch title/abstract/journal/year/authors into an existing paper row."""
    existing = db.conn.execute(
        "SELECT abstract, title FROM papers WHERE pmid = ?", (pmid,)
    ).fetchone()
    if existing is None:
        return

    abstract = pubmed.get("abstract") or ""
    title = pubmed.get("title") or ""
    journal = pubmed.get("journal")
    year = pubmed.get("year")
    authors = pubmed.get("authors")

    updates: list[tuple[str, object]] = []
    if abstract and len(abstract) >= ABSTRACT_MIN_CHARS and (
        not existing["abstract"]
        or len(existing["abstract"]) < ABSTRACT_MIN_CHARS
    ):
        updates.append(("abstract", abstract))
    if title and (
        not existing["title"] or len(existing["title"]) < 10
    ):
        updates.append(("title", title))
    if journal:
        updates.append(("journal", journal))
    if year:
        updates.append(("year", year))
    if authors:
        updates.append(("authors", json.dumps(authors)))

    if not updates:
        return
    set_clause = ", ".join(f"{col} = ?" for col, _ in updates)
    params = [val for _, val in updates] + [pmid]
    db.conn.execute(
        f"UPDATE papers SET {set_clause} WHERE pmid = ?", params
    )
    db.conn.commit()


def abstract_chars(db: Database, pmid: str) -> int:
    row = db.conn.execute(
        "SELECT length(abstract) AS n FROM papers WHERE pmid = ?", (pmid,)
    ).fetchone()
    return int(row["n"] or 0) if row else 0


def render_summary(results: list[RowResult]) -> str:
    lines = [
        "| PMID | trial | overall_rob | abs_chars | pmcid | fulltext_bytes | status |",
        "|------|-------|-------------|-----------|-------|----------------|--------|",
    ]
    for r in results:
        lines.append(
            f"| {r.pmid} | {r.trial_name} | {r.overall_rob} | "
            f"{r.abstract_chars} | {r.pmcid or '-'} | "
            f"{r.fulltext_bytes} | {r.status} |"
        )
    return "\n".join(lines)


async def run(csv_path: Path, verification_set: str, config: Config) -> int:
    rows = parse_csv_rows(csv_path)
    logger.info("Loaded %d rows from %s", len(rows), csv_path)

    db = Database(config.db_path)
    db.initialize()

    # Stage 1: upsert papers with RoB 2 ratings (Cochrane-authoritative).
    for row in rows:
        paper = row_to_paper_dict(row)
        db.upsert_cochrane_paper(paper)

    # Stage 2: fetch PubMed metadata in one batch.
    pmids = [row["pmid"] for row in rows]
    logger.info("Fetching PubMed metadata for %d PMIDs", len(pmids))
    pubmed_results = await fetch_all_pubmed_metadata(config, pmids)

    for pmid, data in pubmed_results.items():
        update_paper_from_pubmed(db, pmid, data)

    missing_from_pubmed = sorted(set(pmids) - set(pubmed_results))
    if missing_from_pubmed:
        logger.warning(
            "PubMed returned no data for %d PMID(s) — these rows "
            "may appear as 'no_abstract' in the summary even though "
            "the root cause is a PubMed batch failure: %s",
            len(missing_from_pubmed), missing_from_pubmed,
        )

    # Stage 3: fetch Europe PMC JATS full text per row, with polite spacing.
    results: list[RowResult] = []
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for row in rows:
            pmid = row["pmid"]
            pmcid = extract_pmcid(row["pmc_url"])
            bytes_written, ft_status = await fetch_fulltext(
                client, pmid, pmcid
            )
            await asyncio.sleep(REQUEST_DELAY_SECONDS)

            ft_ok = ft_status in {"ok", "already_cached"} and bytes_written >= MIN_JATS_BYTES
            ft_path = (
                str(fulltext_path_for(pmid).relative_to(REPO_ROOT))
                if ft_ok
                else None
            )
            abs_chars = abstract_chars(db, pmid)

            status: str
            if abs_chars < ABSTRACT_MIN_CHARS:
                status = "no_abstract"
            elif not ft_ok:
                status = ft_status or "fulltext_error"
            else:
                status = "ok"

            notes: Optional[str] = None
            if status != "ok":
                notes = f"fulltext_status={ft_status}; abs_chars={abs_chars}"

            db.upsert_manually_verified(
                pmid=pmid,
                verification_set=verification_set,
                trial_name=row["trial_name"],
                source_review=row["source_review"],
                fulltext_path=ft_path,
                fulltext_ok=ft_ok,
                notes=notes,
            )

            results.append(
                RowResult(
                    pmid=pmid,
                    trial_name=row["trial_name"],
                    overall_rob=row["overall"],
                    abstract_chars=abs_chars,
                    pmcid=pmcid,
                    fulltext_bytes=bytes_written,
                    fulltext_ok=ft_ok,
                    status=status,
                    notes=notes or "",
                )
            )

    print(render_summary(results))
    failures = [r for r in results if r.status != "ok"]
    if failures:
        logger.warning(
            "%d/%d rows have non-'ok' status; see summary table above",
            len(failures), len(results),
        )
    return 0 if not failures else 1


def _configure_logging() -> None:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        log_dir
        / f"import_rob2_verification_set_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger.info("Logging to %s", log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"Path to the verification-set CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--verification-set", default=DEFAULT_VERIFICATION_SET,
        help=(
            "Tag value written to manually_verified.verification_set "
            f"(default: {DEFAULT_VERIFICATION_SET})"
        ),
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help=(
            "Override the SQLite path from config.Config().db_path. "
            "Useful when the default DB is on a legacy schema and you "
            "want to target a post-methodology database (e.g. "
            "dataset/biasbuster_recovered.db)."
        ),
    )
    args = parser.parse_args()

    _configure_logging()

    config = Config()
    if args.db is not None:
        logger.info("Overriding config.db_path: %s -> %s", config.db_path, args.db)
        config.db_path = str(args.db)
    return asyncio.run(run(args.csv, args.verification_set, config))


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit PubMed PublicationType for every Cochrane-tagged paper in the DB.

Purpose: after the Stage E discovery that 3 non-trial papers had leaked
into the cohort as Stage A LLM extraction artifacts, this script
systematically checks every remaining Cochrane-tagged paper against
PubMed's `<PublicationType>` metadata to find any more non-trial
artifacts. Output is a classification per PMID: trial | non_trial |
ambiguous | unknown, with the PubMedtypes list preserved for review.

Writes: dataset/cochrane_pubtype_audit.csv
Prints: summary counts and a breakdown by Cochrane overall_rob rating.

Usage:
    uv run python scripts/audit_publication_types.py
    uv run python scripts/audit_publication_types.py --include-excluded  # also audit soft-deleted PMIDs
    uv run python scripts/audit_publication_types.py --apply-exclusions  # mark found non-trials as excluded
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biasbuster.utils.pubtype import (  # noqa: E402
    NON_TRIAL_TYPES,
    TRIAL_TYPES,
    classify,
    fetch_publication_types,
)

logger = logging.getLogger("audit_publication_types")

DEFAULT_DB = Path("dataset/biasbuster.db")
DEFAULT_OUTPUT = Path("dataset/cochrane_pubtype_audit.csv")
USER_AGENT = "biasbuster/pubtype-audit"

CSV_FIELDS = [
    "pmid",
    "overall_rob",
    "excluded_before",
    "publication_types",
    "classification",  # trial | non_trial | ambiguous | unknown
    "title",
]


def cochrane_pmids(db_path: Path, include_excluded: bool) -> list[tuple[str, str, int, str]]:
    """Return (pmid, overall_rob, excluded, title) for Cochrane-tagged papers."""
    with sqlite3.connect(str(db_path)) as conn:
        where = "source LIKE 'cochrane%'"
        if not include_excluded:
            where += " AND COALESCE(excluded,0)=0"
        rows = conn.execute(
            f"SELECT pmid, COALESCE(overall_rob,''), COALESCE(excluded,0), "
            f"COALESCE(title,'') FROM papers WHERE {where} ORDER BY pmid"
        ).fetchall()
    return rows


# classify(), parse_publication_types(), and the batch fetcher have moved
# to biasbuster.utils.pubtype. This script is now a thin wrapper that
# builds a CSV/summary report from those primitives.


def write_csv(
    rows: list[tuple[str, str, int, str]],
    types_by_pmid: dict[str, list[str]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for pmid, rob, excluded, title in rows:
            types = types_by_pmid.get(pmid, [])
            writer.writerow({
                "pmid": pmid,
                "overall_rob": rob,
                "excluded_before": excluded,
                "publication_types": "; ".join(types),
                "classification": classify(types),
                "title": title[:120],
            })


def summarize(
    rows: list[tuple[str, str, int, str]],
    types_by_pmid: dict[str, list[str]],
) -> dict:
    class_counts: Counter = Counter()
    class_by_rob: dict[str, Counter] = {}
    non_trial_types: Counter = Counter()
    non_trial_pmids: list[tuple[str, str, str, str]] = []
    for pmid, rob, _excluded, title in rows:
        types = types_by_pmid.get(pmid, [])
        cls = classify(types)
        class_counts[cls] += 1
        class_by_rob.setdefault(rob or "(empty)", Counter())[cls] += 1
        if cls == "non_trial":
            for t in types:
                if t in NON_TRIAL_TYPES:
                    non_trial_types[t] += 1
            non_trial_pmids.append((pmid, rob, "; ".join(types), title[:70]))
    return {
        "class_counts": class_counts,
        "class_by_rob": class_by_rob,
        "non_trial_types": non_trial_types,
        "non_trial_pmids": non_trial_pmids,
    }


def apply_exclusions(db_path: Path, pmids: list[str], reason: str) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        placeholders = ",".join("?" for _ in pmids)
        conn.execute(
            f"UPDATE papers SET excluded=1, excluded_reason=? "
            f"WHERE pmid IN ({placeholders}) AND COALESCE(excluded,0)=0",
            (reason, *pmids),
        )
        conn.commit()
        return conn.total_changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument(
        "--apply-exclusions",
        action="store_true",
        help="Mark all detected non-trial papers as excluded in the DB.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    rows = cochrane_pmids(args.db, include_excluded=args.include_excluded)
    pmids = [r[0] for r in rows]
    logger.info("Cochrane papers to audit: %d (include_excluded=%s)",
                len(pmids), args.include_excluded)

    types_by_pmid = asyncio.run(
        fetch_publication_types(pmids, user_agent=USER_AGENT)
    )
    write_csv(rows, types_by_pmid, args.output)

    report = summarize(rows, types_by_pmid)
    cc = report["class_counts"]
    total = sum(cc.values())
    print()
    print("Classification summary:")
    for cls in ("trial", "non_trial", "ambiguous", "unknown"):
        n = cc.get(cls, 0)
        pct = 100.0 * n / max(1, total)
        print(f"  {cls:<12} {n:>4}  ({pct:.1f}%)")

    print("\nBreakdown by Cochrane overall_rob:")
    print(f"  {'rob':<14} {'trial':>6} {'non_trial':>10} {'ambiguous':>10} {'unknown':>8}")
    for rob, counts in sorted(report["class_by_rob"].items()):
        print(f"  {rob:<14} {counts.get('trial',0):>6} "
              f"{counts.get('non_trial',0):>10} "
              f"{counts.get('ambiguous',0):>10} "
              f"{counts.get('unknown',0):>8}")

    if report["non_trial_types"]:
        print("\nNon-trial PublicationType frequency:")
        for t, n in report["non_trial_types"].most_common():
            print(f"  {t:<35} {n:>3}")

    if report["non_trial_pmids"]:
        print(f"\nDetected {len(report['non_trial_pmids'])} non-trial PMIDs (first 15):")
        for pmid, rob, types, title in report["non_trial_pmids"][:15]:
            print(f"  {pmid} rob={rob:<14} types=[{types}]")
            print(f"           title: {title}")

    print(f"\nFull audit written to: {args.output}")

    if args.apply_exclusions and report["non_trial_pmids"]:
        to_exclude = [p for p, *_ in report["non_trial_pmids"]]
        reason = "non-trial: PubMed PublicationType audit 2026-04-17"
        changed = apply_exclusions(args.db, to_exclude, reason)
        logger.info("Applied excluded=1 to %d papers", changed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

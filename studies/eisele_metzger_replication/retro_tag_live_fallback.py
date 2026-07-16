"""Retro-tag live-path algorithmic-fallback rows as raw_label='FALLBACK'.

Background
----------
Commit 3f1d78d added an algorithmic fallback to ``eval_ollama.parse_response``:
when the model omits the explicit ``judgement`` field but emits valid
``signalling_answers``, the judgement is derived in code via the Cochrane
per-domain rules. Until 2026-07-16 the runners ingested such rows with
``raw_label = <derived judgement>`` and ``parse_status = 'ok'`` — i.e.
indistinguishable from genuine model-emitted judgements. That silently
contaminates the pre-registered "model-emitted judgement" primary metric,
and ``recover_parse_failures.py`` cannot find these rows because it only
scans parse *failures*.

This script re-parses every successful domain-call raw response in
``evaluation_run`` and, where the judgement can only have come from the
algorithmic fallback (no explicit ``judgement`` field in the JSON, but a
derivable one from ``signalling_answers``), tags the matching
``benchmark_judgment`` row ``raw_label='FALLBACK'`` and stamps
``evaluation_run.error`` with the same marker used by both the fixed
runners and ``recover_parse_failures.py``.

Idempotent: rows already tagged FALLBACK are skipped. Rows whose raw
response contains an explicit judgement are never touched.

Usage
-----
    uv run python studies/eisele_metzger_replication/retro_tag_live_fallback.py            # dry run (default)
    uv run python studies/eisele_metzger_replication/retro_tag_live_fallback.py --apply    # write tags
    uv run python studies/eisele_metzger_replication/retro_tag_live_fallback.py \
        --db-path dataset/eisele_metzger_benchmark.spark.db --apply                        # a shard
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[1]
sys.path.insert(0, str(STUDY_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from eval_ollama import (  # noqa: E402
    FALLBACK_ERROR_MSG,
    FALLBACK_RAW_LABEL,
    parse_response,
)

DEFAULT_DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"


def retro_tag(conn: sqlite3.Connection, apply: bool) -> dict[str, int]:
    """Scan successful domain calls; tag fallback-derived rows. Returns counts."""
    cur = conn.cursor()
    rows = cur.execute(
        """SELECT er.rct_id, er.source, er.domain, er.raw_response
           FROM evaluation_run er
           JOIN benchmark_judgment bj
             ON bj.rct_id = er.rct_id AND bj.source = er.source
                AND bj.domain = er.domain
           WHERE er.domain != 'overall'
             AND er.parse_status IN ('ok', 'retry_succeeded')
             AND bj.valid = 1
             AND (bj.raw_label IS NULL OR bj.raw_label != ?)""",
        (FALLBACK_RAW_LABEL,),
    ).fetchall()

    counts = {"scanned": len(rows), "tagged": 0, "unparseable": 0}
    for rct_id, source, domain, raw_response in rows:
        judgment, _rationale, is_fallback = parse_response(
            raw_response or "", "domain", domain_code=domain,
        )
        if judgment is None:
            # Stored as ok but no longer parseable — flag loudly, don't touch.
            counts["unparseable"] += 1
            print(f"[warn] {rct_id} {source} {domain}: parse_status ok but "
                  f"raw_response no longer parses; left untouched",
                  file=sys.stderr)
            continue
        if not is_fallback:
            continue
        counts["tagged"] += 1
        print(f"[tag] {rct_id} {source} {domain}: judgement={judgment} "
              f"was algorithm-derived -> raw_label={FALLBACK_RAW_LABEL}")
        if apply:
            cur.execute(
                """UPDATE benchmark_judgment SET raw_label = ?
                   WHERE rct_id = ? AND source = ? AND domain = ?""",
                (FALLBACK_RAW_LABEL, rct_id, source, domain),
            )
            cur.execute(
                """UPDATE evaluation_run SET error = ?
                   WHERE rct_id = ? AND source = ? AND domain = ?""",
                (FALLBACK_ERROR_MSG, rct_id, source, domain),
            )
    if apply:
        conn.commit()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the tags. Without this flag the script only reports "
             "what it would tag (dry run).",
    )
    args = parser.parse_args()
    if not args.db_path.exists():
        print(f"[error] DB not found at {args.db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(args.db_path)
    try:
        counts = retro_tag(conn, apply=args.apply)
    finally:
        conn.close()

    mode = "APPLIED" if args.apply else "DRY RUN (use --apply to write)"
    print(f"\n[{mode}] scanned={counts['scanned']} "
          f"tagged={counts['tagged']} unparseable={counts['unparseable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

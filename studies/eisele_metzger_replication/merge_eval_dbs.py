"""Merge evaluation results from one or more shard DBs into a canonical DB.

Multi-host workflow (Phase 5 was run on more than one machine):

  Mac runs:   uv run ... run_evaluation.py ... --db-path .../mac.db
  Spark runs: uv run ... run_evaluation.py ... --db-path .../spark.db

After both runs finish, copy the shard DBs back to one machine and run:

  uv run python merge_eval_dbs.py --dest .../eisele_metzger_benchmark.db \\
      --source .../mac.db --source .../spark.db

The merge copies new rows from each source's `evaluation_run` and
`benchmark_judgment` tables into the destination. By design, each host
should have written under non-overlapping `source` labels (e.g. Mac
runs gpt_oss_20b and gemma4_26b; Spark runs qwen3_6_35b). Conflicts on
the (rct_id, source, domain) primary key are handled by INSERT OR
IGNORE — i.e. the destination's existing row wins. Use --replace to
flip that behaviour to INSERT OR REPLACE if you intentionally want
shard B's run to overwrite shard A's.

Rows from `benchmark_rct` are NOT merged: that table is loaded from the
frozen EM CSV via build_benchmark_db.py and is byte-identical across
shards. Same for the `benchmark_judgment` rows from EM-derived sources
(cochrane, em_claude2_*) — those are loaded by build, not produced by
evaluation. The merge therefore only touches rows whose `source` matches
one of the model_short labels in MODEL_REGISTRY.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "studies/eisele_metzger_replication"))

from run_evaluation import MODEL_REGISTRY  # noqa: E402

# Source labels we recognise as evaluation outputs (anything that starts
# with one of the model_short labels). Anything else is left alone.
EVAL_SOURCE_PREFIXES = tuple(MODEL_REGISTRY.keys())


def is_eval_source(source_label: str) -> bool:
    return any(source_label.startswith(f"{p}_") for p in EVAL_SOURCE_PREFIXES)


def merge_one_source(dest: sqlite3.Connection, source_db_path: Path,
                     replace: bool = False) -> tuple[int, int]:
    """Copy evaluation_run + benchmark_judgment rows from src into dest.

    Returns (rows_judgment_inserted, rows_eval_run_inserted).
    """
    if not source_db_path.exists():
        raise FileNotFoundError(source_db_path)

    cur = dest.cursor()
    cur.execute("ATTACH DATABASE ? AS src", (str(source_db_path),))
    try:
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        # Build LIKE filter from prefixes — only copy rows whose source
        # is one of our evaluation outputs.
        prefixes = " OR ".join(["source LIKE ?" for _ in EVAL_SOURCE_PREFIXES])
        params = [f"{p}_%" for p in EVAL_SOURCE_PREFIXES]

        # benchmark_judgment
        before_j = cur.execute("SELECT COUNT(*) FROM benchmark_judgment").fetchone()[0]
        cur.execute(
            f"""{verb} INTO benchmark_judgment
                  (rct_id, source, domain, judgment, rationale, valid, raw_label)
                SELECT rct_id, source, domain, judgment, rationale, valid, raw_label
                FROM src.benchmark_judgment
                WHERE {prefixes}""",
            params,
        )
        after_j = cur.execute("SELECT COUNT(*) FROM benchmark_judgment").fetchone()[0]

        # evaluation_run
        before_e = cur.execute("SELECT COUNT(*) FROM evaluation_run").fetchone()[0]
        cur.execute(
            f"""{verb} INTO evaluation_run
                  (rct_id, source, domain, model_id, protocol, pass_n,
                   started_at, completed_at, duration_seconds,
                   input_tokens, output_tokens,
                   cache_read_tokens, cache_write_tokens,
                   raw_response, parse_status, parse_attempts, error)
                SELECT rct_id, source, domain, model_id, protocol, pass_n,
                       started_at, completed_at, duration_seconds,
                       input_tokens, output_tokens,
                       cache_read_tokens, cache_write_tokens,
                       raw_response, parse_status, parse_attempts, error
                FROM src.evaluation_run
                WHERE {prefixes}""",
            params,
        )
        after_e = cur.execute("SELECT COUNT(*) FROM evaluation_run").fetchone()[0]

        dest.commit()
        return (after_j - before_j, after_e - before_e)
    finally:
        cur.execute("DETACH DATABASE src")


def show_source_breakdown(conn: sqlite3.Connection, label: str) -> None:
    """Print row counts in each table grouped by source for a quick sanity check."""
    cur = conn.cursor()
    print(f"\n## {label} — source breakdown\n")
    print(f"{'source':<45} {'judgments':>10} {'runs':>10}")
    rows = cur.execute(
        """SELECT j.source,
                  (SELECT COUNT(*) FROM benchmark_judgment WHERE source = j.source),
                  (SELECT COUNT(*) FROM evaluation_run WHERE source = j.source)
           FROM benchmark_judgment j
           GROUP BY j.source
           ORDER BY j.source"""
    ).fetchall()
    for src, n_j, n_r in rows:
        marker = "  ← eval" if is_eval_source(src) else ""
        print(f"{src:<45} {n_j:>10} {n_r:>10}{marker}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True,
                        help="Canonical DB to merge into.")
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="Shard DB(s) to read from. Pass --source N times for N shards.")
    parser.add_argument(
        "--replace", action="store_true",
        help="Overwrite existing destination rows on PK conflict (default: keep destination).",
    )
    parser.add_argument(
        "--show-only", action="store_true",
        help="Print the per-source breakdown without performing the merge.",
    )
    args = parser.parse_args()

    if not args.dest.exists():
        print(f"[error] dest DB not found: {args.dest}", file=sys.stderr)
        return 2

    dest_conn = sqlite3.connect(args.dest)
    try:
        if args.show_only:
            show_source_breakdown(dest_conn, f"dest ({args.dest.name})")
            for src in args.source:
                if not src.exists():
                    print(f"[warn] source not found: {src}", file=sys.stderr)
                    continue
                src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
                try:
                    show_source_breakdown(src_conn, f"src ({src.name})")
                finally:
                    src_conn.close()
            return 0

        show_source_breakdown(dest_conn, f"dest BEFORE merge ({args.dest.name})")
        total_j = total_e = 0
        for src in args.source:
            print(f"\n[merge] from {src}")
            n_j, n_e = merge_one_source(dest_conn, src, replace=args.replace)
            print(f"  judgments inserted: {n_j}, eval_run inserted: {n_e}")
            total_j += n_j
            total_e += n_e
        print(f"\n[total] judgments inserted: {total_j}, eval_run inserted: {total_e}")
        show_source_breakdown(dest_conn, f"dest AFTER merge ({args.dest.name})")
    finally:
        dest_conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

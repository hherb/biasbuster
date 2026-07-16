#!/usr/bin/env python3
"""
Compare evaluation databases: local (Claude/GPT-OSS/Sonnet) vs Spark (Gemma4).

Analyzes parse success rates, model agreement, and data quality without modifying.
"""

import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd


def connect_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def analyze_parse_status(db_path: str, name: str) -> None:
    """Analyze parse status distribution."""
    conn = connect_db(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT model_id, protocol, parse_status, COUNT(*) as count
        FROM evaluation_run
        WHERE model_id IS NOT NULL
        GROUP BY model_id, protocol, parse_status
        ORDER BY model_id, protocol, parse_status
    """)

    print(f"\n{'='*80}")
    print(f"{name} — Parse Status Distribution")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Protocol':<12} {'Status':<20} {'Count':>8}")
    print(f"{'-'*70}")

    for row in cursor.fetchall():
        print(
            f"{row['model_id']:<30} {row['protocol']:<12} "
            f"{row['parse_status']:<20} {row['count']:>8}"
        )

    # Summary
    cursor.execute("""
        SELECT parse_status, COUNT(*) as count
        FROM evaluation_run
        GROUP BY parse_status
    """)

    print(f"\n{'Overall Summary':^70}")
    print(f"{'-'*70}")
    for row in cursor.fetchall():
        total = sum(r["count"] for r in cursor.execute(
            "SELECT COUNT(*) as count FROM evaluation_run"
        ).fetchall())
        print(f"  {row['parse_status']:<20} {row['count']:>8} " +
              f"({100*row['count']/total:.1f}%)")

    conn.close()


def compare_model_agreement(local_db: str, spark_db: str) -> None:
    """Compare judgment agreement between models across both DBs."""
    print(f"\n{'='*80}")
    print("Model Agreement: Local vs Spark")
    print(f"{'='*80}")

    local = connect_db(local_db)
    spark = connect_db(spark_db)

    # Get judgments for a few RCTs to see agreement patterns
    local_cursor = local.cursor()
    spark_cursor = spark.cursor()

    # Sample comparison: first 20 RCTs, d1 domain, abstract protocol
    local_cursor.execute("""
        SELECT DISTINCT rct_id FROM benchmark_judgment
        WHERE domain = 'd1' LIMIT 20
    """)
    sample_rcts = [r["rct_id"] for r in local_cursor.fetchall()]

    print(f"\nSampling {len(sample_rcts)} RCTs — domain d1, abstract protocol:")
    print(f"\n{'RCT ID':<20} {'Claude2':<20} {'GPT-OSS20B':<20} {'Gemma4-26B':<20}")
    print(f"{'-'*80}")

    for rct_id in sample_rcts[:10]:  # Show first 10
        local_cursor.execute(
            "SELECT source, judgment FROM benchmark_judgment "
            "WHERE rct_id = ? AND domain = 'd1'",
            (rct_id,),
        )
        local_judgments = {r["source"].split("_")[0]: r["judgment"]
                          for r in local_cursor.fetchall()}

        spark_cursor.execute(
            "SELECT source, judgment FROM benchmark_judgment "
            "WHERE rct_id = ? AND domain = 'd1'",
            (rct_id,),
        )
        spark_judgments = {r["source"].split("_")[0]: r["judgment"]
                          for r in spark_cursor.fetchall()}

        c2 = local_judgments.get("em", "—")[:12]
        gpt = local_judgments.get("gpt", "—")[:12]
        gemma = spark_judgments.get("gemma4", "—")[:12]

        print(f"{rct_id:<20} {str(c2):<20} {str(gpt):<20} {str(gemma):<20}")

    local.close()
    spark.close()


def compare_run_counts(local_db: str, spark_db: str) -> None:
    """Compare total runs and completion rates."""
    print(f"\n{'='*80}")
    print("Run Counts & Completion Rates")
    print(f"{'='*80}")

    for db_path, label in [(local_db, "Local"), (spark_db, "Spark")]:
        conn = connect_db(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT model_id,
                   COUNT(*) as total,
                   SUM(CASE WHEN parse_status = 'ok' THEN 1 ELSE 0 END) as ok,
                   SUM(CASE WHEN parse_status IN ('ok', 'retry_succeeded') THEN 1
                            ELSE 0 END) as successful,
                   SUM(CASE WHEN parse_status = 'parse_failure' THEN 1 ELSE 0 END) as parse_fail,
                   SUM(CASE WHEN parse_status = 'api_error' THEN 1 ELSE 0 END) as api_err
            FROM evaluation_run
            GROUP BY model_id
            ORDER BY model_id
        """)

        print(f"\n{label} Database:")
        print(f"{'Model':<30} {'Total':>8} {'OK':>8} {'Succ.':>8} {'Parse Fail':>12} {'API Err':>10}")
        print(f"{'-'*80}")

        for row in cursor.fetchall():
            total = row["total"]
            ok_pct = 100 * row["ok"] / total if total else 0
            succ_pct = 100 * row["successful"] / total if total else 0
            print(
                f"{row['model_id']:<30} {row['total']:>8} "
                f"{row['ok']:>8} ({ok_pct:>5.1f}%) {row['successful']:>8} "
                f"{row['parse_fail']:>12} {row['api_err']:>10}"
            )

        conn.close()


def check_error_patterns(db_path: str, name: str) -> None:
    """Show error patterns."""
    print(f"\n{'='*80}")
    print(f"{name} — Error Patterns")
    print(f"{'='*80}")

    conn = connect_db(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT model_id, parse_status, error, COUNT(*) as count
        FROM evaluation_run
        WHERE error IS NOT NULL
        GROUP BY model_id, parse_status, error
        ORDER BY count DESC
        LIMIT 20
    """)

    rows = cursor.fetchall()
    if not rows:
        print("No errors found.")
    else:
        print(f"{'Model':<30} {'Status':<20} {'Error Summary':<50} {'Count':>5}")
        print(f"{'-'*105}")
        for row in rows:
            error_summary = (row["error"][:45] + "...") if row["error"] else "—"
            print(
                f"{row['model_id']:<30} {row['parse_status']:<20} "
                f"{error_summary:<50} {row['count']:>5}"
            )

    conn.close()


def main():
    local_db = "dataset/eisele_metzger_benchmark.db"
    spark_db = "dataset/eisele_metzger_benchmark.spark.db"

    # Verify both exist
    for db in [local_db, spark_db]:
        if not Path(db).exists():
            print(f"ERROR: {db} not found")
            return

    # Run analysis
    analyze_parse_status(local_db, "Local (Claude2, GPT-OSS-20B, Sonnet 4.6)")
    analyze_parse_status(spark_db, "Spark (Gemma4-26B, Qwen3.6-35B partial)")

    compare_run_counts(local_db, spark_db)
    check_error_patterns(local_db, "Local")
    check_error_patterns(spark_db, "Spark")

    compare_model_agreement(local_db, spark_db)

    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

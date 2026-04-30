"""Interim analysis after one model × one protocol × all three passes finish.

Purpose: at three checkpoints during Phase 5 (after each model's runs),
report enough numbers to decide whether to continue, fix, or stop. Not
a final analysis — that's Phase 6.

Per pre-reg §8 halt rules:
- parse-failure rate > 20% → halt and revise prompts.
- κ vs Cochrane < 0.10 (worse than EM Claude 2's 0.22) → halt and review.

What this script reports:
1. Coverage: completed calls per (source, parse_status).
2. Per-pass label distribution vs Cochrane.
3. Per-pass κ vs Cochrane (all three weightings, overall + per-domain).
4. Run-to-run pairwise κ across the 3 passes (the LLM-internal noise
   floor — directly comparable to Minozzi 2020 human Fleiss κ = 0.16).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/hherb/src/biasbuster")
sys.path.insert(0, str(PROJECT_ROOT / "studies/eisele_metzger_replication"))

from sanity_check_kappa import cohen_kappa, raw_agreement  # noqa: E402

DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"
DOMAINS = ("d1", "d2", "d3", "d4", "d5", "overall")


def load_pairs(conn: sqlite3.Connection, source_a: str, source_b: str,
               domain: str) -> list[tuple[str, str]]:
    cur = conn.cursor()
    return cur.execute(
        """SELECT a.judgment, b.judgment
           FROM benchmark_judgment a
           JOIN benchmark_judgment b
             ON a.rct_id = b.rct_id AND a.domain = b.domain
           WHERE a.source = ? AND b.source = ? AND a.domain = ?
             AND a.judgment IS NOT NULL AND b.judgment IS NOT NULL
             AND a.valid = 1 AND b.valid = 1""",
        (source_a, source_b, domain),
    ).fetchall()


def coverage_table(conn: sqlite3.Connection, model_short: str, protocol: str) -> None:
    cur = conn.cursor()
    print(f"\n## Coverage and parse status — {model_short} × {protocol}\n")
    print(f"{'source':<40} {'ok':>4} {'retry':>5} {'parse_fail':>10} {'api_err':>7} {'in_flight':>9}")
    for pass_n in (1, 2, 3):
        source = f"{model_short}_{protocol}_pass{pass_n}"
        rows = cur.execute(
            """SELECT parse_status, COUNT(*)
               FROM evaluation_run WHERE source = ?
               GROUP BY parse_status""",
            (source,),
        ).fetchall()
        d = dict(rows)
        ok = d.get("ok", 0)
        retry = d.get("retry_succeeded", 0)
        fail = d.get("parse_failure", 0)
        api = d.get("api_error", 0)
        flight = d.get("in_flight", 0)
        print(f"{source:<40} {ok:>4} {retry:>5} {fail:>10} {api:>7} {flight:>9}")
        n_attempted = ok + retry + fail + api
        if n_attempted >= 30:
            failure_rate = (fail + api) / n_attempted
            tag = " ⚠️ HALT THRESHOLD EXCEEDED" if failure_rate > 0.20 else ""
            print(f"{'  failure_rate:':<40} {failure_rate:>5.1%}{tag}")


def label_distribution(conn: sqlite3.Connection, model_short: str,
                       protocol: str, domain: str) -> None:
    cur = conn.cursor()
    print(f"\n## Label distribution — {domain} (Cochrane vs each pass)\n")
    print(f"{'source':<40} {'low':>5} {'some_concerns':>14} {'high':>5} {'n':>4}")
    for src in ["cochrane",
                f"{model_short}_{protocol}_pass1",
                f"{model_short}_{protocol}_pass2",
                f"{model_short}_{protocol}_pass3"]:
        counts = dict(cur.execute(
            """SELECT judgment, COUNT(*) FROM benchmark_judgment
               WHERE source = ? AND domain = ? AND valid = 1
               GROUP BY judgment""",
            (src, domain),
        ).fetchall())
        n = sum(counts.values())
        print(f"{src:<40} {counts.get('low', 0):>5} "
              f"{counts.get('some_concerns', 0):>14} {counts.get('high', 0):>5} {n:>4}")


def kappa_vs_cochrane(conn: sqlite3.Connection, model_short: str,
                      protocol: str) -> None:
    print(f"\n## κ vs Cochrane — {model_short} × {protocol}\n")
    print(f"{'source':<35} {'domain':<10} {'n':>4} {'rawAgr':>7} "
          f"{'κ_unw':>7} {'κ_lin':>7} {'κ_quad':>7}")
    for pass_n in (1, 2, 3):
        src = f"{model_short}_{protocol}_pass{pass_n}"
        for domain in DOMAINS:
            pairs = load_pairs(conn, "cochrane", src, domain)
            if not pairs:
                continue
            print(
                f"{src:<35} {domain:<10} {len(pairs):>4} "
                f"{raw_agreement(pairs):>7.3f} "
                f"{cohen_kappa(pairs, 'none'):>7.3f} "
                f"{cohen_kappa(pairs, 'linear'):>7.3f} "
                f"{cohen_kappa(pairs, 'quadratic'):>7.3f}"
            )


def run_to_run_kappa(conn: sqlite3.Connection, model_short: str,
                     protocol: str) -> None:
    print(f"\n## Run-to-run κ (LLM-internal noise) — {model_short} × {protocol}\n")
    print("Comparable to Minozzi 2020 human Fleiss κ = 0.16 (untrained reviewers)")
    print("and Minozzi 2021 human κ = 0.42 (with implementation document).\n")
    print(f"{'comparison':<45} {'domain':<10} {'n':>4} {'rawAgr':>7} "
          f"{'κ_unw':>7} {'κ_lin':>7} {'κ_quad':>7}")
    pairs_to_test = [
        (1, 2), (1, 3), (2, 3),
    ]
    for p_a, p_b in pairs_to_test:
        src_a = f"{model_short}_{protocol}_pass{p_a}"
        src_b = f"{model_short}_{protocol}_pass{p_b}"
        for domain in ("overall",):  # keep concise — overall is the headline
            pairs = load_pairs(conn, src_a, src_b, domain)
            if not pairs:
                continue
            label = f"pass{p_a} vs pass{p_b}"
            print(
                f"{label:<45} {domain:<10} {len(pairs):>4} "
                f"{raw_agreement(pairs):>7.3f} "
                f"{cohen_kappa(pairs, 'none'):>7.3f} "
                f"{cohen_kappa(pairs, 'linear'):>7.3f} "
                f"{cohen_kappa(pairs, 'quadratic'):>7.3f}"
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--protocol", required=True, choices=("abstract", "fulltext"))
    args = p.parse_args()

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        coverage_table(conn, args.model, args.protocol)
        label_distribution(conn, args.model, args.protocol, "overall")
        kappa_vs_cochrane(conn, args.model, args.protocol)
        run_to_run_kappa(conn, args.model, args.protocol)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

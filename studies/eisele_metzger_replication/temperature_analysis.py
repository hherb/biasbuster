"""Post-hoc analysis of the temperature sensitivity sweep.

Reads benchmark_judgment rows produced by ``temperature_sweep.py`` —
identifiable by the temperature-tagged source labels
(``{model}_T{X}_{protocol}_pass{N}``) — and computes pairwise
run-to-run κ at each temperature for direct comparison.

The headline question the analysis answers:

    "How much of the run-to-run κ reported in the main study is
     stochastic-decoding floor at default temperature versus genuine
     intra-model stability of judgement?"

Reads from any DB (canonical, Spark shard, or merged). No writes.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "studies/eisele_metzger_replication"))

from sanity_check_kappa import cohen_kappa, raw_agreement  # noqa: E402

DEFAULT_DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"
SOURCE_RE = re.compile(
    r"^(?P<model>[a-z0-9_]+?)_T(?P<temp>\d+(?:p\d+)?)_(?P<protocol>abstract|fulltext)_pass(?P<pass>\d+)$"
)


def decode_temperature(temp_str: str) -> float:
    """Inverse of temperature_sweep.temperature_label.

    >>> decode_temperature("0")
    0.0
    >>> decode_temperature("0p3")
    0.3
    >>> decode_temperature("1p2")
    1.2
    """
    return float(temp_str.replace("p", "."))


def find_temperature_sources(conn: sqlite3.Connection, model: str,
                              protocol: str) -> dict[float, dict[int, str]]:
    """Return {temperature: {pass_n: source_label}} for the given model × protocol."""
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT DISTINCT source FROM benchmark_judgment "
        "WHERE source LIKE ? ORDER BY source",
        (f"{model}_T%_{protocol}_pass%",),
    ).fetchall()
    out: dict[float, dict[int, str]] = defaultdict(dict)
    for (src,) in rows:
        m = SOURCE_RE.match(src)
        if not m or m.group("model") != model or m.group("protocol") != protocol:
            continue
        temp = decode_temperature(m.group("temp"))
        pass_n = int(m.group("pass"))
        out[temp][pass_n] = src
    return dict(out)


def load_pairs(conn: sqlite3.Connection, source_a: str, source_b: str,
               domain: str = "overall") -> list[tuple[str, str]]:
    """Joined (judgement_a, judgement_b) for matched (rct_id, domain) rows."""
    return conn.execute(
        """SELECT a.judgment, b.judgment
           FROM benchmark_judgment a
           JOIN benchmark_judgment b
             ON a.rct_id = b.rct_id AND a.domain = b.domain
           WHERE a.source = ? AND b.source = ? AND a.domain = ?
             AND a.judgment IS NOT NULL AND b.judgment IS NOT NULL
             AND a.valid = 1 AND b.valid = 1""",
        (source_a, source_b, domain),
    ).fetchall()


def run_to_run_for_temperature(conn: sqlite3.Connection,
                                temp_sources: dict[int, str]
                                ) -> dict[str, float | int]:
    """Mean pairwise κ across the available passes for one temperature."""
    pass_nums = sorted(temp_sources)
    n_passes = len(pass_nums)
    if n_passes < 2:
        return {"n_passes": n_passes, "n_pairs": 0,
                "raw_agreement": 0.0,
                "kappa_unweighted": 0.0, "kappa_linear": 0.0, "kappa_quadratic": 0.0}

    raw_agrs = []
    k_unw = []
    k_lin = []
    k_quad = []
    n_pairs_list = []
    for i in range(n_passes):
        for j in range(i + 1, n_passes):
            pairs = load_pairs(conn, temp_sources[pass_nums[i]],
                               temp_sources[pass_nums[j]])
            if not pairs:
                continue
            raw_agrs.append(raw_agreement(pairs))
            k_unw.append(cohen_kappa(pairs, "none"))
            k_lin.append(cohen_kappa(pairs, "linear"))
            k_quad.append(cohen_kappa(pairs, "quadratic"))
            n_pairs_list.append(len(pairs))

    if not raw_agrs:
        return {"n_passes": n_passes, "n_pairs": 0,
                "raw_agreement": 0.0,
                "kappa_unweighted": 0.0, "kappa_linear": 0.0, "kappa_quadratic": 0.0}

    n_comparisons = len(raw_agrs)
    return {
        "n_passes": n_passes,
        "n_comparisons": n_comparisons,
        "n_pairs_mean": sum(n_pairs_list) // n_comparisons,
        "raw_agreement": sum(raw_agrs) / n_comparisons,
        "kappa_unweighted": sum(k_unw) / n_comparisons,
        "kappa_linear": sum(k_lin) / n_comparisons,
        "kappa_quadratic": sum(k_quad) / n_comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--model", default="gpt_oss_20b",
                        help="Model short label (default: gpt_oss_20b).")
    parser.add_argument("--protocol", default="fulltext",
                        choices=("abstract", "fulltext"))
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"[error] DB not found at {args.db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(f"file:{args.db_path}?mode=ro", uri=True)
    try:
        temps = find_temperature_sources(conn, args.model, args.protocol)
        if not temps:
            print(f"[empty] no temperature-tagged sources found for "
                  f"{args.model} × {args.protocol} in {args.db_path}.\n"
                  f"        run temperature_sweep.py first.")
            return 1

        print(f"\n## Temperature-sensitivity sweep — {args.model} × {args.protocol}\n")
        print(f"{'temperature':>11} {'n_passes':>9} {'n_pairs':>8} "
              f"{'rawAgr':>7} {'κ_unw':>7} {'κ_lin':>7} {'κ_quad':>7}")
        print("-" * 65)
        for temp in sorted(temps):
            stats = run_to_run_for_temperature(conn, temps[temp])
            n_pairs = stats.get("n_pairs_mean", 0) or stats.get("n_pairs", 0)
            print(f"{temp:>11.1f} "
                  f"{stats['n_passes']:>9d} "
                  f"{n_pairs:>8d} "
                  f"{stats['raw_agreement']:>7.3f} "
                  f"{stats['kappa_unweighted']:>7.3f} "
                  f"{stats['kappa_linear']:>7.3f} "
                  f"{stats['kappa_quadratic']:>7.3f}")

        # Reference markers from the main study (gpt-oss × fulltext)
        # and the human-vs-human band.
        print()
        print("Reference markers:")
        print("  Main study  (model defaults T=0.8, n=91)  run-to-run κ_quad ≈ 0.44 (gpt-oss × fulltext)")
        print("  Minozzi 2021 (trained humans + ID, Fleiss κ): 0.42")
        print("  Minozzi 2020 (trained humans, no ID, Fleiss κ): 0.16")
        print()
        print("Interpretation:")
        print("  • If κ at T=0 is dramatically higher than at T=0.8 (main study),")
        print("    sampling determinism is contributing materially to run-to-run κ.")
        print("  • If κ at T=0 is similar to T=0.8, the harness's structural")
        print("    constraint (5-category signalling answers + algorithmic")
        print("    worst-wins synthesis) absorbs the temperature contribution.")
        print("  • If κ at T=1.2 is dramatically lower, the harness's stability")
        print("    has an upper temperature limit; useful for prompt-engineering")
        print("    discussions but doesn't undermine the main result at T=0.8.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

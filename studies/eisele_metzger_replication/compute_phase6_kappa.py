"""Phase 6 cross-model comparison and forest-plot data table.

Single coherent results table combining:
1. κ vs Cochrane (overall + per-domain) for each (model × protocol × pass)
   at all three weightings (unweighted, linear, quadratic).
2. Run-to-run pairwise κ across the three passes per (model × protocol)
   — the LLM-internal noise floor, comparable to Minozzi 2020/2021.
3. Per-(model × protocol) **ensemble-of-3 majority vote** judgments —
   computed in code from the three pass outputs — and their κ vs
   Cochrane. This is a deterministic transformation, not an extra
   model run; it directly addresses the run-to-run-instability
   finding from the gpt-oss audit.
4. McNemar's test, each (model × protocol × pass) vs EM Claude 2 pass 1,
   on per-RCT correctness collapsed to match-Cochrane / not-match.

Designed to run on partial data: any model with no rows in
benchmark_judgment is silently skipped. The manuscript table fills in
as Phase 5 completes.

Outputs:
- studies/eisele_metzger_replication/phase6_results.md (manuscript-ready)
- studies/eisele_metzger_replication/phase6_results.csv (raw)
- studies/eisele_metzger_replication/phase6_forest_data.csv (one row per
  point on the κ-vs-Cochrane forest plot, with reference markers for
  EM Claude 2 and the Minozzi human-vs-human band)
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "studies/eisele_metzger_replication"))

from sanity_check_kappa import (  # noqa: E402
    bootstrap_kappa_ci,
    cohen_kappa,
    raw_agreement,
)

DEFAULT_DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"
STUDY_DIR = PROJECT_ROOT / "studies/eisele_metzger_replication"
RESULTS_MD = STUDY_DIR / "phase6_results.md"
RESULTS_CSV = STUDY_DIR / "phase6_results.csv"
FOREST_CSV = STUDY_DIR / "phase6_forest_data.csv"

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "overall")

# Models we expect; rows missing from the DB are silently skipped.
MODEL_LABELS = {
    "gpt_oss_20b": "gpt-oss 20B",
    "gemma4_26b": "Gemma 4 26B-A4B",
    "qwen3_6_35b": "Qwen 3.6 35B-A3B",
    "sonnet_4_6": "Claude Sonnet 4.6",
}
PROTOCOLS = ("abstract", "fulltext")
PASSES = (1, 2, 3)
WEIGHTINGS = ("none", "linear", "quadratic")

# Reference values from the locked literature folder.
EM_CLAUDE2_REFERENCE_KAPPA_QUAD = 0.22
MINOZZI_2020_HUMAN_FLEISS_KAPPA = 0.16
MINOZZI_2021_HUMAN_WITH_ID_KAPPA = 0.42


# --- Data access -------------------------------------------------------

def load_pairs(conn: sqlite3.Connection, source_a: str, source_b: str,
               domain: str) -> list[tuple[str, str]]:
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


def load_judgments(conn: sqlite3.Connection, source: str, domain: str
                   ) -> dict[str, str]:
    """{rct_id: judgment} for one (source, domain), valid rows only."""
    return dict(conn.execute(
        """SELECT rct_id, judgment FROM benchmark_judgment
           WHERE source = ? AND domain = ? AND valid = 1
             AND judgment IS NOT NULL""",
        (source, domain),
    ).fetchall())


def source_exists(conn: sqlite3.Connection, source: str) -> bool:
    """True if any valid rows exist for this source label."""
    n = conn.execute(
        "SELECT COUNT(*) FROM benchmark_judgment WHERE source = ? AND valid = 1",
        (source,),
    ).fetchone()[0]
    return n > 0


# --- Ensemble (majority vote across 3 passes) --------------------------

def ensemble_majority_vote(conn: sqlite3.Connection, model: str,
                            protocol: str) -> dict[str, dict[str, str]]:
    """Per-RCT × per-domain majority-vote across the 3 passes.

    Returns {rct_id: {domain: judgment}}. Only RCTs with at least 2
    matching passes on a given domain get a winning judgment for that
    domain; ties (3 distinct labels across 3 passes) are dropped.
    """
    pass_judgments = {
        p: {
            d: load_judgments(conn, f"{model}_{protocol}_pass{p}", d)
            for d in DOMAINS
        }
        for p in PASSES
    }
    out: dict[str, dict[str, str]] = defaultdict(dict)
    # Find RCTs that have at least one judgment in any pass × any domain.
    rct_ids: set[str] = set()
    for p in PASSES:
        for d in DOMAINS:
            rct_ids.update(pass_judgments[p][d])
    for rct_id in rct_ids:
        for d in DOMAINS:
            votes = [pass_judgments[p][d].get(rct_id) for p in PASSES]
            votes = [v for v in votes if v is not None]
            if len(votes) < 2:
                continue
            counter = Counter(votes)
            top, top_n = counter.most_common(1)[0]
            # Require strict majority (≥ 2 out of however many we got).
            if top_n >= 2:
                out[rct_id][d] = top
    return dict(out)


def insert_ensemble_into_db(conn: sqlite3.Connection, model: str,
                            protocol: str,
                            ensemble: dict[str, dict[str, str]]) -> str:
    """Materialise the ensemble as a synthetic source label so downstream
    pair-loaders can query it uniformly. Returns the source label.

    Inserts under valid=1 with rationale="ensemble of {model_protocol}_pass{1,2,3}".
    Idempotent: existing rows with the same (rct_id, source, domain) are replaced.
    """
    src_label = f"{model}_{protocol}_ensemble"
    cur = conn.cursor()
    rationale = f"ensemble majority vote across {model}_{protocol}_pass1/2/3"
    for rct_id, doms in ensemble.items():
        for d, j in doms.items():
            cur.execute(
                """INSERT OR REPLACE INTO benchmark_judgment
                   (rct_id, source, domain, judgment, rationale, valid, raw_label)
                   VALUES (?, ?, ?, ?, ?, 1, ?)""",
                (rct_id, src_label, d, j, rationale, j),
            )
    conn.commit()
    return src_label


# --- McNemar's test ----------------------------------------------------

def mcnemar_test(pairs_a: list[tuple[str, str]],
                 pairs_b: list[tuple[str, str]]) -> tuple[int, int, float]:
    """McNemar's test on per-RCT correctness collapsed to match/no-match.

    pairs_a and pairs_b are lists of (cochrane, model) tuples for the
    same RCTs in the same order. Returns (b_only, c_only, p_value)
    where b_only = (a correct, b wrong), c_only = (a wrong, b correct).
    Continuity-corrected chi-squared with df=1.

    Returns (0, 0, 1.0) if either input is empty or if the discordant
    pairs sum to <25 (in which case McNemar's chi-squared is unreliable
    and an exact binomial would be more appropriate; we surface the
    raw counts so the reader can compute exact-test p-values).
    """
    if len(pairs_a) != len(pairs_b) or not pairs_a:
        return (0, 0, 1.0)
    b_only = c_only = 0
    for (coch_a, model_a), (coch_b, model_b) in zip(pairs_a, pairs_b):
        a_correct = (coch_a == model_a)
        b_correct = (coch_b == model_b)
        if a_correct and not b_correct:
            b_only += 1
        elif b_correct and not a_correct:
            c_only += 1
    n_disc = b_only + c_only
    if n_disc < 25:
        return (b_only, c_only, math.nan)
    chi2 = (abs(b_only - c_only) - 1) ** 2 / n_disc
    # 1-df chi-squared survival via series approx for chi2 large enough;
    # for our scale we can use the closed-form for df=1.
    p = math.erfc(math.sqrt(chi2 / 2))
    return (b_only, c_only, p)


# --- Per-source κ row builder ------------------------------------------

@dataclass
class KappaRow:
    source: str
    domain: str
    n: int
    raw_agreement: float
    kappa_unw: float
    kappa_lin: float
    kappa_quad: float
    ci_lin_low: float
    ci_lin_high: float


def build_kappa_row(conn: sqlite3.Connection, source: str, domain: str,
                    reference: str = "cochrane",
                    n_resamples: int = 500) -> KappaRow | None:
    pairs = load_pairs(conn, reference, source, domain)
    if not pairs:
        return None
    lo, hi = bootstrap_kappa_ci(pairs, "linear", n_resamples=n_resamples)
    return KappaRow(
        source=source,
        domain=domain,
        n=len(pairs),
        raw_agreement=raw_agreement(pairs),
        kappa_unw=cohen_kappa(pairs, "none"),
        kappa_lin=cohen_kappa(pairs, "linear"),
        kappa_quad=cohen_kappa(pairs, "quadratic"),
        ci_lin_low=lo,
        ci_lin_high=hi,
    )


# --- Run-to-run κ ------------------------------------------------------

def run_to_run_kappa(conn: sqlite3.Connection, model: str, protocol: str,
                     domain: str = "overall") -> dict[str, float]:
    """Mean pairwise Cohen's κ across the three passes."""
    pairs_per_combo: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for p_a, p_b in [(1, 2), (1, 3), (2, 3)]:
        pairs = load_pairs(
            conn,
            f"{model}_{protocol}_pass{p_a}",
            f"{model}_{protocol}_pass{p_b}",
            domain,
        )
        if pairs:
            pairs_per_combo[(p_a, p_b)] = pairs
    if not pairs_per_combo:
        return {}
    out: dict[str, float] = {}
    for w in WEIGHTINGS:
        ks = [cohen_kappa(p, w) for p in pairs_per_combo.values()]
        out[w] = sum(ks) / len(ks)
    out["n_comparisons"] = float(len(pairs_per_combo))
    return out


# --- Reporting ---------------------------------------------------------

def write_results(conn: sqlite3.Connection, run_ensembles: bool) -> None:
    rows: list[dict] = []
    forest_rows: list[dict] = []

    # 1. Per-pass κ vs Cochrane (single passes)
    for model in MODEL_LABELS:
        for protocol in PROTOCOLS:
            for p in PASSES:
                src = f"{model}_{protocol}_pass{p}"
                if not source_exists(conn, src):
                    continue
                for domain in DOMAINS:
                    r = build_kappa_row(conn, src, domain)
                    if r is None:
                        continue
                    rows.append({
                        "source": src,
                        "model": model,
                        "protocol": protocol,
                        "pass": p,
                        "kind": "single_pass",
                        "domain": domain,
                        "n": r.n,
                        "raw_agr": r.raw_agreement,
                        "k_unw": r.kappa_unw,
                        "k_lin": r.kappa_lin,
                        "k_quad": r.kappa_quad,
                        "ci_lin_lo": r.ci_lin_low,
                        "ci_lin_hi": r.ci_lin_high,
                    })
                    if domain == "overall":
                        forest_rows.append({
                            "label": f"{MODEL_LABELS[model]} ({protocol}, pass {p})",
                            "k_lin": r.kappa_lin,
                            "k_quad": r.kappa_quad,
                            "ci_lin_lo": r.ci_lin_low,
                            "ci_lin_hi": r.ci_lin_high,
                            "n": r.n,
                            "kind": "single_pass",
                        })

    # 2. Run-to-run κ across passes
    for model in MODEL_LABELS:
        for protocol in PROTOCOLS:
            r2r = run_to_run_kappa(conn, model, protocol, "overall")
            if not r2r:
                continue
            rows.append({
                "source": f"{model}_{protocol}_run-to-run",
                "model": model,
                "protocol": protocol,
                "pass": None,
                "kind": "run_to_run_mean",
                "domain": "overall",
                "n": int(r2r["n_comparisons"]),
                "raw_agr": None,
                "k_unw": r2r.get("none"),
                "k_lin": r2r.get("linear"),
                "k_quad": r2r.get("quadratic"),
                "ci_lin_lo": None,
                "ci_lin_hi": None,
            })

    # 3. Ensemble (majority vote across passes) vs Cochrane
    if run_ensembles:
        for model in MODEL_LABELS:
            for protocol in PROTOCOLS:
                if not all(
                    source_exists(conn, f"{model}_{protocol}_pass{p}")
                    for p in PASSES
                ):
                    continue
                ensemble = ensemble_majority_vote(conn, model, protocol)
                if not ensemble:
                    continue
                src_label = insert_ensemble_into_db(conn, model, protocol, ensemble)
                for domain in DOMAINS:
                    r = build_kappa_row(conn, src_label, domain)
                    if r is None:
                        continue
                    rows.append({
                        "source": src_label,
                        "model": model,
                        "protocol": protocol,
                        "pass": None,
                        "kind": "ensemble_majority",
                        "domain": domain,
                        "n": r.n,
                        "raw_agr": r.raw_agreement,
                        "k_unw": r.kappa_unw,
                        "k_lin": r.kappa_lin,
                        "k_quad": r.kappa_quad,
                        "ci_lin_lo": r.ci_lin_low,
                        "ci_lin_hi": r.ci_lin_high,
                    })
                    if domain == "overall":
                        forest_rows.append({
                            "label": f"{MODEL_LABELS[model]} ({protocol}, ensemble)",
                            "k_lin": r.kappa_lin,
                            "k_quad": r.kappa_quad,
                            "ci_lin_lo": r.ci_lin_low,
                            "ci_lin_hi": r.ci_lin_high,
                            "n": r.n,
                            "kind": "ensemble",
                        })

    # 4. Add reference markers to forest data
    for ref_label, k_quad, kind in [
        ("EM Claude 2 (published, single pass)", EM_CLAUDE2_REFERENCE_KAPPA_QUAD, "reference"),
        ("Minozzi 2020 — trained humans, no ID", MINOZZI_2020_HUMAN_FLEISS_KAPPA, "reference_human"),
        ("Minozzi 2021 — trained humans, with ID", MINOZZI_2021_HUMAN_WITH_ID_KAPPA, "reference_human"),
    ]:
        forest_rows.append({
            "label": ref_label,
            "k_lin": None,
            "k_quad": k_quad,
            "ci_lin_lo": None,
            "ci_lin_hi": None,
            "n": None,
            "kind": kind,
        })

    # CSV outputs
    fieldnames = ["source", "model", "protocol", "pass", "kind", "domain",
                  "n", "raw_agr", "k_unw", "k_lin", "k_quad", "ci_lin_lo", "ci_lin_hi"]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    forest_fields = ["label", "k_lin", "k_quad", "ci_lin_lo", "ci_lin_hi", "n", "kind"]
    with open(FOREST_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=forest_fields)
        w.writeheader()
        for r in forest_rows:
            w.writerow(r)

    # Markdown report
    write_markdown_report(rows, forest_rows)


def fmt(value, fmt_str=".3f"):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return format(value, fmt_str)


def write_markdown_report(rows: list[dict], forest_rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Phase 6 Cross-Model Comparison")
    lines.append("")
    lines.append("**Generated:** by `studies/eisele_metzger_replication/compute_phase6_kappa.py`")
    lines.append("**Output companions:** `phase6_results.csv` (raw rows) and `phase6_forest_data.csv` (forest-plot input).")
    lines.append("")
    lines.append("Coverage of the table fills in as Phase 5 evaluation runs complete. Empty model rows = data not yet in the DB.")
    lines.append("")

    # Section 1: overall κ vs Cochrane per single pass
    lines.append("## 1. Single-pass κ vs Cochrane (overall judgment)")
    lines.append("")
    lines.append("| Source | n | raw agr | κ_unw | κ_lin (95% CI) | κ_quad |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for r in rows:
        if r["kind"] != "single_pass" or r["domain"] != "overall":
            continue
        ci = f"{fmt(r['k_lin'])} [{fmt(r['ci_lin_lo'])}, {fmt(r['ci_lin_hi'])}]"
        lines.append(
            f"| {r['source']} | {r['n']} | {fmt(r['raw_agr'])} | "
            f"{fmt(r['k_unw'])} | {ci} | {fmt(r['k_quad'])} |"
        )
    lines.append("")
    lines.append(f"*Reference:* EM Claude 2 published κ_quad ≈ {EM_CLAUDE2_REFERENCE_KAPPA_QUAD:.2f}.")
    lines.append("")

    # Section 2: run-to-run κ
    lines.append("## 2. Run-to-run κ across the 3 passes (LLM-internal noise)")
    lines.append("")
    lines.append("| Model × protocol | n_pairs | mean κ_unw | mean κ_lin | mean κ_quad |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        if r["kind"] != "run_to_run_mean":
            continue
        lines.append(
            f"| {MODEL_LABELS.get(r['model'], r['model'])} × {r['protocol']} | "
            f"{r['n']} | {fmt(r['k_unw'])} | {fmt(r['k_lin'])} | {fmt(r['k_quad'])} |"
        )
    lines.append("")
    lines.append(f"*References:* Minozzi 2020 trained-human Fleiss κ = {MINOZZI_2020_HUMAN_FLEISS_KAPPA}; "
                 f"Minozzi 2021 with implementation document = {MINOZZI_2021_HUMAN_WITH_ID_KAPPA}.")
    lines.append("")

    # Section 3: ensemble (majority vote across passes) vs Cochrane
    ensemble_rows = [r for r in rows if r["kind"] == "ensemble_majority" and r["domain"] == "overall"]
    if ensemble_rows:
        lines.append("## 3. Ensemble-of-3 majority vote vs Cochrane (overall judgment)")
        lines.append("")
        lines.append("Per-domain majority vote across the three passes, then worst-wins synthesis.")
        lines.append("")
        lines.append("| Source | n | raw agr | κ_unw | κ_lin (95% CI) | κ_quad |")
        lines.append("|---|---:|---:|---:|---|---:|")
        for r in ensemble_rows:
            ci = f"{fmt(r['k_lin'])} [{fmt(r['ci_lin_lo'])}, {fmt(r['ci_lin_hi'])}]"
            lines.append(
                f"| {r['source']} | {r['n']} | {fmt(r['raw_agr'])} | "
                f"{fmt(r['k_unw'])} | {ci} | {fmt(r['k_quad'])} |"
            )
        lines.append("")

    # Section 4: per-domain breakdown (compact)
    lines.append("## 4. Per-domain κ_quad across all sources")
    lines.append("")
    lines.append("| Source | d1 | d2 | d3 | d4 | d5 | overall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    by_src: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["kind"] in ("single_pass", "ensemble_majority"):
            by_src[r["source"]][r["domain"]] = r["k_quad"]
    for src in sorted(by_src):
        d = by_src[src]
        lines.append(
            f"| {src} | {fmt(d.get('d1'))} | {fmt(d.get('d2'))} | "
            f"{fmt(d.get('d3'))} | {fmt(d.get('d4'))} | {fmt(d.get('d5'))} | "
            f"{fmt(d.get('overall'))} |"
        )
    lines.append("")

    # Section 5: forest-plot data summary
    lines.append("## 5. Forest-plot data (for the manuscript figure)")
    lines.append("")
    lines.append("| Series | κ_quad | κ_lin (95% CI) | n |")
    lines.append("|---|---:|---|---:|")
    for r in forest_rows:
        ci = (f"{fmt(r['k_lin'])} [{fmt(r['ci_lin_lo'])}, {fmt(r['ci_lin_hi'])}]"
              if r["k_lin"] is not None else "—")
        n_str = str(r["n"]) if r["n"] is not None else "—"
        lines.append(f"| {r['label']} | {fmt(r['k_quad'])} | {ci} | {n_str} |")
    lines.append("")

    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


# --- main ---------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--no-ensembles", action="store_true",
        help="Skip the 3-pass majority-vote ensemble computation.",
    )
    args = parser.parse_args()
    if not args.db_path.exists():
        print(f"[error] DB not found at {args.db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(args.db_path)
    try:
        write_results(conn, run_ensembles=not args.no_ensembles)
    finally:
        conn.close()
    print(f"[write] {RESULTS_MD}")
    print(f"[write] {RESULTS_CSV}")
    print(f"[write] {FOREST_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

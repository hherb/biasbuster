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
A ``mcnemar_test`` helper (per-RCT correctness collapsed to
match-Cochrane / not-match) is provided for pairwise significance testing,
but is not emitted in the report: EM Claude 2's per-RCT labels are not in
the DB (only its published aggregate κ ≈ 0.22 is), so there is nothing to
pair against here.

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
from exclusions import (  # noqa: E402
    RECOVERABLE_WRONG_PAPER_RCTS,
    RECOVERY_NOTE_MARKER,
    UNRECOVERABLE_WRONG_PAPER_RCTS,
    WRONG_PAPER_RCTS,
    wrong_paper_filter,
)

DEFAULT_DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"
STUDY_DIR = PROJECT_ROOT / "studies/eisele_metzger_replication"
RESULTS_MD = STUDY_DIR / "phase6_results.md"
RESULTS_CSV = STUDY_DIR / "phase6_results.csv"
FOREST_CSV = STUDY_DIR / "phase6_forest_data.csv"

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "overall")
SIGNALLING_DOMAINS = ("d1", "d2", "d3", "d4", "d5")

# RoB 2 three-level severity ordering. `overall` is the worst domain
# ("high" if any domain is high, "low" only if all are low, else
# "some_concerns") — the same worst-wins rule the per-paper algorithm uses.
SEVERITY_RANK = {"low": 0, "some_concerns": 1, "high": 2}
_RANK_TO_LABEL = {v: k for k, v in SEVERITY_RANK.items()}

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

# When True, algorithm-derived judgements (raw_label='FALLBACK', written by the
# live-path fallback in eval_ollama.parse_response and by recover_parse_failures.py)
# are excluded from every κ computation. That reproduces the pre-registered
# "model-emitted judgement" primary metric; the default (False) reports the
# inclusive numbers. Toggled by the --exclude-fallback CLI flag. Rows with a
# NULL raw_label (e.g. Cochrane ground truth) are always kept.
EXCLUDE_FALLBACK = False

# When True, run the recovered-corpus **sensitivity** analysis: exclude only the
# two unindexed wrong papers (UNRECOVERABLE_WRONG_PAPER_RCTS) instead of all 13
# (WRONG_PAPER_RCTS), so the ~11 recovered-and-re-assessed RCTs re-enter the κ.
# Toggled by --sensitivity, which also guards that those RCTs really were
# recovered (see sensitivity_precondition_failures) — running it against a
# non-recovered DB would count stale wrong-document judgements. Default (False)
# is the pre-registered primary (exclude all 13).
SENSITIVITY_MODE = False


def _active_wrong_paper_set() -> frozenset[str]:
    """The RCTs excluded from κ under the current mode.

    Sensitivity mode drops only the unindexed (unrecoverable) wrong papers; the
    primary drops the full wrong-paper class. Read at query time so the CLI can
    flip ``SENSITIVITY_MODE`` before the loaders run.
    """
    return UNRECOVERABLE_WRONG_PAPER_RCTS if SENSITIVITY_MODE else WRONG_PAPER_RCTS


def recovered_wrong_paper_rcts(conn: sqlite3.Connection) -> frozenset[str]:
    """RCT ids whose ``benchmark_rct.notes`` record a wrong-paper recovery.

    ``recover_wrong_papers.apply_recovery`` appends ``RECOVERY_NOTE_MARKER`` to
    the notes of every RCT it re-fetches. Detecting the marker tells the
    sensitivity guard which recoverable wrong papers have actually been
    corrected in this DB (rows with NULL notes never match).
    """
    rows = conn.execute(
        "SELECT rct_id FROM benchmark_rct WHERE notes LIKE ?",
        (f"%{RECOVERY_NOTE_MARKER}%",),
    ).fetchall()
    return frozenset(r[0] for r in rows)


def sensitivity_precondition_failures(
        conn: sqlite3.Connection) -> frozenset[str]:
    """Recoverable wrong papers NOT yet recovered in this DB.

    The sensitivity κ re-includes ``RECOVERABLE_WRONG_PAPER_RCTS``; that is only
    valid once each has been re-fetched to the correct document and re-assessed
    (otherwise its stale wrong-document judgements would be counted). Returns
    the recoverable RCTs missing the recovery marker — empty when the DB is
    ready. The CLI refuses ``--sensitivity`` when this is non-empty.
    """
    return RECOVERABLE_WRONG_PAPER_RCTS - recovered_wrong_paper_rcts(conn)


def _fallback_filter(*aliases: str) -> str:
    """SQL fragment excluding FALLBACK-tagged rows, or '' when disabled.

    Pass the table alias used for each ``benchmark_judgment`` reference in the
    query (e.g. ``"a"``, ``"b"``); pass ``""`` for an unaliased table.
    """
    if not EXCLUDE_FALLBACK:
        return ""
    parts = []
    for alias in aliases:
        col = f"{alias}.raw_label" if alias else "raw_label"
        parts.append(f"({col} IS NULL OR {col} != 'FALLBACK')")
    return " AND " + " AND ".join(parts)


def load_pairs(conn: sqlite3.Connection, source_a: str, source_b: str,
               domain: str) -> list[tuple[str, str]]:
    """Paired (reference, model) judgments for one domain, ordered by rct_id.

    The ORDER BY is load-bearing, not cosmetic: ``bootstrap_kappa_ci``
    resamples this list by index, so an unspecified SQLite row order makes
    the resulting confidence intervals irreproducible between runs. The
    ensemble sources are rewritten via INSERT OR REPLACE on every run,
    which changes their physical placement and previously caused exactly
    that drift. Cohen's kappa itself is order-invariant, so this affects
    CIs only — never a point estimate.
    """
    wp_sql, wp_params = wrong_paper_filter(
        "a", "b", exclusion_set=_active_wrong_paper_set())
    return conn.execute(
        """SELECT a.judgment, b.judgment
           FROM benchmark_judgment a
           JOIN benchmark_judgment b
             ON a.rct_id = b.rct_id AND a.domain = b.domain
           WHERE a.source = ? AND b.source = ? AND a.domain = ?
             AND a.judgment IS NOT NULL AND b.judgment IS NOT NULL
             AND a.valid = 1 AND b.valid = 1"""
        + _fallback_filter("a", "b")
        + wp_sql
        + " ORDER BY a.rct_id",
        (source_a, source_b, domain, *wp_params),
    ).fetchall()


def load_judgments(conn: sqlite3.Connection, source: str, domain: str
                   ) -> dict[str, str]:
    """{rct_id: judgment} for one (source, domain), valid rows only.

    Wrong-paper RCTs (the active set — all 13 in the primary, only the two
    unindexed ones under --sensitivity) are dropped here, which also keeps them
    out of the ensemble (built from these judgments).
    """
    wp_sql, wp_params = wrong_paper_filter(
        "", exclusion_set=_active_wrong_paper_set())
    return dict(conn.execute(
        """SELECT rct_id, judgment FROM benchmark_judgment
           WHERE source = ? AND domain = ? AND valid = 1
             AND judgment IS NOT NULL"""
        + _fallback_filter("")
        + wp_sql,
        (source, domain, *wp_params),
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
    """Per-RCT ensemble judgments across the 3 passes.

    Returns {rct_id: {domain: judgment}}. The five signalling domains
    (d1–d5) are set by strict majority vote across the passes (≥2 of the
    available passes agree; ties or <2 passes are dropped). The ``overall``
    judgment is then derived from those ensemble domains by the RoB 2
    worst-wins rule — NOT a direct majority vote of the passes' own overall
    labels — so the reported ensemble ``overall`` is consistent with the
    reported ensemble d1–d5. ``overall`` is only emitted when all five
    signalling domains have a majority winner for that RCT.
    """
    pass_judgments = {
        p: {
            d: load_judgments(conn, f"{model}_{protocol}_pass{p}", d)
            for d in SIGNALLING_DOMAINS
        }
        for p in PASSES
    }
    out: dict[str, dict[str, str]] = {}
    # Find RCTs that have at least one judgment in any pass × any domain.
    rct_ids: set[str] = set()
    for p in PASSES:
        for d in SIGNALLING_DOMAINS:
            rct_ids.update(pass_judgments[p][d])
    for rct_id in rct_ids:
        ensemble_domains: dict[str, str] = {}
        for d in SIGNALLING_DOMAINS:
            votes = [pass_judgments[p][d].get(rct_id) for p in PASSES]
            votes = [v for v in votes if v is not None]
            if len(votes) < 2:
                continue
            counter = Counter(votes)
            top, top_n = counter.most_common(1)[0]
            # Require strict majority (≥ 2 out of however many we got).
            if top_n >= 2:
                ensemble_domains[d] = top
        if not ensemble_domains:
            continue
        # Overall = worst of the five ensemble domains (RoB 2 worst-wins),
        # only when every signalling domain has an ensemble winner.
        if all(d in ensemble_domains for d in SIGNALLING_DOMAINS):
            worst_rank = max(
                SEVERITY_RANK[ensemble_domains[d]] for d in SIGNALLING_DOMAINS
            )
            ensemble_domains["overall"] = _RANK_TO_LABEL[worst_rank]
        out[rct_id] = ensemble_domains
    return out


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

def mcnemar_test(preds_a: dict[str, tuple[str, str]],
                 preds_b: dict[str, tuple[str, str]]) -> tuple[int, int, float]:
    """McNemar's test on per-RCT correctness collapsed to match/no-match.

    ``preds_a`` and ``preds_b`` map ``rct_id -> (cochrane, model)`` for the
    two systems being compared. Alignment is by ``rct_id`` key — NOT by row
    order — because the callers build these from independent SQLite queries
    whose row order is unspecified; only RCTs present in both are used.
    Returns (b_only, c_only, p_value) where b_only = (a correct, b wrong),
    c_only = (a wrong, b correct). Continuity-corrected chi-squared, df=1.

    Returns (0, 0, 1.0) if the shared set is empty, or (b_only, c_only, NaN)
    if the discordant pairs sum to <25 (McNemar's chi-squared is unreliable
    there and an exact binomial is more appropriate; we surface the raw
    counts so the reader can compute exact-test p-values).
    """
    shared = preds_a.keys() & preds_b.keys()
    if not shared:
        return (0, 0, 1.0)
    b_only = c_only = 0
    for rct_id in shared:
        coch_a, model_a = preds_a[rct_id]
        coch_b, model_b = preds_b[rct_id]
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

def write_results(conn: sqlite3.Connection, run_ensembles: bool, *,
                  results_md: Path = RESULTS_MD,
                  results_csv: Path = RESULTS_CSV,
                  forest_csv: Path = FOREST_CSV,
                  exclude_fallback: bool = False) -> None:
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
    with open(results_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    forest_fields = ["label", "k_lin", "k_quad", "ci_lin_lo", "ci_lin_hi", "n", "kind"]
    with open(forest_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=forest_fields)
        w.writeheader()
        for r in forest_rows:
            w.writerow(r)

    # Markdown report
    write_markdown_report(rows, forest_rows, results_md=results_md,
                          exclude_fallback=exclude_fallback)


def fmt(value, fmt_str=".3f"):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return format(value, fmt_str)


def write_markdown_report(rows: list[dict], forest_rows: list[dict], *,
                          results_md: Path = RESULTS_MD,
                          exclude_fallback: bool = False) -> None:
    lines: list[str] = []
    lines.append("# Phase 6 Cross-Model Comparison")
    lines.append("")
    lines.append("**Generated:** by `studies/eisele_metzger_replication/compute_phase6_kappa.py`")
    lines.append("**Output companions:** `phase6_results.csv` (raw rows) and `phase6_forest_data.csv` (forest-plot input).")
    lines.append("")
    lines.append("Coverage of the table fills in as Phase 5 evaluation runs complete. Empty model rows = data not yet in the DB.")
    lines.append("")
    if exclude_fallback:
        lines.append(
            "> **STRICT MODE (`--exclude-fallback`):** algorithm-derived "
            "(`raw_label='FALLBACK'`) judgements are excluded — these numbers "
            "reproduce the pre-registered *model-emitted* primary metric."
        )
    else:
        lines.append(
            "> **INCLUSIVE MODE (default):** algorithm-derived "
            "(`raw_label='FALLBACK'`) judgements are included. Re-run with "
            "`--exclude-fallback` for the pre-registered model-emitted primary metric."
        )
    if SENSITIVITY_MODE:
        excl = ", ".join(sorted(UNRECOVERABLE_WRONG_PAPER_RCTS))
        lines.append("")
        lines.append(
            f"> **SENSITIVITY ANALYSIS (`--sensitivity`):** only the unindexed "
            f"wrong papers ({excl}) are excluded; the recovered-and-re-assessed "
            "wrong-paper RCTs are re-included. Secondary to the pre-registered "
            "primary (all 13 wrong papers excluded)."
        )
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
        lines.append("Each signalling domain (d1–d5) is a strict majority vote across the three "
                     "passes; `overall` is then the worst of those five ensemble domains (RoB 2 "
                     "worst-wins), not a direct majority vote of the passes' overall labels.")
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

    results_md.write_text("\n".join(lines), encoding="utf-8")


# --- main ---------------------------------------------------------------

def main() -> int:
    global EXCLUDE_FALLBACK, SENSITIVITY_MODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--no-ensembles", action="store_true",
        help="Skip the 3-pass majority-vote ensemble computation.",
    )
    parser.add_argument(
        "--exclude-fallback", action="store_true",
        help="Exclude algorithm-derived (raw_label='FALLBACK') judgements from "
             "all κ computations, reproducing the pre-registered model-emitted "
             "primary metric. Writes to *.strict.{md,csv} so the inclusive "
             "results are not overwritten.",
    )
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="Recovered-corpus SENSITIVITY analysis: exclude only the two "
             "unindexed wrong papers (RCT030, RCT080) instead of all 13, so the "
             "recovered-and-re-assessed wrong-paper RCTs re-enter the κ. Refuses "
             "to run unless every recoverable wrong paper has been recovered in "
             "the DB (see recover_wrong_papers.py). Writes to "
             "*.sensitivity.{md,csv} so the primary results are not overwritten; "
             "composes with --exclude-fallback (*.sensitivity.strict.{md,csv}).",
    )
    args = parser.parse_args()
    if not args.db_path.exists():
        print(f"[error] DB not found at {args.db_path}", file=sys.stderr)
        return 2

    EXCLUDE_FALLBACK = args.exclude_fallback
    SENSITIVITY_MODE = args.sensitivity

    conn = sqlite3.connect(args.db_path)
    try:
        # Refuse the sensitivity analysis on a DB where the recoverable wrong
        # papers have not actually been recovered — otherwise the re-included
        # RCTs still carry stale wrong-document judgements and the κ is garbage.
        if args.sensitivity:
            missing = sensitivity_precondition_failures(conn)
            if missing:
                ids = " ".join(sorted(missing))
                print(
                    "[error] --sensitivity requires the recoverable wrong "
                    "papers to be recovered + re-assessed first. Not yet "
                    f"recovered in this DB:\n        {', '.join(sorted(missing))}\n"
                    "        Run the owner-gated recovery, e.g.:\n"
                    "        uv run python studies/eisele_metzger_replication/"
                    f"recover_wrong_papers.py apply {ids} --apply\n"
                    "        then re-assess the deleted rows (run_evaluation*.py) "
                    "before re-running --sensitivity.",
                    file=sys.stderr,
                )
                return 2

        tags = []
        if args.sensitivity:
            tags.append("sensitivity")
        if args.exclude_fallback:
            tags.append("strict")
        infix = "." + ".".join(tags) if tags else ""
        results_md = STUDY_DIR / f"phase6_results{infix}.md"
        results_csv = STUDY_DIR / f"phase6_results{infix}.csv"
        forest_csv = STUDY_DIR / f"phase6_forest_data{infix}.csv"

        if args.sensitivity:
            excl = ", ".join(sorted(UNRECOVERABLE_WRONG_PAPER_RCTS))
            print(f"[mode] SENSITIVITY — recovered corpus; excluding only the "
                  f"unindexed wrong papers ({excl})")
        if args.exclude_fallback:
            print("[mode] STRICT — excluding raw_label='FALLBACK' rows")
        if not (args.sensitivity or args.exclude_fallback):
            print("[mode] PRIMARY INCLUSIVE — all 13 wrong papers excluded, "
                  "FALLBACK rows included (use --exclude-fallback for the "
                  "model-emitted primary metric)")

        write_results(conn, run_ensembles=not args.no_ensembles,
                      results_md=results_md, results_csv=results_csv,
                      forest_csv=forest_csv, exclude_fallback=args.exclude_fallback)
    finally:
        conn.close()
    print(f"[write] {results_md}")
    print(f"[write] {results_csv}")
    print(f"[write] {forest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pre-manuscript spot-checks for the EM 2025 replication (runbook §6).

Two audits the manuscript needs before submission — both computed from the
benchmark DB so no number in the drafts is authored by hand (HANDOVER
convention: "all κ / ensemble / synthesis numbers are computed in code, never
by the model"). Read-only over ``dataset/eisele_metzger_benchmark.db``.

1. **Sonnet ``low``-judgement audit** (companion to draft §3.5, which does this
   for gpt-oss). Every RCT where Claude Sonnet 4.6 emits an *overall* ``low`` on
   full text, with its per-domain judgements, the Cochrane gold, and the
   rationale prose, so the reader can confirm the ``low`` calls are
   right-for-the-right-reasons rather than pattern-matching artefacts.

2. **Per-domain run-to-run instability** (companion to draft §3.6, which does
   this for gpt-oss). For all four models × full text, the pass-to-pass
   disagreement rate per signalling domain (d1–d5), to test whether the
   D1-concentrated instability seen in the gpt-oss audit generalises to the
   other three models.

Both audits honour the wrong-paper exclusion set
(``exclusions.WRONG_PAPER_RCTS``) via the shared loaders imported from
``compute_phase6_kappa``. This script never writes to the DB.

Usage::

    uv run python studies/eisele_metzger_replication/premanuscript_spotchecks.py
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "studies/eisele_metzger_replication"))

from compute_phase6_kappa import (  # noqa: E402
    MODEL_LABELS,
    PASSES,
    SIGNALLING_DOMAINS,
    load_pairs,
)
from exclusions import wrong_paper_filter  # noqa: E402
from sanity_check_kappa import cohen_kappa  # noqa: E402

DEFAULT_DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"
STUDY_DIR = PROJECT_ROOT / "studies/eisele_metzger_replication"
RESULTS_MD = STUDY_DIR / "premanuscript_spotcheck_results.md"
INSTABILITY_CSV = STUDY_DIR / "premanuscript_instability.csv"

# Pass pairs used for every run-to-run comparison (matches compute_phase6_kappa).
PASS_PAIRS = ((1, 2), (1, 3), (2, 3))

# The model whose `low` calls the audit inspects (runbook §6.1). gpt-oss's own
# `low` audit is already in draft §3.5; Sonnet is the outstanding one.
LOW_AUDIT_MODEL = "sonnet_4_6"


# --- Pure helpers (unit-tested) ----------------------------------------

def disagreement_stats(pairs: list[tuple[str, str]]) -> tuple[int, int]:
    """``(n_disagree, n_total)`` for a list of ``(label_a, label_b)`` pairs.

    A pair counts as a disagreement when the two labels differ. Empty input
    yields ``(0, 0)``.
    """
    n_total = len(pairs)
    n_disagree = sum(1 for a, b in pairs if a != b)
    return n_disagree, n_total


def disagreement_rate(n_disagree: int, n_total: int) -> float | None:
    """Fraction of compared pairs that disagree, or ``None`` when no pairs."""
    if n_total == 0:
        return None
    return n_disagree / n_total


def dominant_domain(
    per_domain: dict[str, tuple[int, int]],
    order: tuple[str, ...] = SIGNALLING_DOMAINS,
) -> str | None:
    """Domain with the highest pass-to-pass disagreement rate.

    ``per_domain`` maps ``domain -> (n_disagree, n_total)``. Domains with no
    comparable pairs (``n_total == 0``) are ignored. Ties on rate are broken by
    the larger absolute disagreement count, then by position in ``order`` (so
    the result is deterministic). Returns ``None`` when nothing is comparable.
    """
    candidates = [
        (d, n_dis, n_tot)
        for d, (n_dis, n_tot) in per_domain.items()
        if n_tot > 0
    ]
    if not candidates:
        return None
    order_index = {d: i for i, d in enumerate(order)}
    return max(
        candidates,
        key=lambda t: (t[1] / t[2], t[1], -order_index.get(t[0], len(order))),
    )[0]


# --- DB loaders --------------------------------------------------------

@dataclass
class DomainInstability:
    domain: str
    n_disagree: int
    n_total: int
    mean_kappa_quad: float | None

    @property
    def rate(self) -> float | None:
        return disagreement_rate(self.n_disagree, self.n_total)


@dataclass
class ModelInstability:
    model: str
    protocol: str
    per_domain: dict[str, DomainInstability] = field(default_factory=dict)

    @property
    def dominant(self) -> str | None:
        return dominant_domain(
            {d: (di.n_disagree, di.n_total) for d, di in self.per_domain.items()}
        )


def model_instability(
    conn: sqlite3.Connection, model: str, protocol: str
) -> ModelInstability | None:
    """Per-signalling-domain run-to-run instability for one model × protocol.

    For each domain the disagreement count/total is pooled across the three
    pass-pairs; the mean quadratic-weighted κ is the mean pairwise run-to-run κ
    (higher κ = more stable). Returns ``None`` if the model has no rows.
    """
    result = ModelInstability(model=model, protocol=protocol)
    any_data = False
    for domain in SIGNALLING_DOMAINS:
        pooled: list[tuple[str, str]] = []
        kappas: list[float] = []
        for pa, pb in PASS_PAIRS:
            pairs = load_pairs(
                conn,
                f"{model}_{protocol}_pass{pa}",
                f"{model}_{protocol}_pass{pb}",
                domain,
            )
            if not pairs:
                continue
            pooled.extend(pairs)
            kappas.append(cohen_kappa(pairs, "quadratic"))
        if not pooled:
            continue
        any_data = True
        n_dis, n_tot = disagreement_stats(pooled)
        result.per_domain[domain] = DomainInstability(
            domain=domain,
            n_disagree=n_dis,
            n_total=n_tot,
            mean_kappa_quad=(sum(kappas) / len(kappas)) if kappas else None,
        )
    return result if any_data else None


@dataclass
class LowAuditRow:
    rct_id: str
    cochrane_overall: str | None
    cochrane_domains: dict[str, str | None]
    pass_overall: dict[int, str | None]
    pass_domains: dict[int, dict[str, str | None]]
    low_pass_rationales: dict[int, str | None]

    def low_passes(self) -> list[int]:
        """Passes on which this model emitted an overall ``low`` for this RCT."""
        return [p for p, j in sorted(self.pass_overall.items()) if j == "low"]

    def correct_low_passes(self) -> list[int]:
        """``low`` passes whose overall matches the Cochrane gold (== low)."""
        if self.cochrane_overall != "low":
            return []
        return self.low_passes()

    def differing_domains(self, pass_n: int) -> list[str]:
        """Signalling domains where ``pass_n`` differs from the Cochrane gold."""
        doms = self.pass_domains.get(pass_n, {})
        return [
            d for d in SIGNALLING_DOMAINS
            if doms.get(d) is not None
            and self.cochrane_domains.get(d) is not None
            and doms.get(d) != self.cochrane_domains.get(d)
        ]


def _domain_map(
    conn: sqlite3.Connection, source: str, rct_id: str
) -> dict[str, str | None]:
    rows = conn.execute(
        """SELECT domain, judgment FROM benchmark_judgment
           WHERE source = ? AND rct_id = ? AND valid = 1""",
        (source, rct_id),
    ).fetchall()
    return {d: j for d, j in rows}


def load_low_audit(
    conn: sqlite3.Connection, model: str
) -> list[LowAuditRow]:
    """Every full-text RCT where ``model`` emits an overall ``low``, with detail.

    Wrong-paper RCTs are excluded (the shared exclusion set), matching every κ
    loader. One row per RCT; ``pass_overall``/``pass_domains`` carry all three
    passes so the reader sees the run-to-run picture around each ``low`` call.
    """
    wp_sql, wp_params = wrong_paper_filter("")
    low_rcts = [
        r[0]
        for r in conn.execute(
            """SELECT DISTINCT rct_id FROM benchmark_judgment
               WHERE source LIKE ? AND domain = 'overall'
                 AND judgment = 'low' AND valid = 1"""
            + wp_sql,
            (f"{model}_fulltext_pass%", *wp_params),
        ).fetchall()
    ]
    rows: list[LowAuditRow] = []
    for rct_id in sorted(low_rcts):
        cochrane = _domain_map(conn, "cochrane", rct_id)
        pass_overall: dict[int, str | None] = {}
        pass_domains: dict[int, dict[str, str | None]] = {}
        low_rationales: dict[int, str | None] = {}
        for p in PASSES:
            src = f"{model}_fulltext_pass{p}"
            doms = _domain_map(conn, src, rct_id)
            pass_overall[p] = doms.get("overall")
            pass_domains[p] = {d: doms.get(d) for d in SIGNALLING_DOMAINS}
            if doms.get("overall") == "low":
                rationale = conn.execute(
                    """SELECT rationale FROM benchmark_judgment
                       WHERE source = ? AND rct_id = ? AND domain = 'overall'""",
                    (src, rct_id),
                ).fetchone()
                low_rationales[p] = rationale[0] if rationale else None
        rows.append(
            LowAuditRow(
                rct_id=rct_id,
                cochrane_overall=cochrane.get("overall"),
                cochrane_domains={d: cochrane.get(d) for d in SIGNALLING_DOMAINS},
                pass_overall=pass_overall,
                pass_domains=pass_domains,
                low_pass_rationales=low_rationales,
            )
        )
    return rows


# --- Reporting ---------------------------------------------------------

def _fmt_pct(rate: float | None) -> str:
    return "—" if rate is None else f"{rate:.0%}"


def _fmt_kappa(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def write_report(
    low_rows: list[LowAuditRow],
    instabilities: list[ModelInstability],
    *,
    results_md: Path = RESULTS_MD,
    instability_csv: Path = INSTABILITY_CSV,
) -> None:
    lines: list[str] = []
    lines.append("# Pre-manuscript spot-checks (runbook §6)")
    lines.append("")
    lines.append(
        "**Generated by** "
        "`studies/eisele_metzger_replication/premanuscript_spotchecks.py` "
        "(read-only over `dataset/eisele_metzger_benchmark.db`). "
        "Wrong-paper RCTs excluded via `exclusions.WRONG_PAPER_RCTS`."
    )
    lines.append("")

    # --- Audit 1: Sonnet `low`-judgement audit ---
    model_label = MODEL_LABELS.get(LOW_AUDIT_MODEL, LOW_AUDIT_MODEL)
    n_low = sum(len(r.low_passes()) for r in low_rows)
    n_correct = sum(len(r.correct_low_passes()) for r in low_rows)
    lines.append(f"## 1. {model_label} `low`-judgement audit (companion to §3.5)")
    lines.append("")
    lines.append(
        f"{model_label} emits an overall `low` on full text **{n_low} times** "
        f"across the three passes, spread over **{len(low_rows)} RCT(s)**. "
        f"**{n_correct}/{n_low}** of those `low` judgements match the Cochrane "
        "gold overall."
    )
    lines.append("")
    lines.append(
        "| RCT | Cochrane gold | pass 1 / 2 / 3 (overall) | `low` matches? | "
        "domains differing from Cochrane |"
    )
    lines.append("|---|---|---|---|---|")
    for r in low_rows:
        triple = " / ".join(str(r.pass_overall.get(p)) for p in PASSES)
        low_ps = r.low_passes()
        matched = r.correct_low_passes()
        if r.cochrane_overall == "low":
            match_str = f"{len(matched)}/{len(low_ps)} ✓"
        else:
            match_str = f"0/{len(low_ps)} (Cochrane = {r.cochrane_overall})"
        diffs: set[str] = set()
        for p in PASSES:
            diffs.update(r.differing_domains(p))
        diff_str = ", ".join(sorted(diffs)).upper() if diffs else "none"
        lines.append(
            f"| {r.rct_id} | {r.cochrane_overall} | {triple} | "
            f"{match_str} | {diff_str} |"
        )
    lines.append("")
    lines.append("### Rationale prose for each `low` call")
    lines.append("")
    for r in low_rows:
        for p in sorted(r.low_pass_rationales):
            lines.append(
                f"- **{r.rct_id} pass {p}** (Cochrane overall = "
                f"{r.cochrane_overall}): {r.low_pass_rationales[p]}"
            )
    lines.append("")

    # --- Audit 2: per-domain run-to-run instability ---
    lines.append(
        "## 2. Per-domain run-to-run instability, all four models (companion to §3.6)"
    )
    lines.append("")
    lines.append(
        "Pass-to-pass disagreement pooled across the three pass-pairs "
        "(1↔2, 1↔3, 2↔3) per signalling domain, full-text protocol. Higher "
        "disagreement rate = noisier domain; mean κ_quad is the mean pairwise "
        "run-to-run agreement (higher = more stable). **Dominant** = the domain "
        "with the highest disagreement rate."
    )
    lines.append("")
    for inst in instabilities:
        label = MODEL_LABELS.get(inst.model, inst.model)
        dom = inst.dominant
        dom_label = dom.upper() if dom else "—"
        lines.append(f"### {label} × {inst.protocol} — dominant: {dom_label}")
        lines.append("")
        lines.append("| Domain | disagreements | rate | mean κ_quad |")
        lines.append("|---|---|---:|---:|")
        for d in SIGNALLING_DOMAINS:
            di = inst.per_domain.get(d)
            if di is None:
                lines.append(f"| {d.upper()} | — | — | — |")
                continue
            marker = " ⟵ dominant" if d == dom else ""
            lines.append(
                f"| {d.upper()}{marker} | {di.n_disagree}/{di.n_total} | "
                f"{_fmt_pct(di.rate)} | {_fmt_kappa(di.mean_kappa_quad)} |"
            )
        lines.append("")

    results_md.write_text("\n".join(lines), encoding="utf-8")

    # CSV companion for the instability table (one row per model×protocol×domain)
    with open(instability_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["model", "protocol", "domain", "n_disagree", "n_total",
             "disagreement_rate", "mean_kappa_quad", "is_dominant"]
        )
        for inst in instabilities:
            dom = inst.dominant
            for d in SIGNALLING_DOMAINS:
                di = inst.per_domain.get(d)
                if di is None:
                    continue
                w.writerow([
                    inst.model, inst.protocol, d, di.n_disagree, di.n_total,
                    f"{di.rate:.4f}" if di.rate is not None else "",
                    f"{di.mean_kappa_quad:.4f}"
                    if di.mean_kappa_quad is not None else "",
                    int(d == dom),
                ])


# --- main --------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--protocol", default="fulltext",
                        choices=("abstract", "fulltext"),
                        help="Protocol for the instability audit (default: fulltext).")
    args = parser.parse_args()
    if not args.db_path.exists():
        print(f"[error] DB not found at {args.db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(args.db_path)
    try:
        low_rows = load_low_audit(conn, LOW_AUDIT_MODEL)
        instabilities = []
        for model in MODEL_LABELS:
            inst = model_instability(conn, model, args.protocol)
            if inst is not None:
                instabilities.append(inst)
        write_report(low_rows, instabilities)
    finally:
        conn.close()

    print(f"[write] {RESULTS_MD}")
    print(f"[write] {INSTABILITY_CSV}")
    # Console summary
    n_low = sum(len(r.low_passes()) for r in low_rows)
    n_correct = sum(len(r.correct_low_passes()) for r in low_rows)
    print(f"[audit] {LOW_AUDIT_MODEL} low judgements: {n_correct}/{n_low} match Cochrane")
    for inst in instabilities:
        print(f"[instability] {inst.model} × {args.protocol}: dominant = {inst.dominant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

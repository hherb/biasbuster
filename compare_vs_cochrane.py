"""Compare V5A model annotations against Cochrane RoB 2 expert ratings.

Biasbuster deliberately deviates from Cochrane RoB 2 in one important
way: biasbuster treats structural conflict-of-interest as HIGH risk
(triggers a/b/c/d per ``docs/two_step_approach/DESIGN_RATIONALE_COI.md``),
while Cochrane RoB 2 does not assess COI at all. This means raw
overall-severity agreement will understate the pipeline's agreement
with Cochrane on the domains they actually share.

Outputs (Markdown, JSON, CSV) go to
``dataset/annotation_comparison/cochrane_comparison_<date>.{md,json,csv}``.

Mapping used:
  biasbuster.methodology        ↔ max(Cochrane randomization, deviation,
                                       missing_outcome, measurement)
  biasbuster.outcome_reporting  ↔ Cochrane reporting_bias
  biasbuster.overall_severity   ↔ Cochrane overall_rob
  biasbuster.spin               — no Cochrane analogue
  biasbuster.stat_reporting     — partial overlap with reporting_bias
  biasbuster.coi                — no Cochrane analogue (deliberate)

Severity mapping: Cochrane "low" → biasbuster "low"; Cochrane "some
concerns" → biasbuster "moderate"; Cochrane "high" → biasbuster "high".
Biasbuster "none" is folded into "low" for comparison, and "critical"
into "high", so both scales reduce to a common 3-level ordinal.

Usage:
    uv run python compare_vs_cochrane.py \\
        --models anthropic_fulltext_decomposed,ollama:gemma4:26b-a4b-it-q8_0_fulltext_decomposed
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from biasbuster.database import Database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity normalisation — reduce both schemas to a common 3-level ordinal
# ---------------------------------------------------------------------------

# Biasbuster severities → common 3-level:
_BB_TO_3LEVEL = {
    "none": 0,
    "low": 0,
    "moderate": 1,
    "high": 2,
    "critical": 2,
}

# Cochrane RoB 2 ratings → common 3-level:
_COCHRANE_TO_3LEVEL = {
    "low": 0,
    "some concerns": 1,
    "high": 2,
    # Variant spellings observed in the wild
    "some_concerns": 1,
    "unclear": 1,  # treat as middle
}

_3LEVEL_NAMES = ["low", "moderate", "high"]


def _bb_rank(sev: str | None) -> int | None:
    if not sev:
        return None
    return _BB_TO_3LEVEL.get(str(sev).lower().strip())


def _cochrane_rank(sev: str | None) -> int | None:
    if not sev:
        return None
    key = str(sev).lower().strip()
    return _COCHRANE_TO_3LEVEL.get(key)


# ---------------------------------------------------------------------------
# Cochrane methodology compositing
# ---------------------------------------------------------------------------

_COCHRANE_METHODOLOGY_DOMAINS = (
    "randomization_bias",
    "deviation_bias",
    "missing_outcome_bias",
    "measurement_bias",
)


def _cochrane_methodology_rank(paper_row: dict) -> int | None:
    """Take the max rank over the 4 Cochrane procedural domains.

    Biasbuster's methodology domain aggregates attrition, ITT,
    multiplicity, blinding, premature stopping, etc. — which maps to
    the worst of Cochrane's 4 procedural domains (Cochrane splits
    them but the ordinal rating is comparable).
    """
    ranks = []
    for dom in _COCHRANE_METHODOLOGY_DOMAINS:
        r = _cochrane_rank(paper_row.get(dom))
        if r is not None:
            ranks.append(r)
    return max(ranks) if ranks else None


# ---------------------------------------------------------------------------
# Weighted kappa (linear weights, 3-level ordinal)
# ---------------------------------------------------------------------------

def _weighted_kappa(pairs: list[tuple[int, int]], n_levels: int = 3) -> float:
    """Compute linear-weighted Cohen's kappa for ordinal ratings.

    Args:
        pairs: list of (rater_a_rank, rater_b_rank) tuples.
        n_levels: number of ordinal categories (default 3).

    Returns:
        Kappa in [-1, 1]. Returns NaN (as 0.0) if no valid pairs.
    """
    if not pairs:
        return 0.0

    n = len(pairs)
    # Observed agreement with linear weights
    total_weight = (n_levels - 1)
    observed = 0.0
    for a, b in pairs:
        w = 1.0 - abs(a - b) / total_weight
        observed += w
    observed /= n

    # Expected agreement under independence
    count_a = [0] * n_levels
    count_b = [0] * n_levels
    for a, b in pairs:
        count_a[a] += 1
        count_b[b] += 1

    expected = 0.0
    for i in range(n_levels):
        for j in range(n_levels):
            w = 1.0 - abs(i - j) / total_weight
            expected += w * (count_a[i] / n) * (count_b[j] / n)

    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


# ---------------------------------------------------------------------------
# Per-model evaluation against Cochrane
# ---------------------------------------------------------------------------

@dataclass
class ModelCochraneEval:
    """Evaluation of one biasbuster model vs Cochrane expert labels."""
    model_id: str = ""
    n_papers: int = 0
    # Overall agreement
    overall_kappa_raw: float = 0.0          # includes COI-driven divergence
    overall_kappa_adjusted: float = 0.0     # excludes COI-only-HIGH papers
    overall_pairs: list[tuple[str, int, int]] = field(default_factory=list)  # (pmid, bb_rank, cochrane_rank)
    # Per-domain agreement (only directly comparable domains)
    methodology_kappa: float = 0.0
    outcome_reporting_kappa: float = 0.0
    # COI-only analysis
    coi_only_high_count: int = 0            # papers where biasbuster rates HIGH solely due to COI
    coi_only_high_pmids: list[str] = field(default_factory=list)


def _is_coi_only_driver(bb_ann: dict) -> bool:
    """True if biasbuster rates this paper HIGH solely due to COI.

    Used to identify papers where raw overall-severity comparison vs
    Cochrane is unfair (Cochrane does not assess COI).
    """
    overall = str(bb_ann.get("overall_severity", "")).lower()
    if overall not in ("high", "critical"):
        return False

    coi_sev = str(bb_ann.get("conflict_of_interest", {}).get("severity", "")).lower()
    if coi_sev not in ("high", "critical"):
        return False

    # Check that no other domain is at the same level
    for dom in ("statistical_reporting", "spin", "outcome_reporting", "methodology"):
        dom_sev = str(bb_ann.get(dom, {}).get("severity", "")).lower()
        if dom_sev in ("high", "critical"):
            return False  # at least one other domain also high — not COI-only
    return True


def evaluate_model_vs_cochrane(
    model_id: str,
    annotations_by_pmid: dict[str, dict],
    papers_by_pmid: dict[str, dict],
) -> ModelCochraneEval:
    """Compute Cochrane-agreement metrics for one biasbuster model."""
    result = ModelCochraneEval(model_id=model_id)

    overall_pairs_raw: list[tuple[int, int]] = []
    overall_pairs_adjusted: list[tuple[int, int]] = []
    methodology_pairs: list[tuple[int, int]] = []
    outcome_pairs: list[tuple[int, int]] = []

    for pmid, ann in annotations_by_pmid.items():
        paper = papers_by_pmid.get(pmid)
        if not paper or paper.get("source") != "cochrane_rob":
            continue
        result.n_papers += 1

        # Overall severity
        bb_overall = _bb_rank(ann.get("overall_severity"))
        c_overall = _cochrane_rank(paper.get("overall_rob"))
        if bb_overall is not None and c_overall is not None:
            overall_pairs_raw.append((bb_overall, c_overall))
            result.overall_pairs.append((pmid, bb_overall, c_overall))

            if _is_coi_only_driver(ann):
                result.coi_only_high_count += 1
                result.coi_only_high_pmids.append(pmid)
                # Exclude from adjusted comparison
            else:
                overall_pairs_adjusted.append((bb_overall, c_overall))

        # Methodology
        bb_meth = _bb_rank(ann.get("methodology", {}).get("severity"))
        c_meth = _cochrane_methodology_rank(paper)
        if bb_meth is not None and c_meth is not None:
            methodology_pairs.append((bb_meth, c_meth))

        # Outcome reporting ↔ Cochrane reporting_bias
        bb_outcome = _bb_rank(ann.get("outcome_reporting", {}).get("severity"))
        c_rep = _cochrane_rank(paper.get("reporting_bias"))
        if bb_outcome is not None and c_rep is not None:
            outcome_pairs.append((bb_outcome, c_rep))

    result.overall_kappa_raw = _weighted_kappa(overall_pairs_raw)
    result.overall_kappa_adjusted = _weighted_kappa(overall_pairs_adjusted)
    result.methodology_kappa = _weighted_kappa(methodology_pairs)
    result.outcome_reporting_kappa = _weighted_kappa(outcome_pairs)

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_markdown_report(
    evals: list[ModelCochraneEval],
    timestamp: str,
) -> str:
    """Produce a human-readable report of Cochrane comparison."""
    lines = []
    lines.append("# Biasbuster V5A vs Cochrane RoB 2 Expert Labels")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"N papers: {evals[0].n_papers if evals else 0}")
    lines.append("")
    lines.append(
        "Biasbuster's policy deliberately extends Cochrane RoB 2 by "
        "assessing structural conflict-of-interest risk (see "
        "`docs/two_step_approach/DESIGN_RATIONALE_COI.md`). "
        "Cochrane RoB 2 does not assess COI. This means the raw "
        "overall-severity agreement will be suppressed on "
        "industry-funded trials. The 'adjusted' kappa excludes papers "
        "where biasbuster rates HIGH solely due to COI."
    )
    lines.append("")

    # --- Overall agreement table ---
    lines.append("## Overall severity agreement (weighted kappa, 3-level ordinal)")
    lines.append("")
    lines.append("| Model | κ (raw) | κ (COI-adjusted) | COI-only HIGH papers |")
    lines.append("|-------|---------|------------------|----------------------|")
    for e in evals:
        lines.append(
            f"| {e.model_id} | {e.overall_kappa_raw:+.3f} | "
            f"{e.overall_kappa_adjusted:+.3f} | {e.coi_only_high_count} |"
        )
    lines.append("")

    # --- Per-domain (only comparable ones) ---
    lines.append("## Per-domain agreement with Cochrane")
    lines.append("")
    lines.append(
        "Only biasbuster domains with a Cochrane analogue are scored. "
        "`methodology` is compared against `max(randomization, "
        "deviation, missing_outcome, measurement)`; `outcome_reporting` "
        "against `reporting_bias`. `spin`, `statistical_reporting`, "
        "and `conflict_of_interest` have no direct Cochrane RoB 2 "
        "equivalent and are not scored here."
    )
    lines.append("")
    lines.append("| Model | methodology κ | outcome_reporting κ |")
    lines.append("|-------|---------------|---------------------|")
    for e in evals:
        lines.append(
            f"| {e.model_id} | {e.methodology_kappa:+.3f} | "
            f"{e.outcome_reporting_kappa:+.3f} |"
        )
    lines.append("")

    # --- COI-only HIGH papers ---
    lines.append("## Papers where biasbuster rates HIGH solely due to COI")
    lines.append("")
    lines.append(
        "These are excluded from the COI-adjusted overall kappa. Cochrane "
        "cannot agree or disagree on these because Cochrane RoB 2 does "
        "not assess COI."
    )
    lines.append("")
    any_coi = False
    for e in evals:
        if e.coi_only_high_pmids:
            any_coi = True
            pmids_str = ", ".join(e.coi_only_high_pmids)
            lines.append(f"- **{e.model_id}**: {pmids_str}")
    if not any_coi:
        lines.append("  (none)")
    lines.append("")

    # --- Per-paper detail ---
    lines.append("## Per-paper overall severity (biasbuster 3-level vs Cochrane)")
    lines.append("")
    if evals:
        # Use the first model's pair list to get the PMID order
        pmids = [p[0] for p in evals[0].overall_pairs]
        hdr = "| PMID | Cochrane | " + " | ".join(e.model_id for e in evals) + " |"
        sep = "|------|----------|" + "|".join("---" for _ in evals) + "|"
        lines.append(hdr)
        lines.append(sep)
        for pmid in pmids:
            cochrane_rank = None
            cells = []
            for e in evals:
                for p, bb, c in e.overall_pairs:
                    if p == pmid:
                        cochrane_rank = c
                        cells.append(_3LEVEL_NAMES[bb])
                        break
                else:
                    cells.append("—")
            cochrane_cell = _3LEVEL_NAMES[cochrane_rank] if cochrane_rank is not None else "?"
            lines.append(f"| {pmid} | {cochrane_cell} | " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare biasbuster V5A annotations against Cochrane RoB 2 expert labels."
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated model_name values (as stored in the annotations table). "
             "Example: anthropic_fulltext_decomposed,ollama:gemma4:26b-a4b-it-q8_0_fulltext_decomposed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/annotation_comparison",
        help="Directory to write the report files. Default: dataset/annotation_comparison/",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    db = Database()
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    # Load papers keyed by pmid
    papers_by_pmid: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT * FROM papers WHERE source='cochrane_rob'"
    ):
        papers_by_pmid[row["pmid"]] = dict(row)
    logger.info(f"Loaded {len(papers_by_pmid)} Cochrane RoB papers")

    # Evaluate each model
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    evals: list[ModelCochraneEval] = []
    for model_name in model_names:
        rows = conn.execute(
            "SELECT pmid, annotation FROM annotations WHERE model_name = ?",
            (model_name,),
        ).fetchall()
        anns_by_pmid = {r["pmid"]: json.loads(r["annotation"]) for r in rows}
        logger.info(
            f"Model {model_name}: {len(anns_by_pmid)} annotations, "
            f"of which {sum(1 for p in anns_by_pmid if p in papers_by_pmid)} "
            f"are Cochrane papers"
        )
        evals.append(evaluate_model_vs_cochrane(
            model_id=model_name,
            annotations_by_pmid=anns_by_pmid,
            papers_by_pmid=papers_by_pmid,
        ))

    # Write report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")

    md_path = out_dir / f"cochrane_comparison_{date_str}.md"
    report = generate_markdown_report(evals, timestamp=datetime.now().isoformat())
    md_path.write_text(report)
    logger.info(f"Markdown report: {md_path}")

    json_path = out_dir / f"cochrane_comparison_{date_str}.json"
    json_path.write_text(json.dumps(
        [
            {
                "model_id": e.model_id,
                "n_papers": e.n_papers,
                "overall_kappa_raw": e.overall_kappa_raw,
                "overall_kappa_adjusted": e.overall_kappa_adjusted,
                "methodology_kappa": e.methodology_kappa,
                "outcome_reporting_kappa": e.outcome_reporting_kappa,
                "coi_only_high_count": e.coi_only_high_count,
                "coi_only_high_pmids": e.coi_only_high_pmids,
                "overall_pairs": [
                    {"pmid": p, "bb": _3LEVEL_NAMES[b], "cochrane": _3LEVEL_NAMES[c]}
                    for p, b, c in e.overall_pairs
                ],
            }
            for e in evals
        ],
        indent=2,
    ))
    logger.info(f"JSON report: {json_path}")

    csv_path = out_dir / f"cochrane_comparison_{date_str}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "value"])
        for e in evals:
            writer.writerow([e.model_id, "overall_kappa_raw", f"{e.overall_kappa_raw:.4f}"])
            writer.writerow([e.model_id, "overall_kappa_adjusted", f"{e.overall_kappa_adjusted:.4f}"])
            writer.writerow([e.model_id, "methodology_kappa", f"{e.methodology_kappa:.4f}"])
            writer.writerow([e.model_id, "outcome_reporting_kappa", f"{e.outcome_reporting_kappa:.4f}"])
            writer.writerow([e.model_id, "coi_only_high_count", e.coi_only_high_count])
            writer.writerow([e.model_id, "n_papers", e.n_papers])
    logger.info(f"CSV report: {csv_path}")

    print()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

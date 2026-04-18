"""Cochrane RoB 2 faithfulness evaluation harness.

The headline validation for biasbuster's multi-methodology claim: when
the user picks ``--methodology=cochrane_rob2`` on a Cochrane-reviewed
RCT, how closely does the model reproduce the Cochrane expert
assessors' per-domain and overall RoB 2 judgements?

Ground truth lives on the ``papers`` table (populated when Cochrane
reviews are ingested by ``collectors.cochrane_rob``): ``overall_rob``
plus per-domain columns (``randomization_bias``, ``deviation_bias``,
``missing_outcome_bias``, ``measurement_bias``, ``reporting_bias``).
Predictions live in the ``annotations`` table with
``methodology='cochrane_rob2'`` and the per-outcome / per-domain
judgements preserved in the JSON blob.

Report structure (Markdown + JSON sidecar):

1. **Coverage** — how many papers have both expert ratings and a
   cochrane_rob2 prediction for the chosen model.
2. **Overall agreement** — exact-match rate, 3x3 confusion matrix
   (low / some_concerns / high), linear-weighted Cohen's kappa.
3. **Per-domain agreement** — same metrics, per RoB 2 domain.
4. **Discrepancy table** — PMIDs where the prediction disagrees with
   expert ratings, sorted by severity of disagreement (low-vs-high
   pairs worst).

Run as a module::

    uv run python -m biasbuster.evaluation.rob2_faithfulness \\
        --model anthropic --output dataset/annotation_comparison

Emits ``rob2_faithfulness_<model>_<timestamp>.md`` + ``.json``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from biasbuster.database import Database
from biasbuster.methodologies.cochrane_rob2 import (
    METHODOLOGY_VERSION,
    evaluation_mapping_to_ground_truth,
)
from biasbuster.methodologies.cochrane_rob2.schema import (
    ROB2_DOMAIN_DISPLAY,
    ROB2_DOMAIN_SLUGS,
)

logger = logging.getLogger(__name__)

# Stable ordinal ranks for weighted-kappa + MAE computation. The RoB 2
# scale is ordinal: low < some_concerns < high, so a low/high
# disagreement is worse than a low/some_concerns disagreement.
_JUDGEMENT_RANK: dict[str, int] = {
    "low": 0,
    "some_concerns": 1,
    "high": 2,
}

# Display order for confusion matrices — canonical Cochrane order.
_JUDGEMENT_ORDER: tuple[str, ...] = ("low", "some_concerns", "high")


# ---- Data gathering ----------------------------------------------------

@dataclass
class PairedPaper:
    """One paper with both expert ground truth and a model prediction."""

    pmid: str
    title: str
    expert: dict  # {"overall": <j>, "domains": {slug: <j>, ...}}
    prediction: dict  # {"overall": <j>, "domains": {slug: <j>, ...}}
    # Preserve the raw annotation row in case the report wants to surface
    # per-domain rationales or evidence quotes in a future iteration.
    raw_annotation: dict = field(default_factory=dict)


def collect_paired_papers(
    db: Database, model_name: str,
) -> list[PairedPaper]:
    """Return every paper with both expert RoB 2 ground truth and a model
    prediction under ``methodology='cochrane_rob2'`` for the chosen model.

    Papers lacking either side are silently excluded from the output
    (the coverage stats in the report quantify how many were dropped).
    """
    annotations = db.get_annotations(
        model_name=model_name, methodology="cochrane_rob2",
    )
    if not annotations:
        return []
    paired: list[PairedPaper] = []
    for ann in annotations:
        pmid = ann.get("pmid", "")
        if not pmid:
            continue
        paper = db.get_paper(pmid)
        if paper is None:
            logger.warning(
                "annotation for PMID %s references a paper not in the DB; "
                "skipping", pmid,
            )
            continue
        expert = evaluation_mapping_to_ground_truth(paper)
        if expert is None:
            # Missing or incomplete Cochrane expert ratings — paper
            # cannot be scored. Not a bug; documented in the report.
            continue
        prediction = _extract_prediction_view(ann)
        if prediction is None:
            logger.warning(
                "annotation for PMID %s does not parse as a cochrane_rob2 "
                "assessment; skipping", pmid,
            )
            continue
        paired.append(PairedPaper(
            pmid=pmid,
            title=paper.get("title", ""),
            expert=expert,
            prediction=prediction,
            raw_annotation=ann,
        ))
    return paired


def _extract_prediction_view(ann: dict) -> Optional[dict]:
    """Normalise a stored cochrane_rob2 annotation into the ground-truth shape.

    The stored annotation blob is a :class:`RoB2Assessment` serialised
    via ``to_dict()`` (list of outcomes, each with a domains dict). For
    comparison against the expert mapping we reduce it to
    ``{"overall": <j>, "domains": {slug: <j>}}`` using the worst-wins
    rule across outcomes (consistent with what's stored in the DB's
    ``overall_severity`` column).
    """
    overall = ann.get("worst_across_outcomes") or ann.get("overall_severity")
    if not isinstance(overall, str) or overall not in _JUDGEMENT_RANK:
        return None
    outcomes = ann.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        return None
    # Per-domain worst across outcomes (matches worst_case_across_outcomes
    # semantics so we don't accidentally read one outcome's low when
    # another outcome has high for the same domain).
    by_domain: dict[str, str] = {}
    for outcome in outcomes:
        domains = outcome.get("domains") or {}
        for slug in ROB2_DOMAIN_SLUGS:
            dj = domains.get(slug)
            if not isinstance(dj, dict):
                continue
            current = dj.get("judgement")
            if current not in _JUDGEMENT_RANK:
                continue
            prior = by_domain.get(slug)
            if prior is None or _JUDGEMENT_RANK[current] > _JUDGEMENT_RANK[prior]:
                by_domain[slug] = current
    if set(by_domain) != set(ROB2_DOMAIN_SLUGS):
        return None
    return {"overall": overall, "domains": by_domain}


# ---- Metrics -----------------------------------------------------------

@dataclass
class JudgementSeries:
    """Paired expert/prediction series for a single (domain or overall) field.

    Stored as parallel lists of string judgements; the metric helpers
    convert to ordinal ranks lazily so callers can still see the source
    strings in reports.
    """

    expert: list[str] = field(default_factory=list)
    prediction: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.expert)

    def confusion(self) -> dict[str, dict[str, int]]:
        """3x3 confusion matrix keyed by expert × prediction."""
        matrix = {g: {p: 0 for p in _JUDGEMENT_ORDER} for g in _JUDGEMENT_ORDER}
        for g, p in zip(self.expert, self.prediction):
            if g in matrix and p in matrix[g]:
                matrix[g][p] += 1
        return matrix

    def exact_match(self) -> float:
        if not self.expert:
            return 0.0
        hits = sum(1 for g, p in zip(self.expert, self.prediction) if g == p)
        return hits / len(self.expert)

    def within_one(self) -> float:
        """Proportion within one ordinal step (low→some_concerns or vice versa)."""
        if not self.expert:
            return 0.0
        close = sum(
            1 for g, p in zip(self.expert, self.prediction)
            if abs(_JUDGEMENT_RANK[g] - _JUDGEMENT_RANK[p]) <= 1
        )
        return close / len(self.expert)

    def mean_abs_error(self) -> float:
        """Ordinal MAE on the 0/1/2 scale. Low/high is MAE=2; a perfect run is 0."""
        if not self.expert:
            return 0.0
        total = sum(
            abs(_JUDGEMENT_RANK[g] - _JUDGEMENT_RANK[p])
            for g, p in zip(self.expert, self.prediction)
        )
        return total / len(self.expert)

    def weighted_kappa(self) -> float:
        """Linear-weighted Cohen's kappa on the 3-level ordinal scale.

        Returns 1.0 for perfect agreement, 0.0 for chance-level agreement,
        negative for systematic anti-agreement. Uses the same linear-
        weight scheme as ``biasbuster.evaluation.metrics.OrdinalMetrics``
        so reports from both methodologies are comparable in spirit
        (while not directly comparable because the scales differ).
        """
        n = self.n
        if n == 0:
            return 0.0
        n_cats = len(_JUDGEMENT_ORDER)
        rank = {j: i for i, j in enumerate(_JUDGEMENT_ORDER)}
        observed = [[0] * n_cats for _ in range(n_cats)]
        for g, p in zip(self.expert, self.prediction):
            observed[rank[g]][rank[p]] += 1
        row_totals = [sum(row) for row in observed]
        col_totals = [sum(observed[i][j] for i in range(n_cats))
                      for j in range(n_cats)]
        weights = [
            [abs(i - j) / (n_cats - 1) for j in range(n_cats)]
            for i in range(n_cats)
        ]
        obs_w = sum(
            weights[i][j] * observed[i][j]
            for i in range(n_cats) for j in range(n_cats)
        ) / n
        exp_w = sum(
            weights[i][j] * row_totals[i] * col_totals[j]
            for i in range(n_cats) for j in range(n_cats)
        ) / (n * n)
        if exp_w == 0:
            return 1.0 if obs_w == 0 else 0.0
        return 1.0 - (obs_w / exp_w)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "exact_match": round(self.exact_match(), 4),
            "within_one": round(self.within_one(), 4),
            "mean_abs_error": round(self.mean_abs_error(), 4),
            "weighted_kappa": round(self.weighted_kappa(), 4),
            "confusion": self.confusion(),
        }


@dataclass
class FaithfulnessReport:
    """Aggregate metrics + per-paper discrepancies for a faithfulness run."""

    model_name: str
    methodology_version: str
    n_paired: int
    n_model_annotations: int
    overall: JudgementSeries
    per_domain: dict[str, JudgementSeries]
    discrepancies: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "methodology_version": self.methodology_version,
            "n_paired": self.n_paired,
            "n_model_annotations": self.n_model_annotations,
            "overall": self.overall.to_dict(),
            "per_domain": {
                slug: series.to_dict()
                for slug, series in self.per_domain.items()
            },
            "discrepancies": self.discrepancies,
        }


def build_report(
    paired: list[PairedPaper], model_name: str,
    n_model_annotations: int,
) -> FaithfulnessReport:
    """Compute the report from a list of paired papers.

    Discrepancies are sorted worst-first by ordinal distance on the
    overall judgement, ties broken by pmid for reproducibility.
    """
    overall_series = JudgementSeries()
    per_domain: dict[str, JudgementSeries] = {
        slug: JudgementSeries() for slug in ROB2_DOMAIN_SLUGS
    }
    for paper in paired:
        overall_series.expert.append(paper.expert["overall"])
        overall_series.prediction.append(paper.prediction["overall"])
        for slug in ROB2_DOMAIN_SLUGS:
            per_domain[slug].expert.append(paper.expert["domains"][slug])
            per_domain[slug].prediction.append(paper.prediction["domains"][slug])

    discrepancies = []
    for paper in paired:
        expert_o = paper.expert["overall"]
        pred_o = paper.prediction["overall"]
        if expert_o == pred_o:
            continue
        distance = abs(_JUDGEMENT_RANK[expert_o] - _JUDGEMENT_RANK[pred_o])
        per_domain_disagreements = [
            {
                "domain": slug,
                "expert": paper.expert["domains"][slug],
                "prediction": paper.prediction["domains"][slug],
            }
            for slug in ROB2_DOMAIN_SLUGS
            if paper.expert["domains"][slug] != paper.prediction["domains"][slug]
        ]
        discrepancies.append({
            "pmid": paper.pmid,
            "title": paper.title,
            "expert_overall": expert_o,
            "predicted_overall": pred_o,
            "distance": distance,
            "per_domain_disagreements": per_domain_disagreements,
        })
    discrepancies.sort(key=lambda d: (-d["distance"], d["pmid"]))

    return FaithfulnessReport(
        model_name=model_name,
        methodology_version=METHODOLOGY_VERSION,
        n_paired=len(paired),
        n_model_annotations=n_model_annotations,
        overall=overall_series,
        per_domain=per_domain,
        discrepancies=discrepancies,
    )


# ---- Markdown rendering -----------------------------------------------

def _md_confusion_table(conf: dict[str, dict[str, int]]) -> str:
    """Render a 3x3 confusion matrix as a Markdown table."""
    header = "| expert ↓ / predicted → | " + " | ".join(_JUDGEMENT_ORDER) + " |"
    sep = "|" + "---|" * (len(_JUDGEMENT_ORDER) + 1)
    rows = [header, sep]
    for expert_j in _JUDGEMENT_ORDER:
        cells = [str(conf[expert_j][pred_j]) for pred_j in _JUDGEMENT_ORDER]
        rows.append(f"| **{expert_j}** | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _md_metrics_line(series: JudgementSeries) -> str:
    return (
        f"n={series.n}, "
        f"exact_match={series.exact_match():.3f}, "
        f"within_one={series.within_one():.3f}, "
        f"MAE={series.mean_abs_error():.3f}, "
        f"weighted_kappa={series.weighted_kappa():.3f}"
    )


def render_markdown(report: FaithfulnessReport) -> str:
    """Produce the Markdown report body."""
    lines: list[str] = []
    lines.append(f"# Cochrane RoB 2 faithfulness — `{report.model_name}`")
    lines.append("")
    lines.append(
        f"Methodology version: `{report.methodology_version}`. "
        f"Model annotations found: **{report.n_model_annotations}**. "
        f"Paired with Cochrane expert ground truth: **{report.n_paired}**."
    )
    if report.n_paired == 0:
        lines.append("")
        lines.append(
            "_No paired papers — cannot compute agreement. Either no "
            "model annotations exist under `methodology='cochrane_rob2'` "
            "for this model, or none of the annotated papers carry "
            "Cochrane expert RoB 2 ratings._"
        )
        return "\n".join(lines)

    lines.append("")
    lines.append("## Overall judgement agreement")
    lines.append("")
    lines.append(_md_metrics_line(report.overall))
    lines.append("")
    lines.append(_md_confusion_table(report.overall.confusion()))

    lines.append("")
    lines.append("## Per-domain agreement")
    for slug in ROB2_DOMAIN_SLUGS:
        series = report.per_domain[slug]
        lines.append("")
        lines.append(f"### {ROB2_DOMAIN_DISPLAY[slug]}")
        lines.append("")
        lines.append(_md_metrics_line(series))
        lines.append("")
        lines.append(_md_confusion_table(series.confusion()))

    lines.append("")
    lines.append("## Discrepancies (worst-first)")
    if not report.discrepancies:
        lines.append("")
        lines.append("_None — every paired paper's overall judgement matched._")
    else:
        lines.append("")
        lines.append(
            f"Showing {len(report.discrepancies)} paper(s) where the "
            "overall judgement differs. Distance is ordinal: 1 = adjacent "
            "category (low↔some_concerns or some_concerns↔high); 2 = "
            "extreme disagreement (low↔high)."
        )
        lines.append("")
        lines.append("| PMID | distance | expert | predicted | domains in disagreement |")
        lines.append("|---|---|---|---|---|")
        for d in report.discrepancies:
            domain_summary = "; ".join(
                f"{x['domain']} (expert={x['expert']}, pred={x['prediction']})"
                for x in d["per_domain_disagreements"]
            ) or "—"
            lines.append(
                f"| {d['pmid']} | {d['distance']} | {d['expert_overall']} | "
                f"{d['predicted_overall']} | {domain_summary} |"
            )
    lines.append("")
    return "\n".join(lines)


# ---- CLI entry point --------------------------------------------------

def _write_outputs(
    report: FaithfulnessReport, output_dir: Path,
) -> tuple[Path, Path]:
    """Write the Markdown + JSON files and return their paths."""
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"rob2_faithfulness_{report.model_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8",
    )
    return md_path, json_path


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--model", required=True,
        help="Annotator backend slug (e.g. 'anthropic', 'deepseek') to score.",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("dataset/annotation_comparison"),
        help="Directory to write the Markdown + JSON reports.",
    )
    p.add_argument(
        "--db", type=Path, default=None,
        help="Override the DB path (defaults to config.db_path).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)

    db_path: Path
    if args.db is not None:
        db_path = args.db
    else:
        from config import Config
        db_path = Path(Config().db_path)

    db = Database(db_path)
    db.initialize()
    try:
        annotations = db.get_annotations(
            model_name=args.model, methodology="cochrane_rob2",
        )
        paired = collect_paired_papers(db, args.model)
    finally:
        db.close()

    report = build_report(
        paired=paired,
        model_name=args.model,
        n_model_annotations=len(annotations),
    )
    md_path, json_path = _write_outputs(report, args.output)
    logger.info("Markdown report: %s", md_path)
    logger.info("JSON sidecar: %s", json_path)
    if report.n_paired > 0:
        logger.info(
            "overall_exact_match=%.3f, weighted_kappa=%.3f (n=%d paired)",
            report.overall.exact_match(),
            report.overall.weighted_kappa(),
            report.n_paired,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

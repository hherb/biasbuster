"""Methodology-agnostic faithfulness evaluation harness.

Scores an active methodology's stored predictions against human/expert
ground truth held in ``expert_methodology_ratings``. Every methodology
that wants to participate exposes a :class:`FaithfulnessSpec` as the
module-level ``FAITHFULNESS_SPEC`` constant; the harness picks it up
by methodology slug and renders a uniform Markdown + JSON report.

Design choices:

- **Expert ground truth in one place.** Historically RoB 2 expert
  ratings lived as hard-coded columns on ``papers``; QUADAS-2's live
  in ``expert_methodology_ratings``. The RoB 2 columns are backfilled
  into the new table by :mod:`scripts.backfill_rob2_expert_ratings`
  so this harness never branches on methodology when reading truth.
- **Ordinal scale is per-methodology.** RoB 2: low / some_concerns /
  high. QUADAS-2: low / unclear / high. ``judgement_order`` on the
  spec pins the ordering; metrics use the index as the ordinal rank.
- **Prediction shape normalisation is methodology-specific.** RoB 2
  stores a per-outcome list (worst-wins across outcomes); QUADAS-2
  stores a flat per-domain judgement. Each spec provides a
  ``load_prediction_view`` callable that collapses its methodology's
  stored annotation to the common ``{"overall", "domains": {slug: j}}``
  shape.
- **Bias-only (v1).** QUADAS-2 publishes bias *and* applicability
  per domain. The ingested expert ratings only carry bias (the JATS
  tables we parse don't split applicability), so the harness scores
  bias only. Applicability is a planned parallel set of
  :class:`JudgementSeries`.

Run as a module::

    uv run python -m biasbuster.evaluation.methodology_faithfulness \\
        --methodology quadas_2 --model anthropic \\
        --db dataset/biasbuster_recovered.db
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from biasbuster.database import Database

logger = logging.getLogger(__name__)


# ---- Per-methodology configuration ------------------------------------

#: Normalised prediction view shared across methodologies. The
#: ``overall`` judgement and each per-domain judgement is a string
#: drawn from the methodology's ``judgement_order``. Extra keys may
#: appear in a future applicability-aware revision.
PredictionView = dict[str, Any]

#: Expert ground-truth view, same shape as ``PredictionView``.
ExpertView = dict[str, Any]


@dataclass(frozen=True)
class FaithfulnessSpec:
    """Methodology-specific knobs for the generic harness.

    A methodology module opts in by exporting this as the module-level
    ``FAITHFULNESS_SPEC`` constant. The harness imports by methodology
    slug and relies on the spec for all methodology-specific behaviour.
    """

    methodology: str
    methodology_version: str
    display_name: str
    judgement_order: tuple[str, ...]
    domain_slugs: tuple[str, ...]
    domain_display: dict[str, str]
    #: Collapse a stored annotation JSON into the shared prediction
    #: shape. Returns None for un-parseable or incomplete annotations
    #: — those papers drop from the report's denominator.
    load_prediction_view: Callable[[dict], Optional[PredictionView]]


def get_spec(methodology: str) -> FaithfulnessSpec:
    """Look up the :class:`FaithfulnessSpec` for a registered methodology.

    Raises ``LookupError`` when the methodology's package doesn't export
    ``FAITHFULNESS_SPEC`` — that's always a registration bug, not user
    input, so the error is eager rather than wrapped.
    """
    module = importlib.import_module(f"biasbuster.methodologies.{methodology}")
    spec: Optional[FaithfulnessSpec] = getattr(module, "FAITHFULNESS_SPEC", None)
    if spec is None:
        raise LookupError(
            f"methodology {methodology!r} does not expose FAITHFULNESS_SPEC; "
            "the methodology's package must define it at module scope"
        )
    if spec.methodology != methodology:
        raise LookupError(
            f"FAITHFULNESS_SPEC.methodology={spec.methodology!r} does not "
            f"match imported path {methodology!r} — package misnamed?"
        )
    return spec


# ---- Expert-rating loader ---------------------------------------------

def load_expert_view(
    db: Database, spec: FaithfulnessSpec, pmid: str,
) -> Optional[ExpertView]:
    """Read ground truth from ``expert_methodology_ratings`` for one paper.

    Selection rule when multiple rows exist for the same ``(methodology,
    pmid)`` (possible when different reviews rate the same study):

    1. A ``verified=1`` row beats any unverified row.
    2. Otherwise return the first row the DB orders by PK.

    Returns ``None`` when no row exists, when any domain is missing, or
    when a rating is outside the methodology's ``judgement_order``. The
    caller surfaces such papers as unpaired in the report.
    """
    rows = db.get_expert_ratings(
        methodology=spec.methodology, pmid=pmid,
    )
    if not rows:
        return None
    verified = [r for r in rows if r.get("verified") == 1]
    chosen = verified[0] if verified else rows[0]
    if len(rows) > 1:
        logger.info(
            "PMID %s has %d expert ratings for %s; picked rating_source=%s "
            "(verified=%s)",
            pmid, len(rows), spec.methodology,
            chosen["rating_source"], chosen["verified"],
        )
    overall = chosen.get("overall_rating")
    if overall not in spec.judgement_order:
        return None

    raw_domains = chosen.get("domain_ratings") or {}
    if not isinstance(raw_domains, dict):
        return None
    domains: dict[str, str] = {}
    for slug in spec.domain_slugs:
        cell = raw_domains.get(slug)
        if not isinstance(cell, dict):
            return None
        bias = cell.get("bias")
        if bias not in spec.judgement_order:
            return None
        domains[slug] = bias
    return {
        "overall": overall,
        "domains": domains,
        "rating_source": chosen["rating_source"],
        "verified": bool(chosen.get("verified")),
    }


# ---- Paired data ------------------------------------------------------

@dataclass
class PairedPaper:
    """One paper with both expert ground truth and a model prediction."""

    pmid: str
    title: str
    expert: ExpertView
    prediction: PredictionView
    raw_annotation: dict = field(default_factory=dict)


def collect_paired_papers(
    db: Database, spec: FaithfulnessSpec, model_name: str,
) -> list[PairedPaper]:
    """Join annotations × expert ratings for a single methodology + model.

    Papers missing either side are silently excluded; coverage stats in
    the report quantify the drop. Warnings are logged only for truly
    suspicious states (annotation references a PMID not in ``papers``).
    """
    annotations = db.get_annotations(
        model_name=model_name, methodology=spec.methodology,
    )
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
        expert = load_expert_view(db, spec, pmid)
        if expert is None:
            continue
        prediction = spec.load_prediction_view(ann)
        if prediction is None:
            logger.warning(
                "annotation for PMID %s does not parse as a %s assessment; "
                "skipping", pmid, spec.methodology,
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


# ---- Metrics ----------------------------------------------------------

@dataclass
class JudgementSeries:
    """Paired expert/prediction series for a single domain or overall.

    The ordinal scale is carried on the instance so the same metric
    helpers serve methodologies with different vocabularies (RoB 2:
    low/some_concerns/high; QUADAS-2: low/unclear/high).
    """

    judgement_order: tuple[str, ...] = ("low", "some_concerns", "high")
    expert: list[str] = field(default_factory=list)
    prediction: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.expert)

    def _rank(self, j: str) -> int:
        # Position in the ordinal scale. KeyError is a test bug (the
        # caller pushed a judgement the spec didn't declare) not
        # runtime input, so we let it propagate.
        return self.judgement_order.index(j)

    def confusion(self) -> dict[str, dict[str, int]]:
        """3x3 confusion matrix keyed by expert × prediction."""
        order = self.judgement_order
        matrix: dict[str, dict[str, int]] = {
            g: {p: 0 for p in order} for g in order
        }
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
        """Proportion within one ordinal step."""
        if not self.expert:
            return 0.0
        close = sum(
            1 for g, p in zip(self.expert, self.prediction)
            if abs(self._rank(g) - self._rank(p)) <= 1
        )
        return close / len(self.expert)

    def mean_abs_error(self) -> float:
        """Ordinal MAE on the 0..(n-1) scale."""
        if not self.expert:
            return 0.0
        total = sum(
            abs(self._rank(g) - self._rank(p))
            for g, p in zip(self.expert, self.prediction)
        )
        return total / len(self.expert)

    def weighted_kappa(self) -> float:
        """Linear-weighted Cohen's kappa on the ordinal scale."""
        n = self.n
        if n == 0:
            return 0.0
        order = self.judgement_order
        n_cats = len(order)
        rank = {j: i for i, j in enumerate(order)}
        observed = [[0] * n_cats for _ in range(n_cats)]
        for g, p in zip(self.expert, self.prediction):
            observed[rank[g]][rank[p]] += 1
        row_totals = [sum(row) for row in observed]
        col_totals = [
            sum(observed[i][j] for i in range(n_cats))
            for j in range(n_cats)
        ]
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


# ---- Report ----------------------------------------------------------

@dataclass
class FaithfulnessReport:
    """Aggregate metrics + per-paper discrepancies for one harness run."""

    methodology: str
    methodology_version: str
    display_name: str
    model_name: str
    n_paired: int
    n_model_annotations: int
    overall: JudgementSeries
    per_domain: dict[str, JudgementSeries]
    discrepancies: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "methodology": self.methodology,
            "methodology_version": self.methodology_version,
            "display_name": self.display_name,
            "model_name": self.model_name,
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
    paired: list[PairedPaper],
    spec: FaithfulnessSpec,
    model_name: str,
    n_model_annotations: int,
) -> FaithfulnessReport:
    """Compute aggregate series + discrepancy list from paired papers.

    Discrepancies are sorted worst-first by ordinal distance on the
    overall judgement, ties broken by pmid for reproducibility.
    """
    overall_series = JudgementSeries(judgement_order=spec.judgement_order)
    per_domain: dict[str, JudgementSeries] = {
        slug: JudgementSeries(judgement_order=spec.judgement_order)
        for slug in spec.domain_slugs
    }
    for paper in paired:
        overall_series.expert.append(paper.expert["overall"])
        overall_series.prediction.append(paper.prediction["overall"])
        for slug in spec.domain_slugs:
            per_domain[slug].expert.append(paper.expert["domains"][slug])
            per_domain[slug].prediction.append(
                paper.prediction["domains"][slug]
            )

    rank = {j: i for i, j in enumerate(spec.judgement_order)}
    discrepancies: list[dict[str, Any]] = []
    for paper in paired:
        expert_o = paper.expert["overall"]
        pred_o = paper.prediction["overall"]
        if expert_o == pred_o:
            continue
        distance = abs(rank[expert_o] - rank[pred_o])
        per_domain_disagreements = [
            {
                "domain": slug,
                "expert": paper.expert["domains"][slug],
                "prediction": paper.prediction["domains"][slug],
            }
            for slug in spec.domain_slugs
            if paper.expert["domains"][slug]
            != paper.prediction["domains"][slug]
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
        methodology=spec.methodology,
        methodology_version=spec.methodology_version,
        display_name=spec.display_name,
        model_name=model_name,
        n_paired=len(paired),
        n_model_annotations=n_model_annotations,
        overall=overall_series,
        per_domain=per_domain,
        discrepancies=discrepancies,
    )


# ---- Markdown rendering ------------------------------------------------

def _md_confusion_table(
    conf: dict[str, dict[str, int]], order: tuple[str, ...],
) -> str:
    header = "| expert ↓ / predicted → | " + " | ".join(order) + " |"
    sep = "|" + "---|" * (len(order) + 1)
    rows = [header, sep]
    for expert_j in order:
        cells = [str(conf[expert_j][pred_j]) for pred_j in order]
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


def render_markdown(
    report: FaithfulnessReport, spec: FaithfulnessSpec,
) -> str:
    """Produce the Markdown report body."""
    lines: list[str] = []
    lines.append(
        f"# {report.display_name} faithfulness — `{report.model_name}`"
    )
    lines.append("")
    lines.append(
        f"Methodology version: `{report.methodology_version}`. "
        f"Model annotations found: **{report.n_model_annotations}**. "
        f"Paired with expert ground truth: **{report.n_paired}**."
    )
    if report.n_paired == 0:
        lines.append("")
        lines.append(
            "_No paired papers — cannot compute agreement. Either no "
            f"model annotations exist under `methodology='{report.methodology}'` "
            "for this model, or none of the annotated papers carry "
            "expert ratings in `expert_methodology_ratings`._"
        )
        return "\n".join(lines)

    lines.append("")
    lines.append("## Overall judgement agreement")
    lines.append("")
    lines.append(_md_metrics_line(report.overall))
    lines.append("")
    lines.append(_md_confusion_table(
        report.overall.confusion(), spec.judgement_order,
    ))

    lines.append("")
    lines.append("## Per-domain agreement")
    for slug in spec.domain_slugs:
        series = report.per_domain[slug]
        lines.append("")
        lines.append(f"### {spec.domain_display[slug]}")
        lines.append("")
        lines.append(_md_metrics_line(series))
        lines.append("")
        lines.append(_md_confusion_table(
            series.confusion(), spec.judgement_order,
        ))

    lines.append("")
    lines.append("## Discrepancies (worst-first)")
    if not report.discrepancies:
        lines.append("")
        lines.append(
            "_None — every paired paper's overall judgement matched._"
        )
    else:
        lines.append("")
        lines.append(
            "| distance | PMID | expert overall | predicted overall | "
            "per-domain disagreements |"
        )
        lines.append("|---|---|---|---|---|")
        for d in report.discrepancies:
            per_dom = "; ".join(
                f"{x['domain']}: {x['expert']}→{x['prediction']}"
                for x in d["per_domain_disagreements"]
            ) or "—"
            lines.append(
                f"| {d['distance']} | `{d['pmid']}` | {d['expert_overall']}"
                f" | {d['predicted_overall']} | {per_dom} |"
            )
    return "\n".join(lines)


# ---- End-to-end runner ------------------------------------------------

def run_faithfulness(
    db_path: Path,
    methodology: str,
    model_name: str,
    output_dir: Optional[Path] = None,
) -> tuple[FaithfulnessReport, Optional[Path], Optional[Path]]:
    """Run the harness end-to-end and optionally write Markdown + JSON.

    Returns ``(report, markdown_path, json_path)``. When ``output_dir``
    is ``None`` the paths are ``None`` and nothing is written — useful
    for tests and callers that want to render the report themselves.
    """
    spec = get_spec(methodology)
    db = Database(db_path)
    try:
        db.initialize()
        n_model_annotations = len(
            db.get_annotations(model_name=model_name, methodology=methodology)
        )
        paired = collect_paired_papers(db, spec, model_name)
    finally:
        db.close()
    report = build_report(paired, spec, model_name, n_model_annotations)
    md_path: Optional[Path] = None
    json_path: Optional[Path] = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{methodology}_faithfulness_{model_name}_{stamp}"
        md_path = output_dir / f"{stem}.md"
        json_path = output_dir / f"{stem}.json"
        md_path.write_text(render_markdown(report, spec), encoding="utf-8")
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8",
        )
    return report, md_path, json_path


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--methodology", required=True,
        help="Methodology slug (e.g. 'cochrane_rob2', 'quadas_2').",
    )
    p.add_argument(
        "--model", required=True,
        help="LLM backend identifier whose annotations should be scored.",
    )
    p.add_argument(
        "--db", type=Path,
        default=Path("dataset/biasbuster.db"),
        help="DB holding predictions + expert ratings.",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("dataset/annotation_comparison"),
        help="Output directory for the Markdown + JSON report.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    report, md_path, json_path = run_faithfulness(
        args.db, args.methodology, args.model, args.output,
    )
    print(f"Methodology:        {report.display_name}")
    print(f"Model:              {report.model_name}")
    print(f"Model annotations:  {report.n_model_annotations}")
    print(f"Paired (n):         {report.n_paired}")
    if report.n_paired:
        print(f"Overall exact:      {report.overall.exact_match():.3f}")
        print(f"Overall kappa:      {report.overall.weighted_kappa():.3f}")
    if md_path is not None:
        print(f"Markdown: {md_path}")
    if json_path is not None:
        print(f"JSON:     {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

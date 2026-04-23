"""Generate preprint supplementary materials S1–S3 from the annotations DB.

Outputs (under ``docs/papers/supplements/``):

  S1_annotations.jsonl           — one row per AI annotation, with the
                                   paired expert rating attached for
                                   easy side-by-side audit.
  S2_expert_ratings.csv          — flat export of the expert ratings
                                   the AI was benchmarked against.
  S3_audit_table.md              — per (pmid, methodology, domain)
                                   table showing AI rating, expert
                                   rating, and the AI's signalling
                                   answers + justification so a reader
                                   can independently apply the
                                   Cochrane / Whiting decision rule.

Run from the repo root::

    uv run python scripts/dump_preprint_supplements.py \\
        --db dataset/biasbuster_recovered.db
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from biasbuster.database import Database

logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_OUT: Path = REPO_ROOT / "docs" / "papers" / "supplements"

# Display order for rows in the audit table.
_ROB2_DOMAIN_ORDER: tuple[str, ...] = (
    "randomization",
    "deviations_from_interventions",
    "missing_outcome_data",
    "outcome_measurement",
    "selection_of_reported_result",
)
_QUADAS2_DOMAIN_ORDER: tuple[str, ...] = (
    "patient_selection",
    "index_test",
    "reference_standard",
    "flow_and_timing",
)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _fetch_annotations(db: Database) -> list[dict]:
    """Every (pmid, methodology, model) annotation currently in the DB."""
    rows = db.conn.execute(
        """SELECT a.pmid, a.model_name, a.methodology, a.methodology_version,
                  a.overall_severity, a.annotated_at, a.annotation,
                  p.title
             FROM annotations a
             LEFT JOIN papers p USING (pmid)
             WHERE a.methodology IN ('cochrane_rob2', 'quadas_2')
             ORDER BY a.methodology, a.pmid""",
    ).fetchall()
    return [dict(r) for r in rows]


def _fetch_expert_ratings(db: Database) -> list[dict]:
    """Every expert rating row for cochrane_rob2 + quadas_2."""
    rows = db.conn.execute(
        """SELECT methodology, rating_source, study_label, pmid, doi,
                  methodology_version, source_reference,
                  domain_ratings, overall_rating, verified, added_by,
                  added_at
             FROM expert_methodology_ratings
             WHERE methodology IN ('cochrane_rob2', 'quadas_2')
             ORDER BY methodology, rating_source, study_label""",
    ).fetchall()
    return [dict(r) for r in rows]


def _expert_index(
    experts: list[dict],
) -> dict[tuple[str, str], dict]:
    """(methodology, pmid) → latest expert row for that paper.

    If multiple expert sources rate the same paper, the later-added
    row wins. This matches the harness's de-duplication policy.
    """
    index: dict[tuple[str, str], dict] = {}
    for row in experts:
        pmid = row.get("pmid")
        if not pmid:
            continue
        key = (row["methodology"], str(pmid))
        index[key] = row
    return index


def _parse_json_field(value: Any) -> Any:
    """Tolerant JSON decode: accept str or already-parsed, return raw on fail."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


# ---------------------------------------------------------------------------
# S1 — annotations with paired expert ratings
# ---------------------------------------------------------------------------

def write_s1(
    annotations: list[dict], expert_idx: dict[tuple[str, str], dict],
    out_path: Path,
) -> int:
    """JSONL: one record per AI annotation, expert row attached in-line."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ann in annotations:
            record = {
                "pmid": ann["pmid"],
                "title": ann.get("title"),
                "methodology": ann["methodology"],
                "methodology_version": ann.get("methodology_version"),
                "model_name": ann["model_name"],
                "overall_severity": ann.get("overall_severity"),
                "annotated_at": ann.get("annotated_at"),
                "ai_annotation": _parse_json_field(ann.get("annotation")),
                "expert_rating": None,
            }
            expert = expert_idx.get((ann["methodology"], str(ann["pmid"])))
            if expert is not None:
                record["expert_rating"] = {
                    "rating_source": expert.get("rating_source"),
                    "source_reference": expert.get("source_reference"),
                    "study_label": expert.get("study_label"),
                    "overall_rating": expert.get("overall_rating"),
                    "domain_ratings": _parse_json_field(
                        expert.get("domain_ratings"),
                    ),
                    "verified": bool(expert.get("verified")),
                    "added_by": expert.get("added_by"),
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# S2 — expert ratings as a flat CSV
# ---------------------------------------------------------------------------

def write_s2(experts: list[dict], out_path: Path) -> int:
    """One row per (methodology, rating_source, study_label, domain)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "methodology", "rating_source", "source_reference",
            "study_label", "pmid", "doi", "methodology_version",
            "domain", "expert_rating", "overall_rating",
            "verified", "added_by", "added_at",
        ])
        for row in experts:
            domain_ratings = _parse_json_field(row.get("domain_ratings")) or {}
            overall = row.get("overall_rating")
            verified = 1 if row.get("verified") else 0
            base = [
                row["methodology"], row.get("rating_source", ""),
                row.get("source_reference", ""), row.get("study_label", ""),
                row.get("pmid", ""), row.get("doi", ""),
                row.get("methodology_version", ""),
            ]
            tail = [overall or "", verified, row.get("added_by", ""),
                    row.get("added_at", "")]
            if isinstance(domain_ratings, dict) and domain_ratings:
                for slug, rating in sorted(domain_ratings.items()):
                    # domain_ratings values are ``{"bias": "low"}`` etc.
                    expert_val = ""
                    if isinstance(rating, dict):
                        expert_val = rating.get("bias") or rating.get("judgement") or ""
                    elif isinstance(rating, str):
                        expert_val = rating
                    writer.writerow(base + [slug, expert_val, *tail])
                    n += 1
            else:
                # Overall-only row (e.g. ROBINS-I imports).
                writer.writerow(base + ["", "", *tail])
                n += 1
    return n


# ---------------------------------------------------------------------------
# S3 — per (pmid × domain) audit table
# ---------------------------------------------------------------------------

def _rob2_model_domain_rows(annotation: dict) -> list[dict]:
    """Flatten a RoB 2 annotation to one row per domain."""
    out: list[dict] = []
    outcomes = annotation.get("outcomes") or []
    if not outcomes:
        return out
    outcome = outcomes[0]  # decomposed assessor emits a single outcome
    domains = outcome.get("domains") or {}
    for slug in _ROB2_DOMAIN_ORDER:
        dom = domains.get(slug)
        if not isinstance(dom, dict):
            continue
        out.append({
            "domain": slug,
            "model_rating": dom.get("judgement"),
            "signalling_answers": dom.get("signalling_answers") or {},
            "justification": (dom.get("justification") or "").strip(),
        })
    return out


def _quadas2_model_domain_rows(annotation: dict) -> list[dict]:
    """Flatten a QUADAS-2 annotation to one row per domain."""
    out: list[dict] = []
    domains = annotation.get("domains") or {}
    for slug in _QUADAS2_DOMAIN_ORDER:
        dom = domains.get(slug)
        if not isinstance(dom, dict):
            continue
        out.append({
            "domain": slug,
            "model_rating": dom.get("bias_rating"),
            "signalling_answers": dom.get("signalling_answers") or {},
            "justification": (dom.get("justification") or "").strip(),
        })
    return out


def _expert_domain_rating(
    expert: Optional[dict], slug: str,
) -> Optional[str]:
    if expert is None:
        return None
    domain_ratings = _parse_json_field(expert.get("domain_ratings")) or {}
    dom = domain_ratings.get(slug) if isinstance(domain_ratings, dict) else None
    if isinstance(dom, dict):
        return dom.get("bias") or dom.get("judgement")
    if isinstance(dom, str):
        return dom
    return None


def _format_answers(answers: dict) -> str:
    """Compact single-line render for the audit table cell."""
    if not isinstance(answers, dict) or not answers:
        return "—"
    items = sorted(answers.items(), key=lambda kv: kv[0])
    return ", ".join(f"{k}={v}" for k, v in items)


def _truncate(text: str, n: int = 180) -> str:
    text = text.replace("|", "\\|").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def write_s3(
    annotations: list[dict],
    expert_idx: dict[tuple[str, str], dict],
    out_path: Path,
) -> int:
    """Per-domain audit table. Markdown, one row per (paper × domain).

    Columns designed to make the algorithm-rule check the reader
    themselves can run trivial:

      PMID | trial | domain | model | expert | match? | signalling |
      model justification (truncated)

    A reader cross-checks ``signalling`` against the Cochrane / Whiting
    decision rule in the methodology's prompt and decides whether the
    model or the expert is faithful to the algorithm.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    lines: list[str] = []
    lines.append("# Supplement S3 — Per-domain audit table\n")
    lines.append(
        "One row per (paper × domain). Reader protocol: given the "
        "*signalling* column, apply the methodology's decision rule "
        "(see `docs/ANNOTATION_JSON_SPEC.md` §3.3 for RoB 2, §4 for "
        "QUADAS-2) and decide whether the AI rating or the expert "
        "rating is consistent with the tool's algorithm.\n",
    )

    n_rows = 0
    for methodology, rows_for_method in _group_by_methodology(annotations):
        lines.append(f"\n## {methodology}\n")
        lines.append(
            "| PMID | Trial | Domain | AI | Expert | Match? | "
            "Signalling | AI justification (truncated) |",
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|",
        )
        for ann in rows_for_method:
            annotation_blob = _parse_json_field(ann.get("annotation")) or {}
            if not isinstance(annotation_blob, dict):
                continue
            if methodology == "cochrane_rob2":
                model_rows = _rob2_model_domain_rows(annotation_blob)
            else:
                model_rows = _quadas2_model_domain_rows(annotation_blob)
            expert = expert_idx.get((methodology, str(ann["pmid"])))
            # Prefer a human-readable label. The RoB 2 backfill uses pmid
            # as study_label (all-digits); fall back to the paper title in
            # that case so a reader sees "Effects of plyometric..." instead
            # of "32841300".
            trial_name = ""
            if expert is not None:
                candidate = expert.get("study_label") or ""
                if candidate and not candidate.isdigit():
                    trial_name = candidate
            if not trial_name:
                trial_name = ann.get("title") or ""
            trial_short = _truncate(trial_name, 40)

            for row in model_rows:
                slug = row["domain"]
                model_rating = row["model_rating"] or "—"
                expert_rating = _expert_domain_rating(expert, slug) or "—"
                match = "✓" if (
                    model_rating != "—"
                    and expert_rating != "—"
                    and model_rating == expert_rating
                ) else "✗"
                signalling = _format_answers(row["signalling_answers"])
                just = _truncate(row["justification"], 180)
                lines.append(
                    f"| {ann['pmid']} | {trial_short} | {slug} | "
                    f"{model_rating} | {expert_rating} | {match} | "
                    f"{signalling} | {just} |",
                )
                n_rows += 1

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_rows


def _group_by_methodology(
    annotations: Iterable[dict],
) -> list[tuple[str, list[dict]]]:
    """Stable group — preserves insertion order within each methodology."""
    groups: dict[str, list[dict]] = {}
    for ann in annotations:
        groups.setdefault(ann["methodology"], []).append(ann)
    return list(groups.items())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--db", type=Path, required=True,
        help="DB with AI annotations + expert_methodology_ratings.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT}).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    db = Database(args.db)
    try:
        annotations = _fetch_annotations(db)
        experts = _fetch_expert_ratings(db)
    finally:
        db.close()

    expert_idx = _expert_index(experts)

    s1 = args.out / "S1_annotations.jsonl"
    s2 = args.out / "S2_expert_ratings.csv"
    s3 = args.out / "S3_audit_table.md"

    n_s1 = write_s1(annotations, expert_idx, s1)
    n_s2 = write_s2(experts, s2)
    n_s3 = write_s3(annotations, expert_idx, s3)

    logger.info("S1 annotations written: %d rows -> %s", n_s1, s1)
    logger.info("S2 expert ratings written: %d rows -> %s", n_s2, s2)
    logger.info("S3 audit table written: %d rows -> %s", n_s3, s3)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Extract QUADAS-2 expert ground truth from a JATS systematic review.

Reads the risk-of-bias table (usually Table 2) of a Cochrane-style
systematic review of diagnostic accuracy and emits a sidecar JSON that
drives the Step-9-style QUADAS-2 faithfulness harness.

The JATS table exposes rows like::

    AlQusayer (2019) [32] | Unclear | Low  | Low | Low | Unclear

where each row is one included study (or a sub-stratum of a study) and
the columns are QUADAS-2's four domain bias ratings plus the overall.
The study identifier ``[32]`` is a cross-reference into the review's
bibliography; we resolve it to PMID/DOI so the ratings can be joined
to the ``papers`` table.

Also cross-checks each identified paper against a legacy biasbuster
DB to flag entries whose stored ``title`` doesn't plausibly match the
JATS reference — early-warning for corrupted legacy rows where the
wrong paper is associated with a DOI. This lets you decide per-paper
whether to trust the legacy metadata or re-download.

Usage::

    uv run python scripts/extract_quadas2_ground_truth.py \\
        --jats tests/fixtures/cochrane_reviews/jcm-15-01829.xml \\
        --legacy-db dataset/biasbuster.db \\
        --output tests/fixtures/quadas2_ground_truth/jcm-15-01829.json

The output JSON has shape::

    {
      "review": {"pmid": ..., "doi": ..., "title": ...},
      "studies": [
        {
          "label": "AlQusayer (2019)",
          "first_author_surname": "AlQusayer",
          "year": 2019,
          "pmid": null, "doi": "10.5281/zenodo.2542068",
          "quadas2": {
            "patient_selection": {"bias": "unclear"},
            "index_test": {"bias": "low"},
            "reference_standard": {"bias": "low"},
            "flow_and_timing": {"bias": "low"},
            "overall": "unclear"
          },
          "legacy_db": {
            "present": true,
            "title_in_db": "...",
            "title_matches_author": true   // heuristic
          }
        },
        ...
      ]
    }

The ``legacy_db.title_matches_author`` heuristic is a simple
case-insensitive substring match between the extracted first-author
surname and the stored title. It's a sanity check, not authoritative —
a human should verify the flagged entries before treating them as
trusted ground truth.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Mapping from the JATS table's column *position* (0-indexed after the
# study-label column) to the canonical QUADAS-2 domain slug. The four
# systematic reviews we expect to ingest share the Patient Selection /
# Index Test / Reference Standard / Flow and Timing / Overall column
# order per the Whiting 2011 reporting template, but individual reviews
# rename columns (e.g. "Indicator Measurement (Salivary)" for Index
# Test in a salivary-glucose review). We key by position, not by column
# header, to absorb that cosmetic variation.
_DOMAIN_BY_COLUMN_INDEX: dict[int, str] = {
    0: "patient_selection",
    1: "index_test",
    2: "reference_standard",
    3: "flow_and_timing",
    # Position 4 is the overall judgement — handled separately.
}

_RATING_NORMALISE: dict[str, str] = {
    "low": "low",
    "high": "high",
    "unclear": "unclear",
    # Be generous with variant spellings the author might have used.
    "unclear risk": "unclear",
    "high risk": "high",
    "low risk": "low",
}


def _normalise_rating(raw: str) -> Optional[str]:
    """Convert a cell's text to the canonical QUADAS-2 rating or None."""
    key = raw.strip().lower()
    return _RATING_NORMALISE.get(key)


@dataclass
class StudyRow:
    """One row parsed from the QUADAS-2 table."""

    label: str
    first_author_surname: str
    year: Optional[int]
    ref_id: Optional[str]
    pmid: Optional[str] = None
    doi: Optional[str] = None
    #: Expected article title from the JATS bibliography — used to
    #: sanity-check the legacy DB entry.
    expected_title: Optional[str] = None
    bias_ratings: dict[str, str] = field(default_factory=dict)
    overall: Optional[str] = None
    legacy_db: dict[str, Any] = field(default_factory=dict)


def _parse_study_label(label: str) -> tuple[str, Optional[int], Optional[str]]:
    """Pull the first-author surname, year, and bibliography ref id from a label.

    Labels look like ``AlQusayer (2019)`` or ``Borg (1998) — T2DM`` or
    ``Kadashetti (2015) — BG > 200 mg/dL``. We strip the sub-stratum
    suffix (after an em-dash or ASCII dash) and keep the first author
    + year. The ``[32]`` reference marker is omitted here because it's
    carried as a structured ``rid`` attribute on the ``<xref>`` element
    we resolve separately.
    """
    # Strip anything after an em-dash, en-dash, or " - " (sub-stratum).
    main = re.split(r"\s*[—–-]\s*(?=\w)", label, maxsplit=1)[0]
    m = re.match(r"\s*(\S+)\s*\((\d{4})\)", main)
    if not m:
        return label.strip(), None, None
    author = m.group(1).strip()
    year = int(m.group(2))
    return author, year, None


def parse_quadas2_table(
    jats_path: Path, table_label: str = "Table 2",
) -> tuple[list[StudyRow], dict[str, dict[str, Any]]]:
    """Parse the QUADAS-2 table and the bibliography from a JATS XML file.

    Returns ``(study_rows, bibliography)``. The bibliography maps a
    ``ref-id`` (e.g. ``"B32-jcm-15-01829"``) to ``{"pmid": ..., "doi": ...,
    "title": ..., "year": ..., "first_author_surname": ...}``.
    """
    tree = ET.parse(jats_path)
    root = tree.getroot()

    # --- Bibliography ---
    bibliography: dict[str, dict[str, Any]] = {}
    for ref in root.iter("ref"):
        rid = ref.get("id") or ""
        if not rid:
            continue
        entry: dict[str, Any] = {
            "pmid": None, "doi": None, "title": None,
            "year": None, "first_author_surname": None,
        }
        for el in ref.iter():
            if el.tag == "pub-id":
                t = (el.get("pub-id-type") or "").lower()
                if t == "pmid":
                    entry["pmid"] = (el.text or "").strip() or None
                elif t == "doi":
                    entry["doi"] = (el.text or "").strip() or None
            elif el.tag == "article-title" and entry["title"] is None:
                entry["title"] = "".join(el.itertext()).strip() or None
            elif el.tag == "year" and entry["year"] is None:
                try:
                    entry["year"] = int((el.text or "").strip())
                except (TypeError, ValueError):
                    entry["year"] = None
            elif el.tag == "surname" and entry["first_author_surname"] is None:
                entry["first_author_surname"] = (el.text or "").strip() or None
        bibliography[rid] = entry

    # --- The QUADAS-2 table ---
    target_table = None
    for tw in root.iter("table-wrap"):
        label_el = tw.find("label")
        if label_el is not None and (label_el.text or "").strip() == table_label:
            target_table = tw
            break
    if target_table is None:
        raise SystemExit(
            f"Could not find a <table-wrap> with label {table_label!r} "
            f"in {jats_path}"
        )

    studies: list[StudyRow] = []
    tbody = target_table.find(".//tbody")
    if tbody is None:
        raise SystemExit(
            f"Table {table_label!r} has no <tbody> in {jats_path}"
        )
    for tr in tbody.findall("tr"):
        cells = tr.findall("td")
        if len(cells) < 5:  # label + 4 domains + overall = 6 cells minimum
            continue
        # Cell 0 is the study label; subsequent cells are the ratings.
        label_text = "".join(cells[0].itertext()).strip()
        # Strip the "[32]" reference marker from the displayed label
        # for readability — we're capturing ref-id via the xref below.
        display_label = re.sub(r"\s*\[\d+\]", "", label_text).strip()
        author, year, _ = _parse_study_label(display_label)
        xref = cells[0].find(".//xref[@ref-type='bibr']")
        ref_id = xref.get("rid") if xref is not None else None

        # Domain ratings (first four rating columns, position 1-4)
        bias_ratings: dict[str, str] = {}
        for i, slug in _DOMAIN_BY_COLUMN_INDEX.items():
            cell_text = "".join(cells[i + 1].itertext()).strip()
            rating = _normalise_rating(cell_text)
            if rating is None:
                logger.warning(
                    "Row %r: unparseable rating for domain %s: %r",
                    display_label, slug, cell_text,
                )
                continue
            bias_ratings[slug] = rating
        # Overall column is the last rating cell
        overall_raw = "".join(cells[5].itertext()).strip()
        overall = _normalise_rating(overall_raw)

        pmid = None
        doi = None
        expected_title = None
        if ref_id and ref_id in bibliography:
            pmid = bibliography[ref_id].get("pmid")
            doi = bibliography[ref_id].get("doi")
            expected_title = bibliography[ref_id].get("title")

        studies.append(StudyRow(
            label=display_label,
            first_author_surname=author,
            year=year,
            ref_id=ref_id,
            pmid=pmid,
            doi=doi,
            expected_title=expected_title,
            bias_ratings=bias_ratings,
            overall=overall,
        ))
    return studies, bibliography


_WORD_RE = re.compile(r"[A-Za-z]{4,}")


def _title_token_set(title: str) -> set[str]:
    """Return the 4-or-more-letter words of a title, lowercase.

    Short words (<4 letters) are excluded because they're dominated by
    grammatical connectives ("a", "of", "the", "in") that dilute the
    Jaccard overlap between two biomedical titles about the same topic.
    """
    return {w.lower() for w in _WORD_RE.findall(title)}


def _title_similarity(expected: str, actual: str) -> float:
    """Jaccard overlap of content words between two titles (0.0 – 1.0).

    Returns 0.0 if either title is empty or has no content words.
    Used as a correctness sanity check on legacy-DB lookups: two titles
    about the same paper should share most content words; two titles
    about entirely different papers will overlap near zero.
    """
    exp_tokens = _title_token_set(expected or "")
    act_tokens = _title_token_set(actual or "")
    if not exp_tokens or not act_tokens:
        return 0.0
    intersection = exp_tokens & act_tokens
    union = exp_tokens | act_tokens
    return len(intersection) / len(union)


#: Jaccard threshold below which the legacy DB title is deemed a likely
#: mismatch (i.e. the DOI points at the wrong paper). Picked empirically:
#: a title with one or two words rearranged or a sub-title omitted will
#: still land above this; a completely unrelated topic (dermatofibroma
#: vs. salivary glucose) lands near zero.
_TITLE_MATCH_THRESHOLD: float = 0.2


def cross_reference_legacy_db(
    studies: list[StudyRow], legacy_db: Path,
) -> None:
    """Annotate each study with legacy-DB lookup results (in place).

    For each study with a PMID or DOI, check whether the legacy ``papers``
    table has a matching row and compare the stored title against the
    title recorded in the JATS bibliography. A Jaccard overlap below
    ``_TITLE_MATCH_THRESHOLD`` flags the legacy row as likely corrupt
    (the DOI got associated with the wrong paper metadata somewhere in
    the v1 ingest pipeline).
    """
    if not legacy_db.exists():
        raise SystemExit(f"Legacy DB not found: {legacy_db}")
    conn = sqlite3.connect(str(legacy_db))
    try:
        for study in studies:
            hit: Optional[dict[str, Any]] = None
            if study.pmid:
                row = conn.execute(
                    "SELECT pmid, doi, title, source FROM papers WHERE pmid = ?",
                    (study.pmid,),
                ).fetchone()
                if row is not None:
                    hit = {"pmid": row[0], "doi": row[1], "title": row[2],
                           "source": row[3]}
            if hit is None and study.doi:
                row = conn.execute(
                    "SELECT pmid, doi, title, source FROM papers "
                    "WHERE LOWER(doi) = LOWER(?)",
                    (study.doi,),
                ).fetchone()
                if row is not None:
                    hit = {"pmid": row[0], "doi": row[1], "title": row[2],
                           "source": row[3]}
            if hit is None:
                study.legacy_db = {"present": False}
                continue
            title = (hit["title"] or "")
            similarity = _title_similarity(
                study.expected_title or "", title,
            )
            plausible = similarity >= _TITLE_MATCH_THRESHOLD
            study.legacy_db = {
                "present": True,
                "pmid_in_db": hit["pmid"],
                "doi_in_db": hit["doi"],
                "title_in_db": title,
                "source_in_db": hit["source"],
                "expected_title": study.expected_title,
                "title_jaccard_similarity": round(similarity, 3),
                "title_plausible": plausible,
            }
    finally:
        conn.close()


def summarise(
    studies: list[StudyRow],
) -> dict[str, int]:
    """Produce a short counts summary for the log/stdout."""
    n_total = len(studies)
    n_with_pmid = sum(1 for s in studies if s.pmid)
    n_with_doi = sum(1 for s in studies if s.doi)
    n_in_legacy = sum(
        1 for s in studies if s.legacy_db.get("present") is True
    )
    n_plausible = sum(
        1 for s in studies if s.legacy_db.get("title_plausible") is True
    )
    n_corrupt = sum(
        1 for s in studies
        if s.legacy_db.get("present") is True
        and s.legacy_db.get("title_plausible") is False
    )
    return {
        "total_rows": n_total,
        "with_pmid": n_with_pmid,
        "with_doi": n_with_doi,
        "present_in_legacy_db": n_in_legacy,
        "legacy_title_plausible": n_plausible,
        "legacy_likely_corrupt": n_corrupt,
    }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--jats", type=Path, required=True,
        help="Path to the JATS XML of the systematic review containing the "
             "QUADAS-2 risk-of-bias table.",
    )
    p.add_argument(
        "--table-label", default="Table 2",
        help="The <table-wrap><label> text that identifies the QUADAS-2 "
             "table. Default: 'Table 2'.",
    )
    p.add_argument(
        "--legacy-db", type=Path, default=None,
        help="Optional path to the legacy biasbuster.db for cross-referencing "
             "each study. When omitted, the output only contains the parsed "
             "JATS data.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Path to write the ground-truth JSON sidecar.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    studies, bibliography = parse_quadas2_table(
        args.jats, table_label=args.table_label,
    )
    if args.legacy_db is not None:
        cross_reference_legacy_db(studies, args.legacy_db)

    summary: dict[str, int] = summarise(studies)
    payload: dict[str, Any] = {
        "source": {
            "jats_file": str(args.jats),
            "table_label": args.table_label,
        },
        "studies": [asdict(s) for s in studies],
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"  Rows parsed:                       {summary['total_rows']}")
    print(f"  With PMID:                         {summary['with_pmid']}")
    print(f"  With DOI:                          {summary['with_doi']}")
    if args.legacy_db is not None:
        print(f"  Present in legacy DB:              "
              f"{summary['present_in_legacy_db']}")
        print(f"  Titles plausibly match:            "
              f"{summary['legacy_title_plausible']}")
        print(f"  Likely corrupt (title mismatch):   "
              f"{summary['legacy_likely_corrupt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

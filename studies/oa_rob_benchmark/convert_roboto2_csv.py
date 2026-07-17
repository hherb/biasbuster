"""Convert the raw ROBoto2 dataset CSV into normalized ingestion JSON.

The ROBoto2 release ships as a wide CSV (``dataset/roboto2_dataset.csv``)
whose cells embed JSON strings. This module normalises it into the flat
record shape ``ingest_roboto2.parse_record`` consumes: it reads the CSV,
keeps only the rows carrying a human-expert ``manual_assessment`` (the
benchmark ground truth), and emits one flat, self-describing JSON record per
trial for downstream ingestion.

Three deliberate choices, each confirmed against the real file (see the
task discussion that produced this converter):

1. **Ground truth = the experts' recorded judgements.** Each row already
   stores the experts' per-domain ``domain_risk_levels`` and
   ``overall_risk_level``. We carry those verbatim (normalized to the
   canonical level vocabulary) rather than re-deriving them from the raw
   signalling answers via the RoB 2 algorithm — re-derivation disagreed
   with the recorded expert labels on ~36% of domains / ~21% of overalls
   on this dataset, partly because the answers include combined values
   (``N/PN``, ``Y/PY``) and ``None`` the algorithm cannot parse. The raw
   signalling answers are still preserved (unaltered) under ``signalling``
   for provenance and any future analysis.

2. **Identity is title + abstract + authors, NOT ``paper_id``.** ROBoto2's
   ``paper_id`` is an S2ORC/internal identifier (mostly numeric, a few
   ``Author_Year`` strings), and the records contain **no PMID/DOI/PMCID**
   (``identifiers`` is empty). Treating ``paper_id`` as a PMID would fetch
   unrelated documents (e.g. ``2018`` extracted from ``Efendi_2018``). We
   therefore surface the real identity fields so a downstream resolver can
   map each trial to a PubMed record by title, not by a spurious PMID.

3. **Only manual (expert) rows are emitted.** LLM-assisted
   ``roboto2_assessment`` rows are excluded (the ingest litmus drops them
   anyway); they are candidate OA-trial pools, never ground truth.

The conversion is pure and deterministic (no network I/O), so it is safe to
run in-session and re-run idempotently — same CSV in, same JSON out.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Default input/output paths (relative to the repo root).
DEFAULT_CSV_PATH = "dataset/roboto2_dataset.csv"
DEFAULT_OUT_PATH = "dataset/roboto2/roboto2.json"

#: Provenance tag for every record this converter emits.
SOURCE_TAG = "roboto2_manual"

#: RoB 2 ``domain_number`` (1..5) → canonical extractor domain name. Mirrors
#: the ordering of ``rob2_tuple._DOMAINS`` (randomization / deviations /
#: missing_outcome / measurement / reporting) so the emitted ``rob2`` and
#: ``signalling`` keys line up with the rest of the Stage A pipeline.
_DOMAIN_BY_NUMBER: dict[int, str] = {
    1: "randomization",
    2: "deviations",
    3: "missing_outcome",
    4: "measurement",
    5: "reporting",
}

#: Canonical RoB 2 level vocabulary. ``none`` is retained as a distinct value
#: (some domains are recorded as "None" — not assessed / not applicable — in
#: the source) rather than being coerced into a real risk level.
_LEVEL_ALIASES: dict[str, str] = {
    "low": "low",
    "some concerns": "some concerns",
    "some_concerns": "some concerns",
    "high": "high",
    "none": "none",
}

#: CSV placeholder strings that mean "empty" for a given cell.
_EMPTY_CELL = frozenset({"null", "", "none", "[]"})


def _normalise_level(raw: Any) -> str:
    """Map a recorded RoB 2 level to the canonical vocabulary.

    Returns one of ``low`` / ``some concerns`` / ``high`` / ``none``.
    An unrecognised value is logged and returned lower-cased/stripped
    unchanged rather than silently dropped, so a vocabulary drift in a
    future ROBoto2 release surfaces instead of corrupting a label.
    """
    text = str(raw).strip().lower()
    mapped = _LEVEL_ALIASES.get(text)
    if mapped is None:
        logger.warning("convert_roboto2_csv: unrecognised RoB2 level %r", raw)
        return text
    return mapped


def _is_empty_cell(value: str | None) -> bool:
    """True if a CSV cell is one of the recognised empty placeholders."""
    return value is None or value.strip().lower() in _EMPTY_CELL


def _author_names(paper_parse: dict) -> list[str]:
    """Extract ``"First Last"`` author strings from an S2ORC paper parse."""
    names: list[str] = []
    for author in paper_parse.get("authors") or []:
        if not isinstance(author, dict):
            continue
        parts = [str(author.get("first", "")).strip(), str(author.get("last", "")).strip()]
        full = " ".join(p for p in parts if p)
        if full:
            names.append(full)
    return names


def _signalling_by_domain(result: list[dict]) -> dict[str, dict[str, Any]]:
    """Group per-question signalling answers into ``{domain_name: {"d.q": answer}}``.

    Keys are ``"{domain_number}.{question_number}"`` (e.g. ``"1.1"``) to
    match the RoB 2 signalling-question numbering. Answers
    (``expertPrediction``) are preserved verbatim — including combined
    values like ``N/PN`` / ``Y/PY`` and ``None`` — since this field is raw
    provenance, not the ground-truth label.
    """
    by_domain: dict[str, dict[str, Any]] = {name: {} for name in _DOMAIN_BY_NUMBER.values()}
    for entry in result:
        if not isinstance(entry, dict):
            continue
        domain_number = entry.get("domain_number")
        question_number = entry.get("question_number")
        name = _DOMAIN_BY_NUMBER.get(domain_number) if isinstance(domain_number, int) else None
        if name is None or question_number is None:
            continue
        by_domain[name][f"{domain_number}.{question_number}"] = entry.get("expertPrediction")
    return by_domain


def parse_manual_row(row: dict[str, str]) -> dict[str, Any] | None:
    """Convert one CSV row into a normalized ingestion record.

    Returns ``None`` (never raises) for rows without a human
    ``manual_assessment`` — those are LLM-assisted-only or unlabeled rows
    that are not benchmark ground truth. The returned record carries the
    experts' recorded RoB 2 judgements as ground truth, the real identity
    fields (title/abstract/authors) for downstream PMID resolution, and the
    raw signalling answers for provenance.
    """
    manual_cell = row.get("manual_assessment")
    if manual_cell is None or _is_empty_cell(manual_cell):
        return None
    manual = json.loads(manual_cell)
    if not manual:
        return None
    assessment = manual[0]

    levels = assessment.get("domain_risk_levels") or []
    if len(levels) != len(_DOMAIN_BY_NUMBER):
        logger.warning(
            "convert_roboto2_csv: paper_id=%s has %d domain levels (expected %d); skipping",
            row.get("paper_id"), len(levels), len(_DOMAIN_BY_NUMBER),
        )
        return None

    rob2: dict[str, str] = {"overall": _normalise_level(assessment.get("overall_risk_level"))}
    for number, name in _DOMAIN_BY_NUMBER.items():
        rob2[name] = _normalise_level(levels[number - 1])

    signalling = _signalling_by_domain(assessment.get("result") or [])

    parse_cell = row.get("paper_parse")
    has_fulltext = parse_cell is not None and not _is_empty_cell(parse_cell)
    title = abstract = ""
    authors: list[str] = []
    if has_fulltext and parse_cell is not None:
        paper_parse = json.loads(parse_cell)
        title = str(paper_parse.get("title") or "").strip()
        abstract = str(paper_parse.get("abstract") or "").strip()
        authors = _author_names(paper_parse)

    return {
        "paper_id": row.get("paper_id", ""),
        "source": SOURCE_TAG,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "has_fulltext": has_fulltext,
        "rob2": rob2,
        "signalling": signalling,
    }


def convert_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """Read the ROBoto2 CSV and return normalized records for expert rows.

    Preserves CSV row order and emits one record per row carrying a human
    ``manual_assessment``. Rows without one are skipped silently (they are
    LLM-assisted/unlabeled and out of scope for the benchmark ground truth).
    """
    # ROBoto2 cells embed large JSON blobs (full-text parses run to ~90 KB);
    # lift the CSV field-size cap so the reader does not truncate them.
    csv.field_size_limit(sys.maxsize)
    records: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record = parse_manual_row(row)
            if record is not None:
                records.append(record)
    return records


def write_json(records: list[dict[str, Any]], out_path: str | Path) -> None:
    """Write records as pretty-printed UTF-8 JSON, creating parent dirs."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: convert the ROBoto2 CSV to ingestion JSON."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="input ROBoto2 CSV path")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="output JSON path")
    args = parser.parse_args(argv)

    records = convert_csv(args.csv)
    write_json(records, args.out)

    with_fulltext = sum(1 for r in records if r["has_fulltext"])
    logger.info(
        "Wrote %d expert records to %s (%d with full text, %d without)",
        len(records), args.out, with_fulltext, len(records) - with_fulltext,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

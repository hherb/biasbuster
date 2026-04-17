"""Structural bias-assessment table extractor from JATS XML.

Parses ``<table-wrap>`` elements in systematic review JATS to find
per-study risk-of-bias (or quality) assessment tables, identifies which
bias methodology the table uses (RoB 2, QUADAS-2, ROBINS-I, etc.),
and extracts per-study, per-domain ratings.

Two rating signals are extracted independently and cross-checked:
1. **Cell text** — the literal string ("Low", "High", "Some concerns").
2. **CSS background colour** — many review tools (robvis, RevMan)
   colour-code cells: green (#C6EFCE) = low, yellow/amber (#FFEB9C) =
   unclear/some concerns, red (#FFC7CE) = high/serious.

Pure functions — no I/O, no DB access, no state. Unit-testable against
fixture JATS files in ``tests/fixtures/cochrane_reviews/``.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# -- Bias methodology definitions ----------------------------------------

@dataclass(frozen=True)
class BiasMethodology:
    """Definition of a bias assessment tool: its name, domain keywords
    (for header matching), and the valid rating levels it uses."""
    name: str
    display_name: str
    domain_keywords: dict[str, list[str]]  # canonical_name → [header keywords]
    valid_ratings: frozenset[str]
    rating_aliases: dict[str, str]  # normalised alias → canonical rating


ROB2 = BiasMethodology(
    name="rob2",
    display_name="Cochrane Risk of Bias 2 (RoB 2)",
    domain_keywords={
        "randomization": ["randomi"],
        "deviations": ["deviation"],
        "missing_outcome": ["missing"],
        "measurement": ["measurement of"],
        "reporting": ["reported result", "reporting"],
    },
    valid_ratings=frozenset({"low", "some_concerns", "high"}),
    rating_aliases={
        "low risk": "low",
        "low": "low",
        "some concerns": "some_concerns",
        "some_concerns": "some_concerns",
        "unclear": "some_concerns",
        "high risk": "high",
        "high": "high",
    },
)

QUADAS2 = BiasMethodology(
    name="quadas2",
    display_name="QUADAS-2",
    domain_keywords={
        "patient_selection": ["patient", "selection"],
        "index_test": ["index", "indicator"],
        "reference_standard": ["reference"],
        "flow_timing": ["flow", "timing"],
    },
    valid_ratings=frozenset({"low", "unclear", "high"}),
    rating_aliases={
        "low": "low",
        "low risk": "low",
        "unclear": "unclear",
        "high": "high",
        "high risk": "high",
    },
)

ROBINS_I = BiasMethodology(
    name="robins_i",
    display_name="ROBINS-I",
    domain_keywords={
        "confounding": ["confound"],
        "selection": ["selection"],
        "classification": ["classif"],
        "deviations": ["deviation"],
        "missing_data": ["missing"],
        "measurement": ["measur"],
        "reporting": ["report", "select"],
    },
    valid_ratings=frozenset({"low", "moderate", "serious", "critical"}),
    rating_aliases={
        "low": "low",
        "low risk": "low",
        "moderate": "moderate",
        "serious": "serious",
        "high": "serious",
        "critical": "critical",
    },
)

ALL_METHODOLOGIES: list[BiasMethodology] = [ROB2, QUADAS2, ROBINS_I]

# Minimum number of domain columns that must match for a table to be
# considered a bias assessment table for a given methodology.
MIN_DOMAIN_MATCH = 3


# -- CSS colour → rating mapping -----------------------------------------

COLOUR_TO_RATING: dict[str, str] = {
    "#c6efce": "low",       # green
    "#c6efcf": "low",
    "#006100": "low",       # dark green text
    "#ffeb9c": "unclear",   # amber/yellow
    "#9c6500": "unclear",   # dark amber text
    "#ffc7ce": "high",      # red/pink
    "#9c0006": "high",      # dark red text
}


# -- Extracted data structures -------------------------------------------

@dataclass
class ExtractedRating:
    """One cell from a bias assessment table."""
    domain: str           # canonical domain name
    rating_text: str      # normalised rating from cell text
    rating_colour: str    # normalised rating from CSS colour (or "")
    raw_text: str         # original cell text
    raw_style: str        # original CSS style attribute


@dataclass
class ExtractedStudyRow:
    """One row (study) from a bias assessment table."""
    study_id: str         # "FirstAuthor (Year)" as it appears in the table
    overall: ExtractedRating | None
    domains: list[ExtractedRating]
    row_index: int        # 0-based position in the table body


@dataclass
class ExtractedBiasTable:
    """A complete bias assessment table extracted from a review."""
    methodology: BiasMethodology
    table_id: str         # JATS table-wrap id
    table_label: str      # "Table 2" etc.
    caption: str
    column_headers: list[str]       # raw header strings
    domain_mapping: dict[int, str]  # col_index → canonical domain name
    overall_col: int | None         # col_index of the "Overall" column
    studies: list[ExtractedStudyRow]

    # Provenance for the source section
    section_title: str = ""


# -- Core extraction logic -----------------------------------------------

def extract_bias_tables(jats_xml: bytes | str) -> list[ExtractedBiasTable]:
    """Extract all bias assessment tables from a JATS XML document.

    Args:
        jats_xml: Raw JATS XML (bytes or string).

    Returns:
        List of extracted tables, each with its detected methodology,
        column mapping, and per-study ratings. Empty list if no bias
        tables are found.
    """
    if isinstance(jats_xml, bytes):
        jats_xml = jats_xml.decode("utf-8", errors="replace")

    try:
        root = ET.fromstring(jats_xml)
    except ET.ParseError as exc:
        logger.warning("JATS parse error: %s", exc)
        return []

    results: list[ExtractedBiasTable] = []

    for table_wrap in root.iter("table-wrap"):
        table_el = table_wrap.find(".//table")
        if table_el is None:
            continue

        # Get table metadata
        tw_id = table_wrap.get("id", "")
        label_el = table_wrap.find("label")
        label = label_el.text.strip() if label_el is not None and label_el.text else ""
        caption_el = table_wrap.find("caption")
        caption = " ".join(caption_el.itertext()).strip() if caption_el is not None else ""

        # Get column headers
        headers = _extract_headers(table_el)
        if len(headers) < 3:
            continue

        # Score ALL methodologies and pick the best match (most domains)
        best_meth: BiasMethodology | None = None
        best_mapping: dict[int, str] | None = None
        best_overall: int | None = None
        best_score = 0

        for meth in ALL_METHODOLOGIES:
            mapping, overall_col = _match_methodology(headers, meth)
            if mapping is not None and len(mapping) > best_score:
                best_meth = meth
                best_mapping = mapping
                best_overall = overall_col
                best_score = len(mapping)

        if best_meth is not None and best_mapping is not None:
            section_title = _find_parent_section_title(root, table_wrap)
            studies = _extract_study_rows(
                table_el, best_mapping, best_overall, best_meth,
            )
            results.append(ExtractedBiasTable(
                methodology=best_meth,
                table_id=tw_id,
                table_label=label,
                caption=caption[:200],
                column_headers=headers,
                domain_mapping=best_mapping,
                overall_col=best_overall,
                studies=studies,
                section_title=section_title,
            ))

    return results


def _extract_headers(table_el: ET.Element) -> list[str]:
    """Extract column header strings from a <table> element."""
    thead = table_el.find("thead")
    if thead is None:
        return []

    headers: list[str] = []
    # Take the last header row (some tables have multi-row headers)
    header_rows = thead.findall("tr")
    if not header_rows:
        return []
    last_row = header_rows[-1]
    for cell in last_row.findall("th") or last_row.findall("td"):
        text = " ".join(cell.itertext()).strip()
        headers.append(text)
    return headers


def _match_methodology(
    headers: list[str], meth: BiasMethodology
) -> tuple[dict[int, str] | None, int | None]:
    """Try to match column headers against a methodology's domain keywords.

    Returns (domain_mapping, overall_col_index) if enough domains match,
    or (None, None) if no match.
    """
    header_lower = [h.lower() for h in headers]
    mapping: dict[int, str] = {}
    overall_col: int | None = None

    for col_idx, h in enumerate(header_lower):
        if col_idx == 0:
            continue  # skip study-id column
        if "overall" in h:
            overall_col = col_idx
            continue
        for domain_name, keywords in meth.domain_keywords.items():
            if any(kw in h for kw in keywords):
                mapping[col_idx] = domain_name
                break

    if len(mapping) >= MIN_DOMAIN_MATCH:
        return mapping, overall_col
    return None, None


def _extract_study_rows(
    table_el: ET.Element,
    domain_mapping: dict[int, str],
    overall_col: int | None,
    meth: BiasMethodology,
) -> list[ExtractedStudyRow]:
    """Extract per-study ratings from table body rows."""
    tbody = table_el.find("tbody")
    if tbody is None:
        return []

    studies: list[ExtractedStudyRow] = []
    for row_idx, tr in enumerate(tbody.findall("tr")):
        cells = tr.findall("td")
        if not cells:
            continue

        # Study ID from first column
        study_id = " ".join(cells[0].itertext()).strip()
        if not study_id:
            continue

        # Extract domain ratings
        domains: list[ExtractedRating] = []
        for col_idx, domain_name in domain_mapping.items():
            if col_idx < len(cells):
                rating = _extract_cell_rating(cells[col_idx], domain_name, meth)
                domains.append(rating)

        # Extract overall rating
        overall: ExtractedRating | None = None
        if overall_col is not None and overall_col < len(cells):
            overall = _extract_cell_rating(cells[overall_col], "overall", meth)

        studies.append(ExtractedStudyRow(
            study_id=study_id,
            overall=overall,
            domains=domains,
            row_index=row_idx,
        ))

    return studies


def _extract_cell_rating(
    cell: ET.Element, domain: str, meth: BiasMethodology
) -> ExtractedRating:
    """Extract the rating from a single table cell using text + CSS colour."""
    raw_text = " ".join(cell.itertext()).strip()
    raw_style = cell.get("style", "")

    # Rating from text
    rating_text = _normalise_rating_text(raw_text, meth)

    # Rating from CSS background colour
    rating_colour = _rating_from_css(raw_style)

    return ExtractedRating(
        domain=domain,
        rating_text=rating_text,
        rating_colour=rating_colour,
        raw_text=raw_text,
        raw_style=raw_style,
    )


def _normalise_rating_text(text: str, meth: BiasMethodology) -> str:
    """Normalise cell text to a canonical rating string."""
    t = text.strip().lower()
    # Direct match
    if t in meth.rating_aliases:
        return meth.rating_aliases[t]
    # Substring match (for cells like "Low risk of bias")
    for alias, canonical in sorted(
        meth.rating_aliases.items(), key=lambda x: -len(x[0])
    ):
        if alias in t:
            return canonical
    return ""


def _rating_from_css(style: str) -> str:
    """Extract a rating from CSS background-colour in the style attribute."""
    if not style:
        return ""
    # Match background or background-color
    bg_match = re.search(r"background(?:-color)?:\s*([^;\"']+)", style, re.IGNORECASE)
    if not bg_match:
        return ""
    colour = bg_match.group(1).strip().lower()
    return COLOUR_TO_RATING.get(colour, "")


def _find_parent_section_title(root: ET.Element, target: ET.Element) -> str:
    """Walk up from target to find the nearest enclosing <sec> title."""
    # Build parent map
    parent_map: dict[ET.Element, ET.Element] = {}
    for parent in root.iter():
        for child in parent:
            parent_map[child] = parent

    current = target
    while current in parent_map:
        current = parent_map[current]
        if current.tag == "sec":
            title_el = current.find("title")
            if title_el is not None and title_el.text:
                return title_el.text.strip()
    return ""


# -- Convenience: extract + summarise ------------------------------------

def summarise_extraction(tables: list[ExtractedBiasTable]) -> str:
    """Human-readable summary of extracted tables (for logging / manifest)."""
    if not tables:
        return "No bias assessment tables found."
    lines: list[str] = []
    for t in tables:
        lines.append(
            f"{t.table_label} ({t.methodology.display_name}): "
            f"{len(t.studies)} studies, "
            f"{len(t.domain_mapping)} domains + "
            f"{'overall' if t.overall_col is not None else 'no overall'}"
        )
        for s in t.studies[:3]:
            domains_str = ", ".join(
                f"{r.domain}={r.rating_text or r.rating_colour or '?'}"
                for r in s.domains
            )
            overall_str = ""
            if s.overall:
                overall_str = f" overall={s.overall.rating_text or s.overall.rating_colour or '?'}"
            lines.append(f"  {s.study_id}: {domains_str}{overall_str}")
        if len(t.studies) > 3:
            lines.append(f"  ... and {len(t.studies) - 3} more")
    return "\n".join(lines)

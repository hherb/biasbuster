"""JATS reference-list and front-matter parsing for Stage A EM candidate ingest.

Split out of ``ingest_em_candidates.py`` (that module's line-count budget)
per the task brief's split guidance. Pure, offline XML parsing only — no
network I/O, never raises (malformed input yields ``[]``/empty strings).
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

from biasbuster.collectors.study_pmid_resolver import Reference

logger = logging.getLogger(__name__)

#: Non-digit characters trimmed from a ``<label>`` before treating it as a
#: reference number, e.g. "[28]" or "28." both normalise to "28".
_REF_NUMBER_STRIP_RE = re.compile(r"[^0-9]")


def parse_xml(jats_xml: bytes | str) -> ET.Element | None:
    """Parse JATS XML, returning ``None`` (never raising) on malformed input."""
    if isinstance(jats_xml, bytes):
        jats_xml = jats_xml.decode("utf-8", errors="replace")
    try:
        return ET.fromstring(jats_xml)
    except ET.ParseError as exc:
        logger.warning("JATS parse error: %s", exc)
        return None


def _ref_number_from_label(label_text: str) -> str:
    """Strip brackets/punctuation from a ``<label>`` down to its digits."""
    return _REF_NUMBER_STRIP_RE.sub("", label_text)


def _ref_number_from_id(ref_id: str) -> str:
    """Pull the numeric part out of a ``<ref>``'s ``@id`` (e.g. "B28" -> "28")."""
    m = re.search(r"(\d+)", ref_id)
    return m.group(1) if m else ""


def parse_reference_list(jats_xml: bytes) -> list[Reference]:
    """Parse a review's ``<ref-list>/<ref>`` entries into ``Reference`` records.

    For each ``<ref>``: ``ref_number`` comes from ``<label>`` text (digits
    only) if present, else the numeric part of the ``@id`` attribute;
    ``pmid`` from ``<pub-id pub-id-type="pmid">``; ``first_author`` from the
    first ``<surname>`` found (the first author, since JATS author lists are
    ordered); ``year`` from ``<year>``; ``title`` from ``<article-title>``.
    A ``<ref>`` missing a piece leaves that field empty rather than being
    dropped — only totally malformed/unparsable XML yields ``[]``. Never
    raises.
    """
    root = parse_xml(jats_xml)
    if root is None:
        return []

    references: list[Reference] = []
    for ref_list in root.iter("ref-list"):
        for ref in ref_list.findall("ref"):
            label_el = ref.find("label")
            if label_el is not None and label_el.text:
                ref_number = _ref_number_from_label(label_el.text)
            else:
                ref_number = _ref_number_from_id(ref.get("id", ""))

            pmid_el = ref.find(".//pub-id[@pub-id-type='pmid']")
            pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""

            surname_el = ref.find(".//surname")
            first_author = (surname_el.text or "").strip() if surname_el is not None else ""

            year_el = ref.find(".//year")
            year = (year_el.text or "").strip() if year_el is not None else ""

            title_el = ref.find(".//article-title")
            title = " ".join(title_el.itertext()).strip() if title_el is not None else ""

            references.append(Reference(
                ref_number=ref_number, pmid=pmid, first_author=first_author,
                year=year, title=title,
            ))
    return references


def extract_front_ids(root: ET.Element) -> tuple[str, str]:
    """Best-effort ``(pmid, pmcid)`` for the review itself, from front matter."""
    pmid, pmcid = "", ""
    article_meta = root.find(".//front/article-meta")
    if article_meta is None:
        return pmid, pmcid
    for article_id in article_meta.findall("article-id"):
        id_type = article_id.get("pub-id-type", "")
        text = (article_id.text or "").strip()
        if not text:
            continue
        if id_type == "pmid":
            pmid = text
        elif id_type in ("pmcid", "pmc"):
            pmcid = text
    return pmid, pmcid

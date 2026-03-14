"""
PubMed XML Parsing Utilities

Standalone functions for parsing PubMed E-utilities XML responses.
Used by both the retraction watch collector and the pipeline's
RCT abstract fetcher.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Optional

logger = logging.getLogger(__name__)


def parse_pubmed_xml(xml_text: str) -> Optional[dict]:
    """Parse a single PubMed efetch XML response into an article dict.

    Args:
        xml_text: Raw XML string from PubMed efetch.

    Returns:
        Article dict with keys: pmid, doi, title, abstract, journal,
        year, authors, grants, mesh_terms. None if parsing fails.
    """
    try:
        root = ET.fromstring(xml_text)
        article = root.find(".//PubmedArticle")
        if article is None:
            return None
        return _extract_article_data(article)
    except ET.ParseError as e:
        logger.warning(f"XML parse error: {e}")
        return None


def parse_pubmed_xml_batch(xml_text: str) -> dict[str, dict]:
    """Parse a batch PubMed efetch XML response.

    Args:
        xml_text: Raw XML string from PubMed efetch (multiple articles).

    Returns:
        Dict mapping PMID strings to article dicts.
    """
    results: dict[str, dict] = {}
    try:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            data = _extract_article_data(article)
            if data and data.get("pmid"):
                results[data["pmid"]] = data
    except ET.ParseError as e:
        logger.warning(f"XML parse error: {e}")
    return results


def _extract_article_data(article: ET.Element) -> Optional[dict]:
    """Extract key fields from a PubmedArticle XML element.

    Args:
        article: An ElementTree Element for a single PubmedArticle.

    Returns:
        Dict with article metadata, or None on extraction failure.
    """
    try:
        medline = article.find(".//MedlineCitation")
        if medline is None:
            return None

        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        art = medline.find(".//Article")
        if art is None:
            return None

        title_el = art.find("ArticleTitle")
        title = title_el.text if title_el is not None else ""

        # Abstract - may have multiple AbstractText elements (structured)
        abstract_parts: list[str] = []
        abstract_el = art.find("Abstract")
        if abstract_el is not None:
            for at in abstract_el.findall("AbstractText"):
                label = at.get("Label", "")
                text = "".join(at.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)

        abstract = "\n".join(abstract_parts)

        # Journal
        journal_el = art.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""

        # Year
        year_el = medline.find(".//DateCompleted/Year")
        if year_el is None:
            year_el = art.find(".//Journal/JournalIssue/PubDate/Year")
        year = int(year_el.text) if year_el is not None else None

        # Authors with affiliations
        authors: list[dict] = []
        for author in art.findall(".//Author"):
            last = author.find("LastName")
            first = author.find("ForeName")
            affils = [a.text for a in author.findall(".//Affiliation") if a.text]
            authors.append({
                "last": last.text if last is not None else "",
                "first": first.text if first is not None else "",
                "affiliations": affils,
            })

        # Funding / grants
        grants: list[dict] = []
        for grant in medline.findall(".//Grant"):
            gid = grant.find("GrantID")
            agency = grant.find("Agency")
            grants.append({
                "id": gid.text if gid is not None else "",
                "agency": agency.text if agency is not None else "",
            })

        # DOI
        doi = ""
        for eid in art.findall(".//ELocationID"):
            if eid.get("EIdType") == "doi":
                doi = eid.text or ""

        # MeSH terms
        mesh_terms: list[str] = []
        for mesh in medline.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)

        return {
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "authors": authors,
            "grants": grants,
            "mesh_terms": mesh_terms,
        }
    except Exception as e:
        logger.warning(f"Article extraction error: {e}")
        return None

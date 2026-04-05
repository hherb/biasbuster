"""Text chunking for map-reduce bias analysis.

Splits full-text documents into chunks suitable for per-section LLM analysis.
Prefers section-based splitting from JATS structure, falls back to
overlapping token windows for unstructured text.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from bmlib.fulltext.models import JATSArticle, JATSBodySection

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 3000
DEFAULT_OVERLAP_TOKENS = 200

# Rough approximation: 1 token ~ 4 characters for English text.
# Good enough for chunking decisions; exact counts aren't critical.
CHARS_PER_TOKEN = 4


@dataclass
class TextChunk:
    """A chunk of text for per-section bias analysis."""

    section: str
    text: str
    chunk_index: int = 0
    total_chunks: int = 1

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate."""
        return len(self.text) // CHARS_PER_TOKEN


def chunk_jats_article(
    article: JATSArticle,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    jats_xml: bytes | None = None,
) -> list[TextChunk]:
    """Split a parsed JATS article into analysis chunks by section.

    Uses the semantic structure of the article (abstract, introduction,
    methods, results, discussion, etc.) to create coherent chunks.
    Large sections are sub-chunked using token windows.

    When ``jats_xml`` is provided, back-matter (funding, COI disclosures,
    acknowledgments) is extracted and included as a dedicated chunk.
    This is critical because JATS places funding and COI information
    in ``<back>`` rather than ``<body>``.

    Args:
        article: Parsed JATS article with body sections.
        max_tokens: Maximum tokens per chunk.
        jats_xml: Raw JATS XML bytes (optional, for back-matter extraction).

    Returns:
        Ordered list of text chunks.
    """
    max_chars = max_tokens * CHARS_PER_TOKEN
    chunks: list[TextChunk] = []

    # Abstract as first chunk (always included)
    if article.abstract_sections:
        abstract_text = "\n".join(
            f"[{s.title}] {s.content}" if s.title else s.content
            for s in article.abstract_sections
        )
        if abstract_text.strip():
            chunks.append(TextChunk(section="Abstract", text=abstract_text.strip()))

    # Body sections
    for section in article.body_sections:
        section_text = _flatten_section(section)
        if not section_text.strip():
            continue

        section_name = section.title or "Untitled Section"

        if len(section_text) <= max_chars:
            chunks.append(TextChunk(section=section_name, text=section_text.strip()))
        else:
            # Sub-chunk large sections
            sub_chunks = _token_window_chunks(
                section_name, section_text, max_tokens
            )
            chunks.extend(sub_chunks)

    # Back-matter: funding, COI, acknowledgments (from raw XML)
    if jats_xml is not None:
        backmatter = extract_jats_backmatter(jats_xml)
        if backmatter.strip():
            chunks.append(TextChunk(
                section="Funding, COI & Disclosures",
                text=backmatter.strip(),
            ))

    return chunks


def chunk_plain_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[TextChunk]:
    """Split plain text into overlapping token-window chunks.

    Used as fallback when no JATS structure is available (e.g. PDFs).

    Args:
        text: Full document text.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Overlap between adjacent chunks.

    Returns:
        Ordered list of text chunks.
    """
    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    step = max_chars - overlap_chars

    if step <= 0:
        step = max_chars // 2

    if len(text) <= max_chars:
        return [TextChunk(section="Full Text", text=text.strip())]

    chunks: list[TextChunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        # Try to break at paragraph or sentence boundary
        chunk_text = text[start:end]
        if end < len(text):
            chunk_text = _break_at_boundary(chunk_text)
            end = start + len(chunk_text)

        if chunk_text.strip():
            chunks.append(TextChunk(
                section=f"Chunk {idx + 1}",
                text=chunk_text.strip(),
                chunk_index=idx,
            ))
            idx += 1

        start = start + max(len(chunk_text) - overlap_chars, step)

    # Set total_chunks on all
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks


def _flatten_section(section: JATSBodySection, depth: int = 0) -> str:
    """Recursively flatten a JATS body section into plain text.

    Preserves section headings as markdown-style headers for context.
    """
    parts: list[str] = []

    if section.title and depth > 0:
        prefix = "#" * min(depth + 1, 4)
        parts.append(f"{prefix} {section.title}")

    for para in section.paragraphs:
        parts.append(para)

    for sub in section.subsections:
        parts.append(_flatten_section(sub, depth + 1))

    return "\n\n".join(parts)


def _token_window_chunks(
    section_name: str,
    text: str,
    max_tokens: int,
) -> list[TextChunk]:
    """Split a large section into token-window sub-chunks."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = DEFAULT_OVERLAP_TOKENS * CHARS_PER_TOKEN
    step = max_chars - overlap_chars

    chunks: list[TextChunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text = text[start:end]
        if end < len(text):
            chunk_text = _break_at_boundary(chunk_text)
            end = start + len(chunk_text)

        if chunk_text.strip():
            chunks.append(TextChunk(
                section=f"{section_name} (part {idx + 1})",
                text=chunk_text.strip(),
                chunk_index=idx,
            ))
            idx += 1

        start = start + max(len(chunk_text) - overlap_chars, step)

    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks


def extract_jats_backmatter(jats_xml: bytes) -> str:
    """Extract funding, COI, and disclosure sections from JATS XML back-matter.

    JATS articles place funding and conflict-of-interest disclosures in
    ``<back>`` elements (``<funding-group>``, ``<notes>``, ``<ack>``,
    ``<fn-group>``) rather than ``<body>``. These are critical for bias
    assessment but are not included in JATSParser's ``body_sections``.

    Args:
        jats_xml: Raw JATS XML content.

    Returns:
        Plain text with labeled sections for funding, COI, and related
        disclosures. Empty string if none found.
    """
    parts: list[str] = []

    # Decode XML to string for regex-based extraction.
    # Using regex instead of ET.parse because JATS XML often has namespace
    # issues and entity references that break strict XML parsers.
    try:
        xml_text = jats_xml.decode("utf-8", errors="replace")
    except Exception:
        return ""

    # Funding
    funding_match = re.search(
        r"<funding-statement>(.*?)</funding-statement>", xml_text, re.DOTALL
    )
    if funding_match:
        parts.append(f"Funding: {_strip_xml_tags(funding_match.group(1))}")
    else:
        # Try funding-source as fallback
        sources = re.findall(
            r"<funding-source>(.*?)</funding-source>", xml_text, re.DOTALL
        )
        if sources:
            parts.append(f"Funding sources: {'; '.join(_strip_xml_tags(s) for s in sources)}")

    # COI / Conflicts of Interest
    coi_match = re.search(
        r'<notes[^>]*notes-type="COI-statement"[^>]*>(.*?)</notes>',
        xml_text, re.DOTALL,
    )
    if coi_match:
        coi_text = _strip_xml_tags(coi_match.group(1))
        # Remove the title tag content separately
        coi_text = re.sub(r"^\s*Conflicts?\s+of\s+Interest\s*", "", coi_text).strip()
        parts.append(f"Conflicts of Interest: {coi_text}")
    else:
        # Broader search for COI in any <notes> block
        for m in re.finditer(r"<notes[^>]*>(.*?)</notes>", xml_text, re.DOTALL):
            content = m.group(1)
            if re.search(r"conflict|competing\s+interest|COI|disclosure", content, re.IGNORECASE):
                parts.append(f"Conflicts of Interest: {_strip_xml_tags(content)}")
                break

    # Author contributions
    for m in re.finditer(r"<notes[^>]*>(.*?)</notes>", xml_text, re.DOTALL):
        content = m.group(1)
        if re.search(r"<title>Author Contributions</title>", content, re.IGNORECASE):
            parts.append(f"Author Contributions: {_strip_xml_tags(content)}")
            break

    # Acknowledgments (may reveal funding/industry involvement)
    ack_match = re.search(r"<ack>(.*?)</ack>", xml_text, re.DOTALL)
    if ack_match:
        parts.append(f"Acknowledgments: {_strip_xml_tags(ack_match.group(1))}")

    return "\n\n".join(parts)


def _strip_xml_tags(text: str) -> str:
    """Remove XML/HTML tags from text."""
    text = re.sub(r"<title>.*?</title>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _break_at_boundary(text: str) -> str:
    """Try to break text at a paragraph or sentence boundary near the end.

    Looks backwards from the end for paragraph breaks (double newline),
    then sentence-ending punctuation. Returns the text up to the boundary.
    """
    min_keep = len(text) * 3 // 4  # Don't throw away more than 25%

    # Try paragraph break
    last_para = text.rfind("\n\n", min_keep)
    if last_para > 0:
        return text[: last_para + 2]

    # Try sentence break
    for punct in (". ", ".\n", "? ", "! "):
        last_sent = text.rfind(punct, min_keep)
        if last_sent > 0:
            return text[: last_sent + len(punct)]

    return text

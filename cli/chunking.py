"""Text chunking for map-reduce bias analysis.

Splits full-text documents into chunks suitable for per-section LLM analysis.
Prefers section-based splitting from JATS structure, falls back to
overlapping token windows for unstructured text.
"""

from __future__ import annotations

import logging
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
) -> list[TextChunk]:
    """Split a parsed JATS article into analysis chunks by section.

    Uses the semantic structure of the article (abstract, introduction,
    methods, results, discussion, etc.) to create coherent chunks.
    Large sections are sub-chunked using token windows.

    Args:
        article: Parsed JATS article with body sections.
        max_tokens: Maximum tokens per chunk.

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

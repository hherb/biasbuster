"""Tests for cli.chunking — text chunking logic."""

from bmlib.fulltext.models import JATSAbstractSection, JATSArticle, JATSBodySection

from biasbuster.cli.chunking import (
    TextChunk,
    chunk_jats_article,
    chunk_plain_text,
    _flatten_section,
    _break_at_boundary,
)


def _make_article(
    body_sections: list[JATSBodySection],
    abstract_text: str = "Background of the study.",
) -> JATSArticle:
    """Helper to create a minimal JATSArticle for testing."""
    return JATSArticle(
        title="Test Article",
        authors=[],
        journal="Test Journal",
        volume="1",
        issue="1",
        pages="1-10",
        year="2024",
        doi="10.1234/test",
        pmc_id="PMC12345",
        pmid="12345678",
        abstract_sections=[JATSAbstractSection(title="Background", content=abstract_text)],
        body_sections=body_sections,
        figures=[],
        tables=[],
        references=[],
    )


def test_chunk_jats_simple():
    """Article with small sections → one chunk per section."""
    sections = [
        JATSBodySection(title="Introduction", paragraphs=["Intro paragraph."]),
        JATSBodySection(title="Methods", paragraphs=["Methods paragraph."]),
        JATSBodySection(title="Results", paragraphs=["Results paragraph."]),
        JATSBodySection(title="Discussion", paragraphs=["Discussion paragraph."]),
    ]
    article = _make_article(sections)
    chunks = chunk_jats_article(article, max_tokens=3000)

    # Abstract + 4 body sections
    assert len(chunks) == 5
    assert chunks[0].section == "Abstract"
    assert chunks[1].section == "Introduction"
    assert chunks[4].section == "Discussion"


def test_chunk_jats_large_section_splits():
    """A large section gets sub-chunked."""
    # Create a section with ~5000 tokens worth of text (20000 chars)
    big_text = "This is a test paragraph. " * 800  # ~20000 chars
    sections = [
        JATSBodySection(title="Results", paragraphs=[big_text]),
    ]
    article = _make_article(sections)
    chunks = chunk_jats_article(article, max_tokens=1000)

    # Should have abstract + multiple sub-chunks of Results
    assert len(chunks) > 2
    assert chunks[0].section == "Abstract"
    assert "Results" in chunks[1].section


def test_chunk_plain_text_short():
    """Short text → single chunk."""
    text = "This is a short paper."
    chunks = chunk_plain_text(text, max_tokens=1000)
    assert len(chunks) == 1
    assert chunks[0].section == "Full Text"


def test_chunk_plain_text_splits():
    """Long text → multiple overlapping chunks."""
    text = "Sentence number one. " * 1000  # ~21000 chars
    chunks = chunk_plain_text(text, max_tokens=500, overlap_tokens=50)

    assert len(chunks) > 1
    # All chunks have correct total
    for chunk in chunks:
        assert chunk.total_chunks == len(chunks)


def test_flatten_section_with_subsections():
    """Nested subsections are flattened with heading markers."""
    section = JATSBodySection(
        title="Methods",
        paragraphs=["Overview of methods."],
        subsections=[
            JATSBodySection(
                title="Participants",
                paragraphs=["100 patients enrolled."],
            ),
            JATSBodySection(
                title="Statistical Analysis",
                paragraphs=["ANOVA was used."],
            ),
        ],
    )

    text = _flatten_section(section)
    assert "Overview of methods." in text
    assert "## Participants" in text
    assert "100 patients enrolled." in text
    assert "## Statistical Analysis" in text


def test_break_at_boundary_paragraph():
    """Prefers paragraph break when within the min_keep window."""
    # Text needs to be long enough that 75% mark falls before the last \n\n
    text = ("A" * 100) + "\n\n" + ("B" * 100) + "\n\n" + ("C" * 20)
    result = _break_at_boundary(text)
    assert result.endswith("\n\n")


def test_break_at_boundary_sentence():
    """Falls back to sentence break when no paragraph break found."""
    text = "First sentence. Second sentence. Third sentence. Fourth."
    # Set min_keep to require at least 75%
    result = _break_at_boundary(text)
    assert result.endswith(". ")

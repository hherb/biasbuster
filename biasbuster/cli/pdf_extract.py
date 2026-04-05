"""PDF text extraction for the BiasBuster CLI.

Uses pdfplumber for layout-aware text extraction from PDF files.
Falls back to basic extraction if pdfplumber is not installed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string with page breaks preserved.

    Raises:
        ImportError: If pdfplumber is not installed.
        FileNotFoundError: If the file does not exist.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF text extraction. "
            "Install it with: uv add pdfplumber"
        )

    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append(text)
            else:
                logger.debug("No text on page %d of %s", i + 1, pdf_path)

    if not pages:
        logger.warning("No text extracted from any page of %s", pdf_path)
        return ""

    return "\n\n".join(pages)

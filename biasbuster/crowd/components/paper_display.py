"""Paper display component for the crowd annotation UI.

Renders paper metadata (title, abstract, authors, funding, journal, MeSH)
in a readable format. Reuses build_user_message() from the annotator module
for consistency with the production review workflow.
"""

import html
from typing import Any

from nicegui import ui

from biasbuster.annotators import build_user_message


def render_paper(paper: dict, container: ui.element) -> None:
    """Render paper metadata inside the given NiceGUI container.

    Args:
        paper: Paper dict with keys: pmid, title, abstract, authors,
               grants, journal, mesh_terms, year.
        container: NiceGUI element to render into (cleared first).
    """
    container.clear()
    with container:
        pmid = paper.get("pmid", "")
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        # Build metadata dict for build_user_message (excluding retraction data)
        metadata = {
            "authors": paper.get("authors"),
            "grants": paper.get("grants"),
            "journal": paper.get("journal"),
            "mesh_terms": paper.get("mesh_terms"),
        }

        # Use the shared prompt builder (same view as the AI model received)
        prompt_text = build_user_message(pmid, title, abstract, metadata)

        # Render as styled text
        with ui.card().classes("w-full"):
            ui.label("Paper Details").classes("text-lg font-bold mb-2")
            if paper.get("year"):
                ui.label(f"Year: {paper['year']}").classes("text-sm text-grey-7")
            ui.separator()
            ui.html(
                f'<pre style="white-space: pre-wrap; word-wrap: break-word; '
                f'font-family: inherit; font-size: 0.9rem; margin: 0; '
                f'user-select: text;">'
                f"{html.escape(prompt_text)}</pre>"
            ).classes("w-full")


def render_paper_compact(paper: dict, container: ui.element) -> None:
    """Render a compact view with just title and abstract.

    Used as an alternative tab in the annotation view.
    """
    container.clear()
    with container:
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "No abstract available.")

        with ui.card().classes("w-full"):
            ui.label(title).classes("text-lg font-bold")
            if paper.get("journal"):
                ui.label(
                    f"{paper['journal']} ({paper.get('year', '')})"
                ).classes("text-sm text-grey-7 mb-2")
            ui.separator()
            ui.html(
                f'<div style="white-space: pre-wrap; line-height: 1.6; '
                f'user-select: text;">'
                f"{html.escape(abstract)}</div>"
            ).classes("w-full")

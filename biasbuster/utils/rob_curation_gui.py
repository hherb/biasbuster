"""RoB 2 Manual Curation Tool — review-first NiceGUI web application.

Interactive tool for manually entering per-study Cochrane Risk of Bias 2
ratings. The workflow is review-first: the curator opens a systematic
review, reads its RoB 2 traffic-light figure, and enters every trial's
ratings in one session. The tool resolves trial names to PMIDs via
PubMed title-match and persists via the v2 DB invariant.

Queue:
    ``dataset/rob_review_queue.json`` — one entry per review to curate.
    Each entry has the review's identifiers and a list of trials entered
    so far. The curator adds trials as they read the figure.

Usage:
    uv run python -m biasbuster.utils.rob_curation_gui
    uv run python -m biasbuster.utils.rob_curation_gui --port 9090
    uv run python -m biasbuster.utils.rob_curation_gui --queue path/to/queue.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from nicegui import ui

from biasbuster.database import (
    CURRENT_ROB_SOURCE_VERSION,
    VALID_ROB_LEVELS,
    Database,
    RoBInvariantError,
)

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("dataset/biasbuster.db")
DEFAULT_QUEUE = Path("dataset/rob_review_queue.json")

ROB2_DOMAINS = [
    ("randomization_bias", "D1 — Randomisation",
     "Bias from random sequence generation and allocation concealment."),
    ("deviation_bias", "D2 — Deviations",
     "Bias from deviations from intended interventions, failure of blinding."),
    ("missing_outcome_bias", "D3 — Missing data",
     "Bias from attrition, incomplete follow-up, excluded participants."),
    ("measurement_bias", "D4 — Measurement",
     "Bias from outcome assessor blinding or subjective outcome measures."),
    ("reporting_bias", "D5 — Reporting",
     "Bias from selective reporting, outcome switching, analysis flexibility."),
]

RATING_OPTIONS = {"": "—", "low": "Low", "some_concerns": "Some concerns", "high": "High"}
RATING_COLOURS = {"low": "positive", "some_concerns": "warning", "high": "negative"}

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# -- Queue I/O -----------------------------------------------------------

def load_queue(path: Path) -> list[dict]:
    """Load the review queue.

    Each entry: ``{"review_pmid", "review_pmcid", "review_title",
    "review_url", "review_doi", "status", "trials": [...]}``.
    """
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_queue(path: Path, queue: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)


# -- PMID resolution (strict, title-confirmed) --------------------------

def resolve_trial_pmid(first_author: str, year: str, client: httpx.Client | None = None) -> list[dict]:
    """Search PubMed for (author, year). Return candidate dicts with pmid + title.

    Returns up to 10 candidates sorted by PubMed relevance. The caller
    must confirm the match by title. Never silently picks 'first of many'.
    """
    term = f"{first_author}[Author] AND {year}[Date - Publication]"
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": "10",
        "retmode": "json",
    }
    owns = client is None
    if owns:
        client = httpx.Client(timeout=30)
    try:
        resp = client.get(PUBMED_ESEARCH, params=params)
        resp.raise_for_status()
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
        # Fetch titles
        resp2 = client.get(PUBMED_EFETCH, params={
            "db": "pubmed", "id": ",".join(pmids[:10]),
            "rettype": "abstract", "retmode": "xml",
        })
        resp2.raise_for_status()
        root = ET.fromstring(resp2.text)
        candidates = []
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//MedlineCitation/PMID")
            title_el = article.find(".//ArticleTitle")
            if pmid_el is None or pmid_el.text is None:
                continue
            candidates.append({
                "pmid": pmid_el.text.strip(),
                "title": "".join(title_el.itertext()).strip() if title_el is not None else "",
            })
        return candidates
    except Exception as exc:
        logger.warning("PMID resolution failed for %s %s: %s", first_author, year, exc)
        return []
    finally:
        if owns:
            client.close()


# -- URL helpers ---------------------------------------------------------

def _europmc_url(pmid: str) -> str:
    return f"https://europepmc.org/article/MED/{pmid}"


def _pubmed_url(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def _pmc_fulltext_url(pmcid: str) -> str:
    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"


# -- Main app ------------------------------------------------------------

def create_app(db_path: Path, queue_path: Path) -> None:
    """Build the review-first curation interface."""
    db = Database(db_path)
    db.initialize()
    queue = load_queue(queue_path)

    current_review_idx = {"value": 0}

    def pending_indices() -> list[int]:
        return [i for i, r in enumerate(queue) if r.get("status", "pending") != "completed"]

    def current_review() -> dict | None:
        pi = pending_indices()
        if not pi:
            return None
        idx = current_review_idx["value"] % len(pi)
        return queue[pi[idx]]

    def counts() -> dict[str, Any]:
        total_reviews = len(queue)
        completed = sum(1 for r in queue if r.get("status") == "completed")
        total_trials = sum(len(r.get("trials", [])) for r in queue)
        saved_trials = sum(
            1 for r in queue for t in r.get("trials", []) if t.get("saved")
        )
        return {
            "reviews": f"{completed}/{total_reviews}",
            "trials": f"{saved_trials} saved from {total_trials} entered",
        }

    @ui.page("/")
    def main_page():
        # --- Header ---
        with ui.header().classes("items-center justify-between"):
            ui.label("RoB 2 Curation — Review-First").classes("text-xl font-bold")
            status_label = ui.label()

        def refresh_status():
            c = counts()
            status_label.text = f"Reviews: {c['reviews']}  |  Trials: {c['trials']}"

        refresh_status()

        # --- Add review dialog ---
        def show_add_review_dialog():
            with ui.dialog() as dlg, ui.card().classes("w-96"):
                ui.label("Add a Review to the Queue").classes("text-h6")
                pmid_input = ui.input(label="Review PubMed ID (PMID)").classes("w-full")
                pmcid_input = ui.input(label="Review PMC ID (e.g. PMC1234567)").classes("w-full")
                title_input = ui.input(label="Review title").classes("w-full")
                doi_input = ui.input(label="Review DOI (optional)").classes("w-full")
                url_input = ui.input(label="Direct URL (optional — for paywalled PDFs)").classes("w-full")

                with ui.row():
                    def on_add():
                        pmid = pmid_input.value.strip()
                        pmcid = pmcid_input.value.strip()
                        title = title_input.value.strip()
                        if not title:
                            ui.notify("Review title is required", type="warning")
                            return
                        entry = {
                            "review_pmid": pmid,
                            "review_pmcid": pmcid,
                            "review_title": title,
                            "review_doi": doi_input.value.strip(),
                            "review_url": url_input.value.strip(),
                            "status": "pending",
                            "trials": [],
                        }
                        queue.append(entry)
                        save_queue(queue_path, queue)
                        ui.notify(f"Added review: {title[:60]}", type="positive")
                        dlg.close()
                        refresh_status()
                        render_review()

                    ui.button("Add", on_click=on_add, color="positive")
                    ui.button("Cancel", on_click=dlg.close, color="grey")
            dlg.open()

        # --- Content area ---
        with ui.row().classes("w-full q-pa-sm items-center"):
            ui.button("Add Review", on_click=show_add_review_dialog, icon="add_circle", color="primary")
            ui.button("Prev Review", on_click=lambda: nav(-1), icon="chevron_left")
            ui.button("Next Review", on_click=lambda: nav(1), icon="chevron_right")
            ui.button("Mark Review Complete", on_click=lambda: mark_review_complete(), color="positive", icon="check_circle")

        content_area = ui.column().classes("w-full q-pa-md")

        def nav(delta: int):
            pi = pending_indices()
            if not pi:
                return
            current_review_idx["value"] = (current_review_idx["value"] + delta) % len(pi)
            render_review()

        def mark_review_complete():
            review = current_review()
            if review:
                review["status"] = "completed"
                save_queue(queue_path, queue)
                ui.notify("Review marked complete", type="positive")
                refresh_status()
                render_review()

        def render_review():
            content_area.clear()
            review = current_review()
            if review is None:
                with content_area:
                    ui.label("No pending reviews. Add one with the button above.").classes("text-lg text-positive")
                return

            review_pmid = review.get("review_pmid", "")
            review_pmcid = review.get("review_pmcid", "")
            review_title = review.get("review_title", "")
            review_url = review.get("review_url", "")
            trials = review.get("trials", [])

            with content_area:
                # --- Review card ---
                with ui.card().classes("w-full bg-blue-50"):
                    ui.label("Source Review").classes("text-subtitle1 text-bold")
                    ui.label(review_title).classes("text-body1")
                    with ui.row().classes("q-mt-xs"):
                        if review_pmcid:
                            ui.link(
                                f"PMC full text ({review_pmcid})",
                                _pmc_fulltext_url(review_pmcid),
                                new_tab=True,
                            ).classes("text-bold")
                        if review_pmid:
                            ui.link("PubMed", _pubmed_url(review_pmid), new_tab=True)
                        if review_url:
                            ui.link("Direct / PDF link", review_url, new_tab=True)
                        if review.get("review_doi"):
                            ui.link("DOI", f"https://doi.org/{review['review_doi']}", new_tab=True)

                ui.separator()

                # --- RoB 2 quick reference (collapsed) ---
                with ui.expansion("RoB 2 Quick Reference", icon="info"):
                    for _field, label, desc in ROB2_DOMAINS:
                        ui.label(f"{label}: {desc}").classes("text-caption")
                    ui.label(
                        "Ratings: Low = sound methods. Some concerns = raises doubt. "
                        "High = serious problem. Enter exactly what the figure shows."
                    ).classes("text-caption text-bold q-mt-xs")

                ui.separator()

                # --- Existing trials entered for this review ---
                if trials:
                    ui.label(f"Trials entered so far: {len(trials)}").classes("text-subtitle2 q-mt-sm")
                    columns = [
                        {"name": "author", "label": "Author+Year", "field": "study_id", "align": "left"},
                        {"name": "pmid", "label": "PMID", "field": "pmid", "align": "left"},
                        {"name": "title", "label": "Title", "field": "title", "align": "left"},
                        {"name": "overall", "label": "Overall", "field": "overall_rob", "align": "center"},
                        {"name": "d1", "label": "D1", "field": "randomization_bias", "align": "center"},
                        {"name": "d2", "label": "D2", "field": "deviation_bias", "align": "center"},
                        {"name": "d3", "label": "D3", "field": "missing_outcome_bias", "align": "center"},
                        {"name": "d4", "label": "D4", "field": "measurement_bias", "align": "center"},
                        {"name": "d5", "label": "D5", "field": "reporting_bias", "align": "center"},
                        {"name": "saved", "label": "DB", "field": "saved_str", "align": "center"},
                    ]
                    rows = []
                    for t in trials:
                        row = dict(t)
                        row["title"] = (t.get("title") or "")[:50]
                        row["saved_str"] = "yes" if t.get("saved") else "—"
                        rows.append(row)
                    ui.table(columns=columns, rows=rows).classes("w-full q-mt-xs")

                ui.separator()

                # --- Add trial form ---
                ui.label("Add a Trial from this Review's RoB Figure").classes("text-h6 q-mt-md")

                with ui.card().classes("w-full"):
                    with ui.row().classes("items-end"):
                        author_input = ui.input(
                            label="First author surname (from figure)",
                            placeholder="e.g. Smith",
                        ).classes("w-40")
                        year_input = ui.input(
                            label="Year",
                            placeholder="e.g. 2020",
                        ).classes("w-24")
                        resolve_btn = ui.button("Search PubMed", icon="search", color="primary")

                    # Candidate results area
                    candidates_area = ui.column().classes("w-full q-mt-sm")
                    selected_pmid = {"value": "", "title": ""}

                    def on_resolve():
                        author = author_input.value.strip()
                        year = year_input.value.strip()
                        if not author or not year:
                            ui.notify("Enter both author and year", type="warning")
                            return
                        candidates_area.clear()
                        with candidates_area:
                            ui.label("Searching PubMed...").classes("text-caption")
                        candidates = resolve_trial_pmid(author, year)
                        candidates_area.clear()
                        if not candidates:
                            with candidates_area:
                                ui.label(f"No results for '{author} {year}'. Try spelling variants.").classes(
                                    "text-caption text-negative"
                                )
                            return
                        with candidates_area:
                            ui.label(f"{len(candidates)} candidates — click the correct trial:").classes(
                                "text-caption"
                            )
                            for cand in candidates:
                                pmid = cand["pmid"]
                                title = cand["title"]

                                def make_select(p=pmid, t=title):
                                    def select():
                                        selected_pmid["value"] = p
                                        selected_pmid["title"] = t
                                        ui.notify(f"Selected PMID {p}: {t[:60]}", type="info")
                                    return select

                                with ui.row().classes("items-center q-mb-xs"):
                                    ui.button(
                                        f"PMID {pmid}",
                                        on_click=make_select(),
                                        color="grey-4",
                                    ).props("flat dense")
                                    ui.label(title[:90]).classes("text-caption")

                    resolve_btn.on_click(on_resolve)

                    ui.separator().classes("q-my-sm")

                    # Rating dropdowns — compact row
                    ui.label("Ratings (from the figure):").classes("text-subtitle2")
                    domain_selects: dict[str, ui.select] = {}
                    with ui.row().classes("flex-wrap"):
                        for field, label, _desc in ROB2_DOMAINS:
                            sel = ui.select(
                                label=label,
                                options=RATING_OPTIONS,
                                value="",
                            ).classes("w-44")
                            domain_selects[field] = sel
                        overall_select = ui.select(
                            label="OVERALL",
                            options=RATING_OPTIONS,
                            value="",
                        ).classes("w-44 text-bold")

                    with ui.row().classes("q-mt-sm items-end"):
                        curator_input = ui.input(
                            label="Curator name",
                            value=os.environ.get("BIASBUSTER_CURATOR", ""),
                        ).classes("w-40")
                        outcome_input = ui.input(
                            label="Outcome (if per-outcome row)",
                            placeholder="e.g. primary, QoL",
                        ).classes("w-40")
                        notes_input = ui.input(label="Notes", placeholder="optional").classes("w-64")

                    def on_add_trial():
                        pmid = selected_pmid["value"]
                        title = selected_pmid["title"]
                        author = author_input.value.strip()
                        year = year_input.value.strip()

                        if not pmid:
                            ui.notify("Resolve and select a PMID first", type="warning")
                            return

                        ratings = {}
                        missing = []
                        for field, label, _ in ROB2_DOMAINS:
                            v = domain_selects[field].value
                            if not v:
                                missing.append(label)
                            ratings[field] = v
                        ratings["overall_rob"] = overall_select.value
                        if not overall_select.value:
                            missing.append("Overall")
                        if missing:
                            ui.notify(f"Fill all ratings: {', '.join(missing)}", type="warning")
                            return

                        curator = curator_input.value.strip()
                        if not curator:
                            ui.notify("Enter your name", type="warning")
                            return

                        # Build provenance
                        provenance = {
                            "review_pmid": review_pmid,
                            "review_pmcid": review_pmcid,
                            "review_title": review_title,
                            "review_url": review_url,
                            "extraction_method": "manual_curation",
                            "curator": curator,
                            "curated_at": datetime.now(timezone.utc).isoformat(),
                            "study_id_in_figure": f"{author} {year}",
                            "outcome": outcome_input.value.strip(),
                            "notes": notes_input.value.strip(),
                        }

                        paper_dict = {
                            "pmid": pmid,
                            "title": title,
                            "abstract": "",
                            "source": "cochrane_rob",
                            "cochrane_review_pmid": review_pmid or review_pmcid,
                            "cochrane_review_doi": review.get("review_doi", ""),
                            "cochrane_review_title": review_title,
                            "rob_provenance": provenance,
                            "rob_source_version": CURRENT_ROB_SOURCE_VERSION,
                            **ratings,
                        }

                        # Save to DB
                        saved = False
                        try:
                            db.upsert_cochrane_paper_v2(paper_dict)
                            saved = True
                            ui.notify(f"Saved PMID {pmid} to DB", type="positive")
                        except RoBInvariantError as e:
                            ui.notify(f"Invariant error: {e}", type="negative")
                        except Exception as e:
                            ui.notify(f"DB error: {e}", type="negative")

                        # Record in queue (even if DB save failed — preserves entered work)
                        trial_entry = {
                            "study_id": f"{author} {year}",
                            "pmid": pmid,
                            "title": title[:80],
                            "outcome": outcome_input.value.strip(),
                            "saved": saved,
                            **ratings,
                        }
                        review.setdefault("trials", []).append(trial_entry)
                        save_queue(queue_path, queue)

                        # Reset form for next trial
                        selected_pmid["value"] = ""
                        selected_pmid["title"] = ""
                        author_input.value = ""
                        year_input.value = ""
                        for sel in domain_selects.values():
                            sel.value = ""
                        overall_select.value = ""
                        outcome_input.value = ""
                        notes_input.value = ""
                        candidates_area.clear()
                        refresh_status()
                        render_review()

                    ui.button(
                        "Add Trial + Save to DB",
                        on_click=on_add_trial,
                        color="positive",
                        icon="add",
                    ).classes("q-mt-sm")

        render_review()

    @ui.page("/queue")
    def queue_page():
        """Overview of all reviews in the queue."""
        ui.label("Review Queue").classes("text-h5 q-ma-md")
        columns = [
            {"name": "idx", "label": "#", "field": "idx", "align": "left"},
            {"name": "status", "label": "Status", "field": "status", "align": "center"},
            {"name": "title", "label": "Review Title", "field": "title", "align": "left"},
            {"name": "pmid", "label": "PMID", "field": "review_pmid", "align": "left"},
            {"name": "trials", "label": "Trials", "field": "n_trials", "align": "center"},
            {"name": "saved", "label": "Saved", "field": "n_saved", "align": "center"},
        ]
        rows = []
        for i, r in enumerate(queue):
            rows.append({
                "idx": i + 1,
                "status": r.get("status", "pending"),
                "title": (r.get("review_title") or "")[:80],
                "review_pmid": r.get("review_pmid", ""),
                "n_trials": len(r.get("trials", [])),
                "n_saved": sum(1 for t in r.get("trials", []) if t.get("saved")),
            })
        ui.table(columns=columns, rows=rows).classes("w-full q-ma-md")
        ui.link("Back to curation", "/").classes("q-ma-md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--port", type=int, default=8085)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    create_app(args.db, args.queue)
    ui.run(
        title="BiasBuster — RoB 2 Curation",
        port=args.port,
        reload=False,
        show=True,
    )


if __name__ == "__main__":
    main()

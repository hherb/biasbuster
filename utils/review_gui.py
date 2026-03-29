"""
NiceGUI-based Review Tool

Web-based GUI for reviewing and validating bias annotations.
Presents the reviewer with the same prompt the model received (left panel)
and a structured assessment form matching the LLM JSON schema (right panel).

Supports filtering by flagged-for-review status.

Usage:
    uv run python -m utils.review_gui --model anthropic
    uv run python -m utils.review_gui  # shows model picker
"""

import logging
import sys
from pathlib import Path

from nicegui import ui

from annotators import build_user_message
from database import Database
from prompts import REVIEWER_REFERENCE_CARD
from utils.review_form import (
    SEVERITY_OPTIONS,
    build_review_form,
    collect_form_data,
)

logger = logging.getLogger(__name__)

# Filter mode constants
FILTER_ALL = "all"
FILTER_FLAGGED = "flagged"
FILTER_UNVALIDATED = "unvalidated"


def load_annotations_for_review(
    db: Database, model_name: str
) -> list[dict]:
    """Load annotations with paper metadata and human review data."""
    annotations = db.get_annotations_with_paper_data(model_name)
    reviews = {
        r["pmid"]: r for r in db.get_reviews(model_name=model_name)
    }

    rows = []
    for ann in annotations:
        pmid = ann.get("pmid", "")
        stat = ann.get("statistical_reporting", {})
        spin = ann.get("spin", {})
        coi = ann.get("conflict_of_interest", {})
        review = reviews.get(pmid, {})

        rows.append({
            "pmid": pmid,
            "title": str(ann.get("title", "")),
            "overall_severity": ann.get("overall_severity", ""),
            "overall_bias_probability": ann.get("overall_bias_probability", ""),
            "statistical_severity": (
                stat.get("severity", "") if isinstance(stat, dict) else ""
            ),
            "relative_only": (
                stat.get("relative_only", "") if isinstance(stat, dict) else ""
            ),
            "spin_level": (
                spin.get("spin_level", "") if isinstance(spin, dict) else ""
            ),
            "funding_type": (
                coi.get("funding_type", "") if isinstance(coi, dict) else ""
            ),
            "confidence": ann.get("confidence", ""),
            "reasoning_summary": str(ann.get("reasoning", "")),
            "abstract_text": str(ann.get("abstract_text", "")),
            "HUMAN_VALIDATED": "True" if review.get("validated") else "",
            "HUMAN_OVERRIDE_SEVERITY": review.get("override_severity") or "",
            "HUMAN_NOTES": review.get("notes") or "",
            "FLAGGED": bool(review.get("flagged")),
            # Stash full data for the detail panel (not shown in grid)
            "_ann": ann,
            "_paper_metadata": ann.get("_paper_metadata", {}),
            "_human_annotation": review.get("annotation"),
        })
    return rows


def _apply_filter(rows: list[dict], mode: str) -> list[dict]:
    """Return rows matching the filter mode."""
    if mode == FILTER_FLAGGED:
        return [r for r in rows if r.get("FLAGGED")]
    if mode == FILTER_UNVALIDATED:
        return [r for r in rows if r.get("HUMAN_VALIDATED") != "True"]
    return rows


def create_app(
    db: Database, model_name: str, all_models: list[str] | None = None
) -> None:
    """Create the NiceGUI review application."""
    rows = load_annotations_for_review(db, model_name)
    state = {
        "modified": False,
        "rows": rows,
        "model_name": model_name,
        "filter_mode": FILTER_ALL,
        "form_refs": None,
        "current_pmid": None,
    }
    if all_models is None:
        all_models = [m for m in db.get_model_names() if m != "human"]

    # --- Column definitions for AG Grid ---
    column_defs = [
        {
            "field": "FLAGGED", "headerName": "F", "width": 50,
            "sortable": True, "filter": True, "pinned": "left",
            ":cellRenderer": """params =>
                params.value ? '<span style="color:#f44336">&#9873;</span>' : ''
            """,
        },
        {
            "field": "pmid", "headerName": "PMID", "width": 110,
            "sortable": True, "filter": True, "pinned": "left",
        },
        {
            "field": "title", "headerName": "Title", "width": 250,
            "sortable": True, "filter": True, "tooltipField": "title",
        },
        {
            "field": "overall_severity", "headerName": "Severity",
            "width": 100, "sortable": True, "filter": True,
        },
        {
            "field": "overall_bias_probability", "headerName": "Prob",
            "width": 70, "sortable": True, "filter": "agNumberColumnFilter",
        },
        {
            "field": "statistical_severity", "headerName": "Stat",
            "width": 80, "sortable": True, "filter": True,
        },
        {
            "field": "spin_level", "headerName": "Spin",
            "width": 80, "sortable": True, "filter": True,
        },
        {
            "field": "funding_type", "headerName": "Fund",
            "width": 90, "sortable": True, "filter": True,
        },
        {
            "field": "confidence", "headerName": "Conf",
            "width": 70, "sortable": True, "filter": True,
        },
        {
            "field": "HUMAN_VALIDATED", "headerName": "Valid",
            "width": 70, "sortable": True, "filter": True,
        },
        {
            "field": "HUMAN_OVERRIDE_SEVERITY", "headerName": "Ovrd",
            "width": 70, "sortable": True, "filter": True,
        },
    ]

    filtered_rows = _apply_filter(rows, state["filter_mode"])
    grid_options = {
        "columnDefs": column_defs,
        "rowData": filtered_rows,
        "defaultColDef": {"resizable": True, "wrapHeaderText": True},
        "rowSelection": {"mode": "singleRow"},
        "enableCellTextSelection": True,
        "tooltipShowDelay": 300,
        ":getRowStyle": """params => {
            if (params.data.HUMAN_VALIDATED === 'True')
                return {'background-color': '#e8f5e9'};
            if (params.data.FLAGGED && params.data.HUMAN_VALIDATED !== 'True')
                return {'border-left': '4px solid #f44336'};
        }""",
    }

    # --- Build UI ---
    ui.page_title(f"Review: {model_name}")

    # Header
    with ui.header().classes("items-center justify-between q-px-md"):
        with ui.row().classes("items-center gap-4"):
            ui.label("Reviewing:").classes("text-h6")
            model_select = ui.select(
                all_models,
                value=model_name,
                on_change=lambda e: switch_model(e.value),
            ).props("dense outlined dark").classes("min-w-[12rem]")
        with ui.row().classes("items-center gap-2"):
            status_label = ui.label("No changes").classes("text-caption")
            ui.button("Save", on_click=lambda: do_save()).props(
                "icon=save color=primary"
            )

    # Stats bar
    with ui.row().classes("q-px-md q-py-xs items-center gap-4"):
        stats_label = ui.label()

        def update_stats() -> None:
            total = len(state["rows"])
            validated = sum(
                1 for r in state["rows"]
                if r.get("HUMAN_VALIDATED") == "True"
            )
            flagged = sum(1 for r in state["rows"] if r.get("FLAGGED"))
            pct = validated / total * 100 if total > 0 else 0
            showing = len(_apply_filter(state["rows"], state["filter_mode"]))
            stats_label.text = (
                f"Total: {total} | Showing: {showing} | "
                f"Validated: {validated}/{total} ({pct:.0f}%) | "
                f"Flagged: {flagged}"
            )

        update_stats()

    # Filter controls
    with ui.row().classes("q-px-md q-py-xs items-center gap-2"):
        filter_input = ui.input(
            "Quick filter...",
            on_change=lambda e: grid.run_grid_method(
                "setGridOption", "quickFilterText", e.value
            ),
        ).classes("w-48")
        ui.button("Clear", on_click=lambda: (
            filter_input.set_value(""),
            grid.run_grid_method("setGridOption", "quickFilterText", ""),
        )).props("flat size=sm")

        ui.separator().props("vertical")

        def set_filter(mode: str) -> None:
            state["filter_mode"] = mode
            filtered = _apply_filter(state["rows"], mode)
            grid.options["rowData"] = filtered
            grid.update()
            filter_input.set_value("")
            update_stats()

        ui.button("All", on_click=lambda: set_filter(FILTER_ALL)).props(
            "flat size=sm"
        )
        ui.button("Flagged", on_click=lambda: set_filter(FILTER_FLAGGED)).props(
            "flat size=sm color=red"
        )
        ui.button("Unvalidated", on_click=lambda: set_filter(FILTER_UNVALIDATED)).props(
            "flat size=sm color=orange"
        )

        ui.separator().props("vertical")

        ui.button(
            "Next Flagged", on_click=lambda: jump_to_next(flagged_only=True),
        ).props("outline size=sm color=red")
        ui.button(
            "Next Unvalidated", on_click=lambda: jump_to_next(flagged_only=False),
        ).props("outline size=sm color=primary")

        ui.separator().props("vertical")

        async def export_csv() -> None:
            output_path = Path(
                f"dataset/export/{state['model_name']}_review.csv"
            )
            db.export_review_csv(state["model_name"], output_path)
            ui.notify(f"CSV exported to {output_path}", type="positive")

        ui.button("Export CSV", on_click=export_csv).props(
            "flat size=sm color=secondary icon=download"
        )

    # AG Grid (compact, top portion)
    grid = ui.aggrid(grid_options).classes("q-mx-md").style("height: 30vh;")

    # --- Detail: left/right split below grid ---
    detail_placeholder = ui.label(
        "Click a row above to review"
    ).classes("q-px-md q-py-sm text-grey text-h6")

    detail_splitter = ui.splitter(value=50).classes(
        "q-mx-md"
    ).style("height: 60vh;")
    detail_splitter.visible = False

    with detail_splitter:
        # LEFT panel: prompt as model sees it
        with detail_splitter.before:
            with ui.tabs().props("dense") as left_tabs:
                tab_prompt = ui.tab("Prompt (as model sees it)")
                tab_abstract = ui.tab("Abstract only")
                tab_guidelines = ui.tab("Guidelines")
            left_panels = ui.tab_panels(left_tabs, value="Prompt (as model sees it)").classes(
                "w-full"
            ).style("height: calc(60vh - 3rem); overflow-y: auto;")
            with left_panels:
                with ui.tab_panel("Prompt (as model sees it)"):
                    prompt_container = ui.column().classes("w-full")
                with ui.tab_panel("Abstract only"):
                    abstract_container = ui.column().classes("w-full")
                with ui.tab_panel("Guidelines"):
                    with ui.column().classes("w-full"):
                        ui.markdown(
                            REVIEWER_REFERENCE_CARD.replace("\n", "  \n")
                        ).style(
                            "white-space: pre-wrap; font-size: 0.85rem;"
                            " user-select: text;"
                        )

        # RIGHT panel: structured assessment form
        with detail_splitter.after:
            with ui.column().classes("w-full").style(
                "height: 60vh; overflow-y: auto; padding: 0.5rem;"
            ) as right_col:
                # Review action bar at top of form
                with ui.row().classes(
                    "w-full items-center gap-2 q-pb-sm"
                ).style("border-bottom: 1px solid #e0e0e0;"):
                    form_validated = ui.select(
                        ["", "True", "False"],
                        value="",
                        label="Validated",
                    ).classes("min-w-[8rem]")
                    form_notes = ui.input(
                        "Notes",
                    ).classes("flex-grow")
                    ui.button(
                        "Save this paper",
                        on_click=lambda: save_current_paper(),
                    ).props("color=primary size=sm icon=save")

                form_container = ui.column().classes("w-full gap-1")

    # --- Event handlers ---

    def show_detail(row_data: dict) -> None:
        """Populate left/right panels for the selected paper."""
        pmid = row_data.get("pmid", "")
        state["current_pmid"] = pmid
        detail_placeholder.visible = False
        detail_splitter.visible = True

        ann = row_data.get("_ann", {})
        meta = row_data.get("_paper_metadata", {})

        # Build the prompt text the model received
        prompt_text = build_user_message(
            pmid=pmid,
            title=row_data.get("title", ""),
            abstract=row_data.get("abstract_text", ""),
            metadata=meta,
        )

        # LEFT: prompt tab
        prompt_container.clear()
        with prompt_container:
            ui.label(f"PMID: {pmid}").classes("text-bold")
            ui.html(
                f'<pre style="white-space: pre-wrap; word-break: break-word;'
                f' font-size: 0.85rem; user-select: text;'
                f' font-family: monospace;">{_escape_html(prompt_text)}</pre>'
            )

        # LEFT: abstract tab
        abstract_container.clear()
        with abstract_container:
            ui.label(row_data.get("title", "")).classes("text-bold text-body1")
            ui.label(
                row_data.get("abstract_text", "(no abstract available)")
            ).classes("text-body2").style(
                "white-space: pre-wrap; word-break: break-word;"
                " user-select: text;"
            )

        # RIGHT: populate form with human annotation if exists, else model's
        human_ann = row_data.get("_human_annotation")
        if human_ann:
            if isinstance(human_ann, str):
                import json as _json
                try:
                    human_ann = _json.loads(human_ann)
                except (ValueError, TypeError):
                    human_ann = None
        form_ann = human_ann if isinstance(human_ann, dict) else ann
        state["form_refs"] = build_review_form(form_ann, form_container)

        # Populate validated / notes from row
        form_validated.value = row_data.get("HUMAN_VALIDATED", "")
        form_notes.value = row_data.get("HUMAN_NOTES", "")

    async def on_cell_clicked(e) -> None:
        data = e.args
        if data is None:
            return
        event_row = data.get("data") if isinstance(data, dict) else None
        if event_row is None:
            return
        pmid = event_row.get("pmid", "")
        row_data = next(
            (r for r in state["rows"] if r.get("pmid") == pmid),
            event_row,
        )
        show_detail(row_data)

    grid.on("cellClicked", on_cell_clicked)

    def save_current_paper() -> None:
        """Save the structured form data for the currently selected paper."""
        pmid = state.get("current_pmid")
        if not pmid:
            ui.notify("No paper selected", type="warning")
            return
        if state["form_refs"] is None:
            ui.notify("No form data to save", type="warning")
            return

        try:
            annotation = collect_form_data(state["form_refs"])
            validated = form_validated.value == "True"
            override_sev = annotation.get("overall_severity")
            notes = form_notes.value or None

            db.upsert_review(
                pmid=pmid,
                model_name=state["model_name"],
                validated=validated,
                override_severity=override_sev,
                notes=notes,
                annotation=annotation,
            )

            # Update the row in state
            for row in state["rows"]:
                if row.get("pmid") == pmid:
                    row["HUMAN_VALIDATED"] = "True" if validated else ""
                    row["HUMAN_OVERRIDE_SEVERITY"] = override_sev or ""
                    row["HUMAN_NOTES"] = notes or ""
                    row["_human_annotation"] = annotation
                    break

            # Refresh grid display
            filtered = _apply_filter(state["rows"], state["filter_mode"])
            grid.options["rowData"] = filtered
            grid.update()
            update_stats()

            status_label.text = f"Saved PMID {pmid}"
            status_label.classes("text-green", remove="text-red text-grey")
            ui.notify(f"Saved review for PMID {pmid}", type="positive")
        except Exception as exc:
            logger.exception("Save failed for PMID %s", pmid)
            ui.notify(f"Save failed: {exc}", type="negative")

    def do_save() -> None:
        """Save the current paper (header button delegates here)."""
        save_current_paper()

    def switch_model(new_model: str) -> None:
        """Reload grid data for a different model."""
        new_rows = load_annotations_for_review(db, new_model)
        state["rows"] = new_rows
        state["model_name"] = new_model
        state["filter_mode"] = FILTER_ALL
        state["current_pmid"] = None
        state["form_refs"] = None
        grid.options["rowData"] = new_rows
        grid.update()
        ui.page_title(f"Review: {new_model}")
        filter_input.set_value("")
        grid.run_grid_method("setGridOption", "quickFilterText", "")
        update_stats()
        detail_placeholder.visible = True
        detail_splitter.visible = False
        ui.notify(
            f"Loaded {len(new_rows)} annotations for {new_model}",
            type="info",
        )

    def jump_to_next(flagged_only: bool = False) -> None:
        """Jump to the next unvalidated (or flagged+unvalidated) row."""
        visible = _apply_filter(state["rows"], state["filter_mode"])
        for i, row in enumerate(visible):
            if row.get("HUMAN_VALIDATED") == "True":
                continue
            if flagged_only and not row.get("FLAGGED"):
                continue
            grid.run_grid_method("ensureIndexVisible", i, "middle")
            show_detail(row)
            return
        label = "flagged" if flagged_only else "unvalidated"
        ui.notify(f"No more {label} papers!", type="info")

    ui.label(
        "Click a row to review. Left panel shows the prompt as the model sees it. "
        "Right panel is your structured assessment form."
    ).classes("q-px-md text-caption text-grey")


def _escape_html(text: str) -> str:
    """Escape HTML special characters for safe display."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> None:
    """Entry point for the review GUI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GUI for reviewing bias annotations"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name to review"
    )
    parser.add_argument(
        "--db-path", default=None, help="Path to SQLite database",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port for the web server (default: 8080)",
    )
    args = parser.parse_args()

    try:
        from config import Config
        cfg = Config()
    except ImportError:
        cfg = None

    db_path = args.db_path or (cfg.db_path if cfg else "dataset/biasbuster.db")
    db = Database(db_path)
    db.initialize()

    models = [m for m in db.get_model_names() if m != "human"]

    if not models:
        print("No model annotations found in database")
        print("Usage: uv run python -m utils.review_gui --model anthropic")
        db.close()
        sys.exit(1)

    model_name = args.model if args.model in models else models[0]

    create_app(db, model_name, all_models=models)
    ui.run(
        title=f"Review: {model_name}",
        port=args.port,
        reload=False,
        on_air=False,
    )
    db.close()


if __name__ == "__main__":
    main()

"""
NiceGUI-based Review Tool

Web-based GUI for reviewing and validating bias annotations.
Reads from and writes to the SQLite database.

Usage:
    uv run python -m utils.review_gui --model anthropic
    uv run python -m utils.review_gui  # shows model picker
"""

import logging
import sys
from pathlib import Path

from nicegui import ui

from database import Database

logger = logging.getLogger(__name__)

SEVERITY_OPTIONS = ["", "none", "low", "moderate", "high", "critical"]


def load_annotations_for_review(
    db: Database, model_name: str
) -> list[dict]:
    """Load annotations with human review data for the review grid."""
    annotations = db.get_annotations(model_name=model_name)
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
            "title": str(ann.get("title", ""))[:100],
            "overall_severity": ann.get("overall_severity", ""),
            "overall_bias_probability": ann.get("overall_bias_probability", ""),
            "statistical_severity": stat.get("severity", "") if isinstance(stat, dict) else "",
            "relative_only": stat.get("relative_only", "") if isinstance(stat, dict) else "",
            "spin_level": spin.get("spin_level", "") if isinstance(spin, dict) else "",
            "funding_type": coi.get("funding_type", "") if isinstance(coi, dict) else "",
            "confidence": ann.get("confidence", ""),
            "reasoning_summary": str(ann.get("reasoning", ""))[:200],
            "HUMAN_VALIDATED": "True" if review.get("validated") else "",
            "HUMAN_OVERRIDE_SEVERITY": review.get("override_severity") or "",
            "HUMAN_NOTES": review.get("notes") or "",
        })
    return rows


def create_app(db: Database, model_name: str) -> None:
    """Create the NiceGUI review application."""
    rows = load_annotations_for_review(db, model_name)
    state = {"modified": False, "rows": rows, "model_name": model_name}

    # --- Column definitions for AG Grid ---
    column_defs = [
        {
            "field": "pmid",
            "headerName": "PMID",
            "width": 110,
            "sortable": True,
            "filter": True,
            "pinned": "left",
        },
        {
            "field": "title",
            "headerName": "Title",
            "width": 300,
            "sortable": True,
            "filter": True,
            "tooltipField": "title",
        },
        {
            "field": "overall_severity",
            "headerName": "Severity",
            "width": 110,
            "sortable": True,
            "filter": True,
        },
        {
            "field": "overall_bias_probability",
            "headerName": "Bias Prob",
            "width": 100,
            "sortable": True,
            "filter": "agNumberColumnFilter",
        },
        {
            "field": "statistical_severity",
            "headerName": "Stat Sev",
            "width": 100,
            "sortable": True,
            "filter": True,
        },
        {
            "field": "relative_only",
            "headerName": "Rel Only",
            "width": 90,
            "sortable": True,
            "filter": True,
        },
        {
            "field": "spin_level",
            "headerName": "Spin",
            "width": 100,
            "sortable": True,
            "filter": True,
        },
        {
            "field": "funding_type",
            "headerName": "Funding",
            "width": 110,
            "sortable": True,
            "filter": True,
        },
        {
            "field": "confidence",
            "headerName": "Conf",
            "width": 80,
            "sortable": True,
            "filter": True,
        },
        {
            "field": "reasoning_summary",
            "headerName": "Reasoning",
            "width": 250,
            "tooltipField": "reasoning_summary",
        },
        # Editable human review columns
        {
            "field": "HUMAN_VALIDATED",
            "headerName": "Validated",
            "width": 110,
            "editable": True,
            "sortable": True,
            "filter": True,
            "cellEditor": "agSelectCellEditor",
            "cellEditorParams": {"values": ["", "True", "False"]},
        },
        {
            "field": "HUMAN_OVERRIDE_SEVERITY",
            "headerName": "Override Sev",
            "width": 130,
            "editable": True,
            "sortable": True,
            "filter": True,
            "cellEditor": "agSelectCellEditor",
            "cellEditorParams": {"values": SEVERITY_OPTIONS},
        },
        {
            "field": "HUMAN_NOTES",
            "headerName": "Notes",
            "width": 250,
            "editable": True,
            "cellEditor": "agLargeTextCellEditor",
            "cellEditorPopup": True,
        },
    ]

    grid_options = {
        "columnDefs": column_defs,
        "rowData": rows,
        "defaultColDef": {
            "resizable": True,
            "wrapHeaderText": True,
        },
        "rowSelection": "single",
        "animateRows": True,
        "enableCellTextSelection": True,
        "tooltipShowDelay": 300,
        "getRowStyle": """params => {
            if (params.data.HUMAN_VALIDATED === 'True')
                return {'background-color': '#e8f5e9'};
            if (params.data.HUMAN_OVERRIDE_SEVERITY)
                return {'background-color': '#fff8e1'};
        }""",
    }

    # --- Build UI ---
    ui.page_title(f"Review: {model_name}")

    # Header
    with ui.header().classes("items-center justify-between q-px-md"):
        ui.label(f"Reviewing: {model_name} annotations").classes("text-h6")
        with ui.row().classes("items-center gap-2"):
            status_label = ui.label("No changes").classes("text-caption")
            save_btn = ui.button("Save", on_click=lambda: do_save())
            save_btn.props("icon=save color=primary")

    # Stats bar
    with ui.row().classes("q-px-md q-py-sm items-center gap-4"):
        stats_label = ui.label()

        def update_stats():
            total = len(state["rows"])
            validated = sum(
                1
                for r in state["rows"]
                if r.get("HUMAN_VALIDATED") == "True"
            )
            overridden = sum(
                1 for r in state["rows"] if r.get("HUMAN_OVERRIDE_SEVERITY")
            )
            pct = validated / total * 100 if total > 0 else 0
            stats_label.text = (
                f"Total: {total} | "
                f"Validated: {validated}/{total} ({pct:.0f}%) | "
                f"Overridden: {overridden}"
            )

        update_stats()

    # Filter controls
    with ui.row().classes("q-px-md q-py-sm items-center gap-2"):
        filter_input = ui.input(
            "Quick filter...",
            on_change=lambda e: grid.call_api_method(
                "setGridOption", "quickFilterText", e.value
            ),
        ).classes("w-64")
        ui.button(
            "Clear",
            on_click=lambda: (
                filter_input.set_value(""),
                grid.call_api_method("setGridOption", "quickFilterText", ""),
            ),
        ).props("flat size=sm")

        ui.separator().props("vertical")

        async def toggle_unvalidated_filter():
            current = filter_input.value
            if current == "__unvalidated__":
                filter_input.set_value("")
                grid.call_api_method("setGridOption", "quickFilterText", "")
            else:
                unvalidated = [
                    r for r in state["rows"]
                    if r.get("HUMAN_VALIDATED") != "True"
                ]
                grid.options["rowData"] = unvalidated
                grid.update()
                filter_input.set_value("__unvalidated__")

        ui.button(
            "Show Unvalidated Only",
            on_click=toggle_unvalidated_filter,
        ).props("flat size=sm color=orange")

        ui.button(
            "Show All",
            on_click=lambda: (
                grid.options.update({"rowData": state["rows"]}),
                grid.update(),
                filter_input.set_value(""),
            ),
        ).props("flat size=sm")

        ui.button(
            "Next Unvalidated",
            on_click=lambda: jump_to_next_unvalidated(),
        ).props("outline size=sm color=primary")

        ui.separator().props("vertical")

        # CSV export button
        async def export_csv():
            from annotators import REVIEW_CSV_COLUMNS
            output_path = Path(f"dataset/export/{model_name}_review.csv")
            db.export_review_csv(model_name, output_path)
            ui.notify(f"CSV exported to {output_path}", type="positive")

        ui.button(
            "Export CSV",
            on_click=export_csv,
        ).props("flat size=sm color=secondary icon=download")

    # AG Grid
    grid = ui.aggrid(grid_options).classes("q-mx-md").style("height: 55vh;")

    # Detail panel
    with ui.card().classes("q-mx-md q-mt-sm").style("max-height: 30vh; overflow-y: auto;"):
        ui.label("Detail View").classes("text-subtitle1 text-bold")
        detail_container = ui.column().classes("w-full")
        detail_label = ui.label("Click a row to view details").classes(
            "text-grey"
        )

    # --- Event handlers ---
    async def on_cell_changed(e):
        data = e.args
        if data is None:
            return
        row_data = data.get("data", {})
        col = data.get("colId", "")
        new_val = data.get("value", "")
        row_index = data.get("rowIndex")

        if row_index is not None and 0 <= row_index < len(state["rows"]):
            state["rows"][row_index][col] = new_val
        else:
            pmid = row_data.get("pmid", "")
            for row in state["rows"]:
                if row.get("pmid") == pmid:
                    row[col] = new_val
                    break

        state["modified"] = True
        status_label.text = "Unsaved changes"
        status_label.classes("text-red", remove="text-grey text-green")
        update_stats()

    grid.on("cellValueChanged", on_cell_changed)

    async def on_row_selected(e):
        data = e.args
        if data is None:
            return
        row_data = data.get("data") if isinstance(data, dict) else None
        if row_data is None:
            return

        detail_label.visible = False
        detail_container.clear()
        with detail_container:
            with ui.row().classes("gap-4 w-full"):
                with ui.column().classes("w-1/2"):
                    ui.label(f"PMID: {row_data.get('pmid', '')}").classes(
                        "text-bold"
                    )
                    ui.label(
                        f"Title: {row_data.get('title', '')}"
                    ).classes("text-body2")
                    ui.label(
                        f"Severity: {row_data.get('overall_severity', '')} "
                        f"(prob: {row_data.get('overall_bias_probability', '')})"
                    )
                    ui.label(
                        f"Statistical: {row_data.get('statistical_severity', '')} "
                        f"| Spin: {row_data.get('spin_level', '')} "
                        f"| Funding: {row_data.get('funding_type', '')}"
                    )
                    ui.label(
                        f"Confidence: {row_data.get('confidence', '')}"
                    )
                with ui.column().classes("w-1/2"):
                    ui.label("Reasoning:").classes("text-bold")
                    ui.label(
                        row_data.get("reasoning_summary", "(none)")
                    ).classes("text-body2").style(
                        "white-space: pre-wrap; word-break: break-word;"
                    )

    grid.on("rowSelected", on_row_selected)

    def do_save():
        """Save review changes to the database."""
        try:
            for row in state["rows"]:
                pmid = row.get("pmid", "")
                validated_str = row.get("HUMAN_VALIDATED", "")
                override = row.get("HUMAN_OVERRIDE_SEVERITY", "") or None
                notes = row.get("HUMAN_NOTES", "") or None

                if not pmid:
                    continue
                if not validated_str and not override and not notes:
                    continue

                validated = validated_str == "True"
                db.upsert_review(
                    pmid, state["model_name"], validated, override, notes
                )

            state["modified"] = False
            status_label.text = "Saved successfully"
            status_label.classes("text-green", remove="text-red text-grey")
            ui.notify("Saved!", type="positive")
        except Exception as exc:
            ui.notify(f"Save failed: {exc}", type="negative")

    async def jump_to_next_unvalidated():
        for i, row in enumerate(state["rows"]):
            if row.get("HUMAN_VALIDATED") != "True":
                grid.call_api_method("ensureIndexVisible", i, "middle")
                grid.call_api_method("selectIndex", i)
                return
        ui.notify("All rows have been validated!", type="info")

    ui.label(
        "Tip: Double-click Validated/Override/Notes cells to edit in-place"
    ).classes("q-px-md text-caption text-grey")


def main():
    """Entry point for the review GUI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GUI for reviewing bias annotations"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name to review"
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
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

    model_name = args.model
    if not model_name:
        models = db.get_model_names()
        models = [m for m in models if m != "human"]

        if not models:
            print("No model annotations found in database")
            print("Usage: uv run python -m utils.review_gui --model anthropic")
            db.close()
            sys.exit(1)

        if len(models) == 1:
            model_name = models[0]
        else:
            print("Available models:")
            for i, m in enumerate(models, 1):
                print(f"  {i}. {m}")
            print()

            while True:
                try:
                    choice = input(f"Select model (1-{len(models)}): ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        model_name = models[idx]
                        break
                except (ValueError, EOFError):
                    pass
                print(f"Please enter a number between 1 and {len(models)}")

    create_app(db, model_name)
    ui.run(
        title=f"Review: {model_name}",
        port=args.port,
        reload=False,
        on_air=False,
    )
    db.close()


if __name__ == "__main__":
    main()

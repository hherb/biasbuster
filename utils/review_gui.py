"""
NiceGUI-based CSV Review Tool

Web-based GUI for reviewing and validating bias annotation CSVs.
Opens in your browser with an editable AG Grid table.

Usage:
    uv run python -m utils.review_gui dataset/labelled/anthropic/high_suspicion_review.csv
    uv run python -m utils.review_gui  # shows file picker
"""

import csv
import logging
import os
import sys
import tempfile
from pathlib import Path

from nicegui import ui

logger = logging.getLogger(__name__)

SEVERITY_OPTIONS = ["", "none", "low", "moderate", "high", "critical"]

# Import canonical column list from annotators
from annotators import REVIEW_CSV_COLUMNS


def load_csv(csv_path: Path) -> list[dict]:
    """Load review CSV into list of dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def save_csv(rows: list[dict], csv_path: Path) -> None:
    """Save rows back to CSV with atomic write."""
    # Write to temp file first, then rename
    fd, tmp_path = tempfile.mkstemp(
        suffix=".csv", dir=csv_path.parent, prefix=".review_tmp_"
    )
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {col: row.get(col, "") for col in REVIEW_CSV_COLUMNS}
                )
        Path(tmp_path).replace(csv_path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def find_csv_files(labelled_dir: str | Path) -> list[Path]:
    """Find all review CSVs in the labelled directory."""
    labelled_dir = Path(labelled_dir)
    if not labelled_dir.exists():
        return []
    return sorted(labelled_dir.glob("**/*_review.csv"))


def create_app(csv_path: Path) -> None:
    """Create the NiceGUI review application."""
    rows = load_csv(csv_path)
    # Track state
    state = {"modified": False, "rows": rows}

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
    ui.page_title(f"Review: {csv_path.name}")

    # Header
    with ui.header().classes("items-center justify-between q-px-md"):
        ui.label(f"Reviewing: {csv_path}").classes("text-h6")
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
            """Filter grid to show only unvalidated rows."""
            # Use AG Grid quick filter with empty HUMAN_VALIDATED
            current = filter_input.value
            if current == "__unvalidated__":
                filter_input.set_value("")
                grid.call_api_method("setGridOption", "quickFilterText", "")
            else:
                # Filter by setting external filter via row data update
                unvalidated = [
                    r for r in state["rows"]
                    if r.get("HUMAN_VALIDATED") != "True"
                ]
                grid.options["rowData"] = unvalidated
                grid.update()
                filter_input.set_value("__unvalidated__")

        show_unvalidated_btn = ui.button(
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
        """Handle in-grid edits."""
        data = e.args
        if data is None:
            return
        row_data = data.get("data", {})
        col = data.get("colId", "")
        new_val = data.get("value", "")
        row_index = data.get("rowIndex")

        # Update our state — use row index if available, fall back to PMID match
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
        """Show detail panel for selected row."""
        data = e.args
        if data is None:
            return
        # Handle AG Grid selection event
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
        """Save changes to CSV."""
        try:
            save_csv(state["rows"], csv_path)
            state["modified"] = False
            status_label.text = "Saved successfully"
            status_label.classes("text-green", remove="text-red text-grey")
            ui.notify("Saved!", type="positive")
        except Exception as exc:
            ui.notify(f"Save failed: {exc}", type="negative")

    async def jump_to_next_unvalidated():
        """Scroll to next unvalidated row."""
        for i, row in enumerate(state["rows"]):
            if row.get("HUMAN_VALIDATED") != "True":
                # Use AG Grid API to ensure row is visible
                grid.call_api_method("ensureIndexVisible", i, "middle")
                grid.call_api_method("selectIndex", i)
                return
        ui.notify("All rows have been validated!", type="info")

    # Keyboard shortcut info
    ui.label(
        "Tip: Double-click Validated/Override/Notes cells to edit in-place"
    ).classes("q-px-md text-caption text-grey")


def main():
    """Entry point for the review GUI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GUI for reviewing bias annotation CSVs"
    )
    parser.add_argument(
        "csv_path", nargs="?", default=None, help="Path to review CSV"
    )
    parser.add_argument(
        "--labelled-dir",
        default=None,
        help="Browse labelled directory for CSVs",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the web server (default: 8080)",
    )
    args = parser.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        # List available CSVs and let user pick
        try:
            from config import Config
            cfg = Config()
        except ImportError:
            cfg = None

        labelled = args.labelled_dir or (cfg.labelled_dir if cfg else "dataset/labelled")
        csv_files = find_csv_files(labelled)

        if not csv_files:
            print(f"No review CSVs found in {labelled}")
            print("Usage: uv run python -m utils.review_gui <csv_path>")
            sys.exit(1)

        print("Available review CSVs:")
        for i, path in enumerate(csv_files, 1):
            print(f"  {i}. {path}")
        print()

        while True:
            try:
                choice = input(f"Select file (1-{len(csv_files)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(csv_files):
                    csv_path = csv_files[idx]
                    break
            except (ValueError, EOFError):
                pass
            print(f"Please enter a number between 1 and {len(csv_files)}")

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    create_app(csv_path)
    ui.run(
        title=f"Review: {csv_path.name}",
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()

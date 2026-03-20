"""Evaluation tab — run model evaluations and display results.

Launches ``python -m evaluation.run`` as a subprocess, streams log
output, and renders evaluation JSON / comparison Markdown once complete.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nicegui import ui

from gui.process_runner import ProcessRunner

logger = logging.getLogger(__name__)

# Dimensions in the order displayed.
_DIMENSIONS = [
    "statistical_reporting",
    "spin",
    "outcome_reporting",
    "conflict_of_interest",
    "methodology",
]


def _eval_output_dir(state: dict) -> Path:
    """Return an absolute path for evaluation output."""
    project_dir = state.get("project_dir", ".")
    return Path(project_dir) / "eval_results"


def _build_eval_cmd(state: dict, mode: str) -> list[str] | None:
    """Build the evaluation CLI command.  Returns ``None`` on validation error."""
    test_file = state.get("test_file", "")
    model_a = state.get("eval_model_a", "")
    endpoint_a = state.get("eval_endpoint_a", "")

    if not test_file or not model_a or not endpoint_a:
        return None

    cmd = [
        "uv", "run", "python", "-m", "evaluation.run",
        "--test-set", test_file,
        "--model-a", model_a,
        "--endpoint-a", endpoint_a,
        "--mode", mode,
        "--output", str(_eval_output_dir(state)),
    ]

    model_b = state.get("eval_model_b", "")
    endpoint_b = state.get("eval_endpoint_b", "")
    if model_b and endpoint_b:
        cmd += ["--model-b", model_b, "--endpoint-b", endpoint_b]

    return cmd


def _find_eval_json(output_dir: Path) -> list[Path]:
    """Find all ``*_evaluation.json`` files in *output_dir*."""
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("*_evaluation.json"))


def _find_comparison_md(output_dir: Path) -> Path | None:
    """Return the latest comparison Markdown report, if any."""
    if not output_dir.exists():
        return None
    candidates = sorted(
        output_dir.glob("comparison_*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _parse_eval_rows(data: dict) -> list[dict]:
    """Extract per-dimension rows from an evaluation JSON dict."""
    rows = []
    dims = data.get("dimensions", {})
    for dim_key in _DIMENSIONS:
        dim = dims.get(dim_key, {})
        binary = dim.get("binary", {})
        ordinal = dim.get("ordinal", {})
        rows.append({
            "dimension": dim_key.replace("_", " ").title(),
            "f1": f'{binary.get("f1", 0):.3f}',
            "precision": f'{binary.get("precision", 0):.3f}',
            "recall": f'{binary.get("recall", 0):.3f}',
            "kappa": f'{ordinal.get("weighted_kappa", 0):.3f}',
        })
    return rows


def _summary_text(data: dict) -> str:
    """One-line summary from evaluation JSON."""
    overall_b = data.get("overall_binary", {})
    cal = data.get("calibration_error", 0)
    ver = data.get("mean_verification_score", 0)
    f1 = overall_b.get("f1", 0)
    overall_o = data.get("overall_ordinal", {})
    kappa = overall_o.get("weighted_kappa", 0)
    return (
        f"Overall F1: {f1:.3f}  |  Kappa: {kappa:.3f}  "
        f"|  Cal. Error: {cal:.3f}  |  Verification: {ver:.3f}"
    )


def _set_buttons_enabled(buttons: list, enabled: bool) -> None:
    """Enable or disable a list of NiceGUI buttons."""
    for btn in buttons:
        if enabled:
            btn.props(remove="disabled")
        else:
            btn.props("disabled")


def create_evaluation_tab(state: dict) -> None:
    """Build the Evaluation tab UI."""
    runner = ProcessRunner()

    # ── Controls ──────────────────────────────────────────────────────
    with ui.row().classes("items-center gap-4 w-full"):
        baseline_btn = ui.button("Run Baseline (zero-shot)", icon="play_arrow").props(
            "color=primary"
        )
        finetuned_btn = ui.button("Run Fine-Tuned", icon="play_arrow").props(
            "color=teal"
        )
        reanalyse_btn = ui.button("Re-analyse Saved", icon="refresh").props(
            "outline"
        )
        status_badge = ui.badge("Idle", color="grey")

    action_buttons = [baseline_btn, finetuned_btn, reanalyse_btn]

    # ── Log output ────────────────────────────────────────────────────
    with ui.expansion("Evaluation Log", icon="terminal").classes("w-full q-mt-sm"):
        log_widget = ui.log(max_lines=1000).classes("w-full").style("height: 250px;")

    runner.on_output(lambda line: log_widget.push(line.rstrip("\n")))

    # ── Results area ──────────────────────────────────────────────────
    results_container = ui.column().classes("w-full q-mt-md gap-4")

    def display_results() -> None:
        """Scan output directory and render results."""
        results_container.clear()

        out_dir = _eval_output_dir(state)
        eval_files = _find_eval_json(out_dir)
        if not eval_files:
            with results_container:
                ui.label("No evaluation results found.").classes("text-grey")
            return

        with results_container:
            for fpath in eval_files:
                try:
                    data = json.loads(fpath.read_text())
                except (json.JSONDecodeError, OSError):
                    continue

                model_id = data.get("model_id", fpath.stem)
                with ui.card().classes("w-full"):
                    ui.label(model_id).classes("text-subtitle1 text-bold")
                    ui.label(_summary_text(data)).classes("text-caption text-grey")

                    rows = _parse_eval_rows(data)
                    ui.table(
                        columns=[
                            {"name": "dimension", "label": "Dimension", "field": "dimension", "align": "left"},
                            {"name": "f1", "label": "F1", "field": "f1", "align": "center"},
                            {"name": "precision", "label": "Precision", "field": "precision", "align": "center"},
                            {"name": "recall", "label": "Recall", "field": "recall", "align": "center"},
                            {"name": "kappa", "label": "Kappa", "field": "kappa", "align": "center"},
                        ],
                        rows=rows,
                    ).classes("w-full").props("dense flat")

            # Comparison report
            md_path = _find_comparison_md(out_dir)
            if md_path is not None:
                with ui.card().classes("w-full"):
                    ui.label("Comparison Report").classes("text-subtitle1 text-bold")
                    try:
                        md_content = md_path.read_text()
                        ui.markdown(md_content).classes("w-full").style(
                            "user-select: text; max-height: 600px; overflow-y: auto;"
                        )
                    except OSError:
                        ui.label("Could not read comparison report.").classes("text-red")

    # ── Button handlers ───────────────────────────────────────────────
    async def run_eval(mode: str) -> None:
        cmd = _build_eval_cmd(state, mode)
        if cmd is None:
            ui.notify(
                "Configure test set, model name, and endpoint in Settings first",
                type="warning",
            )
            return

        log_widget.clear()
        status_badge.text = "Running..."
        status_badge.props("color=blue")
        _set_buttons_enabled(action_buttons, False)

        project_dir = state.get("project_dir", ".")
        try:
            await runner.start(cmd, cwd=project_dir)
        except RuntimeError as exc:
            ui.notify(str(exc), type="warning")
            _set_buttons_enabled(action_buttons, True)

    def _poll_completion() -> None:
        """Check if the subprocess just finished (runs in UI context)."""
        finished, code = runner.consume_finished()
        if not finished:
            return
        _set_buttons_enabled(action_buttons, True)
        if code == 0:
            status_badge.text = "Complete"
            status_badge.props("color=green")
            ui.notify("Evaluation completed!", type="positive")
            display_results()
        else:
            status_badge.text = "Failed"
            status_badge.props("color=red")
            ui.notify(f"Evaluation failed (exit code {code})", type="negative")

    ui.timer(1.0, _poll_completion)

    baseline_btn.on_click(lambda: run_eval("zero-shot"))
    finetuned_btn.on_click(lambda: run_eval("fine-tuned"))

    async def on_reanalyse() -> None:
        test_file = state.get("test_file", "")
        if not test_file:
            ui.notify("Set test file in Settings first", type="warning")
            return
        out_dir = str(_eval_output_dir(state))
        cmd = [
            "uv", "run", "python", "-m", "evaluation.run",
            "--reanalyse", out_dir,
            "--test-set", test_file,
            "--output", out_dir,
        ]
        log_widget.clear()
        status_badge.text = "Re-analysing..."
        status_badge.props("color=blue")
        _set_buttons_enabled(action_buttons, False)
        project_dir = state.get("project_dir", ".")
        try:
            await runner.start(cmd, cwd=project_dir)
        except RuntimeError as exc:
            ui.notify(str(exc), type="warning")
            _set_buttons_enabled(action_buttons, True)

    reanalyse_btn.on_click(on_reanalyse)

    # Show existing results on load
    display_results()

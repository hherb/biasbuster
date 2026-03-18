"""
BiasBuster Verification Agent — Interactive Demo.

NiceGUI web application that demonstrates the verification agent loop:
enter a PMID or paste an abstract, see the initial bias assessment,
watch verification tools execute, and review the refined assessment.

Usage:
    uv run python -m utils.agent_demo
    uv run python -m utils.agent_demo --model qwen3.5-9b-biasbuster --port 8082
"""

import argparse
import logging
import time
from typing import Any

from nicegui import ui

from agent.config import AgentConfig
from agent.runner import AgentResult, run_agent
from agent.tools import get_tool_display_name

logger = logging.getLogger(__name__)

# Severity → colour mapping for badges
_SEVERITY_COLOURS = {
    "none": "green",
    "low": "green",
    "moderate": "orange",
    "high": "red",
    "critical": "red",
}

# Stage display labels and icons
_STAGE_LABELS = {
    "fetching_abstract": "Fetching abstract from PubMed...",
    "initial_assessment": "Running initial bias assessment...",
    "initial_assessment_done": "Initial assessment complete",
    "parsing_steps": "Parsing verification steps...",
    "parsing_steps_done": "Verification steps parsed",
    "executing_tools": "Executing verification tools...",
    "tool_started": "Running tool...",
    "tool_done": "Tool complete",
    "refining": "Generating refined assessment...",
    "refining_done": "Refined assessment complete",
    "complete": "Agent run complete",
    "error": "Error occurred",
}


def create_app(config: AgentConfig) -> None:
    """Create the NiceGUI verification agent demo."""

    # Shared mutable state
    state: dict[str, Any] = {
        "running": False,
        "result": None,
        "stage_log": [],
        "start_time": 0.0,
    }

    # --- Header ---
    with ui.header().classes("items-center justify-between q-px-md"):
        ui.label("BiasBuster Verification Agent").classes("text-h6")
        status_badge = ui.badge("Ready", color="green").classes("q-ml-md")

    with ui.column().classes("w-full q-pa-md gap-4").style("max-width: 1200px; margin: 0 auto;"):

        # --- Input Section ---
        with ui.card().classes("w-full"):
            ui.label("Input").classes("text-subtitle1 text-bold")
            ui.separator()

            with ui.row().classes("w-full gap-4 items-end"):
                pmid_input = ui.input(
                    label="PMID",
                    placeholder="e.g. 12345678",
                ).classes("w-40")
                ui.label("OR").classes("text-caption q-pb-sm")
                title_input = ui.input(
                    label="Title",
                    placeholder="Paper title (optional with PMID)",
                ).classes("flex-grow")

            abstract_input = ui.textarea(
                label="Abstract (paste here if no PMID)",
                placeholder="Paste the full abstract text...",
            ).classes("w-full").style("min-height: 120px;")

            with ui.row().classes("w-full justify-between items-center"):
                run_button = ui.button("Run Assessment", icon="play_arrow")
                elapsed_label = ui.label("").classes("text-caption text-grey")

        # --- Status Timeline ---
        with ui.card().classes("w-full"):
            ui.label("Status").classes("text-subtitle1 text-bold")
            ui.separator()
            status_container = ui.column().classes("w-full gap-1")

        # --- Results: side by side ---
        with ui.row().classes("w-full gap-4"):
            # Initial assessment
            with ui.card().classes("flex-grow").style("min-width: 45%;"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("Initial Assessment").classes("text-subtitle1 text-bold")
                    initial_severity_badge = ui.badge("", color="grey")
                    initial_severity_badge.visible = False
                ui.separator()
                initial_output = ui.markdown("*Waiting for input...*").classes(
                    "w-full"
                ).style("user-select: text; max-height: 500px; overflow-y: auto;")

            # Refined assessment
            with ui.card().classes("flex-grow").style("min-width: 45%;"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("Refined Assessment").classes("text-subtitle1 text-bold")
                    refined_severity_badge = ui.badge("", color="grey")
                    refined_severity_badge.visible = False
                ui.separator()
                refined_output = ui.markdown("*Waiting for verification...*").classes(
                    "w-full"
                ).style("user-select: text; max-height: 500px; overflow-y: auto;")

        # --- Verification Details ---
        with ui.card().classes("w-full"):
            ui.label("Verification Details").classes("text-subtitle1 text-bold")
            ui.separator()
            tools_container = ui.column().classes("w-full gap-2")

    # --- Helper functions ---

    def _add_status_line(text: str, icon: str = "hourglass_empty", color: str = "blue") -> None:
        """Append a status line to the timeline."""
        with status_container:
            with ui.row().classes("items-center gap-2"):
                ui.icon(icon).classes(f"text-{color}").style("font-size: 18px;")
                ui.label(text).classes("text-body2")
        state["stage_log"].append(text)

    def _update_elapsed() -> None:
        if state["running"] and state["start_time"]:
            elapsed = time.monotonic() - state["start_time"]
            elapsed_label.text = f"Elapsed: {elapsed:.0f}s"

    def _extract_severity(text: str) -> str:
        """Try to extract overall severity from model output."""
        import re
        match = re.search(
            r"(?:overall[_ ]severity|Overall\s+Assessment)[:\s]*"
            r"[*]*\s*(none|low|moderate|high|critical)",
            text, re.IGNORECASE,
        )
        return match.group(1).lower() if match else ""

    def _show_severity_badge(badge: ui.badge, text: str) -> None:
        severity = _extract_severity(text)
        if severity:
            badge.text = severity.upper()
            badge.props(f'color="{_SEVERITY_COLOURS.get(severity, "grey")}"')
            badge.visible = True

    def _render_tool_results(results: list) -> None:
        """Render tool results as expandable cards."""
        tools_container.clear()
        with tools_container:
            for tr in results:
                display_name = get_tool_display_name(tr.tool_name)
                icon = "check_circle" if tr.success else "cancel"
                colour = "green" if tr.success else "red"
                if tr.tool_name == "unsupported":
                    icon = "help_outline"
                    colour = "grey"

                with ui.expansion(
                    text=f"{display_name}: {tr.summary}",
                    icon=icon,
                ).classes("w-full").props(f'header-class="text-{colour}"'):
                    if tr.detail:
                        ui.markdown(tr.detail).style("user-select: text;")
                    if tr.error:
                        ui.label(f"Error: {tr.error}").classes("text-red")

    # --- Stage callback ---

    def on_stage(stage: str, data: Any = None) -> None:
        """Called by the agent runner at each stage."""
        label = _STAGE_LABELS.get(stage, stage)

        if stage == "error":
            _add_status_line(f"Error: {data}", icon="error", color="red")
            return

        if stage == "tool_started" and isinstance(data, dict):
            tool = get_tool_display_name(data.get("tool", ""))
            _add_status_line(f"Running {tool}...", icon="sync", color="blue")
            return

        if stage == "tool_done" and isinstance(data, dict):
            tool = get_tool_display_name(data.get("tool", ""))
            ok = data.get("success", False)
            icon = "check_circle" if ok else "warning"
            colour = "green" if ok else "orange"
            _add_status_line(
                f"{tool}: {data.get('summary', 'done')}",
                icon=icon, color=colour,
            )
            return

        if stage == "complete":
            _add_status_line("Done!", icon="check_circle", color="green")
            return

        if stage.endswith("_done"):
            _add_status_line(label, icon="check_circle", color="green")
        else:
            _add_status_line(label, icon="hourglass_empty", color="blue")

    # --- Run button handler ---

    async def on_run() -> None:
        if state["running"]:
            ui.notify("Already running — please wait.", type="warning")
            return

        pmid = pmid_input.value.strip()
        title = title_input.value.strip()
        abstract = abstract_input.value.strip()

        if not pmid and not abstract:
            ui.notify("Please enter a PMID or paste an abstract.", type="negative")
            return

        # Reset UI
        state["running"] = True
        state["start_time"] = time.monotonic()
        status_badge.text = "Running"
        status_badge.props('color="blue"')
        status_container.clear()
        tools_container.clear()
        initial_output.content = "*Running initial assessment...*"
        refined_output.content = "*Waiting for verification...*"
        initial_severity_badge.visible = False
        refined_severity_badge.visible = False
        elapsed_label.text = "Elapsed: 0s"

        try:
            result: AgentResult = await run_agent(
                config=config,
                pmid=pmid,
                title=title,
                abstract=abstract,
                on_stage=on_stage,
            )

            state["result"] = result

            # Update initial assessment
            if result.initial_assessment:
                initial_output.content = result.initial_assessment
                _show_severity_badge(initial_severity_badge, result.initial_assessment)

            # Update refined assessment
            if result.refined_assessment:
                refined_output.content = result.refined_assessment
                _show_severity_badge(refined_severity_badge, result.refined_assessment)

            # Render tool results
            if result.tool_results:
                _render_tool_results(result.tool_results)

            if result.error:
                status_badge.text = "Error"
                status_badge.props('color="red"')
                ui.notify(f"Agent error: {result.error}", type="negative")
            else:
                status_badge.text = "Complete"
                status_badge.props('color="green"')
                ui.notify(
                    f"Done in {result.total_time_seconds:.0f}s — "
                    f"{len(result.tool_results)} tools executed",
                    type="positive",
                )

        except Exception as exc:
            logger.exception("Agent demo error")
            status_badge.text = "Error"
            status_badge.props('color="red"')
            ui.notify(f"Unexpected error: {exc}", type="negative")
            _add_status_line(f"Unexpected error: {exc}", icon="error", color="red")

        finally:
            state["running"] = False
            elapsed = time.monotonic() - state["start_time"]
            elapsed_label.text = f"Elapsed: {elapsed:.0f}s"

    run_button.on_click(on_run)

    # Elapsed time updater
    ui.timer(1.0, _update_elapsed)


def main() -> None:
    """Entry point for the agent demo."""
    parser = argparse.ArgumentParser(
        description="BiasBuster Verification Agent Demo",
    )
    parser.add_argument(
        "--port", type=int, default=8082,
        help="Port to run the web UI on (default: 8082)",
    )
    parser.add_argument(
        "--ollama-endpoint", default="http://localhost:11434",
        help="Ollama API endpoint (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--model", default="qwen3.5-9b-biasbuster",
        help="Ollama model name (default: qwen3.5-9b-biasbuster)",
    )
    parser.add_argument(
        "--ncbi-api-key", default="",
        help="NCBI API key for higher PubMed rate limits",
    )
    parser.add_argument(
        "--mailto", default="",
        help="Contact email for Crossref/NCBI polite access",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = AgentConfig(
        ollama_endpoint=args.ollama_endpoint,
        model_id=args.model,
        ncbi_api_key=args.ncbi_api_key,
        crossref_mailto=args.mailto,
    )

    create_app(config)
    ui.run(
        title="BiasBuster Verification Agent",
        port=args.port,
        reload=False,
        on_air=False,
    )


if __name__ == "__main__":
    main()

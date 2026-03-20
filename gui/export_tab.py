"""Export tab — merge adapter, GGUF quantisation, and Ollama import.

Each operation runs as its own subprocess so the user can follow
progress in real time.
"""

from __future__ import annotations

import logging
from pathlib import Path

from nicegui import ui

from gui.process_runner import ProcessRunner
from gui.state import save_settings

logger = logging.getLogger(__name__)

QUANTIZATION_OPTIONS = ["Q4_K_M", "Q5_K_M", "q8_0", "f16", "bf16"]


def _merged_dir_for(state: dict) -> str:
    """Infer the merged-model output directory from the training output dir."""
    out_dir = state.get("training_output_dir", "")
    if not out_dir:
        model_key = state.get("model_key", "model")
        backend = state.get("backend", "trl")
        if backend == "mlx":
            out_dir = f"training_output/{model_key}-mlx-lora"
        else:
            out_dir = f"training_output/{model_key}-lora"
    # Derive merged dir by replacing -lora suffix with -merged
    if out_dir.endswith("-lora"):
        return out_dir[:-5] + "-merged"
    return out_dir + "-merged"


def _poll_runner_completion(
    runner: ProcessRunner,
    btn: ui.button,
    status: ui.badge,
    success_text: str,
    success_msg: str,
    fail_msg: str,
) -> None:
    """Create a polling timer that detects subprocess completion.

    Runs in NiceGUI's UI context so ``ui.notify()`` and element
    creation are safe.
    """
    def _check() -> None:
        finished, code = runner.consume_finished()
        if not finished:
            return
        btn.props(remove="disabled")
        if code == 0:
            status.text = success_text
            status.props("color=green")
            ui.notify(success_msg, type="positive")
        else:
            status.text = "Failed"
            status.props("color=red")
            ui.notify(f"{fail_msg} (exit code {code})", type="negative")

    ui.timer(1.0, _check)


async def _start_guarded(
    runner: ProcessRunner,
    cmd: list[str],
    cwd: str,
    btn: ui.button,
    status: ui.badge,
    log: ui.log,
    running_text: str,
) -> None:
    """Start a subprocess with button guard and error handling."""
    log.clear()
    status.text = running_text
    status.props("color=blue")
    btn.props("disabled")
    try:
        await runner.start(cmd, cwd=cwd)
    except RuntimeError as exc:
        ui.notify(str(exc), type="warning")
        btn.props(remove="disabled")
        status.text = "Idle"
        status.props("color=grey")


def create_export_tab(state: dict) -> None:
    """Build the Export tab UI."""
    plat = state.get("platform", {})
    project_dir = state.get("project_dir", ".")

    # ── Section 1: Merge Adapter ──────────────────────────────────────
    merge_runner = ProcessRunner()

    with ui.card().classes("w-full"):
        ui.label("Merge Adapter").classes("text-subtitle1 text-bold")
        ui.label(
            "Merge the LoRA adapter into the base model to create a standalone model."
        ).classes("text-caption text-grey")

        with ui.row().classes("items-center gap-4"):
            merge_btn = ui.button("Merge Adapter", icon="merge_type").props("color=primary")
            merge_status = ui.badge("Idle", color="grey")

        with ui.expansion("Merge Log", icon="terminal").classes("w-full q-mt-sm"):
            merge_log = ui.log(max_lines=300).classes("w-full").style("height: 150px;")

    merge_runner.on_output(lambda line: merge_log.push(line.rstrip("\n")))
    _poll_runner_completion(
        merge_runner, merge_btn, merge_status,
        success_text="Done",
        success_msg="Adapter merged successfully!",
        fail_msg="Merge failed",
    )

    async def on_merge() -> None:
        model_key = state.get("model_key", "")
        backend = state.get("backend", "")
        if not model_key:
            ui.notify("Select a model in the Settings tab first", type="warning")
            return

        if backend == "mlx":
            cmd = [
                "uv", "run", "python", "-m", "training.merge_adapter_mlx",
                "--model", model_key, "--de-quantize",
            ]
        else:
            cmd = [
                "uv", "run", "python", "-m", "training.merge_adapter",
                "--model", model_key,
            ]

        await _start_guarded(
            merge_runner, cmd, project_dir,
            merge_btn, merge_status, merge_log, "Merging...",
        )

    merge_btn.on_click(on_merge)

    # ── Section 2: GGUF Export ────────────────────────────────────────
    gguf_runner = ProcessRunner()

    with ui.card().classes("w-full q-mt-md"):
        ui.label("GGUF Export").classes("text-subtitle1 text-bold")
        ui.label(
            "Convert the merged model to GGUF format with quantisation."
        ).classes("text-caption text-grey")

        # Check prerequisite
        llama_cpp_dir = Path(project_dir) / "llama.cpp"
        if not llama_cpp_dir.exists():
            ui.label(
                "llama.cpp/ directory not found — clone it to enable GGUF export."
            ).classes("text-orange")

        with ui.row().classes("items-center gap-4"):
            def on_quant_change(e) -> None:
                state["quantization"] = e.value
                save_settings(state)

            ui.select(
                options=QUANTIZATION_OPTIONS,
                value=state.get("quantization", "Q4_K_M"),
                label="Quantisation",
                on_change=on_quant_change,
            ).classes("w-40").props("outlined")

            gguf_btn = ui.button("Export GGUF", icon="file_download").props(
                "color=primary"
            )
            gguf_status = ui.badge("Idle", color="grey")

        with ui.expansion("GGUF Log", icon="terminal").classes("w-full q-mt-sm"):
            gguf_log = ui.log(max_lines=300).classes("w-full").style("height: 150px;")

    gguf_runner.on_output(lambda line: gguf_log.push(line.rstrip("\n")))
    _poll_runner_completion(
        gguf_runner, gguf_btn, gguf_status,
        success_text="Done",
        success_msg="GGUF export complete!",
        fail_msg="GGUF export failed",
    )

    async def on_gguf() -> None:
        merged_dir = _merged_dir_for(state)
        if not Path(merged_dir).exists():
            ui.notify(
                "Merged model directory not found — merge the adapter first",
                type="warning",
            )
            return

        quant = state.get("quantization", "Q4_K_M")
        ollama_name = (
            state.get("ollama_model_name", "")
            or f'{state.get("model_key", "model")}-biasbuster'
        )

        cmd = [
            "bash", "training/export_to_ollama.sh",
            merged_dir, ollama_name,
            "--gguf", quant,
        ]

        await _start_guarded(
            gguf_runner, cmd, project_dir,
            gguf_btn, gguf_status, gguf_log, "Exporting...",
        )

    gguf_btn.on_click(on_gguf)

    # ── Section 3: Ollama Import ──────────────────────────────────────
    ollama_runner = ProcessRunner()

    with ui.card().classes("w-full q-mt-md"):
        ui.label("Ollama Import").classes("text-subtitle1 text-bold")
        ui.label(
            "Import the merged model directly into Ollama (full precision, no GGUF)."
        ).classes("text-caption text-grey")

        if not plat.get("has_ollama"):
            ui.label(
                "Ollama not found on PATH — install Ollama to enable this feature."
            ).classes("text-orange")

        with ui.row().classes("items-center gap-4"):
            def on_name_change(e) -> None:
                state["ollama_model_name"] = e.value
                save_settings(state)

            model_name_input = ui.input(
                "Ollama Model Name",
                value=state.get("ollama_model_name", "")
                or f'{state.get("model_key", "model")}-biasbuster',
                on_change=on_name_change,
            ).classes("w-64")

            ollama_btn = ui.button("Import to Ollama", icon="cloud_upload").props(
                "color=primary"
            )
            ollama_status = ui.badge("Idle", color="grey")

        with ui.expansion("Ollama Log", icon="terminal").classes("w-full q-mt-sm"):
            ollama_log = ui.log(max_lines=300).classes("w-full").style("height: 150px;")

    ollama_runner.on_output(lambda line: ollama_log.push(line.rstrip("\n")))
    _poll_runner_completion(
        ollama_runner, ollama_btn, ollama_status,
        success_text="Done",
        success_msg="Ollama import complete!",
        fail_msg="Ollama import failed",
    )

    async def on_ollama() -> None:
        merged_dir = _merged_dir_for(state)
        if not Path(merged_dir).exists():
            ui.notify(
                "Merged model directory not found — merge the adapter first",
                type="warning",
            )
            return

        name = model_name_input.value or f'{state.get("model_key", "model")}-biasbuster'
        cmd = [
            "bash", "training/export_to_ollama.sh",
            merged_dir, name,
        ]

        await _start_guarded(
            ollama_runner, cmd, project_dir,
            ollama_btn, ollama_status, ollama_log, "Importing...",
        )

    ollama_btn.on_click(on_ollama)

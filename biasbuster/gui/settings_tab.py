"""Settings tab — model selector, hyperparameters, and path configuration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from nicegui import ui

from biasbuster.training.configs import MODEL_PRESETS, get_config
from biasbuster.training.configs_mlx import MLX_MODEL_PRESETS, get_mlx_config
from biasbuster.gui.state import save_settings

logger = logging.getLogger(__name__)


async def _show_file_picker(
    start_dir: Path,
    *,
    pattern: str = "*",
) -> Path | None:
    """Show a server-side file picker dialog.

    Lists files and directories in *start_dir* and lets the user navigate
    the filesystem and select a file.  Returns the chosen ``Path`` or
    ``None`` if cancelled.
    """
    result: Path | None = None
    current_dir = start_dir.resolve()

    with ui.dialog().props("maximized=false") as dialog, \
         ui.card().classes("w-96"):
        ui.label("Select File").classes("text-subtitle1 text-bold")
        path_label = ui.label(str(current_dir)).classes(
            "text-caption text-grey w-full"
        ).style("word-break: break-all;")

        file_list = ui.column().classes("w-full").style(
            "max-height: 400px; overflow-y: auto;"
        )

        def populate(directory: Path) -> None:
            nonlocal current_dir
            current_dir = directory.resolve()
            path_label.text = str(current_dir)
            file_list.clear()

            with file_list:
                # Parent directory entry
                if current_dir.parent != current_dir:
                    ui.button(
                        ".. (parent directory)",
                        icon="folder",
                        on_click=lambda _, d=current_dir.parent: populate(d),
                    ).props("flat dense align=left").classes("w-full justify-start")

                try:
                    entries = sorted(
                        current_dir.iterdir(),
                        key=lambda p: (not p.is_dir(), p.name.lower()),
                    )
                except PermissionError:
                    ui.label("Permission denied").classes("text-red")
                    return

                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    if entry.is_dir():
                        ui.button(
                            entry.name,
                            icon="folder",
                            on_click=lambda _, d=entry: populate(d),
                        ).props("flat dense align=left").classes(
                            "w-full justify-start"
                        )
                    elif entry.suffix in (".jsonl", ".json", ".csv", ".tsv", ".txt"):
                        def select(_, f=entry):
                            nonlocal result
                            result = f
                            dialog.close()

                        ui.button(
                            entry.name,
                            icon="description",
                            on_click=select,
                        ).props("flat dense align=left").classes(
                            "w-full justify-start"
                        )

        with ui.row().classes("w-full justify-end q-mt-sm"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

        populate(current_dir)

    await dialog
    return result


# Descriptions for TRL model presets (parallel to MLXModelPreset.description).
_TRL_DESCRIPTIONS: dict[str, str] = {
    "qwen3.5-27b": "Qwen3.5-27B — DGX Spark / high-VRAM GPU",
    "qwen3.5-9b": "Qwen3.5-9B — DGX Spark / mid-range GPU",
    "olmo-3.1-32b": "OLMo-3.1-32B-Instruct — DGX Spark / high-VRAM GPU",
}


def _build_model_options(plat: dict) -> dict[str, str]:
    """Return ``{key: description}`` for models available on this platform.

    On Linux only TRL presets are shown; on macOS only MLX presets.
    On Windows (or unknown OS) all presets are shown for browsing.
    """
    options: dict[str, str] = {}
    is_linux = plat.get("is_linux", False)
    is_macos = plat.get("is_macos", False)

    if is_linux or (not is_linux and not is_macos):
        for key in MODEL_PRESETS:
            options[key] = _TRL_DESCRIPTIONS.get(key, key)
    if is_macos or (not is_linux and not is_macos):
        for key, preset in MLX_MODEL_PRESETS.items():
            options[key] = preset.description
    return options


def _populate_from_preset(
    state: dict,
    model_key: str,
    widgets: dict[str, ui.number | ui.input],
) -> str:
    """Load preset defaults into state and update UI widgets.

    Returns a description string for the model.
    """
    description = ""
    if model_key in MODEL_PRESETS:
        cfg = get_config(model_key)
        state["backend"] = "trl"
        state["learning_rate"] = cfg.learning_rate
        state["num_epochs"] = cfg.num_train_epochs
        state["lora_rank"] = cfg.lora_r
        state["batch_size"] = cfg.per_device_train_batch_size
        state["gradient_accumulation"] = cfg.gradient_accumulation_steps
        state["max_seq_length"] = cfg.max_seq_length
        state["training_output_dir"] = cfg.output_dir
        description = _TRL_DESCRIPTIONS.get(model_key, "")
    elif model_key in MLX_MODEL_PRESETS:
        cfg = get_mlx_config(model_key)
        state["backend"] = "mlx"
        state["learning_rate"] = cfg.learning_rate
        state["num_epochs"] = cfg.num_train_epochs
        state["lora_rank"] = cfg.lora_rank
        state["batch_size"] = cfg.batch_size
        state["gradient_accumulation"] = 1  # MLX has no grad accum
        state["max_seq_length"] = cfg.max_seq_length
        state["training_output_dir"] = cfg.output_dir
        description = MLX_MODEL_PRESETS[model_key].description

    state["model_key"] = model_key

    # Update widget values
    for key, widget in widgets.items():
        if key in state and hasattr(widget, "set_value"):
            widget.set_value(state[key])

    save_settings(state)
    return description


def create_settings_tab(state: dict) -> None:
    """Build the Settings tab UI."""
    plat = state.get("platform", {})
    model_options = _build_model_options(plat)

    # Mutable containers for widget references
    widgets: dict[str, ui.number | ui.input] = {}
    description_label = None
    backend_label = None

    def on_model_change(e) -> None:
        nonlocal description_label, backend_label
        key = e.value
        desc = _populate_from_preset(state, key, widgets)
        if description_label is not None:
            description_label.set_text(desc)
        if backend_label is not None:
            backend_label.set_text(f'Backend: {state.get("backend") or "auto-detect"}')
        ui.notify(f"Loaded defaults for {key}", type="info")

    def on_param_change(param_name: str) -> Callable:
        def handler(e) -> None:
            state[param_name] = e.value
            save_settings(state)
        return handler

    def reset_defaults() -> None:
        key = state.get("model_key", "")
        if key:
            desc = _populate_from_preset(state, key, widgets)
            if description_label is not None:
                description_label.set_text(desc)
            ui.notify("Reset to preset defaults", type="positive")

    # ── Layout ────────────────────────────────────────────────────────
    with ui.row().classes("w-full gap-6"):
        # Left: model selection
        with ui.card().classes("flex-grow"):
            ui.label("Model Selection").classes("text-subtitle1 text-bold")
            ui.select(
                options=model_options,
                value=state.get("model_key") or None,
                label="Base Model",
                on_change=on_model_change,
            ).classes("w-full").props("outlined")
            description_label = ui.label("").classes("text-caption text-grey")
            # Set initial description
            if state.get("model_key"):
                desc = model_options.get(state["model_key"], "")
                description_label.set_text(desc)

            backend_label = ui.label(
                f'Backend: {state.get("backend") or "auto-detect"}'
            ).classes("text-caption text-grey q-mt-sm")

        # Right: hyperparameters
        with ui.card().classes("flex-grow"):
            ui.label("Training Hyperparameters").classes("text-subtitle1 text-bold")

            with ui.row().classes("w-full gap-4"):
                widgets["learning_rate"] = ui.number(
                    "Learning Rate",
                    value=state["learning_rate"],
                    format="%.1e",
                    step=1e-5,
                    min=1e-6,
                    max=1e-2,
                    on_change=on_param_change("learning_rate"),
                ).classes("w-40").tooltip("Peak learning rate for cosine schedule")

                widgets["num_epochs"] = ui.number(
                    "Epochs",
                    value=state["num_epochs"],
                    step=1,
                    min=1,
                    max=20,
                    on_change=on_param_change("num_epochs"),
                ).classes("w-28").tooltip("Number of training epochs")

                widgets["lora_rank"] = ui.number(
                    "LoRA Rank",
                    value=state["lora_rank"],
                    step=8,
                    min=4,
                    max=128,
                    on_change=on_param_change("lora_rank"),
                ).classes("w-28").tooltip("LoRA adapter rank (r)")

            with ui.row().classes("w-full gap-4"):
                widgets["batch_size"] = ui.number(
                    "Batch Size",
                    value=state["batch_size"],
                    step=1,
                    min=1,
                    max=16,
                    on_change=on_param_change("batch_size"),
                ).classes("w-28").tooltip("Per-device batch size")

                widgets["gradient_accumulation"] = ui.number(
                    "Grad Accumulation",
                    value=state["gradient_accumulation"],
                    step=1,
                    min=1,
                    max=32,
                    on_change=on_param_change("gradient_accumulation"),
                ).classes("w-36").tooltip(
                    "Gradient accumulation steps (effective batch = batch_size x this)"
                )

                widgets["max_seq_length"] = ui.number(
                    "Max Seq Length",
                    value=state["max_seq_length"],
                    step=512,
                    min=512,
                    max=16384,
                    on_change=on_param_change("max_seq_length"),
                ).classes("w-36").tooltip("Maximum sequence length for training")

            ui.button("Reset to Defaults", icon="restart_alt", on_click=reset_defaults).props(
                "flat dense size=sm"
            )

    # ── Paths & Endpoints ─────────────────────────────────────────────
    def _file_input_with_picker(
        label: str,
        state_key: str,
        tooltip: str,
        file_types: str = "*.jsonl *.json",
    ) -> ui.input:
        """Create a text input paired with a file-browse button.

        Uses a NiceGUI dialog with a server-side directory listing so it
        works in any browser without native filesystem access.
        """
        inp = ui.input(
            label,
            value=state[state_key],
            on_change=on_param_change(state_key),
        ).classes("flex-grow").tooltip(tooltip)

        async def pick_file() -> None:
            # Resolve starting directory from current value
            current = state.get(state_key, "")
            start_dir = Path(current).parent if current else Path(".")
            if not start_dir.is_absolute():
                start_dir = Path(state.get("project_dir", ".")) / start_dir
            if not start_dir.exists():
                start_dir = Path(state.get("project_dir", "."))

            result = await _show_file_picker(start_dir)
            if result is not None:
                inp.set_value(str(result))
                state[state_key] = str(result)
                save_settings(state)

        ui.button(icon="folder_open", on_click=pick_file).props(
            "flat dense size=sm"
        ).tooltip(f"Browse for {label.lower()}")
        return inp

    with ui.card().classes("w-full q-mt-md"):
        ui.label("Data Paths").classes("text-subtitle1 text-bold")
        with ui.row().classes("w-full gap-4 items-end"):
            _file_input_with_picker(
                "Training Data", "train_file", "Path to training JSONL",
            )
        with ui.row().classes("w-full gap-4 items-end"):
            _file_input_with_picker(
                "Validation Data", "val_file", "Path to validation JSONL",
            )
        with ui.row().classes("w-full gap-4 items-end"):
            _file_input_with_picker(
                "Test Set", "test_file", "Path to test JSONL for evaluation",
            )

    with ui.card().classes("w-full q-mt-md"):
        ui.label("Evaluation Endpoints").classes("text-subtitle1 text-bold")
        with ui.row().classes("w-full gap-4"):
            ui.input(
                "Model A Name",
                value=state["eval_model_a"],
                on_change=on_param_change("eval_model_a"),
            ).classes("w-48")
            ui.input(
                "Model A Endpoint",
                value=state["eval_endpoint_a"],
                on_change=on_param_change("eval_endpoint_a"),
            ).classes("flex-grow")

        with ui.row().classes("w-full gap-4"):
            ui.input(
                "Model B Name (optional)",
                value=state["eval_model_b"],
                on_change=on_param_change("eval_model_b"),
            ).classes("w-48")
            ui.input(
                "Model B Endpoint (optional)",
                value=state["eval_endpoint_b"],
                on_change=on_param_change("eval_endpoint_b"),
            ).classes("flex-grow")

        ui.select(
            options=["zero-shot", "fine-tuned"],
            value=state["eval_mode"],
            label="Evaluation Mode",
            on_change=on_param_change("eval_mode"),
        ).classes("w-48").props("outlined")

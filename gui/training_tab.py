"""Fine-Tuning tab — subprocess-launched training with live monitoring charts.

Reuses ``MetricsReader`` from ``utils.training_monitor`` for proven
incremental ``metrics.jsonl`` polling.  Charts replicate the training
monitor dashboard layout inside a tab panel.
"""

from __future__ import annotations

import logging
from pathlib import Path

from nicegui import ui

from gui.process_runner import ProcessRunner
from utils.training_monitor import MetricsReader, format_eta

logger = logging.getLogger(__name__)

REFRESH_INTERVAL = 3.0  # seconds

# ── Chart configuration constants ────────────────────────────────────
LOSS_CHART_CONFIG = {
    "tooltip": {"trigger": "axis"},
    "legend": {"data": ["train_loss", "eval_loss"], "top": 0, "right": 0},
    "xAxis": {"type": "value", "name": "Step", "nameLocation": "middle", "nameGap": 25},
    "yAxis": {"type": "value", "name": "Loss"},
    "series": [
        {"name": "train_loss", "type": "line", "data": [], "smooth": True,
         "symbol": "none", "lineStyle": {"width": 2}},
        {"name": "eval_loss", "type": "scatter", "data": [],
         "symbolSize": 8, "itemStyle": {"color": "#ee6666"}},
    ],
    "grid": {"left": 60, "right": 20, "bottom": 40, "top": 30},
}

LR_CHART_CONFIG = {
    "tooltip": {"trigger": "axis", "formatter": "{c}"},
    "xAxis": {"type": "value", "name": "Step", "nameLocation": "middle", "nameGap": 25},
    "yAxis": {"type": "value", "name": "LR", "axisLabel": {"formatter": "{value}"}},
    "series": [
        {"name": "learning_rate", "type": "line", "data": [], "smooth": True,
         "symbol": "none", "lineStyle": {"width": 2, "color": "#91cc75"}},
    ],
    "grid": {"left": 80, "right": 20, "bottom": 40, "top": 30},
}

GPU_CHART_CONFIG = {
    "tooltip": {"trigger": "axis"},
    "legend": {"data": ["allocated", "max_allocated"], "top": 0, "right": 0},
    "xAxis": {"type": "value", "name": "Step", "nameLocation": "middle", "nameGap": 25},
    "yAxis": {"type": "value", "name": "GiB"},
    "series": [
        {"name": "allocated", "type": "line", "data": [], "smooth": True,
         "symbol": "none", "lineStyle": {"width": 2}},
        {"name": "max_allocated", "type": "line", "data": [], "smooth": True,
         "symbol": "none", "lineStyle": {"width": 1, "type": "dashed", "color": "#ee6666"}},
    ],
    "grid": {"left": 60, "right": 20, "bottom": 40, "top": 40},
}

GRAD_CHART_CONFIG = {
    "tooltip": {"trigger": "axis"},
    "xAxis": {"type": "value", "name": "Step", "nameLocation": "middle", "nameGap": 25},
    "yAxis": {"type": "value", "name": "Grad Norm"},
    "series": [
        {"name": "grad_norm", "type": "line", "data": [], "smooth": True,
         "symbol": "none", "lineStyle": {"width": 2, "color": "#fac858"}},
    ],
    "grid": {"left": 60, "right": 20, "bottom": 40, "top": 30},
}


NGC_IMAGE = "nvcr.io/nvidia/pytorch:25.11-py3"
HF_CACHE = str(Path.home() / ".cache" / "huggingface")


def _build_training_cmd(state: dict) -> list[str]:
    """Build the subprocess command for the selected backend and model.

    Passes user-modified hyperparameters as CLI overrides so they
    actually take effect (instead of being ignored by ``get_config()``).

    For the TRL backend, builds a ``docker run`` command matching
    ``run_training.sh`` (NGC PyTorch container with GPU access) so that
    torch and CUDA are available.  The MLX backend runs natively via uv.
    """
    model_key = state["model_key"]
    backend = state["backend"]
    project_dir = state.get("project_dir", ".")

    # ── Hyperparameter args (shared across backends) ─────────────────
    hp_args = [
        "--lr", str(state.get("learning_rate", 2e-4)),
        "--epochs", str(int(state.get("num_epochs", 3))),
        "--lora-rank", str(int(state.get("lora_rank", 16))),
        "--batch-size", str(int(state.get("batch_size", 1))),
        "--max-seq-len", str(int(state.get("max_seq_length", 4096))),
    ]

    if backend == "mlx":
        cmd = [
            "uv", "run", "python", "-m", "training.train_lora_mlx",
            "--model", model_key,
            "--data-dir", str(Path(project_dir) / "dataset" / "export" / "alpaca"),
        ]
        cmd += hp_args
    else:  # trl — run inside NGC Docker container
        train_file = state.get("train_file", "dataset/export/alpaca/train.jsonl")
        val_file = state.get("val_file", "dataset/export/alpaca/val.jsonl")
        output_dir = f"training_output/{model_key}-lora"

        hp_args += ["--grad-accum", str(int(state.get("gradient_accumulation", 4)))]

        inner_cmd = (
            "pip install --quiet trl peft datasets 'transformers>=4.57' && "
            "python -m training.train_lora "
            f"--model {model_key} "
            f"--train-file {train_file} "
            f"--val-file {val_file} "
            f"--output-dir {output_dir} "
            + " ".join(hp_args)
        )

        # Resolve project_dir to absolute path for Docker volume mount
        abs_project = str(Path(project_dir).resolve()) if project_dir else str(Path.cwd())

        cmd = [
            "docker", "run", "--gpus", "all", "--rm",
            "--shm-size=16g",
            "-v", f"{abs_project}:/workspace/biasbuster",
            "-v", f"{HF_CACHE}:/root/.cache/huggingface",
            "-w", "/workspace/biasbuster",
            NGC_IMAGE,
            "bash", "-c", inner_cmd,
        ]

    return cmd


def _metrics_path_for(state: dict) -> Path:
    """Determine expected metrics.jsonl path for current model."""
    out_dir = state.get("training_output_dir", "")
    if out_dir:
        return Path(out_dir) / "metrics.jsonl"
    model_key = state.get("model_key", "model")
    backend = state.get("backend", "trl")
    if backend == "mlx":
        return Path(f"training_output/{model_key}-mlx-lora/metrics.jsonl")
    return Path(f"training_output/{model_key}-lora/metrics.jsonl")


def create_training_tab(state: dict) -> None:
    """Build the Fine-Tuning tab UI with live charts."""
    runner = ProcessRunner()
    reader: MetricsReader | None = None

    # ── Controls ──────────────────────────────────────────────────────
    with ui.row().classes("items-center gap-4 w-full"):
        start_btn = ui.button("Start Training", icon="play_arrow").props("color=primary")
        stop_btn = ui.button("Stop", icon="stop").props("color=negative outline")
        stop_btn.visible = False
        status_badge = ui.badge("Idle", color="grey")

    # ── Progress ──────────────────────────────────────────────────────
    with ui.card().classes("w-full q-mt-sm"):
        with ui.row().classes("items-center gap-4 w-full"):
            progress_bar = ui.linear_progress(value=0, show_value=False).classes(
                "flex-grow"
            )
            progress_label = ui.label("Step 0/0 | Epoch 0.0")
            eta_label = ui.label("ETA: —").classes("text-caption text-grey")

    # ── Charts row 1: Loss + LR ──────────────────────────────────────
    with ui.row().classes("w-full q-mt-sm gap-4"):
        with ui.card().classes("flex-grow").style("min-width: 45%;"):
            ui.label("Training Loss").classes("text-subtitle2")
            loss_chart = ui.echart(LOSS_CHART_CONFIG).style("height: 300px;")

        with ui.card().classes("flex-grow").style("min-width: 45%;"):
            ui.label("Learning Rate").classes("text-subtitle2")
            lr_chart = ui.echart(LR_CHART_CONFIG).style("height: 300px;")

    # ── Charts row 2: GPU + Config ────────────────────────────────────
    with ui.row().classes("w-full q-mt-sm gap-4"):
        gpu_card = ui.card().classes("flex-grow").style("min-width: 45%;")
        with gpu_card:
            ui.label("GPU Memory (GiB)").classes("text-subtitle2")
            gpu_chart = ui.echart(GPU_CHART_CONFIG).style("height: 300px;")
        gpu_card.visible = False

        with ui.card().classes("flex-grow").style("min-width: 45%;"):
            ui.label("Hyperparameters").classes("text-subtitle2")
            config_table = ui.table(
                columns=[
                    {"name": "param", "label": "Parameter", "field": "param", "align": "left"},
                    {"name": "value", "label": "Value", "field": "value", "align": "left"},
                ],
                rows=[],
            ).classes("w-full").props("dense flat")

    # ── Gradient norm chart ───────────────────────────────────────────
    grad_card = ui.card().classes("w-full q-mt-sm").style("max-width: 50%;")
    with grad_card:
        ui.label("Gradient Norm").classes("text-subtitle2")
        grad_chart = ui.echart(GRAD_CHART_CONFIG).style("height: 250px;")
    grad_card.visible = False

    # ── Process log ───────────────────────────────────────────────────
    with ui.expansion("Process Log", icon="terminal").classes("w-full q-mt-sm"):
        log_widget = ui.log(max_lines=500).classes("w-full").style("height: 200px;")

    runner.on_output(lambda line: log_widget.push(line.rstrip("\n")))

    # ── Chart update function ─────────────────────────────────────────
    def update_charts() -> None:
        nonlocal reader
        if reader is None:
            return
        if not reader.poll():
            return

        # Status
        if reader.completed:
            status_badge.text = "Completed"
            status_badge.props("color=green")
        elif reader.metrics:
            status_badge.text = "Training..."
            status_badge.props("color=blue")

        # Progress
        total = reader.total_steps
        current = reader.current_step
        if total > 0:
            progress_bar.value = current / total
        progress_label.text = (
            f"Step {current}/{total} | Epoch {reader.current_epoch:.2f}"
        )
        eta_label.text = f"ETA: {format_eta(reader.eta_seconds)}"

        # Loss chart
        train_loss = [
            [m["step"], round(m["loss"], 4)]
            for m in reader.metrics if "loss" in m
        ]
        eval_loss = [
            [m["step"], round(m["eval_loss"], 4)]
            for m in reader.metrics if "eval_loss" in m
        ]
        loss_chart.options["series"][0]["data"] = train_loss
        loss_chart.options["series"][1]["data"] = eval_loss
        loss_chart.update()

        # LR chart
        lr_data = [
            [m["step"], m["learning_rate"]]
            for m in reader.metrics if "learning_rate" in m
        ]
        lr_chart.options["series"][0]["data"] = lr_data
        lr_chart.update()

        # GPU chart
        if reader.has_gpu_data:
            gpu_card.visible = True
            gpu_alloc = [
                [m["step"], m["gpu_mem_gib"]]
                for m in reader.metrics if "gpu_mem_gib" in m
            ]
            gpu_max = [
                [m["step"], m["gpu_max_mem_gib"]]
                for m in reader.metrics if "gpu_max_mem_gib" in m
            ]
            gpu_chart.options["series"][0]["data"] = gpu_alloc
            gpu_chart.options["series"][1]["data"] = gpu_max
            gpu_chart.update()

        # Gradient norm chart
        grad_data = [
            [m["step"], round(m["grad_norm"], 4)]
            for m in reader.metrics if "grad_norm" in m
        ]
        if grad_data:
            grad_card.visible = True
            grad_chart.options["series"][0]["data"] = grad_data
            grad_chart.update()

        # Config table (once)
        if reader.header and not config_table.rows:
            config = reader.header.get("config", {})
            config_table.rows = [
                {"param": k, "value": str(v)} for k, v in config.items()
            ]
            config_table.update()

    # ── Button handlers ───────────────────────────────────────────────
    async def on_start() -> None:
        nonlocal reader
        model_key = state.get("model_key", "")
        if not model_key:
            ui.notify("Select a model in the Settings tab first", type="warning")
            return

        cmd = _build_training_cmd(state)
        metrics_file = _metrics_path_for(state)

        # Ensure output dir exists
        metrics_file.parent.mkdir(parents=True, exist_ok=True)

        reader = MetricsReader(metrics_file)
        status_badge.text = "Starting..."
        status_badge.props("color=blue")
        start_btn.visible = False
        stop_btn.visible = True
        progress_bar.value = 0

        project_dir = state.get("project_dir", ".")
        try:
            await runner.start(cmd, cwd=project_dir)
        except RuntimeError as exc:
            ui.notify(str(exc), type="warning")
            start_btn.visible = True
            stop_btn.visible = False

    async def on_stop() -> None:
        await runner.stop()
        start_btn.visible = True
        stop_btn.visible = False
        if not (reader and reader.completed):
            status_badge.text = "Stopped"
            status_badge.props("color=orange")

    def _poll_completion() -> None:
        """Check if the training subprocess just finished (runs in UI context)."""
        finished, code = runner.consume_finished()
        if not finished:
            return
        start_btn.visible = True
        stop_btn.visible = False
        if code == 0:
            status_badge.text = "Completed"
            status_badge.props("color=green")
            ui.notify("Training completed successfully!", type="positive")
        elif runner.status != "stopped":
            status_badge.text = "Failed"
            status_badge.props("color=red")
            ui.notify(f"Training failed (exit code {code})", type="negative")

    start_btn.on_click(on_start)
    stop_btn.on_click(on_stop)

    # Start polling timers (charts + completion detection)
    ui.timer(REFRESH_INTERVAL, update_charts)
    ui.timer(1.0, _poll_completion)

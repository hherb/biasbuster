"""
Real-time training monitoring dashboard.

NiceGUI-based web UI that reads metrics.jsonl (written by
training.callbacks.MetricsLoggerCallback) and displays live loss curves,
learning rate schedule, GPU memory, and training progress.

Usage:
    uv run python -m utils.training_monitor --metrics-dir training_output/qwen3.5-27b-lora
    uv run python -m utils.training_monitor --metrics-file path/to/metrics.jsonl
    uv run python -m utils.training_monitor --port 8081 --refresh 5
"""

import json
import logging
import time
from pathlib import Path

from nicegui import ui

logger = logging.getLogger(__name__)


class MetricsReader:
    """Incrementally reads a metrics.jsonl file, tracking file position."""

    def __init__(self, metrics_path: Path):
        self.path = metrics_path
        self._last_pos: int = 0
        self.header: dict | None = None
        self.metrics: list[dict] = []
        self.completed: bool = False
        self._start_time: float | None = None

    def poll(self) -> bool:
        """Read new lines since last poll. Returns True if new data found."""
        if not self.path.exists():
            return False
        new_data = False
        with open(self.path, "r") as f:
            f.seek(self._last_pos)
            raw = f.read()
        if not raw:
            return False
        # Only process complete lines (last chunk may be a partial write)
        if raw.endswith("\n"):
            lines = raw.split("\n")
            self._last_pos += len(raw.encode())
        else:
            last_newline = raw.rfind("\n")
            if last_newline == -1:
                return False  # no complete line yet
            lines = raw[: last_newline + 1].split("\n")
            self._last_pos += len(raw[: last_newline + 1].encode())
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_type = entry.get("type")
            if entry_type == "header":
                self.header = entry
                self._start_time = time.time()
                new_data = True
            elif entry_type == "metrics":
                self.metrics.append(entry)
                if self._start_time is None:
                    self._start_time = time.time()
                new_data = True
            elif entry_type == "completed":
                self.completed = True
                new_data = True
        return new_data

    @property
    def total_steps(self) -> int:
        if self.header:
            return self.header.get("total_steps", 0)
        return 0

    @property
    def current_step(self) -> int:
        if self.metrics:
            return self.metrics[-1].get("step", 0)
        return 0

    @property
    def current_epoch(self) -> float:
        if self.metrics:
            return self.metrics[-1].get("epoch", 0.0) or 0.0
        return 0.0

    @property
    def eta_seconds(self) -> float | None:
        if not self._start_time or not self.total_steps or not self.current_step:
            return None
        progress = self.current_step / self.total_steps
        if progress < 0.05 or progress >= 1:
            return None
        elapsed = time.time() - self._start_time
        return elapsed * (1 - progress) / progress

    @property
    def has_gpu_data(self) -> bool:
        return any("gpu_mem_gib" in m for m in self.metrics)


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:.0f}h {minutes:.0f}m"


def create_app(reader: MetricsReader, refresh_interval: float) -> None:
    """Create the NiceGUI training monitor application."""

    ui.page_title("Training Monitor")

    # --- Header ---
    with ui.header().classes("items-center justify-between q-px-md"):
        ui.label("Training Monitor").classes("text-h6")
        status_badge = ui.badge("Waiting for data...", color="grey")

    # --- Progress section ---
    with ui.card().classes("q-mx-md q-mt-md w-full").style("max-width: calc(100% - 32px);"):
        with ui.row().classes("items-center gap-4 w-full"):
            progress_bar = ui.linear_progress(value=0, show_value=False).classes(
                "flex-grow"
            )
            progress_label = ui.label("Step 0/0 | Epoch 0.0")
            eta_label = ui.label("ETA: —").classes("text-caption text-grey")

    # --- Charts row 1: Loss + LR ---
    with ui.row().classes("q-mx-md q-mt-sm w-full gap-4").style(
        "max-width: calc(100% - 32px);"
    ):
        with ui.card().classes("flex-grow").style("min-width: 45%;"):
            ui.label("Training Loss").classes("text-subtitle2")
            loss_chart = ui.echart({
                "tooltip": {"trigger": "axis"},
                "legend": {"data": ["train_loss", "eval_loss"]},
                "xAxis": {"type": "value", "name": "Step", "nameLocation": "middle", "nameGap": 25},
                "yAxis": {"type": "value", "name": "Loss"},
                "series": [
                    {"name": "train_loss", "type": "line", "data": [], "smooth": True,
                     "symbol": "none", "lineStyle": {"width": 2}},
                    {"name": "eval_loss", "type": "scatter", "data": [],
                     "symbolSize": 8, "itemStyle": {"color": "#ee6666"}},
                ],
                "grid": {"left": 60, "right": 20, "bottom": 40, "top": 40},
            }).style("height: 300px;")

        with ui.card().classes("flex-grow").style("min-width: 45%;"):
            ui.label("Learning Rate").classes("text-subtitle2")
            lr_chart = ui.echart({
                "tooltip": {"trigger": "axis", "formatter": "{c}"},
                "xAxis": {"type": "value", "name": "Step", "nameLocation": "middle", "nameGap": 25},
                "yAxis": {"type": "value", "name": "LR", "axisLabel": {"formatter": "{value}"}},
                "series": [
                    {"name": "learning_rate", "type": "line", "data": [], "smooth": True,
                     "symbol": "none", "lineStyle": {"width": 2, "color": "#91cc75"}},
                ],
                "grid": {"left": 80, "right": 20, "bottom": 40, "top": 30},
            }).style("height: 300px;")

    # --- Charts row 2: GPU memory + Config ---
    with ui.row().classes("q-mx-md q-mt-sm w-full gap-4").style(
        "max-width: calc(100% - 32px);"
    ):
        gpu_card = ui.card().classes("flex-grow").style("min-width: 45%;")
        with gpu_card:
            ui.label("GPU Memory (GiB)").classes("text-subtitle2")
            gpu_chart = ui.echart({
                "tooltip": {"trigger": "axis"},
                "legend": {"data": ["allocated", "max_allocated"]},
                "xAxis": {"type": "value", "name": "Step", "nameLocation": "middle", "nameGap": 25},
                "yAxis": {"type": "value", "name": "GiB"},
                "series": [
                    {"name": "allocated", "type": "line", "data": [], "smooth": True,
                     "symbol": "none", "lineStyle": {"width": 2}},
                    {"name": "max_allocated", "type": "line", "data": [], "smooth": True,
                     "symbol": "none", "lineStyle": {"width": 1, "type": "dashed", "color": "#ee6666"}},
                ],
                "grid": {"left": 60, "right": 20, "bottom": 40, "top": 40},
            }).style("height: 300px;")
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

    # --- Grad norm chart ---
    with ui.row().classes("q-mx-md q-mt-sm w-full gap-4").style(
        "max-width: calc(100% - 32px);"
    ):
        grad_card = ui.card().classes("flex-grow").style("min-width: 45%;")
        with grad_card:
            ui.label("Gradient Norm").classes("text-subtitle2")
            grad_chart = ui.echart({
                "tooltip": {"trigger": "axis"},
                "xAxis": {"type": "value", "name": "Step", "nameLocation": "middle", "nameGap": 25},
                "yAxis": {"type": "value", "name": "Grad Norm"},
                "series": [
                    {"name": "grad_norm", "type": "line", "data": [], "smooth": True,
                     "symbol": "none", "lineStyle": {"width": 2, "color": "#fac858"}},
                ],
                "grid": {"left": 60, "right": 20, "bottom": 40, "top": 30},
            }).style("height: 250px;")
        grad_card.visible = False

    def update():
        """Poll for new data and refresh all UI elements."""
        if not reader.poll():
            # Check if file appeared
            if not reader.path.exists():
                status_badge.text = "Waiting for data..."
                status_badge.props("color=grey")
            return

        # --- Status ---
        if reader.completed:
            status_badge.text = "Training completed"
            status_badge.props("color=green")
        elif reader.metrics:
            status_badge.text = "Training in progress"
            status_badge.props("color=blue")

        # --- Progress ---
        total = reader.total_steps
        current = reader.current_step
        if total > 0:
            progress_bar.value = current / total
        progress_label.text = (
            f"Step {current}/{total} | Epoch {reader.current_epoch:.2f}"
        )
        eta_label.text = f"ETA: {_format_eta(reader.eta_seconds)}"

        # --- Loss chart ---
        train_loss_data = [
            [m["step"], round(m["loss"], 4)]
            for m in reader.metrics if "loss" in m
        ]
        eval_loss_data = [
            [m["step"], round(m["eval_loss"], 4)]
            for m in reader.metrics if "eval_loss" in m
        ]
        loss_chart.options["series"][0]["data"] = train_loss_data
        loss_chart.options["series"][1]["data"] = eval_loss_data
        loss_chart.update()

        # --- LR chart ---
        lr_data = [
            [m["step"], m["learning_rate"]]
            for m in reader.metrics if "learning_rate" in m
        ]
        lr_chart.options["series"][0]["data"] = lr_data
        lr_chart.update()

        # --- GPU chart ---
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

        # --- Grad norm chart ---
        grad_data = [
            [m["step"], round(m["grad_norm"], 4)]
            for m in reader.metrics if "grad_norm" in m
        ]
        if grad_data:
            grad_card.visible = True
            grad_chart.options["series"][0]["data"] = grad_data
            grad_chart.update()

        # --- Config table ---
        if reader.header and not config_table.rows:
            config = reader.header.get("config", {})
            config_table.rows = [
                {"param": k, "value": str(v)} for k, v in config.items()
            ]
            config_table.update()

    ui.timer(refresh_interval, update)


def _find_latest_metrics(base_dir: str) -> Path | None:
    """Find the most recently modified metrics.jsonl under base_dir."""
    base = Path(base_dir)
    if not base.exists():
        return None
    candidates = sorted(base.rglob("metrics.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time training monitoring dashboard"
    )
    parser.add_argument(
        "--metrics-dir",
        default=None,
        help="Directory containing metrics.jsonl (searches recursively)",
    )
    parser.add_argument(
        "--metrics-file",
        default=None,
        help="Explicit path to metrics.jsonl",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port for the web server (default: 8081)",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=3.0,
        help="Polling interval in seconds (default: 3)",
    )
    args = parser.parse_args()

    # Resolve metrics path
    if args.metrics_file:
        metrics_path = Path(args.metrics_file)
    elif args.metrics_dir:
        metrics_path = Path(args.metrics_dir) / "metrics.jsonl"
    else:
        # Auto-detect from training_output/
        found = _find_latest_metrics("training_output")
        if found:
            metrics_path = found
            print(f"Auto-detected: {metrics_path}")
        else:
            metrics_path = Path("training_output/metrics.jsonl")
            print(f"No metrics found yet — will watch: {metrics_path}")

    print(f"Monitoring: {metrics_path}")
    print(f"Dashboard: http://localhost:{args.port}")

    reader = MetricsReader(metrics_path)
    create_app(reader, args.refresh)
    ui.run(
        title="Training Monitor",
        port=args.port,
        reload=False,
        on_air=False,
    )


if __name__ == "__main__":
    main()

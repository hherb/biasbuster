"""
MLX training metrics callback that writes to a JSONL log file.

Bridges the MLX-lm callback protocol (on_train_loss_report / on_val_loss_report)
to the same metrics.jsonl format consumed by utils/training_monitor.py.

Output format matches training/callbacks.py exactly:
    {"type":"header","total_steps":500,"num_epochs":3,"config":{...},"timestamp":"..."}
    {"type":"metrics","step":10,"epoch":0.12,"loss":2.41,"learning_rate":1.8e-4,...}
    {"type":"completed","step":500,"timestamp":"..."}
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

try:
    from mlx_lm.tuner.callbacks import TrainingCallback
except ImportError:
    # When running outside Apple Silicon (e.g. smoke test, type checking),
    # provide a minimal base class so the module can be imported.
    class TrainingCallback:  # type: ignore[no-redef]
        pass

logger = logging.getLogger(__name__)

# Bytes to GiB divisor
_BYTES_TO_GIB = 1024 ** 3


class MLXMetricsLoggerCallback(TrainingCallback):
    """Writes MLX training metrics to metrics.jsonl for the training monitor.

    Implements the two callback methods that mlx_lm.tuner.train() invokes:
        - on_train_loss_report(info: dict)
        - on_val_loss_report(info: dict)

    Args:
        output_dir: Directory to write metrics.jsonl into.
        total_iters: Total training iterations (for the header).
        num_epochs: Number of training epochs (for the header).
        iters_per_epoch: Iterations per epoch (for synthesizing epoch field).
        extra_config: Additional config entries for the header record.
    """

    def __init__(
        self,
        output_dir: str,
        total_iters: int,
        num_epochs: int,
        iters_per_epoch: int,
        extra_config: dict | None = None,
        resume: bool = False,
    ):
        self.metrics_path = Path(output_dir) / "metrics.jsonl"
        self.iters_per_epoch = max(iters_per_epoch, 1)
        self._resume = resume
        self._write_header(total_iters, num_epochs, extra_config or {})

    def _write_line(self, data: dict, mode: str = "a") -> None:
        """Write a single JSON line. Use mode='w' to truncate first."""
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, mode) as f:
            f.write(json.dumps(data) + "\n")
            f.flush()

    def _write_header(
        self, total_iters: int, num_epochs: int, extra_config: dict
    ) -> None:
        """Write the header record at the start of training."""
        # Fresh run: truncate + write header atomically (single open with 'w')
        header_mode = "a" if self._resume else "w"
        if header_mode == "w":
            logger.info("Truncating stale metrics file for fresh run")
        self._write_line({
            "type": "header",
            "total_steps": total_iters,
            "num_epochs": num_epochs,
            "config": extra_config,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, mode=header_mode)
        logger.info("MLX metrics logging to %s", self.metrics_path)

    def on_train_loss_report(self, info: dict) -> None:
        """Called by mlx_lm.tuner.train() every steps_per_report iterations.

        Expected info keys: iteration, train_loss, learning_rate,
        iterations_per_second, tokens_per_second, trained_tokens, peak_memory.
        """
        step = info.get("iteration", 0)
        entry: dict = {
            "type": "metrics",
            "step": step,
            "epoch": round(step / self.iters_per_epoch, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "loss": info.get("train_loss"),
            "learning_rate": info.get("learning_rate"),
        }
        # Unified memory usage (Apple Silicon) — map to gpu_mem_gib for monitor
        peak_mem = info.get("peak_memory")
        if peak_mem is not None:
            entry["gpu_mem_gib"] = round(peak_mem / _BYTES_TO_GIB, 2)

        # MLX-specific extras
        for key in ("iterations_per_second", "tokens_per_second", "trained_tokens"):
            if key in info:
                entry[key] = info[key]

        self._write_line(entry)

    def on_val_loss_report(self, info: dict) -> None:
        """Called by mlx_lm.tuner.train() every steps_per_eval iterations.

        Expected info keys: iteration, val_loss, val_time.
        """
        step = info.get("iteration", 0)
        self._write_line({
            "type": "metrics",
            "step": step,
            "epoch": round(step / self.iters_per_epoch, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "eval_loss": info.get("val_loss"),
        })

    def finalize(self, final_step: int) -> None:
        """Write the completion record after training ends."""
        self._write_line({
            "type": "completed",
            "step": final_step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("MLX training completed — metrics log finalized")


if __name__ == "__main__":
    """Smoke test: write synthetic MLX metrics.jsonl for dashboard testing."""
    import math
    import tempfile

    out = Path(tempfile.mkdtemp()) / "mlx_test_run"
    out.mkdir()

    total_iters = 200
    epochs = 3
    iters_per_epoch = total_iters // epochs

    cb = MLXMetricsLoggerCallback(
        output_dir=str(out),
        total_iters=total_iters,
        num_epochs=epochs,
        iters_per_epoch=iters_per_epoch,
        extra_config={
            "model_name_or_path": "mlx-community/Qwen3.5-9B-4bit",
            "lora_rank": 16,
            "lora_scale": 32.0,
            "platform": "apple_silicon_mlx",
        },
    )

    for step in range(10, total_iters + 1, 10):
        base_loss = 2.5 * math.exp(-step / 80) + 0.3
        loss = base_loss + 0.05 * math.sin(step / 7)
        warmup_end = int(total_iters * 0.1)
        if step < warmup_end:
            lr = 2e-4 * step / warmup_end
        else:
            progress = (step - warmup_end) / (total_iters - warmup_end)
            lr = 2e-4 * 0.5 * (1 + math.cos(math.pi * progress))

        cb.on_train_loss_report({
            "iteration": step,
            "train_loss": round(loss, 4),
            "learning_rate": lr,
            "iterations_per_second": 0.8,
            "tokens_per_second": 3200,
            "trained_tokens": step * 4096,
            "peak_memory": 12 * _BYTES_TO_GIB,
        })

        if step % 50 == 0:
            cb.on_val_loss_report({
                "iteration": step,
                "val_loss": round(loss + 0.05, 4),
                "val_time": 12.5,
            })

    cb.finalize(total_iters)

    print(f"Wrote synthetic MLX metrics to: {out / 'metrics.jsonl'}")
    with open(out / "metrics.jsonl") as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print(f"First line: {lines[0].strip()[:120]}...")
    print(f"Last line:  {lines[-1].strip()}")

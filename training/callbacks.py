"""
Training metrics callback that writes to a JSONL log file.

Designed to run inside the NGC Docker container with zero extra dependencies
beyond what TRL/transformers already provide. The companion NiceGUI dashboard
(utils/training_monitor.py) reads this file from the host via volume mount.

Output format (one JSON object per line):
    {"type":"header","total_steps":500,"num_epochs":3,"config":{...},"timestamp":"..."}
    {"type":"metrics","step":10,"epoch":0.12,"loss":2.41,"learning_rate":1.8e-4,...}
    {"type":"completed","step":500,"timestamp":"..."}
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

try:
    from transformers import TrainerCallback
except ImportError:
    # When running outside the training container (e.g. smoke test on host),
    # provide a minimal base class so the module can be imported.
    class TrainerCallback:  # type: ignore[no-redef]
        pass

logger = logging.getLogger(__name__)


class MetricsLoggerCallback(TrainerCallback):
    """Appends training metrics to a JSONL file at each logging step."""

    def __init__(self, output_dir: str, extra_config: dict | None = None):
        self.metrics_path = Path(output_dir) / "metrics.jsonl"
        self.extra_config = extra_config or {}

    def _append_line(self, data: dict) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(data) + "\n")
            f.flush()

    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = args.max_steps if args.max_steps > 0 else state.max_steps
        config = {
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": str(args.lr_scheduler_type),
            "warmup_ratio": args.warmup_ratio,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "bf16": args.bf16,
            "max_grad_norm": args.max_grad_norm,
            "num_train_epochs": args.num_train_epochs,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
        }
        config.update(self.extra_config)
        self._append_line({
            "type": "header",
            "total_steps": total_steps,
            "num_epochs": args.num_train_epochs,
            "config": config,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(f"Metrics logging to {self.metrics_path}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {
            "type": "metrics",
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch is not None else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for key in ("loss", "eval_loss", "learning_rate", "grad_norm",
                     "eval_runtime", "eval_samples_per_second"):
            if key in logs:
                entry[key] = logs[key]

        # GPU memory stats
        try:
            import torch
            if torch.cuda.is_available():
                entry["gpu_mem_gib"] = round(
                    torch.cuda.memory_allocated() / (1024 ** 3), 2
                )
                entry["gpu_max_mem_gib"] = round(
                    torch.cuda.max_memory_allocated() / (1024 ** 3), 2
                )
        except ImportError:
            pass

        self._append_line(entry)

    def on_train_end(self, args, state, control, **kwargs):
        self._append_line({
            "type": "completed",
            "step": state.global_step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Training completed — metrics log finalized")


if __name__ == "__main__":
    """Smoke test: write synthetic metrics.jsonl for dashboard testing."""
    import math
    import tempfile

    out = Path(tempfile.mkdtemp()) / "test_run"
    out.mkdir()

    cb = MetricsLoggerCallback(str(out), extra_config={
        "model_name_or_path": "test/model",
        "lora_r": 16,
        "lora_alpha": 32,
        "max_seq_length": 4096,
    })

    # Simulate header
    class FakeArgs:
        max_steps = 200
        learning_rate = 2e-4
        lr_scheduler_type = "cosine"
        warmup_ratio = 0.1
        per_device_train_batch_size = 1
        gradient_accumulation_steps = 4
        bf16 = True
        max_grad_norm = 1.0
        num_train_epochs = 3
        logging_steps = 10
        save_steps = 50
        eval_steps = 50

    class FakeState:
        max_steps = 200
        global_step = 0
        epoch = 0.0

    cb.on_train_begin(FakeArgs(), FakeState(), None)

    # Simulate training
    total = 200
    warmup = int(total * 0.1)
    for step in range(10, total + 1, 10):
        FakeState.global_step = step
        FakeState.epoch = round(step / total * 3, 4)
        # Synthetic loss: starts high, decays with noise
        base_loss = 2.5 * math.exp(-step / 80) + 0.3
        loss = base_loss + 0.05 * math.sin(step / 7)
        # Cosine LR with warmup
        if step < warmup:
            lr = 2e-4 * step / warmup
        else:
            progress = (step - warmup) / (total - warmup)
            lr = 2e-4 * 0.5 * (1 + math.cos(math.pi * progress))
        logs = {"loss": round(loss, 4), "learning_rate": lr, "grad_norm": round(0.8 + 0.3 * math.sin(step / 5), 3)}
        if step % 50 == 0:
            logs["eval_loss"] = round(loss + 0.05, 4)
        cb.on_log(FakeArgs(), FakeState, None, logs=logs)

    FakeState.global_step = total
    cb.on_train_end(FakeArgs(), FakeState, None)

    print(f"Wrote synthetic metrics to: {out / 'metrics.jsonl'}")
    with open(out / "metrics.jsonl") as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print(f"First line: {lines[0].strip()[:120]}...")
    print(f"Last line:  {lines[-1].strip()}")

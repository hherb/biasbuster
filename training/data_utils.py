"""
Dataset loading and formatting for LoRA fine-tuning.

Reads the alpaca-format JSONL exported by export.py and converts each example
into tokenizer-specific chat messages that TRL's SFTTrainer can consume.

Also provides alpaca-to-chat format conversion for MLX-lm fine-tuning.
"""

import json
from pathlib import Path

from datasets import Dataset


def load_alpaca_jsonl(path: str | Path) -> Dataset:
    """Load an alpaca-format JSONL file into a HuggingFace Dataset.

    Each line is expected to have: {"system": ..., "instruction": ..., "input": ..., "output": ...}
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def make_formatting_func(tokenizer):
    """Return a function that converts alpaca records to chat-templated strings.

    The returned function is passed to SFTTrainer as ``formatting_func``.
    Each call receives a single example dict and returns a list containing one
    formatted string (SFTTrainer expects a list).
    """

    def _format(example):
        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    return _format


# ---------------------------------------------------------------------------
# Alpaca → Chat JSONL conversion (for MLX-lm)
# ---------------------------------------------------------------------------

# MLX-lm expects split names: train.jsonl, valid.jsonl, test.jsonl
_SPLIT_NAME_MAP = {"val": "valid"}


def alpaca_to_chat_jsonl(alpaca_path: Path, output_path: Path) -> int:
    """Convert an alpaca-format JSONL file to MLX-lm chat-format JSONL.

    Alpaca format:
        {"system": ..., "instruction": ..., "input": ..., "output": ...}
    Chat format (MLX-lm):
        {"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}

    Note: The alpaca ``input`` field (always empty in BiasBuster data) is not
    included in the chat messages — only system, instruction, and output are used.

    Args:
        alpaca_path: Path to the source alpaca JSONL file.
        output_path: Path to write the chat-format JSONL file.

    Returns:
        Number of examples converted.

    Raises:
        ValueError: If any record is missing required keys.
    """
    required_keys = {"system", "instruction", "output"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(alpaca_path) as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = required_keys - rec.keys()
            if missing:
                raise ValueError(
                    f"{alpaca_path}:{line_num}: missing keys {missing}"
                )
            messages = [
                {"role": "system", "content": rec["system"]},
                {"role": "user", "content": rec["instruction"]},
                {"role": "assistant", "content": rec["output"]},
            ]
            fout.write(json.dumps({"messages": messages}) + "\n")
            count += 1
    return count


def ensure_chat_data(
    alpaca_dir: str | Path,
    chat_dir: str | Path,
) -> Path:
    """Convert all alpaca JSONL splits to MLX-lm chat format if needed.

    Handles the naming convention difference: alpaca uses ``val.jsonl``,
    MLX-lm expects ``valid.jsonl``.

    Conversion is skipped for a split if the output file already exists and
    is newer than the source file.

    Args:
        alpaca_dir: Directory containing {train,val,test}.jsonl.
        chat_dir: Directory to write {train,valid,test}.jsonl.

    Returns:
        Path to the chat_dir (for chaining).
    """
    alpaca_dir = Path(alpaca_dir)
    chat_dir = Path(chat_dir)
    chat_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        src = alpaca_dir / f"{split}.jsonl"
        if not src.exists():
            continue
        out_name = _SPLIT_NAME_MAP.get(split, split)
        dst = chat_dir / f"{out_name}.jsonl"

        # Skip if destination is up-to-date
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue

        count = alpaca_to_chat_jsonl(src, dst)
        print(f"  Converted {src.name} → {dst.name} ({count} examples)")

    return chat_dir

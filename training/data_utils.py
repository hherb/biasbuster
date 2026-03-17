"""
Dataset loading and formatting for LoRA fine-tuning.

Reads the alpaca-format JSONL exported by export.py and converts each example
into tokenizer-specific chat messages that TRL's SFTTrainer can consume.
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

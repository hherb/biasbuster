"""
Dataset loading and formatting for LoRA fine-tuning.

Reads the alpaca-format JSONL exported by export.py and converts each example
into tokenizer-specific chat messages that TRL's SFTTrainer can consume.

Also provides alpaca-to-chat format conversion for MLX-lm fine-tuning.

Note on Harmony format (GPT-OSS):
    GPT-OSS uses a multi-channel response format (Harmony) at the template
    level.  However, attention-only LoRA cannot teach the model to use
    Harmony's channel tokens (``<|channel|>analysis`` vs ``<|channel|>final``).
    The model falls back to producing literal ``<think>`` tags as text.

    Therefore, GPT-OSS training uses **single-channel** format: everything
    (including ``<think>`` blocks) goes into the ``content`` field, which
    the Harmony template renders into the ``final`` channel only.  The model
    learns ``<think>`` as a text pattern it can reproduce at inference.

    See ``docs/MISTAKES_TO_ROUND_3.md`` §5 for the full analysis.
"""

import json
import re
from pathlib import Path

from datasets import Dataset


# ---------------------------------------------------------------------------
# Think-block splitting for Harmony-style chat templates
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)


def _split_think_block(output: str) -> tuple[str, str | None]:
    """Split a ``<think>...</think>`` prefix from the model output.

    Args:
        output: Raw assistant output that may start with a think block.

    Returns:
        (content, thinking) — *content* is the output after the think block
        (or the full output if no block), *thinking* is the extracted
        reasoning text or ``None``.
    """
    m = _THINK_RE.match(output)
    if m:
        thinking = m.group(1).strip()
        content = output[m.end():].strip()
        if not content:
            # Think-only output — keep everything in content to avoid
            # training on an empty final channel.
            return output.strip(), None
        return content, thinking
    return output, None


def validate_harmony_channels(tokenizer, example: dict) -> None:
    """Verify that a Harmony tokenizer renders the final channel correctly.

    For single-channel training (GPT-OSS), we keep everything in ``content``
    so only the ``final`` channel should be present.  Silently returns for
    non-Harmony tokenizers (e.g. Qwen, OLMo).

    Args:
        tokenizer: A HuggingFace tokenizer with ``apply_chat_template``.
        example: A single alpaca-format record with ``system``, ``instruction``,
                 and ``output`` keys.

    Raises:
        ValueError: If the tokenizer renders ``<|channel|>`` tokens but the
                    expected ``final`` channel is missing.
    """
    fmt_fn = make_formatting_func(tokenizer)
    rendered = fmt_fn(example)
    # Non-Harmony tokenizers won't contain channel tokens — skip silently
    if "<|channel|>" not in rendered:
        return
    if "<|channel|>final" not in rendered:
        raise ValueError(
            "Harmony template detected but <|channel|>final is missing. "
            "The content field may be empty.\n"
            f"Rendered (first 500 chars): {rendered[:500]}"
        )
    if "<|channel|>analysis" in rendered:
        import logging
        logging.getLogger(__name__).warning(
            "Harmony analysis channel detected in training data. "
            "GPT-OSS single-channel training should NOT produce analysis "
            "channel tokens. Check that make_formatting_func() is not "
            "splitting <think> blocks into the thinking field."
        )


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


def _is_harmony_tokenizer(tokenizer) -> bool:
    """Detect whether a tokenizer uses the Harmony response format (GPT-OSS).

    Renders a minimal assistant message and checks for ``<|channel|>`` tokens.
    """
    try:
        test = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": "test"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return "<|channel|>" in test
    except Exception:
        return False


def make_formatting_func(tokenizer):
    """Return a function that converts alpaca records to chat-templated strings.

    The returned function is passed to SFTTrainer as ``formatting_func``.
    Each call receives a single example dict and returns a list containing one
    formatted string (SFTTrainer expects a list).

    For Harmony tokenizers (GPT-OSS), ``<think>`` blocks are kept as literal
    text in ``content`` (single-channel).  Attention-only LoRA cannot learn
    Harmony channel tokens, so the model must see ``<think>`` as text it can
    reproduce.  Non-Harmony models also keep ``<think>`` in ``content``
    (no-op — the thinking field is ignored by non-Harmony templates).
    """
    harmony = _is_harmony_tokenizer(tokenizer)

    def _format(example):
        # Single-channel: keep <think> as literal text in content.
        # The Harmony template renders this into the final channel only,
        # which matches inference (the model produces literal <think> tags).
        assistant_msg: dict = {"role": "assistant", "content": example["output"]}
        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["instruction"]},
            assistant_msg,
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
            # Single-channel: keep <think> as literal text in content.
            # MLX-lm tokenizers apply chat templates at training time;
            # the thinking field is only used by Harmony templates, which
            # we've determined attention-only LoRA cannot learn.
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

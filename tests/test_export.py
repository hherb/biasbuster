"""Tests for export module."""

import json
import pytest
from pathlib import Path

from export import (
    to_alpaca_format,
    to_sharegpt_format,
    to_openai_chat_format,
    build_thinking_chain,
    build_structured_response,
    export_dataset,
    SYSTEM_PROMPT,
)


class TestToAlpacaFormat:
    """Tests for to_alpaca_format function."""

    def test_required_keys(self, sample_annotation):
        result = to_alpaca_format(sample_annotation)
        assert "system" in result
        assert "instruction" in result
        assert "input" in result
        assert "output" in result

    def test_system_prompt_included(self, sample_annotation):
        result = to_alpaca_format(sample_annotation)
        assert result["system"] == SYSTEM_PROMPT

    def test_input_is_empty_string(self, sample_annotation):
        result = to_alpaca_format(sample_annotation)
        assert result["input"] == ""

    def test_instruction_contains_title_and_pmid(self, sample_annotation):
        result = to_alpaca_format(sample_annotation)
        assert "Test Drug Trial" in result["instruction"]
        assert "ANN001" in result["instruction"]

    def test_thinking_included_by_default(self, sample_annotation):
        result = to_alpaca_format(sample_annotation)
        assert "<think>" in result["output"]
        assert "</think>" in result["output"]

    def test_thinking_excluded(self, sample_annotation):
        result = to_alpaca_format(sample_annotation, include_thinking=False)
        assert "<think>" not in result["output"]

    def test_output_contains_structured_response(self, sample_annotation):
        result = to_alpaca_format(sample_annotation)
        assert "Statistical Reporting" in result["output"]


class TestToSharegptFormat:
    """Tests for to_sharegpt_format function."""

    def test_conversations_key(self, sample_annotation):
        result = to_sharegpt_format(sample_annotation)
        assert "conversations" in result

    def test_three_turns(self, sample_annotation):
        result = to_sharegpt_format(sample_annotation)
        convs = result["conversations"]
        assert len(convs) == 3
        assert convs[0]["from"] == "system"
        assert convs[1]["from"] == "human"
        assert convs[2]["from"] == "gpt"

    def test_system_message(self, sample_annotation):
        result = to_sharegpt_format(sample_annotation)
        assert result["conversations"][0]["value"] == SYSTEM_PROMPT

    def test_human_message_contains_abstract(self, sample_annotation):
        result = to_sharegpt_format(sample_annotation)
        assert "Test abstract text" in result["conversations"][1]["value"]

    def test_thinking_in_gpt_response(self, sample_annotation):
        result = to_sharegpt_format(sample_annotation)
        assert "<think>" in result["conversations"][2]["value"]


class TestToOpenaiChatFormat:
    """Tests for to_openai_chat_format function."""

    def test_messages_key(self, sample_annotation):
        result = to_openai_chat_format(sample_annotation)
        assert "messages" in result

    def test_three_messages(self, sample_annotation):
        result = to_openai_chat_format(sample_annotation)
        msgs = result["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_content_key_in_messages(self, sample_annotation):
        result = to_openai_chat_format(sample_annotation)
        for msg in result["messages"]:
            assert "content" in msg


class TestBuildThinkingChain:
    """Tests for build_thinking_chain function."""

    def test_includes_think_tags(self, sample_annotation):
        chain = build_thinking_chain(sample_annotation)
        assert chain.startswith("<think>")
        assert chain.endswith("</think>")

    def test_uses_reasoning_field(self, sample_annotation):
        chain = build_thinking_chain(sample_annotation)
        assert "only reports relative measures" in chain

    def test_builds_from_domains_when_no_reasoning(self):
        annotation = {
            "statistical_reporting": {
                "relative_only": True,
                "evidence_quotes": ["HR 0.50"],
            },
            "spin": {"spin_level": "high", "focus_on_secondary_when_primary_ns": True},
            "conflict_of_interest": {"funding_type": "industry"},
        }
        chain = build_thinking_chain(annotation)
        assert "<think>" in chain
        assert "relative measures" in chain
        assert "industry" in chain.lower()

    def test_empty_annotation(self):
        chain = build_thinking_chain({})
        assert "<think>" in chain
        assert "</think>" in chain


class TestExportDataset:
    """Tests for export_dataset function."""

    def test_creates_split_files(self, tmp_path, sample_annotation):
        annotations = [sample_annotation] * 20
        export_dataset(annotations, tmp_path, fmt="alpaca")
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "val.jsonl").exists()
        assert (tmp_path / "test.jsonl").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_split_proportions(self, tmp_path, sample_annotation):
        annotations = [sample_annotation] * 100
        export_dataset(annotations, tmp_path, fmt="alpaca")
        train_lines = (tmp_path / "train.jsonl").read_text().strip().split("\n")
        val_lines = (tmp_path / "val.jsonl").read_text().strip().split("\n")
        test_lines = (tmp_path / "test.jsonl").read_text().strip().split("\n")
        assert len(train_lines) == 80
        assert len(val_lines) == 10
        assert len(test_lines) == 10

    def test_metadata_written(self, tmp_path, sample_annotation):
        annotations = [sample_annotation] * 10
        export_dataset(annotations, tmp_path, fmt="sharegpt")
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["format"] == "sharegpt"
        assert meta["total_examples"] == 10
        assert meta["seed"] == 42

    def test_alpaca_format_valid_jsonl(self, tmp_path, sample_annotation):
        export_dataset([sample_annotation] * 5, tmp_path, fmt="alpaca")
        for line in (tmp_path / "train.jsonl").read_text().strip().split("\n"):
            item = json.loads(line)
            assert "system" in item
            assert "instruction" in item

    def test_sharegpt_format_valid_jsonl(self, tmp_path, sample_annotation):
        export_dataset([sample_annotation] * 5, tmp_path, fmt="sharegpt")
        for line in (tmp_path / "train.jsonl").read_text().strip().split("\n"):
            item = json.loads(line)
            assert "conversations" in item

    def test_openai_chat_format(self, tmp_path, sample_annotation):
        export_dataset([sample_annotation] * 5, tmp_path, fmt="openai_chat")
        for line in (tmp_path / "train.jsonl").read_text().strip().split("\n"):
            item = json.loads(line)
            assert "messages" in item

    def test_empty_annotations(self, tmp_path):
        export_dataset([], tmp_path, fmt="alpaca")
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "train.jsonl").read_text() == ""

    def test_deterministic_with_seed(self, tmp_path, sample_annotation):
        annotations = [
            {**sample_annotation, "pmid": f"P{i}"} for i in range(20)
        ]
        export_dataset(annotations, tmp_path / "run1", fmt="alpaca", seed=42)
        export_dataset(annotations, tmp_path / "run2", fmt="alpaca", seed=42)
        assert (tmp_path / "run1" / "train.jsonl").read_text() == (
            tmp_path / "run2" / "train.jsonl"
        ).read_text()

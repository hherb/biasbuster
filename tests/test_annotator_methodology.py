"""Byte-identity + plumbing tests for the methodology-aware BaseAnnotator.

Primary property under test: when ``methodology`` is ``None`` (the default,
pre-methodology callers), the annotator sends an LLM call with a system
prompt and user message that are *byte-identical* to what the legacy
code path produced. This is the safety net for Step 4 — refactoring the
prompt layer must not regress the existing biasbuster pipeline.

Secondary properties:

- Full-text-required methodologies refuse abstract-only annotators with
  :class:`FullTextRequiredError`.
- The methodology kwarg threads through ``annotate_batch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from biasbuster import prompts as _prompts_v1
from biasbuster.annotators import (
    BaseAnnotator,
    build_user_message as _legacy_build_user_message,
)
from biasbuster.methodologies import (
    FullTextRequiredError,
    Methodology,
    get_methodology,
)


@dataclass
class FakeAnnotator(BaseAnnotator):
    """A BaseAnnotator subclass with a mocked transport.

    Captures every ``_call_llm`` invocation so tests can assert on the
    exact system prompt + user message that would have hit the wire.
    """

    model: str = "fake-model"
    max_retries: int = 1
    canned_response: Optional[str] = None
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    async def _call_llm(
        self, system_prompt: str, user_message: str, pmid: str = "",
    ) -> Optional[str]:
        self.calls.append((system_prompt, user_message, pmid))
        return self.canned_response

    async def __aenter__(self) -> "FakeAnnotator":
        return self

    async def __aexit__(self, *args) -> None:
        return None


def _valid_biasbuster_json() -> str:
    """A minimally-valid biasbuster annotation JSON response."""
    return (
        '{"overall_severity": "low", "overall_bias_probability": 0.2, '
        '"confidence": "high", "reasoning": "seems fine", '
        '"statistical_reporting": {"severity": "low"}, '
        '"spin": {"spin_level": "none", "severity": "none"}, '
        '"outcome_reporting": {"severity": "low"}, '
        '"conflict_of_interest": {"severity": "low"}, '
        '"methodology": {"severity": "low"}, '
        '"recommended_verification_steps": ["Cross-check CT.gov"]}'
    )


class TestByteIdentityWithBiasbusterDefault:
    """methodology=None must produce the exact legacy prompt + user message."""

    @pytest.mark.asyncio
    async def test_annotate_abstract_system_prompt_is_canonical_constant(
        self,
    ) -> None:
        fa = FakeAnnotator(canned_response=_valid_biasbuster_json())
        await fa.annotate_abstract(
            pmid="T1",
            title="Effect of TestDrug on mortality",
            abstract="Randomized placebo-controlled trial of 500 patients.",
            metadata={"journal": "NEJM"},
        )
        assert len(fa.calls) == 1
        system_prompt, _, _ = fa.calls[0]
        # Byte-identity: the system prompt sent to _call_llm must be the
        # exact same string object (and same bytes) as the legacy constant.
        assert system_prompt is _prompts_v1.ANNOTATION_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_annotate_abstract_user_message_matches_legacy_builder(
        self,
    ) -> None:
        """The refactored path's user message equals the legacy function's output.

        The legacy code called ``build_user_message(pmid, title, abstract,
        metadata)``. The refactored code calls the biasbuster methodology
        adapter which merges ``paper`` + ``enrichment`` into the metadata
        bag for the same legacy function. Those two code paths must
        produce the same bytes.
        """
        pmid = "T1"
        title = "Effect of TestDrug on mortality"
        abstract = "Randomized placebo-controlled trial of 500 patients."
        metadata = {
            "journal": "NEJM",
            "authors": [
                {"last": "Smith", "first": "J.", "affiliations": ["X Hospital"]}
            ],
            "mesh_terms": ["Randomized Controlled Trial"],
        }

        fa = FakeAnnotator(canned_response=_valid_biasbuster_json())
        await fa.annotate_abstract(
            pmid=pmid, title=title, abstract=abstract, metadata=metadata,
        )
        _, user_message, _ = fa.calls[0]

        expected = _legacy_build_user_message(pmid, title, abstract, metadata)
        assert user_message == expected

    @pytest.mark.asyncio
    async def test_explicit_biasbuster_methodology_is_equivalent_to_none(
        self,
    ) -> None:
        """Passing methodology=biasbuster explicitly must not change output."""
        pmid, title, abstract = "T1", "T", "A"
        metadata = {"journal": "NEJM"}

        fa_default = FakeAnnotator(canned_response=_valid_biasbuster_json())
        fa_explicit = FakeAnnotator(canned_response=_valid_biasbuster_json())

        await fa_default.annotate_abstract(
            pmid=pmid, title=title, abstract=abstract, metadata=metadata,
        )
        await fa_explicit.annotate_abstract(
            pmid=pmid, title=title, abstract=abstract, metadata=metadata,
            methodology=get_methodology("biasbuster"),
        )
        assert fa_default.calls[0] == fa_explicit.calls[0]

    @pytest.mark.asyncio
    async def test_annotation_is_stamped_with_methodology_metadata(
        self,
    ) -> None:
        """Refactored path attaches _methodology / _methodology_version.

        The legacy code did not attach these — they are additive metadata
        so downstream comparison harnesses can tell which methodology
        produced the row without re-deriving it.
        """
        fa = FakeAnnotator(canned_response=_valid_biasbuster_json())
        result = await fa.annotate_abstract(
            pmid="T1", title="T", abstract="A", metadata=None,
        )
        assert result is not None
        assert result["_methodology"] == "biasbuster"
        assert result["_methodology_version"] == "v5a"


class TestFullTextGuard:
    """Methodologies that require full text refuse abstract-only annotators."""

    @pytest.mark.asyncio
    async def test_full_text_methodology_refuses_annotate_abstract(
        self,
    ) -> None:
        """A full-text-only methodology must raise on annotate_abstract()."""
        # Build a fake methodology with requires_full_text=True but all
        # the pass-through callables of biasbuster (so only the guard
        # differs).
        bb = get_methodology("biasbuster")
        full_text_only = Methodology(
            name="fake_full_text_only",
            display_name="Fake",
            version="v0",
            applicable_study_designs=frozenset({"rct_parallel"}),
            requires_full_text=True,
            orchestration="decomposed_full_text",
            severity_rollup_levels=("low", "high"),
            status="active",
            build_system_prompt=bb.build_system_prompt,
            build_user_message=bb.build_user_message,
            parse_output=bb.parse_output,
            aggregate=bb.aggregate,
            check_applicability=bb.check_applicability,
            evaluation_mapping_to_ground_truth=bb.evaluation_mapping_to_ground_truth,
        )
        fa = FakeAnnotator(canned_response=_valid_biasbuster_json())
        with pytest.raises(FullTextRequiredError) as exc:
            await fa.annotate_abstract(
                pmid="T1", title="T", abstract="A",
                metadata=None, methodology=full_text_only,
            )
        assert exc.value.pmid == "T1"
        assert exc.value.methodology == "fake_full_text_only"

    @pytest.mark.asyncio
    async def test_biasbuster_default_does_not_trigger_full_text_guard(
        self,
    ) -> None:
        """Biasbuster is abstract-capable; the guard must be a no-op."""
        fa = FakeAnnotator(canned_response=_valid_biasbuster_json())
        # No raise; annotation returns normally.
        result = await fa.annotate_abstract(
            pmid="T1", title="T", abstract="A", metadata=None,
        )
        assert result is not None


class TestFullTextGuardInAgenticAndDecomposed:
    """Defence-in-depth: agentic + decomposed guards refuse abstract-only callers.

    The CLI argparse validator + annotate_paper pre-flight already reject
    non-biasbuster methodologies on these paths, but a future direct
    caller that bypasses the CLI would not hit those checks. The
    ``_assert_full_text`` call inside each method is the backstop.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method_name", [
        "annotate_full_text_agentic",
        "annotate_full_text_decomposed",
    ])
    async def test_empty_sections_raises_when_full_text_required(
        self, method_name: str,
    ) -> None:
        bb = get_methodology("biasbuster")
        full_text_only = Methodology(
            name=f"fake_full_text_only_{method_name}",
            display_name="Fake",
            version="v0",
            applicable_study_designs=frozenset({"rct_parallel"}),
            requires_full_text=True,
            orchestration="decomposed_full_text",
            severity_rollup_levels=("low", "high"),
            status="active",
            build_system_prompt=bb.build_system_prompt,
            build_user_message=bb.build_user_message,
            parse_output=bb.parse_output,
            aggregate=bb.aggregate,
            check_applicability=bb.check_applicability,
            evaluation_mapping_to_ground_truth=bb.evaluation_mapping_to_ground_truth,
        )
        fa = FakeAnnotator(canned_response=_valid_biasbuster_json())
        method = getattr(fa, method_name)
        with pytest.raises(FullTextRequiredError) as exc:
            await method(
                pmid="T1", title="T", sections=[], metadata=None,
                methodology=full_text_only,
            )
        assert exc.value.pmid == "T1"


class TestBatchThreading:
    """annotate_batch passes the methodology through to each per-item call."""

    @pytest.mark.asyncio
    async def test_batch_threads_methodology_into_annotate_fn(self) -> None:
        fa = FakeAnnotator(canned_response=_valid_biasbuster_json())
        items = [
            {"pmid": "T1", "title": "A", "abstract": "Abs1",
             "metadata": None},
            {"pmid": "T2", "title": "B", "abstract": "Abs2",
             "metadata": None},
        ]
        results = await fa.annotate_batch(
            items=items, concurrency=1, delay=0,
            two_call=False,  # route to annotate_abstract (refactored path)
        )
        assert len(results) == 2
        # Both items should have been stamped with methodology metadata.
        for r in results:
            assert r["_methodology"] == "biasbuster"

    @pytest.mark.asyncio
    async def test_batch_explicit_methodology_is_threaded(self) -> None:
        bb = get_methodology("biasbuster")
        fa = FakeAnnotator(canned_response=_valid_biasbuster_json())
        items = [{"pmid": "T1", "title": "A", "abstract": "Abs1",
                  "metadata": None}]
        results = await fa.annotate_batch(
            items=items, concurrency=1, delay=0,
            two_call=False, methodology=bb,
        )
        assert len(results) == 1
        assert results[0]["_methodology"] == "biasbuster"

"""LLM-driven Cochrane RoB 2 decomposed assessor.

One LLM call per bias domain, then deterministic rollup. Reuses
:meth:`BaseAnnotator._extract_full_text_sections` for Stage 1 extraction
because the extraction task ("pull structured facts from the methods /
results / supplement / protocol") is methodology-agnostic.

Stage 2 is the RoB 2-specific part: for each of the five domains we
issue a focused LLM call with the domain's signalling questions and
parse the response into a :class:`RoB2DomainJudgement`. The per-outcome
and overall rollups use pure functions from :mod:`.algorithm` so the
LLM cannot disagree with the Cochrane worst-wins rule.

v1 MVP limitation: one synthetic "primary outcome" per paper. The
per-outcome/per-result granularity the Cochrane Handbook specifies is
out of scope for the scaffold step; the schema already carries a list
of outcomes so Step 8/9 can populate multiple without migration.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from biasbuster.annotators import parse_llm_json

from .algorithm import aggregate_outcome, worst_case_across_outcomes
from .prompts import build_system_prompt, domain_stage_name
from .schema import (
    EvidenceQuote,
    ROB2_DOMAIN_DISPLAY,
    ROB2_DOMAIN_SLUGS,
    RoB2Assessment,
    RoB2DomainJudgement,
    RoB2Judgement,
    RoB2OutcomeJudgement,
    SignallingAnswer,
    VALID_JUDGEMENTS,
    VALID_SIGNALLING_ANSWERS,
)

if TYPE_CHECKING:
    from biasbuster.annotators import BaseAnnotator

logger = logging.getLogger(__name__)

# Upper bound on LLM retries per domain call. Matches the annotator's
# default max_retries so behaviour is consistent with the biasbuster path.
_MAX_RETRIES_PER_DOMAIN: int = 3

# Default outcome label when the paper doesn't declare one explicitly.
# Step 8+ will extend this to enumerate multiple outcomes from the
# Stage 1 extraction.
_DEFAULT_OUTCOME_LABEL: str = "primary outcome"
_DEFAULT_RESULT_LABEL: str = "as reported"


def _build_domain_user_message(
    pmid: str,
    title: str,
    extraction: dict,
    domain_slug: str,
) -> str:
    """Assemble the user message for a single RoB 2 domain call.

    Structure:
    - Identify the paper.
    - Give the LLM the full structured extraction (Stage 1 output) as
      JSON so it can quote verbatim from the relevant sections.
    - Name the domain under examination so the LLM focuses.

    The system prompt supplies the signalling questions + algorithm;
    the user message is purely context. This split keeps the prompts
    constants reusable across papers.
    """
    display_name = ROB2_DOMAIN_DISPLAY[domain_slug]
    parts = [
        f"PMID: {pmid}",
        f"Title: {title}",
        "",
        f"Focus domain: {display_name}",
        "",
        "Structured extraction from the paper (Stage 1 output):",
        "```json",
        json.dumps(extraction, indent=2, default=str),
        "```",
        "",
        "Answer the signalling questions for this domain only, then emit "
        "the per-domain judgement in JSON as specified by your system prompt.",
    ]
    return "\n".join(parts)


def _coerce_signalling_answer(value: object) -> Optional[SignallingAnswer]:
    """Return a well-formed SignallingAnswer or None if the value is invalid."""
    if isinstance(value, str) and value in VALID_SIGNALLING_ANSWERS:
        return value  # type: ignore[return-value]
    return None


def _coerce_judgement(value: object) -> Optional[RoB2Judgement]:
    if isinstance(value, str) and value in VALID_JUDGEMENTS:
        return value  # type: ignore[return-value]
    return None


def _parse_evidence_quotes(raw: object) -> list[EvidenceQuote]:
    """Best-effort parse of the ``evidence_quotes`` list from an LLM blob.

    Tolerates missing ``section``, non-string ``text``, and missing fields.
    Quotes with empty or non-string ``text`` are dropped. This runs on
    untrusted LLM output so defensive parsing is warranted.
    """
    quotes: list[EvidenceQuote] = []
    if not isinstance(raw, list):
        return quotes
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        section = item.get("section")
        if not isinstance(section, str):
            section = None
        quotes.append(EvidenceQuote(text=text.strip(), section=section))
    return quotes


def _parse_domain_response(
    raw: str, domain_slug: str, pmid: str,
) -> Optional[RoB2DomainJudgement]:
    """Parse a domain-call LLM response into a :class:`RoB2DomainJudgement`.

    Returns ``None`` on unrecoverable parse or schema failure — the
    caller retries or gives up, matching the biasbuster annotator's
    retry pattern.
    """
    # Reuse the biasbuster JSON repair layer to be tolerant of stray
    # markdown fences, trailing commas, etc. The ``pmid`` threads into
    # log lines for traceability.
    blob = parse_llm_json(raw, pmid=pmid, require_annotation_fields=False) \
        if "require_annotation_fields" in parse_llm_json.__code__.co_varnames \
        else _loose_parse_json(raw)
    if blob is None:
        # parse_llm_json rejected — try the looser parser as a fallback.
        blob = _loose_parse_json(raw)
    if blob is None:
        # Surface the raw response for diagnosis. Without this an operator
        # only sees retry warnings and has no way to tell whether the LLM
        # emitted prose, malformed JSON, or hit a token limit.
        snippet = raw if len(raw) <= 500 else raw[:500] + "...[truncated]"
        logger.warning(
            "PMID %s: RoB 2 domain %s: JSON parse failed. "
            "Raw response (first 500 chars): %s",
            pmid, domain_slug, snippet,
        )
        return None

    returned_domain = blob.get("domain")
    if returned_domain not in (domain_slug, None):
        logger.warning(
            "PMID %s: RoB 2 domain %r: LLM returned domain=%r, coercing "
            "to expected slug", pmid, domain_slug, returned_domain,
        )

    # Signalling answers — accept any keys the LLM emits; our validator
    # will filter to the well-formed ones.
    raw_answers = blob.get("signalling_answers") or {}
    if not isinstance(raw_answers, dict):
        logger.warning(
            "PMID %s: RoB 2 domain %s: signalling_answers is not a dict, "
            "got %r", pmid, domain_slug, type(raw_answers).__name__,
        )
        return None
    answers: dict[str, SignallingAnswer] = {}
    for k, v in raw_answers.items():
        coerced = _coerce_signalling_answer(v)
        if coerced is not None:
            answers[str(k)] = coerced

    # Accept the American spelling "judgment" as an alias for the British
    # "judgement" used in the Cochrane handbook. Sonnet 4.6 occasionally
    # emits the former despite the prompt schema specifying the latter.
    judgement = _coerce_judgement(blob.get("judgement"))
    if judgement is None:
        judgement = _coerce_judgement(blob.get("judgment"))

    # Distinguish two failure modes:
    #   (a) judgement field MISSING entirely ⇒ model hedged. Per the
    #       Cochrane RoB 2 algorithm the documented default for
    #       ambiguity is "some_concerns" ("otherwise" / "everything
    #       else" in every per-domain prompt's algorithm block).
    #       Fall back to that rather than aborting the whole paper.
    #       Observed: PMID 36101416 / deviations_from_interventions
    #       persistently omitted the field across 3 retries.
    #   (b) judgement field PRESENT but invalid (e.g. "maybe") ⇒
    #       model is confused. Don't paper over that — return None
    #       and let the caller retry or skip.
    if judgement is None:
        judgement_field_present = (
            "judgement" in blob or "judgment" in blob
        )
        if judgement_field_present:
            logger.warning(
                "PMID %s: RoB 2 domain %s: judgement field present "
                "but invalid (judgement=%r, judgment=%r); skipping "
                "domain", pmid, domain_slug,
                blob.get("judgement"), blob.get("judgment"),
            )
            return None
        if isinstance(answers, dict) and answers:
            logger.warning(
                "PMID %s: RoB 2 domain %s: judgement field missing "
                "(blob keys=%s) but %d signalling answers are present; "
                "defaulting to 'some_concerns' per Cochrane's "
                "ambiguity rule.",
                pmid, domain_slug, sorted(blob.keys()), len(answers),
            )
            judgement = "some_concerns"  # type: ignore[assignment]
        else:
            logger.warning(
                "PMID %s: RoB 2 domain %s: no judgement AND no "
                "signalling answers (blob keys=%s); skipping domain",
                pmid, domain_slug, sorted(blob.keys()),
            )
            return None

    justification = blob.get("justification") or ""
    if not isinstance(justification, str):
        justification = str(justification)

    return RoB2DomainJudgement(
        domain=domain_slug,
        signalling_answers=answers,
        judgement=judgement,
        justification=justification.strip(),
        evidence_quotes=_parse_evidence_quotes(blob.get("evidence_quotes")),
    )


def _loose_parse_json(raw: str) -> Optional[dict]:
    """Fallback JSON parser tolerant to markdown fences and prose wrapping.

    Resolution order:
      1. Direct ``json.loads`` after whitespace strip.
      2. Strip a leading/trailing markdown ``````` fence and retry.
      3. Brace-balance scan: find the first ``{``, walk forward tracking
         brace depth (skipping over string contents), and try to parse
         the substring up to and including its matching ``}``. Catches
         the common case where the model leads with prose then emits a
         JSON object, despite our prompt asking for JSON-only output.
    """
    text = raw.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip a markdown fence if present and retry.
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        fenced = "\n".join(lines)
        try:
            return json.loads(fenced)
        except (json.JSONDecodeError, TypeError):
            pass

    # Brace-balance scan: extract the first complete top-level object.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, TypeError):
                    return None
    return None


class CochraneRoB2Assessor:
    """Decomposed RoB 2 assessor.

    Usage::

        assessor = CochraneRoB2Assessor(annotator)
        result = await assessor.assess(
            pmid=pmid, title=title, sections=sections, metadata=metadata,
        )
        if result is not None:
            db.insert_annotation(
                pmid, model_name, result,
                methodology="cochrane_rob2",
                methodology_version=METHODOLOGY_VERSION,
            )

    The assessor is transport-agnostic — any :class:`BaseAnnotator`
    subclass (LLMAnnotator, OpenAICompatAnnotator, BmlibAnnotator)
    provides ``_call_llm`` and ``_extract_full_text_sections`` uniformly.
    """

    def __init__(
        self,
        annotator: "BaseAnnotator",
        *,
        max_retries_per_domain: int = _MAX_RETRIES_PER_DOMAIN,
    ) -> None:
        self.annotator = annotator
        self.max_retries_per_domain = max_retries_per_domain

    async def _call_domain(
        self, pmid: str, title: str, extraction: dict, domain_slug: str,
    ) -> Optional[RoB2DomainJudgement]:
        """Issue one LLM call for a single RoB 2 domain; parse and return."""
        system_prompt = build_system_prompt(domain_stage_name(domain_slug))
        user_message = _build_domain_user_message(
            pmid, title, extraction, domain_slug,
        )
        for attempt in range(self.max_retries_per_domain):
            raw = await self.annotator._call_llm(
                system_prompt, user_message, pmid=pmid,
            )
            if raw is None:
                return None
            parsed = _parse_domain_response(raw, domain_slug, pmid)
            if parsed is not None:
                return parsed
            logger.warning(
                "PMID %s: RoB 2 domain %s parse failed "
                "(attempt %d/%d), retrying",
                pmid, domain_slug,
                attempt + 1, self.max_retries_per_domain,
            )
        logger.error(
            "PMID %s: RoB 2 domain %s: all %d parse attempts failed",
            pmid, domain_slug, self.max_retries_per_domain,
        )
        return None

    async def assess(
        self,
        pmid: str,
        title: str,
        sections: list[tuple[str, str]],
        metadata: Optional[dict] = None,
    ) -> Optional[RoB2Assessment]:
        """Run the full RoB 2 decomposed flow and return the assessment.

        Returns ``None`` on unrecoverable failure (extraction failed, or
        any domain call failed after retries). The caller is responsible
        for logging + skipping.
        """
        del metadata  # reserved for future use (e.g. trial registry links)

        # Stage 1: shared extraction (same call as biasbuster's v3/v4/v5a
        # full-text flows; no RoB 2-specific logic here).
        extraction_result = await self.annotator._extract_full_text_sections(
            pmid, title, sections,
        )
        if extraction_result is None:
            return None
        extraction, _section_extractions, _merge_conflicts, _failed = \
            extraction_result

        # Stage 2: per-domain calls. One sequential call per domain so a
        # failure on domain N doesn't waste calls on domains N+1..5.
        domains: dict[str, RoB2DomainJudgement] = {}
        for slug in ROB2_DOMAIN_SLUGS:
            judgement = await self._call_domain(pmid, title, extraction, slug)
            if judgement is None:
                logger.error(
                    "PMID %s: aborting RoB 2 assessment — domain %s failed",
                    pmid, slug,
                )
                return None
            domains[slug] = judgement

        # Deterministic rollup: the LLM does not get to override the
        # worst-wins rule for the outcome-level judgement.
        outcome_overall = aggregate_outcome(domains)
        outcome = RoB2OutcomeJudgement(
            outcome_label=_DEFAULT_OUTCOME_LABEL,
            result_label=_DEFAULT_RESULT_LABEL,
            domains=domains,
            overall_judgement=outcome_overall,
            overall_rationale=_synthesize_overall_rationale(domains),
        )
        paper_overall = worst_case_across_outcomes([outcome])

        from . import METHODOLOGY_VERSION
        return RoB2Assessment(
            pmid=pmid,
            outcomes=[outcome],
            methodology_version=METHODOLOGY_VERSION,
            worst_across_outcomes=paper_overall,
        )


def _synthesize_overall_rationale(
    domains: dict[str, RoB2DomainJudgement],
) -> str:
    """Generate a deterministic 1-line rationale naming the dominant domains.

    The Cochrane worst-wins rule means the overall judgement is driven by
    the highest-severity domain(s). We name them so downstream readers
    see why the outcome-level judgement landed where it did, without
    an additional LLM call (one fewer token-cost source for the MVP).
    """
    by_level: dict[str, list[str]] = {"low": [], "some_concerns": [], "high": []}
    for slug, d in domains.items():
        by_level[d.judgement].append(ROB2_DOMAIN_DISPLAY[slug])
    if by_level["high"]:
        return (
            "Outcome-level judgement is 'high' because the following "
            "domain(s) were rated high: " + "; ".join(by_level["high"]) + "."
        )
    if by_level["some_concerns"]:
        return (
            "Outcome-level judgement is 'some_concerns' because the "
            "following domain(s) were rated some_concerns: "
            + "; ".join(by_level["some_concerns"]) + "."
        )
    return "All five domains were rated low; outcome-level judgement is low."

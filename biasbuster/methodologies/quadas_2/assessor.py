"""LLM-driven QUADAS-2 decomposed assessor.

Mirror of :class:`biasbuster.methodologies.cochrane_rob2.assessor.CochraneRoB2Assessor`
adapted to QUADAS-2's four-domain, 2-D output shape:

- Stage 1: shared full-text section extraction.
- Stage 2: one LLM call per QUADAS-2 domain. Domains 1-3 return both a
  bias rating and an applicability rating; domain 4 returns bias only.
- Rollup: worst-wins per dimension (bias + applicability separately).

Any domain-level parse failure aborts the whole assessment; partial
QUADAS-2 is worse than no QUADAS-2 because it would be mistaken for a
complete expert-style report.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from .algorithm import worst_applicability, worst_bias
from .prompts import build_system_prompt, domain_stage_name
from .schema import (
    QUADAS2_APPLICABILITY_DOMAINS,
    QUADAS2_DOMAIN_DISPLAY,
    QUADAS2_DOMAIN_SLUGS,
    QUADAS2Assessment,
    QUADAS2DomainJudgement,
    QUADASEvidenceQuote,
    QUADASRating,
    QUADASSignallingAnswer,
    VALID_QUADAS_RATINGS,
    VALID_QUADAS_SIGNALLING,
)

if TYPE_CHECKING:
    from biasbuster.annotators import BaseAnnotator

logger = logging.getLogger(__name__)

_MAX_RETRIES_PER_DOMAIN: int = 3


def _build_domain_user_message(
    pmid: str,
    title: str,
    extraction: dict,
    domain_slug: str,
) -> str:
    """Compose the user message for a single QUADAS-2 domain call.

    Identical structure to the Cochrane RoB 2 equivalent (minus the
    domain-focus wording difference), so a future refactor could
    extract a common helper. Kept local for now — the two methodologies
    have slightly different sweet spots for what context to include.
    """
    display = QUADAS2_DOMAIN_DISPLAY[domain_slug]
    parts = [
        f"PMID: {pmid}",
        f"Title: {title}",
        "",
        f"Focus domain: {display}",
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


def _coerce_signalling_answer(
    value: object,
) -> Optional[QUADASSignallingAnswer]:
    """Normalise an LLM-emitted signalling answer to the canonical slug.

    Tolerates capitalisation variants (``Yes`` → ``yes``) because the
    Whiting et al. template uses title case but JSON typically emits
    lowercase. Unknown values return ``None`` so the caller can drop
    them instead of letting a malformed answer through.
    """
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in VALID_QUADAS_SIGNALLING:
            return lowered  # type: ignore[return-value]
    return None


def _coerce_rating(value: object) -> Optional[QUADASRating]:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in VALID_QUADAS_RATINGS:
            return lowered  # type: ignore[return-value]
    return None


def _parse_evidence_quotes(raw: object) -> list[QUADASEvidenceQuote]:
    """Defensive parse of evidence-quotes list from untrusted LLM output."""
    quotes: list[QUADASEvidenceQuote] = []
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
        quotes.append(QUADASEvidenceQuote(
            text=text.strip(), section=section,
        ))
    return quotes


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


def _parse_domain_response(
    raw: str, domain_slug: str, pmid: str,
) -> Optional[QUADAS2DomainJudgement]:
    """Parse a single QUADAS-2 domain LLM response into a typed judgement."""
    blob = _loose_parse_json(raw)
    if blob is None:
        # Surface the raw response for diagnosis. Without this an operator
        # only sees the "parse failed" retry warnings and has no way to
        # tell whether the LLM emitted prose, malformed JSON, or hit a
        # token limit. Truncate to keep logs readable.
        snippet = raw if len(raw) <= 500 else raw[:500] + "...[truncated]"
        logger.warning(
            "PMID %s: QUADAS-2 domain %s: JSON parse failed. "
            "Raw response (first 500 chars): %s",
            pmid, domain_slug, snippet,
        )
        return None

    # Signalling answers
    raw_answers = blob.get("signalling_answers") or {}
    if not isinstance(raw_answers, dict):
        logger.warning(
            "PMID %s: QUADAS-2 domain %s: signalling_answers is not a "
            "dict (got %r)",
            pmid, domain_slug, type(raw_answers).__name__,
        )
        return None
    answers: dict[str, QUADASSignallingAnswer] = {}
    for k, v in raw_answers.items():
        coerced = _coerce_signalling_answer(v)
        if coerced is not None:
            answers[str(k)] = coerced

    bias = _coerce_rating(blob.get("bias_rating"))
    if bias is None:
        logger.warning(
            "PMID %s: QUADAS-2 domain %s: missing/invalid bias_rating "
            "(got %r); skipping domain",
            pmid, domain_slug, blob.get("bias_rating"),
        )
        return None

    # Applicability is domain-specific: required for 1-3, forbidden for 4.
    if domain_slug in QUADAS2_APPLICABILITY_DOMAINS:
        applicability = _coerce_rating(blob.get("applicability"))
        if applicability is None:
            logger.warning(
                "PMID %s: QUADAS-2 domain %s requires applicability; "
                "got %r. Skipping domain.",
                pmid, domain_slug, blob.get("applicability"),
            )
            return None
    else:
        applicability = None
        if blob.get("applicability") is not None:
            logger.info(
                "PMID %s: QUADAS-2 domain %s: LLM emitted an applicability "
                "field where none is allowed; discarding.",
                pmid, domain_slug,
            )

    justification = blob.get("justification") or ""
    if not isinstance(justification, str):
        justification = str(justification)

    return QUADAS2DomainJudgement(
        domain=domain_slug,
        signalling_answers=answers,
        bias_rating=bias,
        applicability=applicability,
        justification=justification.strip(),
        evidence_quotes=_parse_evidence_quotes(blob.get("evidence_quotes")),
    )


class QUADAS2Assessor:
    """Decomposed QUADAS-2 assessor.

    Usage mirrors :class:`CochraneRoB2Assessor`:

        assessor = QUADAS2Assessor(annotator)
        result = await assessor.assess(
            pmid=pmid, title=title, sections=sections, metadata=metadata,
        )
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
    ) -> Optional[QUADAS2DomainJudgement]:
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
                "PMID %s: QUADAS-2 domain %s parse failed "
                "(attempt %d/%d), retrying",
                pmid, domain_slug,
                attempt + 1, self.max_retries_per_domain,
            )
        logger.error(
            "PMID %s: QUADAS-2 domain %s: all %d parse attempts failed",
            pmid, domain_slug, self.max_retries_per_domain,
        )
        return None

    async def assess(
        self,
        pmid: str,
        title: str,
        sections: list[tuple[str, str]],
        metadata: Optional[dict] = None,
    ) -> Optional[QUADAS2Assessment]:
        """Run the full QUADAS-2 decomposed flow on a paper."""
        del metadata

        extraction_result = await self.annotator._extract_full_text_sections(
            pmid, title, sections,
        )
        if extraction_result is None:
            return None
        extraction, _section_extractions, _merge_conflicts, _failed = \
            extraction_result

        domains: dict[str, QUADAS2DomainJudgement] = {}
        for slug in QUADAS2_DOMAIN_SLUGS:
            judgement = await self._call_domain(pmid, title, extraction, slug)
            if judgement is None:
                logger.error(
                    "PMID %s: aborting QUADAS-2 assessment — domain %s failed",
                    pmid, slug,
                )
                return None
            domains[slug] = judgement

        bias_overall = worst_bias(domains)
        applicability_overall = worst_applicability(domains)

        from . import METHODOLOGY_VERSION
        return QUADAS2Assessment(
            pmid=pmid,
            domains=domains,
            methodology_version=METHODOLOGY_VERSION,
            worst_bias=bias_overall,
            worst_applicability=applicability_overall,
        )

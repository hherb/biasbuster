"""v5A Decomposed Assessor — per-domain LLM override calls.

Replaces the v4 agentic loop (``AssessmentAgent``) with a sequence of
focused LLM calls. For each domain the mechanical assessor flagged at
moderate or higher AND marked as overridable, one narrow LLM call
asks: "does this rule genuinely apply to THIS paper?". The LLM's
output is a 3-field JSON: ``{decision, target_severity, reason}``.

The final assessment JSON is then assembled in Python from the
mechanical draft + the per-domain decisions. Hard-rule enforcement
still runs (via ``enforce_hard_rules``) as a belt-and-braces check.

Design rationale: see ``docs/three_step_approach/V5A_DECOMPOSED.md``.
The empirical motivation (small models ignoring the v4 REVIEW
scaffold) is in ``docs/three_step_approach/OVERVIEW.md``.

Supports two provider backends, identical to AssessmentAgent:
  - ``provider="anthropic"``: direct ``anthropic.AsyncAnthropic`` SDK.
  - ``provider="bmlib"``: bmlib's ``LLMClient.chat()`` (for any
    bmlib-supported provider, notably Ollama for local models).
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic

from biasbuster.annotators import repair_json, strip_markdown_fences
from biasbuster.assessment import assess_extraction
from biasbuster.assessment_agent_enforcement import enforce_hard_rules
from biasbuster.prompts_v5a import (
    DOMAIN_OVERRIDE_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    build_domain_override_user_message,
    build_synthesis_user_message,
)


def _parse_simple_json(text: str) -> Optional[dict]:
    """Parse a small JSON object from LLM output.

    Unlike ``parse_llm_json`` (which validates the full v3 30-field
    annotation schema and rejects anything missing fields), this is a
    lightweight parser for the V5A per-domain decision blob (3 fields)
    and the synthesis text. Tries direct parse, then a repair pass,
    then returns None.
    """
    text = strip_markdown_fences(text or "").strip()
    if not text:
        return None
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        result = json.loads(repair_json(text))
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DomainDecision:
    """One per-domain override decision from Stage 3."""
    domain: str
    mechanical_severity: str
    decision: str = "keep"                # "keep" or "downgrade"
    target_severity: str = ""             # final severity after decision
    reason: str = ""
    parse_error: Optional[str] = None     # non-None if the LLM output
                                          # failed to parse; we fall
                                          # back to "keep"


@dataclass
class DecomposedResult:
    """Full result of one decomposed-assessment run on one paper."""
    assessment: Optional[dict[str, Any]] = None
    mechanical_assessment: Optional[dict[str, Any]] = None
    domain_decisions: list[DomainDecision] = field(default_factory=list)
    enforcement_notes: list[str] = field(default_factory=list)
    n_llm_calls: int = 0                  # count of Stage 3 + Stage 5 calls


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
_ELEVATED = {"moderate", "high", "critical"}


def _rank(severity: Optional[str]) -> int:
    if not severity:
        return 0
    return _SEVERITY_RANK.get(str(severity).lower(), 0)


# ---------------------------------------------------------------------------
# DecomposedAssessor
# ---------------------------------------------------------------------------

class DecomposedAssessor:
    """v5A decomposed bias assessor.

    Use via ``async with DecomposedAssessor(...) as da: await da.assess(...)``.

    Pipeline (per paper):
      Stage 2 — run ``assess_extraction()`` (Python, no LLM)
      Stage 3 — for each elevated + overridable domain, one focused
                LLM call asking "keep or downgrade?"
      Stage 4 — (not implemented in V5A initial cut; verification
                tools can be added later if per-domain decisions
                request them)
      Stage 5 — apply decisions deterministically, run
                ``enforce_hard_rules``, generate reasoning (one small
                LLM call or Python-only fallback) and recommended
                verification steps
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens_per_call: int = 1024,
        temperature: float = 0.0,
        provider: str = "anthropic",
        synthesise_reasoning: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens_per_call = max_tokens_per_call
        self.temperature = temperature
        self.provider = provider
        self.synthesise_reasoning = synthesise_reasoning
        self._anthropic_client: Optional[anthropic.AsyncAnthropic] = None
        self._bmlib_client: Any = None

    async def __aenter__(self) -> "DecomposedAssessor":
        if self.provider == "anthropic":
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        elif self.provider == "bmlib":
            from bmlib.llm import LLMClient
            self._bmlib_client = LLMClient(
                anthropic_api_key=self.api_key or None,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}")
        return self

    async def __aexit__(self, *args) -> None:
        if self._anthropic_client is not None:
            await self._anthropic_client.close()
            self._anthropic_client = None
        self._bmlib_client = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def assess(
        self,
        pmid: str,
        title: str,
        extraction: dict[str, Any],
    ) -> DecomposedResult:
        """Run the full decomposed pipeline on a single paper."""
        if self.provider == "anthropic" and self._anthropic_client is None:
            raise RuntimeError("DecomposedAssessor must be used as an async context manager")
        if self.provider == "bmlib" and self._bmlib_client is None:
            raise RuntimeError("DecomposedAssessor must be used as an async context manager")

        result = DecomposedResult()

        # Stage 2: mechanical assessment (Python only)
        mechanical = assess_extraction(extraction)
        result.mechanical_assessment = copy.deepcopy(mechanical)

        provenance = mechanical.get("_provenance", {}) or {}
        domain_severities: dict[str, str] = provenance.get("domain_severities", {}) or {}
        domain_overridable: dict[str, bool] = provenance.get("domain_overridable", {}) or {}
        domain_rationales: dict[str, str] = provenance.get("domain_rationales", {}) or {}

        # Identify which domains need LLM review
        domains_to_review = [
            d for d, sev in domain_severities.items()
            if sev in _ELEVATED and domain_overridable.get(d, True)
        ]
        logger.info(
            f"[v5a {pmid}] Stage 2 mechanical draft: "
            f"{domain_severities}, "
            f"overridable={domain_overridable}, "
            f"reviewing {len(domains_to_review)} domain(s): {domains_to_review}"
        )

        # Stage 3: per-domain override decisions (parallel)
        extraction_json = json.dumps(extraction, indent=2, default=str)
        if domains_to_review:
            coros = [
                self._decide_one_domain(
                    pmid=pmid,
                    title=title,
                    domain=d,
                    mechanical_severity=domain_severities[d],
                    rationale=domain_rationales.get(d, ""),
                    other_severities=domain_severities,
                    extraction_json=extraction_json,
                )
                for d in domains_to_review
            ]
            decisions = await asyncio.gather(*coros, return_exceptions=True)
            for d, out in zip(domains_to_review, decisions):
                if isinstance(out, Exception):
                    logger.error(f"[v5a {pmid}] {d} review raised {out!r}; keeping mechanical")
                    result.domain_decisions.append(DomainDecision(
                        domain=d,
                        mechanical_severity=domain_severities[d],
                        target_severity=domain_severities[d],
                        reason=f"LLM call failed ({type(out).__name__}); keeping mechanical",
                        parse_error=repr(out),
                    ))
                else:
                    result.domain_decisions.append(out)
                    result.n_llm_calls += 1

        # Stage 5: deterministic synthesis
        final_assessment = self._apply_decisions(
            mechanical=copy.deepcopy(mechanical),
            decisions=result.domain_decisions,
        )

        # Hard-rule enforcement (belt-and-braces, mostly a no-op since
        # we already filtered non-overridable domains before Stage 3)
        final_assessment, notes = enforce_hard_rules(
            final_assessment, result.mechanical_assessment,
        )
        result.enforcement_notes.extend(notes)

        # Optional Stage-5 synthesis LLM call for the `reasoning` field
        if self.synthesise_reasoning:
            try:
                summary = await self._synthesise_reasoning(
                    pmid=pmid,
                    title=title,
                    final_assessment=final_assessment,
                )
                if summary:
                    final_assessment["reasoning"] = summary
                    result.n_llm_calls += 1
            except Exception as exc:
                logger.warning(
                    f"[v5a {pmid}] synthesis call failed: {exc!r}; keeping "
                    f"Python-generated reasoning"
                )

        # Recommended verification steps: derived deterministically
        # from which rules fired. For V5A we don't call verification
        # tools; the steps recommend them for downstream human review.
        final_assessment["recommended_verification_steps"] = (
            self._build_verification_steps(domain_severities, domain_rationales)
        )

        result.assessment = final_assessment
        return result

    # ------------------------------------------------------------------
    # Stage 3: per-domain LLM review
    # ------------------------------------------------------------------

    async def _decide_one_domain(
        self,
        pmid: str,
        title: str,
        domain: str,
        mechanical_severity: str,
        rationale: str,
        other_severities: dict[str, str],
        extraction_json: str,
    ) -> DomainDecision:
        """One focused LLM call asking whether to override this domain."""
        user_msg = build_domain_override_user_message(
            domain=domain,
            mechanical_severity=mechanical_severity,
            rationale=rationale,
            focused_extraction=extraction_json,
            pmid=pmid,
            title=title,
            other_severities=other_severities,
        )

        raw_text = await self._llm_call(
            system=DOMAIN_OVERRIDE_SYSTEM_PROMPT,
            user=user_msg,
        )

        parsed = _parse_simple_json(raw_text)
        if not isinstance(parsed, dict):
            logger.warning(
                f"[v5a {pmid}] {domain} review: JSON parse failed; "
                f"keeping mechanical severity. raw={raw_text[:200]!r}"
            )
            return DomainDecision(
                domain=domain,
                mechanical_severity=mechanical_severity,
                target_severity=mechanical_severity,
                reason=f"LLM output did not parse as JSON; keeping mechanical",
                parse_error="json_parse_failed",
            )

        decision = str(parsed.get("decision", "keep")).strip().lower()
        target = str(parsed.get("target_severity", mechanical_severity)).strip().lower()
        reason = str(parsed.get("reason", "")).strip()

        # Validate: only "keep" or "downgrade"; target_severity must
        # be <= mechanical_severity; reject upgrades
        if decision not in ("keep", "downgrade"):
            logger.warning(
                f"[v5a {pmid}] {domain} review: unknown decision "
                f"{decision!r}; forcing keep"
            )
            decision = "keep"
            target = mechanical_severity

        if target not in _SEVERITY_RANK:
            logger.warning(
                f"[v5a {pmid}] {domain} review: invalid target_severity "
                f"{target!r}; forcing keep"
            )
            target = mechanical_severity
            decision = "keep"

        if _rank(target) > _rank(mechanical_severity):
            logger.warning(
                f"[v5a {pmid}] {domain} review: LLM tried to UPGRADE "
                f"{mechanical_severity} → {target}; forcing keep"
            )
            target = mechanical_severity
            decision = "keep"

        # If decision is "keep" but target != mechanical, trust decision
        if decision == "keep":
            target = mechanical_severity

        logger.info(
            f"[v5a {pmid}] {domain}: {mechanical_severity} → {target} "
            f"({decision}) — {reason[:80]}"
        )

        return DomainDecision(
            domain=domain,
            mechanical_severity=mechanical_severity,
            decision=decision,
            target_severity=target,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Stage 5: assemble final assessment from mechanical draft + decisions
    # ------------------------------------------------------------------

    def _apply_decisions(
        self,
        mechanical: dict[str, Any],
        decisions: list[DomainDecision],
    ) -> dict[str, Any]:
        """Apply per-domain decisions to the mechanical draft.

        Returns a new assessment dict with:
          - severities updated from decisions
          - overall_severity recomputed (max rule)
          - overall_bias_probability recomputed from new severities
          - _overrides array populated for every applied downgrade
          - _annotation_mode set to "decomposed_v5a"
        """
        from biasbuster.assessment.aggregate import (
            PROBABILITY_ANCHORS,
            SEVERITY_RANK as DSR,
            DomainSeverity,
            OverallSeverity,
            _compute_overall_probability,
        )

        assessment = mechanical  # mutate in place; caller passes a copy
        overrides_list: list[dict] = []
        new_domain_severities: dict[str, DomainSeverity] = {}

        # Start from mechanical severities for all 5 domains
        for dom, block in list(assessment.items()):
            if not isinstance(block, dict):
                continue
            if "severity" in block and dom in (
                "statistical_reporting", "spin", "outcome_reporting",
                "conflict_of_interest", "methodology",
            ):
                sev_str = str(block.get("severity", "none")).lower()
                try:
                    new_domain_severities[dom] = DomainSeverity(sev_str)
                except ValueError:
                    new_domain_severities[dom] = DomainSeverity.NONE

        # Apply decisions
        for dec in decisions:
            block = assessment.get(dec.domain)
            if not isinstance(block, dict):
                continue
            if dec.target_severity != dec.mechanical_severity:
                block["severity"] = dec.target_severity
                try:
                    new_domain_severities[dec.domain] = DomainSeverity(dec.target_severity)
                except ValueError:
                    pass
                overrides_list.append({
                    "domain": dec.domain,
                    "mechanical_severity": dec.mechanical_severity,
                    "final_severity": dec.target_severity,
                    "reason": dec.reason,
                    "source": "v5a_per_domain_llm",
                })

        # Recompute overall severity (max rule) and probability
        if new_domain_severities:
            worst = max(new_domain_severities.values(), key=lambda s: DSR[s])
            overall = OverallSeverity(worst.value)
            assessment["overall_severity"] = overall.value
            assessment["overall_bias_probability"] = _compute_overall_probability(
                overall, new_domain_severities,
            )

        assessment["_overrides"] = overrides_list
        assessment["_annotation_mode"] = "decomposed_v5a"
        return assessment

    # ------------------------------------------------------------------
    # Stage 5 optional synthesis LLM call (for the `reasoning` field)
    # ------------------------------------------------------------------

    async def _synthesise_reasoning(
        self,
        pmid: str,
        title: str,
        final_assessment: dict[str, Any],
    ) -> str:
        """One small LLM call to produce a 2-3 sentence reasoning summary."""
        domain_severities = {
            d: str(final_assessment.get(d, {}).get("severity", "none"))
            for d in ("statistical_reporting", "spin", "outcome_reporting",
                     "conflict_of_interest", "methodology")
        }
        user_msg = build_synthesis_user_message(
            pmid=pmid,
            title=title,
            overall_severity=str(final_assessment.get("overall_severity", "none")),
            overall_probability=float(final_assessment.get("overall_bias_probability", 0.0)),
            domain_severities=domain_severities,
            overrides=final_assessment.get("_overrides", []) or [],
        )
        return (await self._llm_call(
            system=SYNTHESIS_SYSTEM_PROMPT,
            user=user_msg,
        )).strip()

    # ------------------------------------------------------------------
    # Deterministic recommended verification steps
    # ------------------------------------------------------------------

    @staticmethod
    def _build_verification_steps(
        domain_severities: dict[str, str],
        domain_rationales: dict[str, str],
    ) -> list[str]:
        """Generate a list of recommended verification actions.

        Driven deterministically by which domains are elevated.
        Avoids an LLM call for this boilerplate.
        """
        steps: list[str] = []

        coi_sev = domain_severities.get("conflict_of_interest", "none")
        if coi_sev in _ELEVATED:
            steps.append(
                "Check CMS Open Payments (openpaymentsdata.cms.gov) for "
                "physician-industry financial ties of the named authors."
            )
            steps.append(
                "Check ORCID records for undisclosed industry affiliations "
                "or recent employment history of the corresponding and "
                "senior authors."
            )
            steps.append(
                "Check Europe PMC funder metadata for funding sources not "
                "mentioned in the paper's disclosure statement."
            )

        meth_sev = domain_severities.get("methodology", "none")
        if meth_sev in _ELEVATED:
            steps.append(
                "Check ClinicalTrials.gov for the registered protocol to "
                "compare registered primary outcome(s) and analysis plan "
                "against what the paper reports."
            )

        outcome_sev = domain_severities.get("outcome_reporting", "none")
        if outcome_sev in _ELEVATED:
            steps.append(
                "Compare paper-reported primary outcome against the "
                "registered primary outcome in ClinicalTrials.gov or the "
                "equivalent trial registry."
            )

        stat_sev = domain_severities.get("statistical_reporting", "none")
        if stat_sev in _ELEVATED:
            steps.append(
                "Re-run an effect-size audit to verify reported relative "
                "effect sizes are not artificially inflated by absent "
                "baseline-risk reporting."
            )

        return steps

    # ------------------------------------------------------------------
    # Provider-agnostic single-shot LLM call (no tools)
    # ------------------------------------------------------------------

    async def _llm_call(self, system: str, user: str) -> str:
        """Issue one plain-text LLM completion. No tools, JSON parsed post-hoc."""
        if self.provider == "anthropic":
            return await self._anthropic_call(system, user)
        return await self._bmlib_call(system, user)

    async def _anthropic_call(self, system: str, user: str) -> str:
        assert self._anthropic_client is not None
        response = await self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens_per_call,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", "") or "")
        return "".join(text_parts)

    async def _bmlib_call(self, system: str, user: str) -> str:
        from bmlib.llm import LLMMessage

        assert self._bmlib_client is not None

        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=user),
        ]

        def _sync_call():
            return self._bmlib_client.chat(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens_per_call,
            )

        response = await asyncio.to_thread(_sync_call)
        return response.content or ""

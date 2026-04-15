"""BiasBuster prompt definitions — v5A Decomposed Pipeline.

V5A replaces the v4 agentic loop with a sequence of focused LLM calls,
one per elevated domain. Each call has a tiny, well-defined task: "for
THIS domain, with THIS rule firing, on THIS paper — should the
mechanical severity be downgraded? Keep it, or downgrade, with a
paper-specific reason."

Design rationale: see ``docs/three_step_approach/V5A_DECOMPOSED.md``.
The empirical motivation is the failed v4 agentic runs on gemma4-26B
and gpt-oss-20B (zero REVIEW blocks produced out of ten runs); see
``docs/three_step_approach/OVERVIEW.md`` §"The problem v5 addresses".

Contrast with v4:
- v4 asked ONE LLM call to review all 5 domains AND emit a 30-field JSON.
- v5A asks ONE focused LLM call PER elevated domain. Output is a
  3-field JSON. The 30-field final JSON is assembled in Python from the
  mechanical draft + the per-domain decisions.

Only overridable domains get a per-domain call. Non-overridable domains
(COI HIGH from structural triggers a/b/c/d) skip the LLM entirely —
the mechanical severity is final. See DESIGN_RATIONALE_COI.md.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Per-domain override prompts
# ---------------------------------------------------------------------------
#
# Each domain has its own template with its own "legitimate" /
# "illegitimate" override criteria. The criteria are the same policy
# as the v4 prompt (prompts_v4.py lines 91-135) but split per-domain so
# each LLM call only sees what's relevant to the single domain under
# review. The goal is to minimise the surface area the small model has
# to reason over.

DOMAIN_OVERRIDE_SYSTEM_PROMPT = """\
You are reviewing ONE bias-risk domain on ONE clinical trial. A
Python-coded mechanical rule has fired for this domain; your job is to
decide whether that rule genuinely applies to THIS specific paper, or
whether this paper is a legitimate exception.

Framing: this is a *risk* assessment, not a forensic finding. A HIGH
rating means "a reader should independently verify before accepting",
not "this paper is wrong". Do not downgrade severities just because
the paper looks well-conducted overall — the mechanical rule
represents a real structural concern unless contextually inapplicable.

You will receive:
  - The domain under review
  - The mechanical severity the rule produced
  - The specific rule that fired and the extracted values that
    triggered it
  - A short paper-level summary for context
  - A list of LEGITIMATE override reasons for this domain
  - A list of ILLEGITIMATE override reasons for this domain

Return EXACTLY one JSON object, nothing else:

{
  "decision": "keep" | "downgrade",
  "target_severity": "none" | "low" | "moderate" | "high" | "critical",
  "reason": "<= 80 words, must cite a specific fact from THIS paper"
}

Rules:
  - If the rule genuinely applies → decision="keep", target_severity =
    the mechanical severity, reason = one-sentence confirmation.
  - If this paper is a legitimate exception → decision="downgrade",
    target_severity = your judged severity (must be lower than
    mechanical), reason = paper-specific structural fact that makes
    the mechanical rule inapplicable.
  - NEVER upgrade. target_severity <= mechanical severity always.
  - NEVER downgrade merely because the paper "explains" the issue or
    "looks reasonable overall". Downgrade requires a specific
    structural reason.
  - If you're uncertain, keep the mechanical severity. When in doubt,
    trust the rule.
"""


# Per-domain override criteria — the policy lists split out so each
# call only loads the criteria relevant to its single domain.
DOMAIN_OVERRIDE_CRITERIA: dict[str, dict[str, list[str]]] = {
    "statistical_reporting": {
        "legitimate": [
            "The paper reports both absolute AND relative effect sizes "
            "but the extractor missed the absolute values. Check the "
            "extraction for effect_size_quotes before downgrading.",
            "The 'inflated effect size' is a correctly-reported "
            "per-protocol result in a paper that clearly labels it as "
            "per-protocol and also presents the ITT estimate.",
        ],
        "illegitimate": [
            "The paper 'explains' why it used relative-only reporting. "
            "Relative-only is still a reporting-bias risk regardless.",
            "Effect sizes look clinically plausible — that's not the "
            "same as the reporting being unbiased.",
        ],
    },
    "spin": {
        "legitimate": [
            "title_spin fired on a descriptive title that contains a "
            "listed clinical verb by coincidence (e.g. 'Evaluation of X's "
            "effect on Y' where 'effect' is the noun, not a therapeutic "
            "claim).",
            "'inappropriate_extrapolation' fired but the paper's "
            "conclusion is carefully hedged with explicit uncertainty "
            "language and scope limitations matching the trial design.",
        ],
        "illegitimate": [
            "The abstract conclusion promotes a positive finding from a "
            "secondary or subgroup analysis while the primary was null. "
            "Keep spin = moderate or high — this IS textbook spin.",
            "The paper 'stops short' of claiming causation but still "
            "uses directional language ('associated with' where "
            "randomisation would allow stronger claims) — keep.",
        ],
    },
    "outcome_reporting": {
        "legitimate": [
            "A 'surrogate without validation' flag fired on a surrogate "
            "that IS well-validated (e.g. HbA1c for diabetes trials, "
            "LDL for statin trials) and the paper cites the validation "
            "literature explicitly.",
            "A 'composite not disaggregated' flag fired but the paper "
            "does in fact report each component separately — the "
            "extractor missed it. Check composite_components_disaggregated.",
        ],
        "illegitimate": [
            "The surrogate 'makes sense clinically' — that is not the "
            "same as validated. Keep the flag.",
            "The paper is an early-phase trial and surrogates are "
            "'appropriate for phase'. Keep — phase does not change the "
            "reporting-bias risk.",
        ],
    },
    "conflict_of_interest": {
        "legitimate": [
            "The mechanical rule mis-classified public funding as "
            "industry due to an extraction error — verify via the "
            "extraction's funding_type before downgrading.",
        ],
        "illegitimate": [
            "The trial is well-conducted and the reported results look "
            "plausible. That is not the policy — COI is about "
            "structural risk, not proof of bias. Keep.",
            "Authors disclosed the COI transparently. Disclosure does "
            "not reduce the structural risk. Keep.",
        ],
        "note": [
            "If you reach this prompt for COI, the domain IS "
            "overridable — the non-overridable structural triggers "
            "(a/b/c/d per DESIGN_RATIONALE_COI.md) are filtered out "
            "before this call is made. Still, override discipline is "
            "strict: a paper-specific structural reason is required.",
        ],
    },
    "methodology": {
        "legitimate": [
            "The multiplicity-correction rule fires on an explicitly "
            "EXPLORATORY secondary analysis of a previously-published "
            "RCT. The paper itself acknowledges its exploratory nature. "
            "The multiplicity standard is customarily lenient for "
            "exploratory analyses.",
            "The differential_attrition rule fires but the paper has "
            "robust ITT analysis with multiple-imputation sensitivity.",
            "The per_protocol_only rule fires but the paper in fact "
            "reports an ITT analysis as its primary — the extractor "
            "mis-classified. Check extraction.analysis fields.",
        ],
        "illegitimate": [
            "The multiplicity rule fires AND the paper's abstract or "
            "conclusion prominently features a positive finding from "
            "the uncorrected analysis. This is textbook selective "
            "reporting. Keep.",
            "High attrition but the paper 'explains' it as expected. "
            "The explanation does not undo the bias. Keep.",
            "The sample-size rule fires but the paper 'acknowledges' "
            "the limitation. Acknowledgement does not undo the "
            "underpowering. Keep.",
        ],
    },
}


DOMAIN_OVERRIDE_USER_TEMPLATE = """\
Domain under review: {domain}
Mechanical severity: {severity}
Rule that fired: {rationale}

The specific extracted facts that triggered this rule:
```json
{focused_extraction}
```

Paper-level context:
  PMID: {pmid}
  Title: {title}
  Overall mechanical assessment (for context only — do NOT try to
  second-guess other domains, you are only reviewing {domain}):
    statistical_reporting: {other_sev_statistical_reporting}
    spin:                  {other_sev_spin}
    outcome_reporting:     {other_sev_outcome_reporting}
    conflict_of_interest:  {other_sev_conflict_of_interest}
    methodology:           {other_sev_methodology}

LEGITIMATE reasons to downgrade {domain} (examples):
{legitimate_list}

ILLEGITIMATE reasons (do NOT apply these):
{illegitimate_list}

Now emit the JSON decision.
"""


def build_domain_override_user_message(
    domain: str,
    mechanical_severity: str,
    rationale: str,
    focused_extraction: str,
    pmid: str,
    title: str,
    other_severities: dict[str, str],
) -> str:
    """Render the per-domain override prompt for one domain.

    Args:
        domain: The domain under review (e.g. ``"methodology"``).
        mechanical_severity: The severity the mechanical rule produced.
        rationale: ``_provenance.domain_rationales[domain]``.
        focused_extraction: Pre-serialised JSON of the extraction
            subset this domain inspects. Typically the full extraction
            for now (models tolerate it and selecting the "minimal"
            per-domain subset risks omitting relevant context). May be
            tightened later if latency demands.
        pmid: Paper identifier.
        title: Paper title.
        other_severities: dict mapping each of the 5 domain names to
            its mechanical severity string. Used to show the LLM the
            overall picture without inviting cross-domain
            second-guessing.

    Returns:
        Formatted user-message string.
    """
    criteria = DOMAIN_OVERRIDE_CRITERIA.get(domain, {})
    legit = criteria.get("legitimate", [])
    illegit = criteria.get("illegitimate", [])
    legit_str = "\n".join(f"  - {item}" for item in legit) if legit else "  (none listed)"
    illegit_str = "\n".join(f"  - {item}" for item in illegit) if illegit else "  (none listed)"

    return DOMAIN_OVERRIDE_USER_TEMPLATE.format(
        domain=domain,
        severity=mechanical_severity,
        rationale=rationale,
        focused_extraction=focused_extraction,
        pmid=pmid,
        title=title,
        other_sev_statistical_reporting=other_severities.get("statistical_reporting", "?"),
        other_sev_spin=other_severities.get("spin", "?"),
        other_sev_outcome_reporting=other_severities.get("outcome_reporting", "?"),
        other_sev_conflict_of_interest=other_severities.get("conflict_of_interest", "?"),
        other_sev_methodology=other_severities.get("methodology", "?"),
        legitimate_list=legit_str,
        illegitimate_list=illegit_str,
    )


# ---------------------------------------------------------------------------
# Optional Stage-5 synthesis prompt — generates a 2-3 sentence summary
# ---------------------------------------------------------------------------
#
# After Stage 3 per-domain decisions and Stage 5 deterministic
# assembly, the final assessment needs a `reasoning` field. Python can
# generate a boilerplate reasoning from the rationales (same as
# aggregate.py `_build_reasoning`), but an optional LLM pass can
# produce a more readable 2-3 sentence summary. This prompt is kept
# short and constrained so small models handle it reliably.

SYNTHESIS_SYSTEM_PROMPT = """\
You are summarising a completed bias-risk assessment. Write a concise
2-3 sentence summary (<=120 words) that explains the overall rating
in plain language. Cite the specific domains driving the rating.
Do not add new judgments — only summarise what's already in the data.
Return plain text, no JSON, no markdown.
"""


SYNTHESIS_USER_TEMPLATE = """\
PMID: {pmid}
Title: {title}
Overall severity: {overall_severity}
Overall bias probability: {overall_probability}

Per-domain severities (after any overrides):
  statistical_reporting: {sev_statistical_reporting}
  spin:                  {sev_spin}
  outcome_reporting:     {sev_outcome_reporting}
  conflict_of_interest:  {sev_conflict_of_interest}
  methodology:           {sev_methodology}

Overrides applied (if any):
{overrides_summary}

Write the 2-3 sentence summary now.
"""


def build_synthesis_user_message(
    pmid: str,
    title: str,
    overall_severity: str,
    overall_probability: float,
    domain_severities: dict[str, str],
    overrides: list[dict],
) -> str:
    """Render the Stage-5 synthesis prompt."""
    if overrides:
        lines = []
        for ov in overrides:
            lines.append(
                f"  - {ov.get('domain', '?')}: "
                f"{ov.get('mechanical_severity', '?')} → "
                f"{ov.get('final_severity', '?')} "
                f"({ov.get('reason', 'no reason given')[:80]})"
            )
        overrides_summary = "\n".join(lines)
    else:
        overrides_summary = "  (none — mechanical draft accepted unchanged)"

    return SYNTHESIS_USER_TEMPLATE.format(
        pmid=pmid,
        title=title,
        overall_severity=overall_severity,
        overall_probability=overall_probability,
        sev_statistical_reporting=domain_severities.get("statistical_reporting", "?"),
        sev_spin=domain_severities.get("spin", "?"),
        sev_outcome_reporting=domain_severities.get("outcome_reporting", "?"),
        sev_conflict_of_interest=domain_severities.get("conflict_of_interest", "?"),
        sev_methodology=domain_severities.get("methodology", "?"),
        overrides_summary=overrides_summary,
    )

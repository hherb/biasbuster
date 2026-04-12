"""
Bias Assessment Scorer

Parses free-text model outputs into structured assessments, then
compares against human-validated ground truth labels.

Handles two output formats:
1. Structured (JSON) - when model follows the annotation schema
2. Free-text - when model gives prose assessment (requires heuristic parsing)

Each bias dimension is scored independently to reveal per-domain strengths.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "DimensionScore",
    "ParsedAssessment",
    "parse_model_output",
    "attach_ground_truth",
    "SEVERITY_ORDER",
    "SEVERITY_LABELS",
    "severity_to_int",
]


# ---- Severity mapping ----

SEVERITY_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
SEVERITY_LABELS = list(SEVERITY_ORDER.keys())


def severity_to_int(s: str) -> int:
    return SEVERITY_ORDER.get(s.lower().strip(), -1)


# ---- Structured output from a single bias dimension ----

@dataclass
class DimensionScore:
    """Parsed score for a single bias dimension."""
    dimension: str = ""
    predicted_severity: str = "none"
    ground_truth_severity: str = "none"
    predicted_binary: bool = False    # any concern vs none
    ground_truth_binary: bool = False
    predicted_flags: dict = field(default_factory=dict)   # dimension-specific booleans
    ground_truth_flags: dict = field(default_factory=dict)
    confidence: str = "unknown"
    raw_text: str = ""  # the relevant portion of model output


@dataclass
class ParsedAssessment:
    """Complete parsed assessment from model output."""
    pmid: str = ""
    model_id: str = ""
    raw_output: str = ""
    thinking_text: str = ""        # content of <think> block if present
    parse_success: bool = False

    # Per-dimension scores
    statistical_reporting: DimensionScore = field(
        default_factory=lambda: DimensionScore(dimension="statistical_reporting")
    )
    spin: DimensionScore = field(
        default_factory=lambda: DimensionScore(dimension="spin")
    )
    outcome_reporting: DimensionScore = field(
        default_factory=lambda: DimensionScore(dimension="outcome_reporting")
    )
    conflict_of_interest: DimensionScore = field(
        default_factory=lambda: DimensionScore(dimension="conflict_of_interest")
    )
    methodology: DimensionScore = field(
        default_factory=lambda: DimensionScore(dimension="methodology")
    )

    # Overall
    overall_severity: str = "none"
    overall_bias_probability: float = 0.0

    # Verification quality
    verification_steps_mentioned: list[str] = field(default_factory=list)
    mentions_open_payments: bool = False
    mentions_clinicaltrials_gov: bool = False
    mentions_orcid: bool = False
    mentions_retraction_watch: bool = False
    mentions_europmc: bool = False
    verification_score: float = 0.0  # 0-1, how many sources mentioned


def parse_model_output(raw_output: str, pmid: str = "", model_id: str = "") -> ParsedAssessment:
    """
    Parse a model's free-text or JSON output into a structured ParsedAssessment.
    Tries JSON first, falls back to heuristic regex parsing.
    """
    result = ParsedAssessment(pmid=pmid, model_id=model_id, raw_output=raw_output)

    # Extract reasoning text — supports multiple formats:
    # 1. v3 training format: <think>...</think> XML tags
    # 2. v4 agentic format: "reasoning" field in the JSON annotation
    think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
    if think_match:
        result.thinking_text = think_match.group(1).strip()
    else:
        # Try to extract "reasoning" from the JSON (v4 agentic output)
        try:
            data = json.loads(raw_output) if raw_output.strip().startswith("{") else {}
            reasoning = data.get("reasoning", "")
            if reasoning:
                result.thinking_text = str(reasoning).strip()
        except (json.JSONDecodeError, TypeError):
            pass

    # Try JSON parse first (strip think block and markdown fences)
    json_text = raw_output
    if think_match:
        json_text = raw_output[think_match.end():]
    json_text = json_text.strip()
    if json_text.startswith("```"):
        json_text = re.sub(r'^```(?:json)?\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text)

    try:
        data = json.loads(json_text)
        result = _parse_from_json(data, result)
        result.parse_success = True
    except (json.JSONDecodeError, ValueError):
        # JSON may have trailing text after the closing brace — try to
        # extract just the JSON object (common when model appends prose).
        json_extracted = False
        if json_text.lstrip().startswith("{"):
            brace_count = 0
            for i, ch in enumerate(json_text):
                if ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            data = json.loads(json_text[: i + 1])
                            result = _parse_from_json(data, result)
                            result.parse_success = True
                            json_extracted = True
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
        if not json_extracted:
            # Fall back to heuristic parsing on post-think text only,
            # so reasoning in <think> blocks doesn't pollute matches.
            parse_text = raw_output
            if think_match:
                parse_text = raw_output[think_match.end():]
            result = _parse_from_text(parse_text, result)
            result.parse_success = True  # May be partial

    # Score verification source mentions
    result = _score_verification_mentions(result)

    return result


_scorer_logger = logging.getLogger(__name__)

# Fallback mapping: when the model produces a severity but no numeric
# probability, infer a reasonable probability from the severity level.
# See docs/ROUND_4.md for context.
_SEVERITY_TO_PROBABILITY = {
    "none": 0.05,
    "low": 0.25,
    "moderate": 0.55,
    "high": 0.80,
    "critical": 0.95,
}


def _safe_dict(val, default_severity="none"):
    """Ensure a dimension value is a dict; coerce strings/None to a stub."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        return {"severity": val}
    if val is not None:
        _scorer_logger.warning("Unexpected dimension type %s: %r", type(val).__name__, val)
    return {"severity": default_severity}


def _normalize_json(data: dict) -> dict:
    """Unwrap nested model output and normalize dimension key names.

    Models wrap their assessments in various top-level keys
    (bias_assessment, assessment, dimensions) and may prefix dimension
    names with numbers ("1. Statistical Reporting").  This function
    flattens to the canonical key names the scorer expects.
    """
    # Unwrap one level of nesting
    for wrapper_key in ("bias_assessment", "assessment", "dimensions"):
        if wrapper_key in data and isinstance(data[wrapper_key], dict):
            inner = data[wrapper_key]
            # Merge inner keys into top level, preserving top-level overrides
            merged = {**inner, **{k: v for k, v in data.items() if k != wrapper_key}}
            data = merged
            break

    # Normalize numbered/variant key names → canonical keys
    key_map = {
        "statistical_reporting": "statistical_reporting",
        "statistical reporting": "statistical_reporting",
        "spin": "spin",
        "outcome_reporting": "outcome_reporting",
        "outcome reporting": "outcome_reporting",
        "conflict_of_interest": "conflict_of_interest",
        "conflict of interest": "conflict_of_interest",
        "methodology": "methodology",
        "methodological_red_flags": "methodology",
        "methodological red flags": "methodology",
    }
    normalized = {}
    for k, v in data.items():
        # Strip leading number+punctuation ("1. ", "2) ", etc.)
        clean = re.sub(r"^\d+[\.\)\-\s]+\s*", "", k).strip().lower()
        canon = key_map.get(clean)
        if canon:
            normalized[canon] = v
        else:
            normalized[k] = v

    # Also look for severity/rating normalization within each dimension
    for dim_key in ("statistical_reporting", "spin", "outcome_reporting",
                     "conflict_of_interest", "methodology"):
        dim = normalized.get(dim_key)
        if isinstance(dim, dict) and "severity" not in dim and "rating" in dim:
            dim["severity"] = dim["rating"]

    return normalized


def _parse_from_json(data: dict, result: ParsedAssessment) -> ParsedAssessment:
    """Parse from a JSON-formatted model response."""
    data = _normalize_json(data)

    # Statistical reporting
    sr = _safe_dict(data.get("statistical_reporting", {}))
    result.statistical_reporting.predicted_severity = sr.get("severity", "none")
    result.statistical_reporting.predicted_binary = sr.get("severity", "none") != "none"
    result.statistical_reporting.predicted_flags = {
        "relative_only": sr.get("relative_only", False),
        "absolute_reported": sr.get("absolute_reported", False),
        "nnt_reported": sr.get("nnt_reported", False),
        "baseline_risk_reported": sr.get("baseline_risk_reported", False),
        "selective_p_values": sr.get("selective_p_values", False),
        "subgroup_emphasis": sr.get("subgroup_emphasis", False),
    }

    # Spin
    sp = _safe_dict(data.get("spin", {}))
    result.spin.predicted_severity = sp.get("severity", "none")
    result.spin.predicted_binary = sp.get("severity", "none") != "none"
    result.spin.predicted_flags = {
        "spin_level": sp.get("spin_level", "none"),
        "conclusion_matches_results": sp.get("conclusion_matches_results", True),
        "focus_on_secondary_when_primary_ns": sp.get("focus_on_secondary_when_primary_ns", False),
        "causal_language_from_observational": sp.get("causal_language_from_observational", False),
        "inappropriate_extrapolation": sp.get("inappropriate_extrapolation", False),
        "title_spin": sp.get("title_spin", False),
    }

    # Outcome reporting
    oc = _safe_dict(data.get("outcome_reporting", {}))
    result.outcome_reporting.predicted_severity = oc.get("severity", "none")
    result.outcome_reporting.predicted_binary = oc.get("severity", "none") != "none"
    result.outcome_reporting.predicted_flags = {
        "primary_outcome_type": oc.get("primary_outcome_type", "unclear"),
        "surrogate_without_validation": oc.get("surrogate_without_validation", False),
        "composite_not_disaggregated": oc.get("composite_not_disaggregated", False),
    }

    # COI
    ci = _safe_dict(data.get("conflict_of_interest", {}))
    result.conflict_of_interest.predicted_severity = ci.get("severity", "none")
    result.conflict_of_interest.predicted_binary = ci.get("severity", "none") != "none"
    result.conflict_of_interest.predicted_flags = {
        "funding_type": ci.get("funding_type", "not_reported"),
        "funding_disclosed_in_abstract": ci.get("funding_disclosed_in_abstract", False),
        "industry_author_affiliations": ci.get("industry_author_affiliations", False),
        "coi_disclosed": ci.get("coi_disclosed", False),
    }

    # Methodology
    mt = _safe_dict(data.get("methodology", {}))
    result.methodology.predicted_severity = mt.get("severity", "none")
    result.methodology.predicted_binary = mt.get("severity", "none") != "none"
    result.methodology.predicted_flags = {
        "inappropriate_comparator": mt.get("inappropriate_comparator", False),
        "per_protocol_only": mt.get("per_protocol_only", False),
        "premature_stopping": mt.get("premature_stopping", False),
        "enrichment_design": mt.get("enrichment_design", False),
        "short_follow_up": mt.get("short_follow_up", False),
    }

    # Overall
    result.overall_severity = data.get("overall_severity") or "none"
    obp = data.get("overall_bias_probability") or 0.0
    if isinstance(obp, dict):
        # Model returned a nested object; try to extract a numeric value
        obp = obp.get("probability") or obp.get("value") or obp.get("score") or 0.0
    try:
        result.overall_bias_probability = float(obp)
    except (TypeError, ValueError):
        result.overall_bias_probability = 0.0

    # Fallback: infer probability from severity when model omits the number
    if result.overall_bias_probability == 0.0 and result.overall_severity != "none":
        result.overall_bias_probability = _SEVERITY_TO_PROBABILITY.get(
            result.overall_severity, 0.0
        )

    # Verification steps
    vs = data.get("recommended_verification_steps", [])
    result.verification_steps_mentioned = vs if isinstance(vs, list) else []

    return result


def _parse_from_text(text: str, result: ParsedAssessment) -> ParsedAssessment:
    """
    Heuristic parsing from free-text model output.
    Looks for section headers and severity keywords.
    """
    text_lower = text.lower()

    # Parse each dimension by looking for section headers
    dimension_map = {
        "statistical_reporting": [
            "statistical reporting", "effect size", "relative risk",
            "absolute risk", "nnt", "reporting bias",
        ],
        "spin": ["spin", "boutron", "conclusion", "misrepresentation"],
        "outcome_reporting": [
            "outcome", "surrogate", "endpoint", "composite",
            "outcome switching", "outcome reporting",
        ],
        "conflict_of_interest": [
            "conflict of interest", "coi", "funding", "sponsor",
            "disclosure", "industry",
        ],
        "methodology": [
            "methodolog", "comparator", "blinding", "per-protocol",
            "premature stop", "enrichment", "red flag",
        ],
    }

    for dim_name, keywords in dimension_map.items():
        dim_score: DimensionScore = getattr(result, dim_name)
        # Find text near dimension keywords (limit to ~500 chars to avoid crossing sections)
        for keyword in keywords:
            pattern = rf'{keyword}.{{0,500}}?\b(none|low|moderate|high|critical)\b'
            match = re.search(pattern, text_lower)
            if match:
                dim_score.predicted_severity = match.group(1)
                dim_score.predicted_binary = match.group(1) != "none"
                dim_score.raw_text = match.group()[:300]
                break

    # Parse specific flags from text
    result.statistical_reporting.predicted_flags = {
        "relative_only": bool(re.search(
            r'(?:only|sole|exclusive)\s+(?:\w+\s+){0,3}relative', text_lower
        )),
        "absolute_reported": bool(re.search(
            r'absolute\s+(?:risk|reduction|difference)', text_lower
        )),
        "nnt_reported": "nnt" in text_lower or "number needed to treat" in text_lower,
        "baseline_risk_reported": "baseline risk" in text_lower or "control group" in text_lower,
    }

    result.spin.predicted_flags = {
        "conclusion_matches_results": not bool(re.search(
            r'(?:conclusions?\s+(?:do\s+not|don\'t|does\s+not)\s+match|'
            r'overstate|misleading|spin)', text_lower
        )),
        "focus_on_secondary_when_primary_ns": bool(re.search(
            r'(?:secondary|subgroup).*(?:primary.*(?:not significant|non-significant)|'
            r'(?:not significant|non-significant).*primary)', text_lower
        )),
    }

    # Overall severity — use the LAST match to prefer the final assessment
    # over incidental mentions earlier in the text (e.g., "overall population").
    overall_matches = list(re.finditer(
        r'overall[\s_].*?\b(none|low|moderate|high|critical)\b', text_lower
    ))
    if overall_matches:
        result.overall_severity = overall_matches[-1].group(1)

    # Bias probability
    prob_match = re.search(r'(?:bias\s+)?probability[:\s]+(\d+(?:\.\d+)?)\s*%?', text_lower)
    if prob_match:
        val = float(prob_match.group(1))
        result.overall_bias_probability = val / 100 if val > 1 else val

    # Fallback: infer probability from severity when model omits the number
    if result.overall_bias_probability == 0.0 and result.overall_severity != "none":
        result.overall_bias_probability = _SEVERITY_TO_PROBABILITY.get(
            result.overall_severity, 0.0
        )

    return result


def _score_verification_mentions(result: ParsedAssessment) -> ParsedAssessment:
    """Score how many verification sources the model used or recommended.

    Checks three evidence channels:
    1. Text mentions in raw_output and recommended_verification_steps
       (v3 text-based annotations)
    2. Tool call names from ``_agent_tool_calls`` stored in the
       annotation JSON (v4 agentic annotations)
    3. The ``reasoning`` field which may reference verification results
    """
    # Build a combined text corpus from all text-bearing fields
    text_lower = (
        result.raw_output.lower()
        + " " + " ".join(result.verification_steps_mentioned).lower()
    )

    # For v4 agentic annotations, extract tool call names from
    # the raw_output JSON (which contains the full annotation dict).
    # The _agent_tool_calls field is a list of [tool_name, args] pairs.
    tool_names_used: set[str] = set()
    try:
        data = json.loads(result.raw_output)
        for tc in data.get("_agent_tool_calls", []):
            if isinstance(tc, (list, tuple)) and len(tc) >= 1:
                tool_names_used.add(str(tc[0]).lower())
        # Also fold the reasoning field into the text corpus
        reasoning = data.get("reasoning", "")
        if reasoning:
            text_lower += " " + reasoning.lower()
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # Map source keys to text patterns AND v4 tool names
    sources = {
        "open_payments": {
            "text": [
                "open payments", "openpaymentsdata", "cms open",
                "sunshine act", "physician payments",
            ],
            "tools": ["check_open_payments"],
        },
        "clinicaltrials_gov": {
            "text": [
                "clinicaltrials.gov", "clinical trials registry",
                "nct", "trial registry", "registered outcome",
            ],
            "tools": ["check_clinicaltrials"],
        },
        "orcid": {
            "text": ["orcid"],
            "tools": ["check_orcid"],
        },
        "retraction_watch": {
            "text": [
                "retraction watch", "retraction database",
                "crossref retraction",
            ],
            "tools": ["check_retraction_status"],
        },
        "europmc": {
            "text": [
                "europe pmc", "europepmc", "europmc",
                "europe pubmed central",
            ],
            "tools": ["check_europmc_funding"],
        },
    }

    extra_sources = {
        "who_ictrp": {
            "text": ["who ictrp", "who trial", "international clinical trials registry"],
            "tools": [],
        },
        "effect_size_audit": {
            "text": ["effect size audit", "effect-size audit"],
            "tools": ["run_effect_size_audit"],
        },
        "cochrane_rob": {
            "text": ["cochrane risk of bias", "rob 2", "cochrane rob"],
            "tools": [],
        },
    }

    count = 0
    total = len(sources) + len(extra_sources)

    field_map = {
        "open_payments": "mentions_open_payments",
        "clinicaltrials_gov": "mentions_clinicaltrials_gov",
        "orcid": "mentions_orcid",
        "retraction_watch": "mentions_retraction_watch",
        "europmc": "mentions_europmc",
    }

    for source_key, spec in sources.items():
        found = (
            any(p in text_lower for p in spec["text"])
            or any(t in tool_names_used for t in spec["tools"])
        )
        field_name = field_map.get(source_key)
        if field_name and hasattr(result, field_name):
            setattr(result, field_name, found)
        if found:
            count += 1

    for _, spec in extra_sources.items():
        if (
            any(p in text_lower for p in spec["text"])
            or any(t in tool_names_used for t in spec["tools"])
        ):
            count += 1

    result.verification_score = count / total if total > 0 else 0.0
    return result


def attach_ground_truth(
    parsed: ParsedAssessment, ground_truth: dict
) -> ParsedAssessment:
    """
    Attach human-validated ground truth labels to a parsed assessment.
    ground_truth can be either:
    - Native annotation schema (with statistical_reporting, spin, etc. keys)
    - Alpaca format (with "output" field containing the annotated text/JSON)
    """
    gt = ground_truth

    # If this looks like Alpaca format, parse the output field to extract labels
    if "output" in gt and "statistical_reporting" not in gt:
        gt_parsed = parse_model_output(
            raw_output=gt["output"], pmid=parsed.pmid, model_id="ground_truth"
        )
        # Copy parsed ground truth severities into the expected dict format
        gt = {
            "statistical_reporting": {"severity": gt_parsed.statistical_reporting.predicted_severity,
                                       **gt_parsed.statistical_reporting.predicted_flags},
            "spin": {"severity": gt_parsed.spin.predicted_severity,
                     **gt_parsed.spin.predicted_flags},
            "outcome_reporting": {"severity": gt_parsed.outcome_reporting.predicted_severity,
                                   **gt_parsed.outcome_reporting.predicted_flags},
            "conflict_of_interest": {"severity": gt_parsed.conflict_of_interest.predicted_severity,
                                      **gt_parsed.conflict_of_interest.predicted_flags},
            "methodology": {"severity": gt_parsed.methodology.predicted_severity,
                            **gt_parsed.methodology.predicted_flags},
            "overall_severity": gt_parsed.overall_severity,
        }

    # Statistical reporting
    sr_gt = gt.get("statistical_reporting", {})
    parsed.statistical_reporting.ground_truth_severity = sr_gt.get("severity", "none")
    parsed.statistical_reporting.ground_truth_binary = sr_gt.get("severity", "none") != "none"
    parsed.statistical_reporting.ground_truth_flags = {
        "relative_only": sr_gt.get("relative_only", False),
        "absolute_reported": sr_gt.get("absolute_reported", False),
        "nnt_reported": sr_gt.get("nnt_reported", False),
        "baseline_risk_reported": sr_gt.get("baseline_risk_reported", False),
        "selective_p_values": sr_gt.get("selective_p_values", False),
        "subgroup_emphasis": sr_gt.get("subgroup_emphasis", False),
    }

    # Spin
    sp_gt = gt.get("spin", {})
    parsed.spin.ground_truth_severity = sp_gt.get("severity", "none")
    parsed.spin.ground_truth_binary = sp_gt.get("severity", "none") != "none"
    parsed.spin.ground_truth_flags = {
        "spin_level": sp_gt.get("spin_level", "none"),
        "conclusion_matches_results": sp_gt.get("conclusion_matches_results", True),
        "focus_on_secondary_when_primary_ns": sp_gt.get(
            "focus_on_secondary_when_primary_ns", False
        ),
    }

    # Outcome reporting
    oc_gt = gt.get("outcome_reporting", {})
    parsed.outcome_reporting.ground_truth_severity = oc_gt.get("severity", "none")
    parsed.outcome_reporting.ground_truth_binary = oc_gt.get("severity", "none") != "none"
    parsed.outcome_reporting.ground_truth_flags = {
        "primary_outcome_type": oc_gt.get("primary_outcome_type", "unclear"),
        "surrogate_without_validation": oc_gt.get("surrogate_without_validation", False),
    }

    # COI
    ci_gt = gt.get("conflict_of_interest", {})
    parsed.conflict_of_interest.ground_truth_severity = ci_gt.get("severity", "none")
    parsed.conflict_of_interest.ground_truth_binary = ci_gt.get("severity", "none") != "none"
    parsed.conflict_of_interest.ground_truth_flags = {
        "funding_type": ci_gt.get("funding_type", "not_reported"),
        "industry_author_affiliations": ci_gt.get("industry_author_affiliations", False),
    }

    # Methodology
    mt_gt = gt.get("methodology", {})
    parsed.methodology.ground_truth_severity = mt_gt.get("severity", "none")
    parsed.methodology.ground_truth_binary = mt_gt.get("severity", "none") != "none"

    # Overall
    parsed.overall_severity = gt.get("overall_severity", parsed.overall_severity)

    return parsed

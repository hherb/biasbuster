"""Output formatting for the BiasBuster CLI.

Produces JSON or Markdown reports from bias assessment results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def format_json(
    assessment: dict[str, Any],
    metadata: dict[str, Any],
    verification: dict[str, Any] | None = None,
) -> str:
    """Format assessment as a JSON string.

    Args:
        assessment: Parsed bias assessment (5-domain + overall).
        metadata: Paper and analysis metadata.
        verification: Optional verification results.

    Returns:
        Pretty-printed JSON string.
    """
    output = {
        "metadata": metadata,
        "assessment": assessment,
        "verification": verification,
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


def format_markdown(
    assessment: dict[str, Any],
    metadata: dict[str, Any],
    verification: dict[str, Any] | None = None,
) -> str:
    """Format assessment as a human-readable Markdown report.

    Args:
        assessment: Parsed bias assessment (5-domain + overall).
        metadata: Paper and analysis metadata.
        verification: Optional verification results.

    Returns:
        Markdown-formatted report string.
    """
    lines: list[str] = []

    # Header
    title = metadata.get("title", "Unknown")
    lines.append(f"# Bias Assessment Report")
    lines.append("")
    lines.append(f"**Paper:** {title}")
    if metadata.get("pmid"):
        lines.append(f"**PMID:** {metadata['pmid']}")
    if metadata.get("doi"):
        lines.append(f"**DOI:** {metadata['doi']}")
    lines.append(f"**Model:** {metadata.get('model', 'unknown')}")
    lines.append(f"**Content:** {metadata.get('content_type', 'unknown')}")
    lines.append(f"**Date:** {metadata.get('timestamp', '')}")
    if metadata.get("verified"):
        lines.append("**Verification:** Enabled")
    lines.append("")

    # Overall summary
    severity = assessment.get("overall_severity", "unknown").upper()
    probability = assessment.get("overall_bias_probability", 0.0)
    confidence = assessment.get("confidence", "unknown").upper()

    lines.append("## Overall Assessment")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Severity | **{severity}** |")
    lines.append(f"| Bias Probability | {probability:.0%} |")
    lines.append(f"| Confidence | {confidence} |")
    lines.append("")

    if assessment.get("reasoning"):
        lines.append(f"**Reasoning:** {assessment['reasoning']}")
        lines.append("")

    # Per-domain sections
    _add_domain_section(lines, "Statistical Reporting", assessment.get("statistical_reporting", {}))
    _add_domain_section(lines, "Spin", assessment.get("spin", {}))
    _add_domain_section(lines, "Outcome Reporting", assessment.get("outcome_reporting", {}))
    _add_domain_section(lines, "Conflict of Interest", assessment.get("conflict_of_interest", {}))
    _add_domain_section(lines, "Methodology", assessment.get("methodology", {}))

    # Recommended verification steps
    rec_steps = assessment.get("recommended_verification_steps", [])
    if rec_steps:
        lines.append("## Recommended Verification Steps")
        lines.append("")
        for step in rec_steps:
            lines.append(f"- {step}")
        lines.append("")

    # Verification results
    if verification:
        lines.append("## Verification Results")
        lines.append("")

        tool_results = verification.get("tool_results", [])
        if tool_results:
            for tr in tool_results:
                status = "Pass" if tr.get("success") else "Fail"
                tool_name = tr.get("tool", "unknown")
                lines.append(f"### {tool_name} [{status}]")
                lines.append("")
                if tr.get("error"):
                    lines.append(f"**Error:** {tr['error']}")
                elif tr.get("summary"):
                    lines.append(tr["summary"])
                if tr.get("detail"):
                    lines.append("")
                    lines.append(tr["detail"])
                lines.append("")

        refined = verification.get("refined_assessment")
        if refined:
            refined_severity = refined.get("overall_severity", "unknown").upper()
            refined_prob = refined.get("overall_bias_probability", 0.0)
            lines.append("### Refined Assessment (Post-Verification)")
            lines.append("")
            lines.append(f"| Metric | Initial | Refined |")
            lines.append(f"|--------|---------|---------|")
            lines.append(f"| Severity | {severity} | **{refined_severity}** |")
            lines.append(f"| Bias Probability | {probability:.0%} | **{refined_prob:.0%}** |")
            lines.append("")

            if refined.get("reasoning"):
                lines.append(f"**Refined Reasoning:** {refined['reasoning']}")
                lines.append("")

    lines.append("---")
    lines.append("*Generated by BiasBuster CLI*")

    return "\n".join(lines)


def build_metadata(
    identifier: str,
    identifier_type: str,
    title: str,
    model: str,
    content_type: str,
    verified: bool,
    pmid: str = "",
    doi: str = "",
) -> dict[str, Any]:
    """Build the metadata dict for output formatting."""
    return {
        "identifier": identifier,
        "identifier_type": identifier_type,
        "title": title,
        "pmid": pmid,
        "doi": doi,
        "model": model,
        "verified": verified,
        "content_type": content_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _add_domain_section(
    lines: list[str],
    heading: str,
    domain: dict[str, Any],
) -> None:
    """Append a per-domain section to the Markdown report."""
    if not domain:
        return

    severity = domain.get("severity", "unknown").upper()
    lines.append(f"## {heading}")
    lines.append("")
    lines.append(f"**Severity:** {severity}")
    lines.append("")

    # Domain-specific flags as a table
    flags: list[tuple[str, Any]] = []
    skip_keys = {"severity", "evidence_quotes", "reasoning"}
    for key, value in domain.items():
        if key in skip_keys:
            continue
        if isinstance(value, bool):
            flags.append((key, "Yes" if value else "No"))
        elif isinstance(value, str):
            flags.append((key, value))

    if flags:
        lines.append("| Flag | Value |")
        lines.append("|------|-------|")
        for name, value in flags:
            display_name = name.replace("_", " ").title()
            lines.append(f"| {display_name} | {value} |")
        lines.append("")

    # Evidence quotes
    quotes = domain.get("evidence_quotes", [])
    if quotes:
        lines.append("**Evidence:**")
        for quote in quotes:
            lines.append(f"> {quote}")
            lines.append("")

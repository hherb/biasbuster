"""Tests for cli.formatting — output formatters."""

import json

from biasbuster.cli.formatting import build_metadata, format_json, format_markdown


def _sample_assessment():
    """Minimal valid assessment dict."""
    return {
        "statistical_reporting": {"severity": "moderate", "relative_only": True, "absolute_reported": False},
        "spin": {"severity": "low", "spin_level": "low", "conclusion_matches_results": True},
        "outcome_reporting": {"severity": "none", "primary_outcome_type": "patient_centred"},
        "conflict_of_interest": {"severity": "high", "funding_type": "industry", "coi_disclosed": False},
        "methodology": {"severity": "low", "inappropriate_comparator": False},
        "overall_severity": "moderate",
        "overall_bias_probability": 0.45,
        "reasoning": "Industry funding with undisclosed COI and relative-only reporting.",
        "confidence": "medium",
        "recommended_verification_steps": ["Check ClinicalTrials.gov", "Check CMS Open Payments"],
    }


def _sample_metadata():
    return build_metadata(
        identifier="12345678",
        identifier_type="pmid",
        title="Test Paper Title",
        model="anthropic:claude-sonnet-4-6",
        content_type="abstract",
        verified=False,
        pmid="12345678",
        doi="10.1234/test",
    )


def test_format_json_valid():
    """JSON output is valid and contains all keys."""
    output = format_json(_sample_assessment(), _sample_metadata())
    parsed = json.loads(output)
    assert "metadata" in parsed
    assert "assessment" in parsed
    assert "verification" in parsed
    assert parsed["metadata"]["pmid"] == "12345678"
    assert parsed["assessment"]["overall_severity"] == "moderate"
    assert parsed["verification"] is None


def test_format_json_with_verification():
    """JSON output includes verification when provided."""
    verification = {
        "tool_results": [{"tool": "clinicaltrials_gov", "success": True, "summary": "Found trial."}],
        "refined_assessment": {"overall_severity": "high", "overall_bias_probability": 0.7},
    }
    output = format_json(_sample_assessment(), _sample_metadata(), verification)
    parsed = json.loads(output)
    assert parsed["verification"] is not None
    assert len(parsed["verification"]["tool_results"]) == 1


def test_format_markdown_structure():
    """Markdown output contains expected sections."""
    output = format_markdown(_sample_assessment(), _sample_metadata())

    assert "# Bias Assessment Report" in output
    assert "**PMID:** 12345678" in output
    assert "## Overall Assessment" in output
    assert "MODERATE" in output
    assert "## Statistical Reporting" in output
    assert "## Spin" in output
    assert "## Conflict of Interest" in output
    assert "## Methodology" in output
    assert "## Recommended Verification Steps" in output
    assert "Check ClinicalTrials.gov" in output


def test_format_markdown_with_verification():
    """Markdown includes verification section when provided."""
    verification = {
        "tool_results": [{"tool": "clinicaltrials_gov", "success": True, "summary": "Found trial NCT001."}],
        "refined_assessment": {
            "overall_severity": "high",
            "overall_bias_probability": 0.7,
            "reasoning": "Confirmed by registry data.",
        },
    }
    output = format_markdown(_sample_assessment(), _sample_metadata(), verification)

    assert "## Verification Results" in output
    assert "clinicaltrials_gov" in output
    assert "Refined Assessment" in output


def test_build_metadata_fields():
    """build_metadata produces all expected fields."""
    meta = _sample_metadata()
    assert meta["identifier"] == "12345678"
    assert meta["identifier_type"] == "pmid"
    assert meta["model"] == "anthropic:claude-sonnet-4-6"
    assert meta["verified"] is False
    assert "timestamp" in meta

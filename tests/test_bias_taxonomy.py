"""Tests for schemas.bias_taxonomy."""

import json
import pytest
from biasbuster.schemas.bias_taxonomy import (
    BiasAssessment,
    StatisticalReportingBias,
    SpinAssessment,
    OutcomeReporting,
    ConflictOfInterest,
    MethodologicalRedFlags,
    Severity,
    SpinLevel,
    FundingType,
    format_training_example,
)


class TestBiasAssessmentFromDict:
    """Tests for BiasAssessment.from_dict."""

    def test_basic_fields(self):
        data = {
            "pmid": "123",
            "title": "Test",
            "abstract_text": "Abstract here",
            "overall_severity": "high",
            "overall_bias_probability": 0.9,
        }
        ba = BiasAssessment.from_dict(data)
        assert ba.pmid == "123"
        assert ba.title == "Test"
        assert ba.overall_bias_probability == 0.9

    def test_nested_dataclasses(self):
        data = {
            "pmid": "456",
            "statistical_reporting": {
                "relative_only": True,
                "severity": "high",
            },
            "spin": {
                "spin_level": "moderate",
                "conclusion_matches_results": False,
            },
            "conflict_of_interest": {
                "funding_type": "industry",
                "coi_disclosed": False,
            },
        }
        ba = BiasAssessment.from_dict(data)
        assert ba.statistical_reporting.relative_only is True
        assert ba.spin.conclusion_matches_results is False
        assert ba.conflict_of_interest.funding_type == "industry"

    def test_unknown_keys_ignored(self):
        data = {
            "pmid": "789",
            "unknown_field": "should be ignored",
            "statistical_reporting": {
                "relative_only": True,
                "nonexistent_field": 42,
            },
        }
        ba = BiasAssessment.from_dict(data)
        assert ba.pmid == "789"
        assert ba.statistical_reporting.relative_only is True

    def test_empty_dict(self):
        ba = BiasAssessment.from_dict({})
        assert ba.pmid == ""
        assert ba.overall_bias_probability == 0.0


class TestBiasAssessmentToDict:
    """Tests for BiasAssessment.to_dict."""

    def test_roundtrip(self):
        original = BiasAssessment(
            pmid="RT001",
            title="Roundtrip Test",
            overall_severity=Severity.HIGH,
            overall_bias_probability=0.75,
            statistical_reporting=StatisticalReportingBias(
                relative_only=True,
                severity=Severity.HIGH,
            ),
        )
        d = original.to_dict()
        assert d["pmid"] == "RT001"
        assert d["statistical_reporting"]["relative_only"] is True

    def test_serializable_to_json(self):
        ba = BiasAssessment(pmid="JSON001", title="JSON Test")
        json_str = ba.to_json()
        parsed = json.loads(json_str)
        assert parsed["pmid"] == "JSON001"

    def test_all_domains_present(self):
        ba = BiasAssessment()
        d = ba.to_dict()
        assert "statistical_reporting" in d
        assert "spin" in d
        assert "outcome_reporting" in d
        assert "conflict_of_interest" in d
        assert "methodology" in d


class TestFormatTrainingExample:
    """Tests for format_training_example function."""

    def test_required_keys(self):
        ba = BiasAssessment(
            pmid="FMT001",
            title="Format Test",
            abstract_text="Test abstract.",
            reasoning="This is a reasoning chain.",
        )
        result = format_training_example(ba)
        assert "system" in result
        assert "instruction" in result
        assert "output" in result
        assert "pmid" in result

    def test_thinking_chain_in_output(self):
        ba = BiasAssessment(
            pmid="FMT002",
            reasoning="Step by step analysis.",
        )
        result = format_training_example(ba)
        assert "<think>" in result["output"]
        assert "Step by step analysis." in result["output"]
        assert "</think>" in result["output"]

    def test_instruction_contains_abstract(self):
        ba = BiasAssessment(
            pmid="FMT003",
            title="My Title",
            abstract_text="My abstract text here.",
        )
        result = format_training_example(ba)
        assert "My Title" in result["instruction"]
        assert "My abstract text here." in result["instruction"]

    def test_severity_domains_in_output(self):
        ba = BiasAssessment(
            pmid="FMT004",
            statistical_reporting=StatisticalReportingBias(
                relative_only=True,
                severity=Severity.HIGH,
                evidence_quotes=["HR 0.50"],
            ),
            overall_severity=Severity.HIGH,
            overall_bias_probability=0.8,
        )
        result = format_training_example(ba)
        assert "Statistical Reporting" in result["output"]
        assert "HIGH" in result["output"]

    def test_verification_steps_in_output(self):
        ba = BiasAssessment(
            pmid="FMT005",
            recommended_verification_steps=["Check ClinicalTrials.gov"],
        )
        result = format_training_example(ba)
        assert "ClinicalTrials.gov" in result["output"]

    def test_provenance_fields(self):
        ba = BiasAssessment(
            pmid="FMT006",
            source="retraction_watch",
            human_validated=True,
        )
        result = format_training_example(ba)
        assert result["source"] == "retraction_watch"
        assert result["human_validated"] is True

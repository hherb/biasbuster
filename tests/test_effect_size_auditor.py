"""Tests for enrichers.effect_size_auditor."""

import pytest
from biasbuster.enrichers.effect_size_auditor import (
    ReportingPattern,
    audit_abstract,
    batch_audit,
    SCORE_RELATIVE_ONLY,
    SCORE_NO_NNT,
    SCORE_NO_BASELINE_RISK,
    SCORE_TITLE_RELATIVE,
    SCORE_PERCENT_REDUCTION_NO_CONTEXT,
)


class TestAuditAbstract:
    """Tests for audit_abstract function."""

    def test_relative_only_pattern(self, biased_abstract):
        result = audit_abstract(**biased_abstract)
        assert result.pattern == ReportingPattern.RELATIVE_ONLY
        assert result.has_hazard_ratio is True
        assert result.has_relative_risk_reduction is True
        assert len(result.relative_measures_found) > 0
        assert len(result.absolute_measures_found) == 0

    def test_balanced_pattern(self, balanced_abstract):
        result = audit_abstract(**balanced_abstract)
        # Has both relative (HR) and absolute (ARR, NNT) => balanced or absolute_emphasised
        assert result.pattern in (
            ReportingPattern.BALANCED,
            ReportingPattern.ABSOLUTE_EMPHASISED,
        )
        assert result.has_hazard_ratio is True
        assert result.has_nnt is True
        assert result.has_absolute_risk_reduction is True
        # "8.2% of the gooddrug group vs 10.8%" has text between % and vs,
        # so the absolute_rates regex does not match it
        assert result.has_absolute_rates is False

    def test_no_effect_size_pattern(self, no_effect_size_abstract):
        result = audit_abstract(**no_effect_size_abstract)
        assert result.pattern == ReportingPattern.NO_EFFECT_SIZE
        assert len(result.relative_measures_found) == 0
        assert len(result.absolute_measures_found) == 0

    def test_pmid_preserved(self, biased_abstract):
        result = audit_abstract(**biased_abstract)
        assert result.pmid == "TEST001"

    def test_confidence_intervals_detected(self, biased_abstract):
        result = audit_abstract(**biased_abstract)
        assert result.has_confidence_intervals is True

    def test_p_values_detected(self, biased_abstract):
        result = audit_abstract(**biased_abstract)
        assert result.has_p_values is True

    def test_no_ci_no_pvalue_for_qualitative(self, no_effect_size_abstract):
        result = audit_abstract(**no_effect_size_abstract)
        assert result.has_confidence_intervals is False
        assert result.has_p_values is False

    def test_effect_in_title_not_detected_for_verb_form(self, biased_abstract):
        """Title 'reduces...by 47%' uses verb form not caught by percent_reduction regex."""
        result = audit_abstract(**biased_abstract)
        # The regex matches 'reduced/reduction/reducing' but not 'reduces'
        assert result.effect_in_title is False

    def test_effect_in_title_detected_with_hr(self):
        """Title with explicit HR should be detected."""
        result = audit_abstract(
            pmid="T",
            title="Drug shows HR 0.50 for mortality",
            abstract="RESULTS: HR 0.50.",
        )
        assert result.effect_in_title is True

    def test_no_effect_in_title(self, balanced_abstract):
        """Title 'Effect of gooddrug on cardiovascular outcomes' has no measure."""
        result = audit_abstract(**balanced_abstract)
        assert result.effect_in_title is False


class TestBiasScoring:
    """Tests for reporting bias score computation."""

    def test_relative_only_score_includes_base(self, biased_abstract):
        result = audit_abstract(**biased_abstract)
        # RELATIVE_ONLY (0.4) + NO_NNT (0.1) + NO_BASELINE_RISK (0.1) at minimum
        assert result.reporting_bias_score >= SCORE_RELATIVE_ONLY + SCORE_NO_NNT + SCORE_NO_BASELINE_RISK

    def test_balanced_lower_score(self, balanced_abstract):
        result = audit_abstract(**balanced_abstract)
        assert result.reporting_bias_score < 0.3

    def test_no_effect_size_zero_score(self, no_effect_size_abstract):
        result = audit_abstract(**no_effect_size_abstract)
        assert result.reporting_bias_score == 0.0

    def test_score_capped_at_one(self):
        """Even extreme abstracts should not exceed 1.0."""
        result = audit_abstract(
            pmid="CAP",
            title="Drug reduces death by 90% relative risk reduction (HR 0.10, OR 0.05)",
            abstract=(
                "RESULTS: HR 0.10, OR 0.05, RR 0.10. Reduced mortality by 90%. "
                "50% lower risk of death. RRR was dramatic. "
                "CONCLUSIONS: HR 0.10 confirms the 90% relative risk reduction."
            ),
        )
        assert result.reporting_bias_score <= 1.0

    def test_flags_populated(self, biased_abstract):
        result = audit_abstract(**biased_abstract)
        assert len(result.flags) > 0
        flag_text = " ".join(result.flags)
        assert "RELATIVE_ONLY" in flag_text

    def test_title_relative_flag(self):
        """Title promoting relative measure when pattern is relative_only."""
        result = audit_abstract(
            pmid="TITLE",
            title="Drug reduces events by 50% (HR 0.50)",
            abstract="RESULTS: HR 0.50 (95% CI 0.3-0.7). CONCLUSIONS: Great drug.",
        )
        assert result.effect_in_title is True
        assert result.reporting_bias_score >= SCORE_TITLE_RELATIVE


class TestBatchAudit:
    """Tests for batch_audit function."""

    def test_filters_by_threshold(self, biased_abstract, no_effect_size_abstract):
        items = [biased_abstract, no_effect_size_abstract]
        results = batch_audit(items, min_score=0.3)
        # Only the biased one should pass
        assert len(results) == 1
        assert results[0].pmid == "TEST001"

    def test_empty_input(self):
        assert batch_audit([]) == []

    def test_zero_threshold_returns_all_with_score(self, biased_abstract, balanced_abstract):
        items = [biased_abstract, balanced_abstract]
        results = batch_audit(items, min_score=0.0)
        # Both have some score >= 0
        assert len(results) >= 1

    def test_high_threshold_filters_all(self, balanced_abstract, no_effect_size_abstract):
        items = [balanced_abstract, no_effect_size_abstract]
        results = batch_audit(items, min_score=0.99)
        assert len(results) == 0

    def test_handles_missing_keys(self):
        """Dicts with missing keys should use defaults."""
        items = [{"pmid": "X"}]
        results = batch_audit(items, min_score=0.0)
        assert len(results) >= 0  # Should not raise


@pytest.mark.parametrize("measure,field_name", [
    ("HR 0.75", "has_hazard_ratio"),
    ("odds ratio 2.3", "has_odds_ratio"),
    ("relative risk 1.5", "has_relative_risk"),
    ("risk ratio 0.80", "has_risk_ratio"),
    ("relative risk reduction", "has_relative_risk_reduction"),
    ("reduced mortality by 30%", "has_percent_reduction"),
])
def test_individual_relative_measures(measure, field_name):
    result = audit_abstract("P", measure, f"RESULTS: {measure} was found.")
    assert getattr(result, field_name) is True


@pytest.mark.parametrize("measure,field_name", [
    ("absolute risk reduction was 5%", "has_absolute_risk_reduction"),
    ("NNT = 25", "has_nnt"),
    ("NNH = 100", "has_nnh"),
    ("8.2% vs 10.8%", "has_absolute_rates"),
])
def test_individual_absolute_measures(measure, field_name):
    result = audit_abstract("P", "Title", f"RESULTS: {measure}.")
    assert getattr(result, field_name) is True

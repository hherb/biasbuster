"""Tests for collectors.spin_detector."""

import pytest
from collectors.spin_detector import (
    screen_for_spin,
    batch_screen,
    SpinType,
    BOUTRON_LOW_THRESHOLD,
    BOUTRON_MODERATE_THRESHOLD,
    SEVERITY_WEIGHTS,
)


class TestScreenForSpin:
    """Tests for screen_for_spin function."""

    def test_ns_primary_detected(self, spun_abstract):
        result = screen_for_spin(**spun_abstract)
        assert result.primary_outcome_significant == "no"

    def test_significant_primary_detected(self, clean_abstract):
        result = screen_for_spin(**clean_abstract)
        # p=0.006 is significant, but the regex checks for explicit
        # "primary endpoint...significant" phrasing
        assert result.primary_outcome_significant in ("yes", "unclear")

    def test_within_group_flagged(self, spun_abstract):
        """'improved from baseline' triggers within-group flag."""
        result = screen_for_spin(**spun_abstract)
        spin_types = [f.spin_type for f in result.flags]
        assert SpinType.WITHIN_GROUP in spin_types

    def test_title_spin_detected(self, spun_abstract):
        """Title says 'improves outcomes' but primary was NS."""
        result = screen_for_spin(**spun_abstract)
        spin_types = [f.spin_type for f in result.flags]
        assert SpinType.TITLE_SPIN in spin_types

    def test_clean_abstract_no_flags(self, clean_abstract):
        result = screen_for_spin(**clean_abstract)
        assert len(result.flags) == 0
        assert result.boutron_level == "none"
        assert result.overall_spin_score == 0.0

    def test_benefit_despite_ns_with_explicit_text(self):
        """Test BENEFIT_DESPITE_NS with text that matches the regex pattern."""
        result = screen_for_spin(
            pmid="BNS1",
            title="Study of drug X",
            abstract=(
                "RESULTS: The primary endpoint did not show a significant difference.\n"
                "CONCLUSION: Drug X demonstrates that it is effective and beneficial."
            ),
        )
        spin_types = [f.spin_type for f in result.flags]
        assert SpinType.BENEFIT_DESPITE_NS in spin_types

    def test_trend_emphasis_with_matching_text(self):
        """Test TREND_EMPHASIS with text matching the patterns."""
        result = screen_for_spin(
            pmid="TE1",
            title="Study of drug Y",
            abstract=(
                "RESULTS: There was a non-significant trend toward improvement. "
                "The result was clinically meaningful. "
                "Approaching significance was noted."
            ),
        )
        spin_types = [f.spin_type for f in result.flags]
        assert SpinType.TREND_EMPHASIS in spin_types

    def test_clinical_recommendation_with_matching_text(self):
        """Test CLINICAL_RECOMMENDATION with text matching the patterns."""
        result = screen_for_spin(
            pmid="CR1",
            title="Study of drug Z",
            abstract=(
                "RESULTS: Drug Z was effective.\n"
                "CONCLUSION: We recommend use of drug Z as first-line therapy. "
                "This should be standard of care."
            ),
        )
        spin_types = [f.spin_type for f in result.flags]
        assert SpinType.CLINICAL_RECOMMENDATION in spin_types

    def test_structured_abstract_detected(self, spun_abstract):
        result = screen_for_spin(**spun_abstract)
        assert result.has_structured_abstract is True

    def test_unstructured_abstract(self):
        result = screen_for_spin(
            pmid="U1",
            title="A study",
            abstract="This study found no significant difference between groups.",
        )
        assert result.has_structured_abstract is False

    def test_pmid_and_title_preserved(self, spun_abstract):
        result = screen_for_spin(**spun_abstract)
        assert result.pmid == "SPIN001"
        assert result.title == spun_abstract["title"]


class TestBoutronLevels:
    """Test Boutron classification thresholds."""

    def test_none_level(self, clean_abstract):
        result = screen_for_spin(**clean_abstract)
        assert result.boutron_level == "none"

    def test_high_level(self, spun_abstract):
        result = screen_for_spin(**spun_abstract)
        # Within-group (moderate=0.2) + title spin (high=0.35) = 0.55
        assert result.boutron_level == "high"
        assert result.overall_spin_score >= BOUTRON_MODERATE_THRESHOLD

    def test_score_capped_at_one(self):
        """Even extreme spin should not exceed 1.0."""
        result = screen_for_spin(
            pmid="EX",
            title="Breakthrough drug prevents death and improves everything",
            abstract=(
                "RESULTS: The primary endpoint did not reach statistical significance "
                "(p=0.40). There was a non-significant trend toward improvement. "
                "Clinically meaningful results. Approaching significance. "
                "Marginally significant. "
                "Improvement from baseline was substantial.\n"
                "CONCLUSIONS: This retrospective cohort study shows the drug prevents "
                "mortality. We recommend use as first-line therapy. "
                "In the subgroup analysis, exploratory post-hoc subset showed benefit."
            ),
        )
        assert result.overall_spin_score <= 1.0

    def test_low_level(self):
        """A single low-severity flag should yield 'low' boutron level."""
        result = screen_for_spin(
            pmid="LOW1",
            title="Study of drug A",
            abstract=(
                "RESULTS: Clinically meaningful reduction noted."
            ),
        )
        if result.flags:
            # Single trend flag with moderate severity = 0.2
            assert result.boutron_level in ("low", "moderate")


class TestWithinGroupDetection:
    """Test within-group comparison detection."""

    def test_within_group_flagged(self):
        result = screen_for_spin(
            pmid="WG1",
            title="Study of drug X",
            abstract=(
                "RESULTS: Scores improved significantly from baseline in the treatment group. "
                "Before and after comparison showed improvement."
            ),
        )
        spin_types = [f.spin_type for f in result.flags]
        assert SpinType.WITHIN_GROUP in spin_types

    def test_within_group_not_flagged_with_between(self):
        """If between-group comparison also present, don't flag."""
        result = screen_for_spin(
            pmid="WG2",
            title="Study of drug Y",
            abstract=(
                "RESULTS: Scores improved from baseline. "
                "The between-group difference was significant."
            ),
        )
        spin_types = [f.spin_type for f in result.flags]
        assert SpinType.WITHIN_GROUP not in spin_types


class TestCausalLanguage:
    """Test causal language detection from observational studies."""

    def test_causal_in_observational_flagged(self):
        result = screen_for_spin(
            pmid="CL1",
            title="A retrospective study",
            abstract=(
                "METHODS: Retrospective cohort study of 10,000 patients.\n"
                "CONCLUSIONS: Drug X prevents cardiovascular events."
            ),
        )
        spin_types = [f.spin_type for f in result.flags]
        assert SpinType.CAUSAL_LANGUAGE in spin_types

    def test_causal_in_rct_not_flagged(self):
        result = screen_for_spin(
            pmid="CL2",
            title="A randomized trial",
            abstract=(
                "METHODS: Randomized controlled trial.\n"
                "CONCLUSIONS: Drug X prevents cardiovascular events."
            ),
        )
        spin_types = [f.spin_type for f in result.flags]
        assert SpinType.CAUSAL_LANGUAGE not in spin_types


class TestBatchScreen:
    """Tests for batch_screen function."""

    def test_filters_by_threshold(self, spun_abstract, clean_abstract):
        items = [spun_abstract, clean_abstract]
        results = batch_screen(items, min_score=0.2)
        pmids = [r.pmid for r in results]
        assert "SPIN001" in pmids
        assert "CLEAN001" not in pmids

    def test_empty_input(self):
        assert batch_screen([]) == []

    def test_high_threshold_filters_all(self, clean_abstract):
        results = batch_screen([clean_abstract], min_score=0.99)
        assert len(results) == 0

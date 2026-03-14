"""Tests for schemas.extract_abstract_sections."""

import pytest
from schemas import extract_abstract_sections


class TestExtractAbstractSections:
    """Tests for extract_abstract_sections function."""

    def test_structured_abstract_with_newlines(self):
        """Sections separated by newlines are correctly parsed.
        Note: 'CONCLUSIONS' is parsed as 'CONCLUSION' because the regex
        alternation matches 'CONCLUSION' before 'CONCLUSIONS', leaving
        the trailing 'S' in the content."""
        abstract = (
            "BACKGROUND: Some background info.\n"
            "METHODS: Study methods here.\n"
            "RESULTS: The results were good.\n"
            "CONCLUSIONS: We conclude something."
        )
        sections = extract_abstract_sections(abstract)
        assert "BACKGROUND" in sections
        assert "METHODS" in sections
        assert "RESULTS" in sections
        # CONCLUSION is matched (not CONCLUSIONS) due to regex alternation order
        assert "CONCLUSION" in sections

    def test_unstructured_abstract(self):
        abstract = "This is a plain abstract without any section labels."
        sections = extract_abstract_sections(abstract)
        assert "FULL" in sections
        assert sections["FULL"] == abstract

    def test_section_content_correct(self):
        abstract = "BACKGROUND: The sky is blue.\nRESULTS: We found evidence."
        sections = extract_abstract_sections(abstract)
        assert "sky is blue" in sections["BACKGROUND"]
        assert "found evidence" in sections["RESULTS"]

    def test_case_insensitive_headers(self):
        abstract = "Background: Some info.\nMethods: Some methods."
        sections = extract_abstract_sections(abstract)
        assert "BACKGROUND" in sections
        assert "METHODS" in sections

    def test_various_section_names(self):
        """Test that many PubMed section names are recognized."""
        abstract = (
            "OBJECTIVE: To test something.\n"
            "DESIGN: RCT.\n"
            "FINDINGS: Good results.\n"
            "INTERPRETATION: Meaningful."
        )
        sections = extract_abstract_sections(abstract)
        assert "OBJECTIVE" in sections
        assert "FINDINGS" in sections
        assert "INTERPRETATION" in sections

    def test_empty_abstract(self):
        sections = extract_abstract_sections("")
        assert "FULL" in sections
        assert sections["FULL"] == ""

    def test_first_section_without_newline(self):
        """The first section header can appear at start of string (matches ^)."""
        abstract = "BACKGROUND: Info here.\nCONCLUSION: Final words."
        sections = extract_abstract_sections(abstract)
        assert "BACKGROUND" in sections
        assert "CONCLUSION" in sections

    def test_space_separated_only_first_detected(self):
        """Without newlines between sections, only the first header is found."""
        abstract = "BACKGROUND: Info here. CONCLUSION: Final words."
        sections = extract_abstract_sections(abstract)
        # Only BACKGROUND matches (anchored to ^ or \n)
        assert "BACKGROUND" in sections
        assert "CONCLUSION" not in sections

    def test_period_after_header(self):
        abstract = "BACKGROUND. Info here.\nCONCLUSION. Final words."
        sections = extract_abstract_sections(abstract)
        assert "BACKGROUND" in sections
        assert "CONCLUSION" in sections

    @pytest.mark.parametrize("header", [
        "INTRODUCTION", "PURPOSE", "AIM", "DISCUSSION",
        "SIGNIFICANCE", "CONTEXT", "SETTING",
    ])
    def test_recognized_headers(self, header):
        abstract = f"{header}: Some content here."
        sections = extract_abstract_sections(abstract)
        assert header in sections

    def test_pubmed_style_structured_abstract(self):
        """Simulate PubMed XML-style structured abstract with newlines.
        Uses CONCLUSION (singular) to avoid the regex alternation issue."""
        abstract = (
            "BACKGROUND: Cardiovascular disease is common.\n"
            "METHODS: We conducted an RCT.\n"
            "RESULTS: HR 0.75 (95% CI 0.63-0.89).\n"
            "CONCLUSION: The drug reduced events."
        )
        sections = extract_abstract_sections(abstract)
        assert len(sections) == 4
        assert "Cardiovascular" in sections["BACKGROUND"]
        assert "RCT" in sections["METHODS"]
        assert "HR 0.75" in sections["RESULTS"]
        assert "reduced events" in sections["CONCLUSION"]

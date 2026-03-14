"""Tests for collectors.pubmed_xml."""

import pytest
from collectors.pubmed_xml import parse_pubmed_xml, parse_pubmed_xml_batch


class TestParsePubmedXml:
    """Tests for parse_pubmed_xml with single articles."""

    def test_basic_fields(self, pubmed_single_xml):
        result = parse_pubmed_xml(pubmed_single_xml)
        assert result is not None
        assert result["pmid"] == "12345678"
        assert result["title"] == "Effect of TestDrug on mortality"
        assert result["journal"] == "Test Journal"
        assert result["doi"] == "10.1234/test.2024"
        assert result["year"] == 2024

    def test_structured_abstract(self, pubmed_single_xml):
        result = parse_pubmed_xml(pubmed_single_xml)
        assert "BACKGROUND" in result["abstract"]
        assert "RESULTS" in result["abstract"]
        assert "We tested a drug." in result["abstract"]

    def test_authors_parsed(self, pubmed_single_xml):
        result = parse_pubmed_xml(pubmed_single_xml)
        assert len(result["authors"]) == 1
        assert result["authors"][0]["last"] == "Smith"
        assert result["authors"][0]["first"] == "John"
        assert "University of Testing" in result["authors"][0]["affiliations"]

    def test_grants_parsed(self, pubmed_single_xml):
        result = parse_pubmed_xml(pubmed_single_xml)
        assert len(result["grants"]) == 1
        assert result["grants"][0]["agency"] == "NHLBI"
        assert result["grants"][0]["id"] == "R01-HL12345"

    def test_mesh_terms_parsed(self, pubmed_single_xml):
        result = parse_pubmed_xml(pubmed_single_xml)
        assert "Mortality" in result["mesh_terms"]

    def test_invalid_xml_returns_none(self):
        result = parse_pubmed_xml("<invalid>xml<broken")
        assert result is None

    def test_no_article_returns_none(self):
        result = parse_pubmed_xml(
            '<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>'
        )
        assert result is None

    def test_missing_optional_fields(self):
        xml = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>99999</PMID>
      <Article>
        <ArticleTitle>Minimal article</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""
        result = parse_pubmed_xml(xml)
        assert result is not None
        assert result["pmid"] == "99999"
        assert result["abstract"] == ""
        assert result["journal"] == ""
        assert result["doi"] == ""
        assert result["year"] is None
        assert result["authors"] == []
        assert result["grants"] == []
        assert result["mesh_terms"] == []

    def test_unstructured_abstract(self):
        xml = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>11111</PMID>
      <Article>
        <ArticleTitle>Simple study</ArticleTitle>
        <Abstract>
          <AbstractText>This is a plain unstructured abstract with no labels.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""
        result = parse_pubmed_xml(xml)
        assert result is not None
        assert "unstructured abstract" in result["abstract"]
        # No label prefix
        assert ":" not in result["abstract"].split("unstructured")[0]


class TestParsePubmedXmlBatch:
    """Tests for parse_pubmed_xml_batch with multiple articles."""

    def test_multiple_articles(self):
        xml = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article><ArticleTitle>Article One</ArticleTitle></Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>222</PMID>
      <Article><ArticleTitle>Article Two</ArticleTitle></Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""
        results = parse_pubmed_xml_batch(xml)
        assert len(results) == 2
        assert "111" in results
        assert "222" in results
        assert results["111"]["title"] == "Article One"
        assert results["222"]["title"] == "Article Two"

    def test_empty_set(self):
        xml = '<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>'
        results = parse_pubmed_xml_batch(xml)
        assert results == {}

    def test_invalid_xml_returns_empty(self):
        results = parse_pubmed_xml_batch("<broken xml!!!>")
        assert results == {}

    def test_single_article_in_batch(self, pubmed_single_xml):
        results = parse_pubmed_xml_batch(pubmed_single_xml)
        assert len(results) == 1
        assert "12345678" in results

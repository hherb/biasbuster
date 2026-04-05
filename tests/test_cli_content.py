"""Tests for cli.content — identifier classification."""

import pytest

from biasbuster.cli.content import classify_identifier


def test_classify_pmid_bare():
    """Bare integer is classified as PMID."""
    id_type, value = classify_identifier("12345678")
    assert id_type == "pmid"
    assert value == "12345678"


def test_classify_pmid_prefixed():
    """PMID: prefix is handled."""
    id_type, value = classify_identifier("PMID:12345678")
    assert id_type == "pmid"
    assert value == "12345678"

    id_type, value = classify_identifier("pmid: 12345678")
    assert id_type == "pmid"
    assert value == "12345678"


def test_classify_doi_bare():
    """DOI starting with 10. is detected."""
    id_type, value = classify_identifier("10.1016/j.example.2024.01.001")
    assert id_type == "doi"
    assert value == "10.1016/j.example.2024.01.001"


def test_classify_doi_prefixed():
    """doi: prefix is stripped."""
    id_type, value = classify_identifier("doi:10.1016/j.example.2024")
    assert id_type == "doi"
    assert value == "10.1016/j.example.2024"


def test_classify_doi_url():
    """DOI URL is parsed."""
    id_type, value = classify_identifier("https://doi.org/10.1016/j.example.2024")
    assert id_type == "doi"
    assert value == "10.1016/j.example.2024"

    id_type, value = classify_identifier("https://dx.doi.org/10.1016/j.example.2024")
    assert id_type == "doi"
    assert value == "10.1016/j.example.2024"


def test_classify_file_pdf(tmp_path):
    """PDF file path is detected."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"dummy")
    id_type, value = classify_identifier(str(pdf))
    assert id_type == "file"


def test_classify_file_xml(tmp_path):
    """XML file path is detected."""
    xml = tmp_path / "paper.xml"
    xml.write_text("<article/>")
    id_type, value = classify_identifier(str(xml))
    assert id_type == "file"


def test_classify_doi_takes_precedence_over_directory(tmp_path, monkeypatch):
    """DOI starting with 10. is classified as DOI even if a matching dir exists."""
    # Create a directory named "10.1016" to test precedence
    doi_dir = tmp_path / "10.1016"
    doi_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    id_type, value = classify_identifier("10.1016/j.example.2024")
    assert id_type == "doi"
    assert value == "10.1016/j.example.2024"


def test_classify_unknown():
    """Unrecognised input raises ValueError."""
    with pytest.raises(ValueError, match="Cannot classify"):
        classify_identifier("not_a_valid_identifier!")

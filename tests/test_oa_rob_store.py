import pytest
from studies.oa_rob_benchmark.store import BenchmarkStore, LitmusError
from scripts.audit_oa_rob_benchmark import audit_benchmark
from studies.oa_rob_benchmark.export import export_redistributable


def _valid_item():
    return {
        "trial_pmid": "111", "trial_pmcid": "PMC1", "trial_doi": "10.1/x",
        "trial_title": "Trial A", "trial_license": "CC-BY-4.0",
        "license_redistributable": True, "non_commercial": False,
        "no_derivatives": False, "fulltext_path": "/cache/111.jats.xml",
        "rob2_overall": "high", "rob2_d1": "low", "rob2_d2": "low",
        "rob2_d3": "some concerns", "rob2_d4": "low", "rob2_d5": "high",
        "per_outcome_variant": False, "label_source": "roboto2",
        "source_review_pmid": "999", "source_review_pmcid": "PMC999",
        "table_index": 0, "row_index": 3, "resolution_method": "bracket_ref",
        "similarity_score": 1.0, "pubtype_check": "trial",
        "extraction_method": "structural_table", "manual_verified": False,
    }


def test_upsert_valid_item(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    assert store.upsert_item(_valid_item()) is True
    assert store.count() == 1


def test_reject_non_redistributable_license(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    bad = _valid_item() | {"license_redistributable": False}
    with pytest.raises(LitmusError):
        store.upsert_item(bad)


def test_reject_partial_tuple(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    bad = _valid_item() | {"rob2_d4": ""}
    with pytest.raises(LitmusError):
        store.upsert_item(bad)


def test_reject_non_trial_pubtype(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    bad = _valid_item() | {"pubtype_check": "non_trial"}
    with pytest.raises(LitmusError):
        store.upsert_item(bad)


def test_reject_missing_fulltext_or_provenance(tmp_path):
    store = BenchmarkStore(str(tmp_path / "b.db"))
    with pytest.raises(LitmusError):
        store.upsert_item(_valid_item() | {"fulltext_path": ""})
    with pytest.raises(LitmusError):
        store.upsert_item(_valid_item() | {"source_review_pmid": ""})


def test_reopen_does_not_drop_rows(tmp_path):
    p = str(tmp_path / "b.db")
    BenchmarkStore(p).upsert_item(_valid_item())
    assert BenchmarkStore(p).count() == 1   # second open must not DROP


def test_upsert_ignores_extra_non_schema_key(tmp_path):
    """An item dict carrying a key outside the schema must still upsert.

    Tasks 5/6 pass item dicts that may include helper fields (e.g. debug
    metadata) not present as columns in ``benchmark_item``. Those extra
    keys must be silently filtered out of the INSERT rather than causing
    a sqlite3.OperationalError for an unknown column.
    """
    store = BenchmarkStore(str(tmp_path / "b.db"))
    item = _valid_item() | {"debug_note": "not a real column"}
    assert store.upsert_item(item) is True
    assert store.count() == 1


def test_audit_flags_bad_row():
    good = _valid_item()
    bad = _valid_item() | {"trial_pmid": "222", "pubtype_check": "non_trial"}
    violations = audit_benchmark([good, bad])
    assert any("222" in v for v in violations)
    assert all("111" not in v for v in violations)


def test_export_keeps_provenance_and_flags_nc_nd():
    row = _valid_item() | {"non_commercial": True, "no_derivatives": True}
    out = export_redistributable([row])
    assert out[0]["non_commercial"] is True
    assert "source_review_prose" not in out[0]
    assert out[0]["rob2_overall"] == "high"
    assert "fulltext_path" not in out[0]


def test_export_drops_non_redistributable_item():
    row = _valid_item() | {"license_redistributable": False}
    assert export_redistributable([row]) == []

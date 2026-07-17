from studies.oa_rob_benchmark.manual_gate import render_manifest, MANUAL_SAMPLE_SIZE


def _item(pmid):
    return {"trial_pmid": pmid, "trial_title": f"Trial {pmid}",
            "source_review_pmid": "999", "source_review_pmcid": "PMC999",
            "table_index": 0, "row_index": 3,
            "rob2_overall": "high", "rob2_d1": "low", "rob2_d2": "low",
            "rob2_d3": "some concerns", "rob2_d4": "low", "rob2_d5": "high",
            "resolution_method": "bracket_ref", "similarity_score": 1.0,
            "trial_license": "CC-BY-4.0", "label_source": "cochrane_review"}

def test_manifest_lists_up_to_sample_size_rows():
    md = render_manifest([_item(str(i)) for i in range(30)])
    assert md.count("Trial ") == MANUAL_SAMPLE_SIZE
    assert "some concerns" in md
    assert "PMC999" in md


def test_manifest_samples_across_label_sources():
    """Both provenance pools must be surfaced even when one dominates.

    Regression for the manual-gate sampling fix: a plain ``items[:20]``
    slice over 30 ``roboto2`` rows followed by 3 ``cochrane_review`` rows
    would show only ``roboto2`` rows and never surface the riskier
    structural-extraction pool. Stratified round-robin must include every
    ``cochrane_review`` row while still capping at the sample size.
    """
    roboto2 = [_item(f"r{i}") | {"label_source": "roboto2"} for i in range(30)]
    cochrane = [_item(f"c{i}") | {"label_source": "cochrane_review"} for i in range(3)]
    md = render_manifest(roboto2 + cochrane)
    assert md.count("Trial ") == MANUAL_SAMPLE_SIZE
    for i in range(3):
        assert f"(PMID c{i})" in md  # every minority-pool row surfaced

def test_manifest_shows_all_six_fields_per_row():
    md = render_manifest([_item("111")])
    for field in ("overall", "D1", "D2", "D3", "D4", "D5"):
        assert field in md

def test_manifest_handles_none_valued_fields_without_crashing():
    # ROBoto2-sourced rows leave table_index/row_index/similarity_score
    # (and possibly trial_title/source_review_pmcid) as None.
    item = _item("222") | {
        "trial_title": None, "source_review_pmcid": None,
        "table_index": None, "row_index": None, "similarity_score": None,
    }
    md = render_manifest([item])
    assert "222" in md
    assert "None" not in md

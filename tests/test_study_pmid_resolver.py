from biasbuster.collectors.study_pmid_resolver import (
    Reference, resolve_study_pmid, TITLE_SIMILARITY_THRESHOLD,
)

REFS = [
    Reference("28", "111", "Smith", "2020", "Effect of drug A on outcome X"),
    Reference("29", "222", "Smith", "2019", "A different Smith trial on Y"),
    Reference("30", "333", "Jones", "2021", "Jones pragmatic trial of Z"),
]

def test_bracket_ref_wins_directly():
    r = resolve_study_pmid("Smith 2020", "28", REFS)
    assert r.pmid == "111"
    assert r.method == "bracket_ref"

def test_author_year_title_disambiguates_two_smiths():
    # No bracket number; two Smiths — title similarity picks the right one.
    r = resolve_study_pmid("Smith 2020 Effect of drug A on outcome X", "", REFS)
    assert r.pmid == "111"
    assert r.method == "author_year_title"
    assert r.similarity >= TITLE_SIMILARITY_THRESHOLD

def test_surname_only_is_rejected_not_guessed():
    # Bare surname, no year/title evidence, multiple Smiths → unresolved.
    r = resolve_study_pmid("Smith", "", REFS)
    assert r.method == "unresolved"
    assert r.pmid == ""

def test_below_threshold_is_unresolved():
    r = resolve_study_pmid("Smith 2020 totally unrelated wording here", "", REFS,
                           threshold=0.95)
    assert r.method == "unresolved"

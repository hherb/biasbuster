# Test Fixtures — Systematic Review JATS Files

These JATS XML files are used as test fixtures for the structural
bias-table extractor (`biasbuster/collectors/rob_table_extractor.py`).

## Files

- `jcm-15-01829.xml` — Haba et al. (2026), "Salivary Glucose Testing
  for Diabetes Mellitus: A Systematic Review and Meta-Analysis."
  J Clin Med. DOI: 10.3390/jcm15051829. Uses **QUADAS-2** bias
  assessment (4 domains + overall). Table 2 has structured per-study
  ratings with CSS colour-coded backgrounds. 25 included studies.

## Adding new fixtures

Place JATS XML files here from CDSR or other OA reviews. Name them
by their article ID (journal abbreviation + volume + page/elocation).
Tag each with the bias methodology used (RoB 2, QUADAS-2, ROBINS-I)
in this README.

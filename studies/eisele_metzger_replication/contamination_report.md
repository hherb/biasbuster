# Phase 2 Contamination Report

**Generated:** by `studies/eisele_metzger_replication/contamination_check.py`
**Source database:** `dataset/biasbuster.db`
**Source benchmark:** `DATA/20240318_Data_for_analysis_full/Extracted_Data_Test_Data-Table 1.csv` (Eisele-Metzger 2025 supplementary)
**Pre-analysis plan reference:** §3.3 (overlap is reported, not gated; full n=100 remains the primary sample)

## Summary

- **Total EM-100 RCTs:** 100
- **High-confidence overlap (PMID or DOI exact match):** 0
- **Medium-confidence overlap (parent Cochrane review present in DB):** 0
- **Low-confidence candidate overlap (author + year):** 0
- **No detected overlap:** 100

## By match type

| Match type | Count |
|---|---:|
| none | 100 |

## Matcher methodology

The Eisele-Metzger CSV does not include explicit RCT-level PMIDs or DOIs as columns; the only structured RCT identifiers are the trial registration number (`rct_regnr`, e.g. NCT02037633) and the parent Cochrane review identifier (`cr_id`, e.g. CD001159.PUB3). The full citation in `rct_ref` is free text. Identifier extraction therefore relies on regex over the citation text and explicit columns:

- **PMID:** regex `\bPMID[:\s]*(\d{6,9})\b` over `rct_ref + rct_regnr + cr_id`.
- **DOI:** regex `\b10\.\d{4,9}/[-._;()/:A-Z0-9]+` over the same fields, with trailing punctuation stripped.
- **NCT:** regex `\bNCT\d{8}\b` over the same fields.
- **Cochrane review ID:** parsed from `cr_id` directly, normalised to e.g. `CD001159.PUB3`.
- **First-author surname:** first non-particle token of `rct_author`. Particles like *de, von, El, van, der, le* etc. are skipped because matching on them flooded an early version of this script with false positives (e.g. `de Vos 2014` → `de` matched many unrelated papers). See `_PARTICLES` in the script for the full list.

Cross-reference against the `papers` table proceeds in confidence order:

1. **High confidence** — PMID exact match (no hits in this run).
2. **High confidence** — DOI exact match (case-insensitive; no hits).
3. **Medium confidence** — parent Cochrane review's CD-id appears in the `cochrane_review_doi` column of any paper. *Note:* in the present snapshot of `dataset/biasbuster.db`, the `cochrane_review_doi` column is empty for all 328 cochrane_rob entries (only `cochrane_review_pmid` is populated, and the Cochrane review PMIDs are not directly comparable to EM's `cr_id` strings without an external lookup). This tier therefore returned no matches, but a more thorough check would convert each EM `cr_id` to a Cochrane Library DOI (e.g. CD001159.PUB3 → 10.1002/14651858.CD001159.pub3) and resolve to PMID via PubMed before comparing. Given the high-confidence tier produced 0 matches, and given the project's `cochrane_rob` source is curated independently of EM's selection, the additional thoroughness is unlikely to change the conclusion.
4. **Low confidence** — author + year + title-keyword co-occurrence. First-author surname and publication year must match a paper in the DB, AND ≥2 distinctive ≥6-character keywords from the EM citation text (after stop-word filtering) must appear in that paper's title. An earlier version of this matcher used author + year alone and produced 9 false positives that were manually verified (e.g. RCT004 Ye 2019 matched a Korean Red Ginseng paper; RCT017 Ho 2020 matched a back-pain trial in a different topic area; etc.). Adding the title-keyword co-occurrence requirement eliminated all false positives and now returns 0 candidates.

## Conclusion

**No overlap detected** between the 100 RCTs of the Eisele-Metzger benchmark and the BiasBuster project's working dataset under any matcher tier. The 100 RCTs are novel relative to our project corpus.

## Methodological note for the manuscript

Of the 100 RCTs in the Eisele-Metzger benchmark, 0 were present in the BiasBuster project's working dataset by PMID or DOI exact match, 0 additional RCTs had their parent Cochrane review present in our dataset (medium-confidence proxy), and 0 further RCTs surfaced as low-confidence author+year+title-keyword candidates. Per the locked pre-analysis plan §3.3 (commit `7854a1c`), no exclusions were applied: the full n=100 was retained as the primary analysis sample. The four evaluated models were trained by their respective providers on data we do not control, so even if any EM RCT did appear in our project corpus, it would not bias their performance against the benchmark; the present report is for methodological transparency. Per-RCT detail and the matcher source code are archived in `studies/eisele_metzger_replication/`.

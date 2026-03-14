# BMLibrarian Bias Detection Dataset Builder

A toolkit for building curated training datasets to fine-tune LLMs for detecting
bias in biomedical abstracts. Designed for use with BMLibrarian.

## Architecture

```
bias_dataset_builder/
├── collectors/
│   ├── retraction_watch.py    # Retracted papers via Crossref API
│   ├── cochrane_rob.py        # Cochrane Risk of Bias assessments
│   ├── spin_detector.py       # Heuristic pre-screening for spin indicators
│   └── clinicaltrials_gov.py  # Outcome switching detection via registry
├── enrichers/
│   ├── author_coi.py          # Author conflict-of-interest verification
│   ├── funding_checker.py     # Funding source classification
│   └── effect_size_auditor.py # Relative vs absolute reporting analysis
├── annotators/
│   └── llm_prelabel.py        # API-based pre-labelling with Claude
├── schemas/
│   └── bias_taxonomy.py       # Structured bias taxonomy and labels
├── config.py                  # Configuration and API keys
├── pipeline.py                # Orchestration pipeline
└── export.py                  # Export to fine-tuning formats (Alpaca, ShareGPT)
```

## Bias Taxonomy

The model is trained on a multi-dimensional bias assessment:

1. **Statistical Reporting Bias** (your key insight)
   - Sole/emphasis on relative risk reduction without absolute measures
   - Missing NNT/NNH
   - Baseline risk omission
   - Selective p-value reporting

2. **Spin in Conclusions**
   - Claims not supported by primary outcome
   - Inappropriate causal language from observational data
   - Focus on secondary/subgroup analyses when primary failed
   - Boutron classification: none/low/moderate/high

3. **Outcome Reporting**
   - Surrogate vs patient-centred outcomes
   - Outcome switching (vs registry)
   - Composite endpoint disaggregation missing

4. **Conflict of Interest Signals**
   - Industry funding without disclosure
   - Author-pharma payment patterns
   - Ghost authorship indicators

5. **Methodological Red Flags**
   - Inappropriate comparator (placebo when active exists)
   - Enrichment design without acknowledgment
   - Per-protocol only (no ITT)
   - Premature stopping

## Verification Sources (for model training)

The model should learn WHERE to look for corroboration:

- **CMS Open Payments** (openpaymentsdata.cms.gov) - US physician payments
- **ClinicalTrials.gov** - Registered outcomes vs published outcomes
- **WHO ICTRP** - International trial registry search
- **Crossref/Retraction Watch** - Retraction status
- **ORCID** - Author affiliation history
- **EuroPMC funding data** - Funder metadata
- **Cochrane RoB database** - Expert risk assessments

## Usage

```bash
# Set up environment and install dependencies
uv sync

# Configure
cp config.example.py config.py
# Edit config.py with your API keys

# Run collection pipeline
uv run python pipeline.py --collect-all --output dataset/

# Pre-label with Claude
uv run python -m annotators.llm_prelabel --input dataset/raw/ --output dataset/labelled/

# Export for fine-tuning
uv run python export.py --format alpaca --input dataset/labelled/ --output training_data.jsonl
```

# 3. Heuristic Enrichment

**What you'll do:** Run automated heuristic analysis on collected abstracts to classify them by suspicion level before sending them for LLM annotation.

## Run Enrichment

```bash
uv run python pipeline.py --stage enrich
```

## What Enrichment Does

### Effect Size Auditing

The effect size auditor scans each PubMed RCT abstract for statistical reporting patterns using regex-based detection:

**Relative measures detected:** Hazard Ratio (HR), Odds Ratio (OR), Relative Risk (RR), Relative Risk Reduction (RRR), percentage reductions ("reduced by X%")

**Absolute measures detected:** Absolute Risk Reduction (ARR), Number Needed to Treat (NNT), baseline event rates, raw event counts ("23/150 vs 45/148")

Each abstract receives a `reporting_bias_score` from 0.0 to 1.0 based on the presence or absence of these patterns. The score drives a three-tier classification:

| Suspicion Level | Score Range | Meaning |
|-----------------|-------------|---------|
| **high** | >= 0.3 | Likely reports only relative measures without absolute context |
| **medium** | 0.1 -- 0.3 | Mixed reporting or unclear |
| **low** | < 0.1 | Balanced statistical reporting |

**Config:**
- `high_suspicion_threshold` (default: 0.3)
- `low_suspicion_threshold` (default: 0.1)

### Outcome Switching Detection

For high-suspicion papers that mention a ClinicalTrials.gov NCT ID, the enricher:

1. Extracts the NCT ID from the abstract text
2. Fetches the registered trial protocol from ClinicalTrials.gov
3. Compares registered primary outcomes against published outcomes
4. Flags discrepancies: switched primary outcomes, omitted outcomes, newly added outcomes, retrospective registration

**Config:**
- `outcome_switching_check_limit` (default: 100) -- maximum high-suspicion papers to check against the registry

## What Gets Stored

Results are stored in the `enrichments` table:

| Field | Description |
|-------|-------------|
| `pmid` | PubMed ID (primary key) |
| `suspicion_level` | "high", "medium", or "low" |
| `reporting_bias_score` | 0.0 to 1.0 |
| `effect_size_audit` | JSON with pattern classification, flags, measures found |
| `outcome_switching` | JSON with registry comparison results (if NCT ID found) |

## Verify Results

```bash
uv run python -c "
from database import Database
db = Database('dataset/biasbuster.db')
with db:
    rows = db.conn.execute(
        'SELECT suspicion_level, COUNT(*) FROM enrichments GROUP BY suspicion_level'
    ).fetchall()
    for level, count in rows:
        print(f'{level}: {count}')
"
```

Typical distribution: 30-40% high suspicion, reflecting the well-documented prevalence of relative-only reporting in clinical trial abstracts.

## Why This Matters

Enrichment serves two purposes:

1. **Efficient annotation** -- high-suspicion abstracts are prioritized for LLM annotation, directing resources toward the most informative training examples
2. **Heuristic context** -- the effect size audit results are included in the LLM prompt during annotation, giving the model additional signal to assess statistical reporting bias

## Next Step

[LLM Annotation](04_annotation.md) -- send abstracts to LLMs for structured bias assessment across five domains.

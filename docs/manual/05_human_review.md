# 5. Creating Ground Truth (Human Review)

**What you'll do:** Use the web-based review tool to validate LLM annotations, creating human-verified ground truth for training and evaluation.

## Launch the Review Tool

```bash
# Auto-detect available models
uv run python -m utils.review_gui

# Specify a model directly
uv run python -m utils.review_gui --model anthropic

# Custom port
uv run python -m utils.review_gui --model anthropic --port 8080
```

Open the displayed URL (typically `http://localhost:8080`) in your browser.

## The Review Interface

### Annotation Grid

The main view shows all annotations in a sortable, filterable table:

| Column | Description |
|--------|-------------|
| PMID | PubMed identifier (click row for details) |
| Title | Paper title |
| Overall Severity | Model's aggregate severity rating |
| Bias Probability | Model's probability estimate (0.0--1.0) |
| Statistical Severity | Severity for statistical reporting domain |
| Spin Level | Boutron spin classification |
| Funding Type | Industry/public/mixed/not reported |
| Confidence | Model's self-assessed confidence |
| **HUMAN_VALIDATED** | Your validation checkbox (True/False) |
| **HUMAN_OVERRIDE_SEVERITY** | Override severity if you disagree |
| **HUMAN_NOTES** | Your freeform notes |

### Detail Panel

Click any row to expand a tabbed detail view showing:

- **Abstract** -- full abstract text
- **Title** -- paper title
- **Reasoning** -- the model's step-by-step assessment

### Navigation

- **Quick filter** -- text search across all columns
- **Show Unvalidated Only** -- filter to items you haven't reviewed yet
- **Next Unvalidated** -- jump to the next unreviewed annotation
- **Show All** -- clear filters

## Review Workflow

1. **Start with high-confidence annotations.** Sort by confidence or bias probability to prioritize clear-cut cases.

2. **For each annotation:**
   - Read the abstract in the detail panel
   - Check the model's reasoning
   - If you agree: set `HUMAN_VALIDATED` to True
   - If you disagree on severity: set `HUMAN_OVERRIDE_SEVERITY` to your assessment
   - Add `HUMAN_NOTES` for edge cases or teaching points

3. **Focus on disagreements.** If using multiple models, switch between them using the model dropdown to find cases where models disagree -- these are the most valuable for ground truth.

4. **Save periodically.** Changes are saved to the database as you edit. The tool also supports CSV export for offline review.

## Export to CSV

Click the Export button in the toolbar to save a CSV snapshot:

```
dataset/export/{model_name}_review.csv
```

This is useful for sharing review data with collaborators or importing into spreadsheet tools.

## How Many to Review?

For initial fine-tuning, aim to validate at least 100-200 annotations across the severity spectrum. Prioritize:

- All retracted paper annotations (known positives)
- Cochrane RoB papers (have expert ground truth to compare against)
- High-suspicion papers with high model confidence
- A sample of low-suspicion papers (to validate negatives)

## Database Storage

Human reviews are stored in the `human_reviews` table:

| Field | Description |
|-------|-------------|
| `pmid` | PubMed ID |
| `model_name` | Which model's annotation was reviewed |
| `validated` | 1 (validated) or 0 (not yet reviewed) |
| `override_severity` | Your corrected severity, or NULL if you agree |
| `notes` | Your freeform notes |
| `reviewed_at` | Timestamp |

## Next Step

[Exporting Training Data](06_export.md) -- convert validated annotations into fine-tuning formats with reasoning chains and verification steps.

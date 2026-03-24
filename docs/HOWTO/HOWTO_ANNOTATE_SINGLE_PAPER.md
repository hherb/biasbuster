# HOWTO: Annotate a Single Paper

Use `annotate_single_paper.py` when you want to add an individual paper to the
dataset outside the batch pipeline. Common scenarios:

- You found an interesting paper and want to include it in training data
- A paper failed during a batch annotation run and you want to retry it
- You want to re-annotate a paper with a different model or after prompt changes

## Quick Start

```bash
# Annotate a paper already in the database
uv run python annotate_single_paper.py --pmid 41271640

# Import a new paper by DOI and annotate it
uv run python annotate_single_paper.py --doi 10.1016/j.jad.2024.01.042
```

## What the Script Does

The script runs the full pipeline for one paper:

```
DOI ──resolve──> PMID ──fetch──> PubMed ──store──> DB ──enrich──> DB ──annotate──> DB
                   │                                 │
                   └── skipped if --pmid given        └── skipped if already in DB
```

1. **Resolve** (DOI only) — Converts DOI to PMID via the NCBI ID Converter API
2. **Fetch** — Downloads the paper from PubMed (title, abstract, authors, MeSH terms, etc.)
3. **Store** — Inserts into the `papers` table with the `--source` label
4. **Validate** — Rejects bare retraction notices and papers without abstracts
5. **Enrich** — Runs the effect-size audit heuristic, stores suspicion level in `enrichments`
6. **Annotate** — Sends to the chosen LLM backend, stores the structured 5-domain assessment

If the paper is already in the database, steps 2-3 are skipped. If an annotation
already exists for that model, the script reports this and exits (unless `--force`
is given).

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pmid PMID` | — | PubMed ID (mutually exclusive with `--doi`) |
| `--doi DOI` | — | DOI to resolve (mutually exclusive with `--pmid`) |
| `--model {anthropic,deepseek}` | `deepseek` | Which LLM backend to use |
| `--source SOURCE` | `manual_import` | Source label for newly imported papers |
| `--force` | off | Delete existing annotation and re-annotate |

## Examples

### Retry a failed annotation

If a paper failed during a batch run (e.g. transient API error), it won't have
an annotation in the database. Simply re-run:

```bash
uv run python annotate_single_paper.py --pmid 41271640
```

The paper is already in the DB, so it skips the fetch and goes straight to
enrichment and annotation.

### Re-annotate with a different model

```bash
# First annotation was with DeepSeek
uv run python annotate_single_paper.py --pmid 41271640

# Add an Anthropic annotation for the same paper
uv run python annotate_single_paper.py --pmid 41271640 --model anthropic
```

Each model's annotation is stored separately (keyed by `(pmid, model_name)`),
so both can coexist.

### Re-annotate with the same model

If you want to replace an existing annotation (e.g. after a prompt change):

```bash
uv run python annotate_single_paper.py --pmid 41271640 --force
```

This deletes the existing annotation for that model before re-running.

### Import a paper you found on PubMed

```bash
uv run python annotate_single_paper.py --pmid 38947261
```

The script fetches the paper from PubMed, stores it as `source=manual_import`,
enriches it, and annotates it.

### Import by DOI

```bash
uv run python annotate_single_paper.py --doi 10.1016/j.jad.2024.01.042
```

The DOI is resolved to a PMID via the NCBI ID Converter, then the same
fetch → enrich → annotate flow runs.

### Tag the source

If you know the paper belongs to a specific source category:

```bash
uv run python annotate_single_paper.py --doi 10.1016/j.example.2024 --source cochrane_rob
```

This sets `source=cochrane_rob` in the `papers` table, so the paper is grouped
correctly in batch operations and exports.

## Troubleshooting

### "Could not resolve DOI to a PMID"

The NCBI ID Converter doesn't cover all DOIs. Try finding the PMID manually on
PubMed and use `--pmid` instead.

### "No abstract — cannot annotate"

The paper exists in PubMed but has no abstract text. This is common for
editorials, letters, and some older papers. These cannot be assessed for bias.

### "Bare retraction/withdrawal notice"

The paper is a retraction notice, not the original research. The script
correctly rejects these. To annotate the original retracted paper, find its
PMID (not the retraction notice's PMID) and use that instead.

### "Already annotated, skipping"

An annotation already exists for this model. Use `--force` to replace it, or
use `--model` to annotate with a different backend.

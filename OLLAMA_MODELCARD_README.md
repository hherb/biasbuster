# BiasBuster — Bias Detection in Biomedical Research Abstracts

A fine-tuned [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) model that detects bias in clinical trial abstracts across five evidence-based domains. Unlike general-purpose classifiers, BiasBuster produces step-by-step reasoning chains and recommends specific verification databases where each claim can be checked.

## What It Does

Given a clinical trial abstract, BiasBuster:

1. **Reasons step-by-step** through each bias domain inside `<think>` tags
2. **Assigns severity** (NONE / LOW / MODERATE / HIGH / CRITICAL) per domain
3. **Cites evidence** from the abstract text
4. **Recommends verification steps** pointing to specific databases and search terms

## Five Bias Domains

| Domain | What It Catches |
|--------|----------------|
| **Statistical Reporting** | Sole reliance on relative measures (RRR, OR, HR) without absolute measures (ARR, NNT, baseline risk) |
| **Spin** | Conclusions that overstate or misrepresent actual results (Boutron taxonomy) |
| **Outcome Reporting** | Surrogate endpoints, composite endpoints, outcome switching from registered protocol |
| **Conflict of Interest** | Industry funding without disclosure, undisclosed author affiliations with sponsors |
| **Methodological Red Flags** | Inappropriate comparators, enrichment designs, per-protocol without ITT, premature stopping, short follow-up |

## Usage

```bash
ollama run hherb/qwen3.5-9b-biasbuster5
```

Then paste a clinical trial abstract. The model will produce a `<think>` reasoning block followed by a structured JSON assessment.

### Example Prompt

```
Assess the following abstract for bias:

Background: We conducted a randomized, double-blind trial of Drug X versus placebo
in 1,200 patients with moderate-to-severe heart failure. The primary endpoint was
the composite of cardiovascular death or hospitalization for heart failure.

Results: Drug X reduced the primary composite endpoint (HR 0.72, 95% CI 0.59-0.88,
p=0.001). The study was funded by PharmaCorp Inc.

Conclusions: Drug X significantly reduces cardiovascular events and should be
considered for routine clinical use in heart failure management.
```

### Example Output

```
<think>
Statistical reporting: The abstract reports HR 0.72 (95% CI 0.59-0.88) — a relative
measure only. No absolute risk reduction, NNT, or baseline event rates provided.
The reader cannot assess clinical significance without external data...

Spin: The conclusion recommends "routine clinical use" based on a single trial with
only a composite endpoint. The language overstates the evidence...

Conflict of interest: Funded by PharmaCorp Inc. No author-level COI disclosure.
No information about author affiliations with the sponsor...
</think>

{
  "statistical_reporting": {
    "severity": "moderate",
    "flags": {"relative_only": true, "absolute_reported": false, "nnt_reported": false},
    "evidence": "HR 0.72 (95% CI 0.59-0.88, p=0.001) with no absolute measures",
    "reasoning": "Only relative risk reduction reported; baseline event rate absent"
  },
  "spin": {
    "severity": "high",
    "flags": {"conclusion_matches_results": false},
    "evidence": "Concludes 'should be considered for routine clinical use'",
    "reasoning": "Recommends clinical adoption from a single trial without acknowledging limitations"
  },
  ...
  "verification_steps": [
    "Search ClinicalTrials.gov for the trial registration to compare registered vs published endpoints",
    "Check CMS Open Payments (openpaymentsdata.cms.gov) for author payment records from PharmaCorp",
    "Search ORCID for author affiliation histories with PharmaCorp",
    "Access full text via Europe PMC to review funding and COI disclosure sections"
  ]
}
```

## Performance

Evaluated on 144 held-out test abstracts (80/10/10 split).

### Binary Detection (Any Bias Present vs. None)

| Metric | Score |
|--------|:-----:|
| **F1** | **0.924** |
| **Recall** | **0.950** |
| Precision | 0.898 |
| Accuracy | 0.868 |

### Per-Domain Binary F1

| Domain | F1 | Precision | Recall |
|--------|:--:|:---------:|:------:|
| Outcome Reporting | 0.839 | 0.771 | 0.922 |
| Spin (Boutron) | 0.826 | 0.750 | 0.918 |
| Statistical Reporting | 0.806 | 0.713 | 0.926 |
| Methodology | 0.737 | 0.658 | 0.839 |
| Conflict of Interest | 0.698 | 0.677 | 0.720 |

### Severity Grading (5-level ordinal)

| Metric | Score |
|--------|:-----:|
| Within-One Accuracy | 76.4% |
| Cohen's Kappa (weighted) | 0.124 |

### Output Quality

| Metric | Score |
|--------|:-----:|
| Thinking chains present | 100% |
| Mean reasoning length | 1,289 characters |
| Parse failures | 0% |
| Verification source score | 0.495 |

### Comparison: Fine-tuned 9B vs. Zero-shot Baseline

| Configuration | Binary F1 | Recall | Spin F1 | Verification |
|--------------|:---------:|:------:|:-------:|:------------:|
| **BiasBuster (9B fine-tuned)** | **0.924** | **0.950** | **0.826** | **0.495** |
| Qwen3.5-9B zero-shot (v3 prompt) | 0.895 | 0.904 | 0.321 | 0.464 |

Fine-tuning dramatically improves spin detection (F1 0.321 -> 0.826) and produces structured reasoning chains (0% -> 100% thinking present).

## Verification Databases

The model recommends checks against these specific resources:

- **ClinicalTrials.gov** (99.3% citation rate) — registered vs. published outcomes, sponsor identity, protocol amendments
- **ORCID** (100%) — author employment histories revealing undisclosed affiliations
- **Europe PMC** (100%) — full-text funding and COI disclosure sections
- **Retraction Watch / Crossref** (100%) — post-publication corrections and retractions
- **CMS Open Payments** (21.5%) — undisclosed industry payments to US-based investigators

## Training Details

| Property | Value |
|----------|-------|
| **Base model** | Qwen/Qwen3.5-9B |
| **Method** | LoRA (rank 32, alpha 64, dropout 0.08) |
| **Training data** | 1,235 curated clinical trial abstracts |
| **Data sources** | Retracted papers (Crossref), Cochrane RoB (Europe PMC), PubMed RCTs, ClinicalTrials.gov |
| **Annotation** | Claude + human review |
| **Hardware** | NVIDIA DGX Spark (GB10 Blackwell, 128 GB unified memory) |
| **Training time** | ~2.5 hours (927 steps, 3 epochs) |
| **Precision** | bfloat16 |
| **Framework** | TRL (SFTTrainer) + PEFT |
| **Learning rate** | 2e-4 (cosine with 10% warmup) |
| **Max sequence length** | 4,096 tokens |

## Known Limitations

- **Severity grading is weak** (kappa 0.124): The model reliably detects bias presence but struggles to grade severity, tending to default to MODERATE. This is a class imbalance issue in training data (HIGH/CRITICAL are 3-5% of labels).
- **Abstract-only**: Assesses only the abstract text, not full papers with methods sections, supplementary tables, or detailed disclosures.
- **CMS Open Payments underrecommended** (21.5%): The model underrecommends payment verification for LOW/MODERATE COI cases.
- **Not a replacement for expert review**: BiasBuster is a screening tool that flags abstracts for closer human scrutiny and points reviewers where to look.

## Intended Use

- Screening clinical trial abstracts for potential bias before full-text review
- Assisting systematic reviewers by prioritizing which studies need the closest scrutiny
- Teaching researchers to recognize common bias patterns
- Research integrity workflows that need scalable first-pass assessment

## Out of Scope

- Definitive bias determination without human verification
- Non-clinical-trial study designs (observational, case reports, etc.)
- Full-text analysis (trained on abstracts only)
- Regulatory or legal decisions about research integrity

## Verification Agent

The [BiasBuster repository](https://github.com/hherb/biasbuster) includes a **verification agent** that closes the loop between bias detection and evidence gathering. The model is trained to cite specific databases in its verification recommendations — the agent parses these tool-use recommendations from the model's output and executes them against real databases automatically:

1. **Initial assessment** — the fine-tuned model analyses an abstract and recommends verification steps citing specific databases
2. **Tool routing** — the agent parses verification steps from the model output (JSON or markdown) and routes each to a concrete tool via pattern matching on database names
3. **Concurrent execution** — tools run in parallel against live APIs:
   - **ClinicalTrials.gov** — fetch trial registration, detect outcome switching between registered and published endpoints
   - **CMS Open Payments** — search for undisclosed industry payments to investigators
   - **ORCID** — check author employment histories for undisclosed industry affiliations
   - **Europe PMC** — extract funding and grant metadata from full-text indexing
   - **Retraction Watch / Crossref** — check for retractions, corrections, expressions of concern
   - **Effect size audit** — local heuristic analysis of statistical reporting completeness
4. **Refined assessment** — the model receives all tool results and produces a final, evidence-backed bias assessment

This works because the fine-tuned model reliably names the target databases in its output, enabling keyword-based routing without a secondary LLM call.

```bash
# Clone and set up
git clone https://github.com/hherb/biasbuster.git
cd biasbuster && uv sync

# Run the agent on a single abstract by PMID
uv run python -m agent --pmid 12345678

# Evaluate this model against a test set
uv run python -m evaluation.run \
  --test-set dataset/export/alpaca/test.jsonl \
  --model-a hherb/qwen3.5-9b-biasbuster5 \
  --endpoint-a http://localhost:11434 \
  --mode fine-tuned
```

## Links

- **Source code & agent harness**: [github.com/hherb/biasbuster](https://github.com/hherb/biasbuster)
- **Part of**: BMLibrarian — automated biomedical literature review

## Citation

```bibtex
@software{biasbuster2026,
  title={BiasBuster: Teaching Small Language Models to Detect Bias in Medical Research},
  author={Herb, Horst},
  year={2026},
  url={https://github.com/hherb/biasbuster}
}
```

## License

Apache 2.0

---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - bias-detection
  - biomedical
  - clinical-trials
  - research-integrity
  - lora
  - qlora
  - qwen3.5
  - medical
base_model: Qwen/Qwen3.5-9B
datasets:
  - custom
metrics:
  - f1
  - recall
  - precision
  - accuracy
---

# BiasBuster: Bias Detection in Biomedical Research Abstracts

BiasBuster is a fine-tuned Qwen3.5-9B model that detects bias in clinical trial abstracts across five evidence-based domains. Unlike general-purpose classifiers, BiasBuster produces step-by-step reasoning chains and recommends specific verification databases where each claim can be checked.

## Model Description

| Property | Value |
|----------|-------|
| **Base model** | Qwen/Qwen3.5-9B |
| **Method** | LoRA (rank 32, alpha 64) |
| **Training data** | 1,235 curated clinical trial abstracts |
| **Task** | 5-domain bias assessment with severity grading and verification steps |
| **Hardware** | NVIDIA DGX Spark (GB10 Blackwell, 128 GB unified memory) |
| **Training time** | ~2.5 hours (927 steps, 3 epochs) |
| **Precision** | bfloat16 |

### What It Does

Given a clinical trial abstract, BiasBuster:

1. **Reasons step-by-step** through each bias domain inside `<think>` tags
2. **Assigns severity** (NONE / LOW / MODERATE / HIGH / CRITICAL) per domain
3. **Cites evidence** from the abstract text
4. **Recommends verification steps** pointing to specific databases and search terms

### Five Bias Domains

| Domain | What It Catches |
|--------|----------------|
| **Statistical Reporting** | Sole reliance on relative measures (RRR, OR, HR) without absolute measures (ARR, NNT, baseline risk) |
| **Spin** | Conclusions that overstate or misrepresent actual results (Boutron taxonomy) |
| **Outcome Reporting** | Surrogate endpoints, composite endpoints, outcome switching from registered protocol |
| **Conflict of Interest** | Industry funding without disclosure, undisclosed author affiliations with sponsors |
| **Methodological Red Flags** | Inappropriate comparators, enrichment designs, per-protocol without ITT, premature stopping, short follow-up |

### Verification Databases

The model learns to recommend checks against these specific resources:

- **ClinicalTrials.gov** — registered vs. published outcomes, sponsor identity, protocol amendments
- **CMS Open Payments** — undisclosed industry payments to investigators
- **ORCID** — author employment histories revealing undisclosed affiliations
- **Europe PMC** — full-text funding and COI disclosure sections
- **Retraction Watch / Crossref** — post-publication corrections and retractions
- **Medicines Australia / EFPIA** — non-US physician payment data

## Performance

Evaluated on 144 held-out test abstracts (80/10/10 split, seed 42).

### Binary Detection (Biased vs. Not Biased)

| Metric | Score |
|--------|:-----:|
| **F1** | **0.924** |
| **Recall** | **0.950** |
| Precision | 0.898 |
| Accuracy | 0.896 |

### Per-Domain Binary F1

| Domain | F1 |
|--------|:--:|
| Statistical Reporting | 0.806 |
| Spin (Boutron) | 0.826 |
| Outcome Reporting | 0.839 |
| Conflict of Interest | 0.698 |
| Methodology | 0.737 |

### Severity Grading (5-level ordinal)

| Metric | Score |
|--------|:-----:|
| Cohen's Kappa (weighted) | 0.124 |
| Within-One Accuracy | 76.4% |
| Mean Absolute Error | 1.076 levels |

### Output Quality

| Metric | Score |
|--------|:-----:|
| Thinking chains present | 100% |
| Mean reasoning length | 1,289 characters |
| Parse failures | 0% |
| Verification source score | 0.495 |

### Comparison with Larger Models

| Configuration | Params | Binary F1 | Recall | Ordinal Kappa | Verification |
|--------------|:------:|:---------:|:------:|:-------------:|:------------:|
| Qwen3.5-27B zero-shot | 27B | 0.989 | 1.000 | 0.021 | 0.539 |
| OLMo-3.1-32B fine-tuned | 32B | 0.952 | 0.920 | 0.285 | 0.368 |
| **BiasBuster (Qwen3.5-9B)** | **9B** | **0.924** | **0.950** | 0.124 | 0.495 |

The 9B model matches 32B on binary detection (F1 gap: 0.028), beats it on recall (0.950 vs 0.920), and produces significantly better verification recommendations (0.495 vs 0.368).

## Usage

### With Ollama

```bash
ollama run biasbuster
```

Then paste or pipe a clinical trial abstract. The model will produce a `<think>` reasoning block followed by a structured 5-domain assessment.

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("hherb/biasbuster-qwen3.5-9b")
tokenizer = AutoTokenizer.from_pretrained("hherb/biasbuster-qwen3.5-9b")

system_prompt = """You are a biomedical research integrity analyst. Given a clinical trial abstract,
assess it for potential bias across five domains: Statistical Reporting, Spin,
Outcome Reporting, Conflict of Interest, and Methodological Red Flags.
For each domain, assign a severity level (NONE/LOW/MODERATE/HIGH/CRITICAL),
cite evidence from the abstract, and recommend specific verification steps."""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Assess the following abstract for bias:\n\n" + abstract_text}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=4096)
print(tokenizer.decode(outputs[0]))
```

### Output Format

The model produces structured output with reasoning:

```
<think>
Statistical reporting: The abstract reports HR 0.67 (95% CI 0.52-0.86)
without absolute risk reduction or NNT. No baseline event rates provided...

Spin: The conclusion states "significantly improved outcomes" but the
primary endpoint was not statistically significant (p=0.12)...

Verification: ClinicalTrials.gov should be checked for NCT01234567.
CMS Open Payments should be searched for the lead author...
</think>

## Statistical Reporting: MODERATE
- relative_only: true
- Evidence: "HR 0.67 (95% CI 0.52-0.86, p=0.002)"
- No absolute measures reported

## Spin: HIGH
- spin_level: high
- Evidence: Conclusion recommends clinical use despite non-significant primary endpoint

...

## Recommended Verification Steps
- Search ClinicalTrials.gov for NCT01234567 to compare registered vs published outcomes
- Check CMS Open Payments for Dr. Smith given industry sponsorship
- Search ORCID for author affiliation histories
```

## Training Details

### Data Pipeline

Training data was built using a multi-source ground truth approach:

1. **Retracted papers** (Crossref / Retraction Watch) — known positive examples of flawed research
2. **Cochrane Risk of Bias assessments** (Europe PMC) — expert-level structured judgments
3. **Heuristic-mined PubMed RCTs** — scored by effect-size auditing and funding classification heuristics
4. **ClinicalTrials.gov** — outcome switching detection via registered vs. published endpoint comparison

Abstracts were annotated by Claude with structured 5-domain assessments using operational definitions, then human-reviewed before export to training format.

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 32 |
| Alpha | 64 |
| Dropout | 0.08 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Weight decay | 0.02 |
| Label smoothing | 0.05 |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-4 (cosine with 10% warmup) |
| Effective batch size | 4 (1 per device x 4 gradient accumulation) |
| Epochs | 3 |
| Max sequence length | 4096 tokens |
| Precision | bfloat16 |
| Gradient clipping | 1.0 |
| Framework | TRL (SFTTrainer) + PEFT |

### Training Dynamics

- Training loss: 13.1 → 3.85 (gradual decline, no saturation)
- Eval loss: 1.267 → 1.101 (still declining at epoch 3)
- GPU memory: ~18 GiB allocated, ~43 GiB peak
- 927 optimization steps

## Known Limitations

### Severity Grading (Ordinal Kappa = 0.124)

The model reliably detects the *presence* of bias but struggles to grade its *severity*. It exhibits "moderate collapse" — defaulting to MODERATE for any non-NONE case — due to class imbalance in the training data (MODERATE is 35-50% of labels; HIGH/CRITICAL are 3-5%). This is a data limitation, not a model capacity issue.

### CMS Open Payments Citation Rate (22%)

The model underrecommends CMS Open Payments verification, particularly for LOW/MODERATE COI cases. The training signal for this database is concentrated at HIGH severity (100% citation rate) with weak signal at lower severities (13-16%).

### Abstract-Only Assessment

The model assesses only the abstract text. Full-text analysis — methods sections, funding disclosures, supplementary statistical tables — would enable more thorough assessment. Abstracts contain only a fraction of the information available in the full paper.

### Test Set Size

The evaluation test set (144 examples) is relatively small. Per-dimension metrics should be interpreted with appropriate confidence intervals.

### Not a Replacement for Expert Review

BiasBuster is a screening tool — a first pass that flags abstracts deserving careful human scrutiny and points the reviewer where to look. It is not a replacement for systematic review methodology or peer review.

## Intended Use

- **Screening clinical trial abstracts** for potential bias before full-text review
- **Assisting systematic reviewers** by prioritizing which studies need the closest scrutiny
- **Teaching and training** researchers to recognize common bias patterns
- **Research integrity** workflows that need scalable first-pass assessment

## Out of Scope

- Definitive bias determination without human verification
- Assessment of non-clinical-trial study designs (observational, case reports, etc.)
- Full-text analysis (trained on abstracts only)
- Regulatory or legal decisions about research integrity

## Citation

```bibtex
@software{biasbuster2026,
  title={BiasBuster: Teaching Small Language Models to Detect Bias in Medical Research},
  author={Herb, Horst},
  year={2026},
  url={https://github.com/hherb/biasbuster}
}
```

## Acknowledgments

Part of the BMLibrarian project for automated biomedical literature review. Built on the NVIDIA DGX Spark platform with Qwen3.5-9B, LoRA fine-tuning via TRL/PEFT, and Ollama for inference.

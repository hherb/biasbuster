# BiasBuster: A Multi-Source Pipeline for Building Training Datasets to Detect Bias in Biomedical Research Abstracts

**Authors:** Herb, Horst; Brinkmann, Bernd; Herb, Frithjof; Herb, Hagen; Claude(Opus 4.6)

**Corresponding author:** Horst Herb <hherb@consensus-ai.org>

**Word count:** [~5,500 main text - consider trimming for submission target]

---

## Abstract

**Background:** Bias in biomedical research remains a pervasive threat to evidence-based medicine. While tools exist for expert-guided risk-of-bias assessment, automated screening at scale has been limited by the absence of large, high-quality training datasets with multi-dimensional bias annotations.

**Objective:** We describe BiasBuster, an open-source pipeline for constructing curated training datasets to fine-tune large language models (LLMs) for detecting bias in biomedical abstracts across five domains: statistical reporting, spin, outcome reporting, conflict of interest, and methodology.

**Methods:** The pipeline integrates five data sources: retracted papers via Crossref/Retraction Watch (known-biased positives), Cochrane Risk of Bias 2.0 assessments via Europe PMC (expert ground truth), PubMed randomized controlled trials filtered by MeSH domain (general population), ClinicalTrials.gov registry data (outcome switching detection), and CMS Open Payments/ORCID (conflict of interest verification). Abstracts undergo heuristic enrichment (effect size auditing, funding classification) before structured annotation by multiple LLMs using identical prompts. Inter-model agreement is evaluated with Cohen's kappa, McNemar's test, and Wilcoxon signed-rank tests. Training data is exported in three fine-tuning formats with chain-of-thought reasoning and actionable verification steps.

**Results:** We present the system architecture, annotation taxonomy, and evaluation across four iterative fine-tuning runs plus expanded baseline comparison. Zero-shot baselines spanning 8B to 32B parameters were evaluated: Qwen 3.5-27B and OLMo-3.1-32B-Instruct achieved binary F1 of 0.989 with perfect recall but near-chance severity calibration (weighted κ = 0.021 and 0.066); OpenAI's gpt-oss:20b (a 21B Mixture-of-Experts model with 3.6B active parameters per token) achieved F1 of 0.918 with κ = 0.158 and verification score of 0.591 without any fine-tuning; IBM's granite3.3:8b failed catastrophically (F1 = 0.022, recall = 1.1%). LoRA fine-tuning of OLMo-3.1-32B improved severity grading (κ 0.066 → 0.285) and COI detection (F1 0.667 → 0.927) but degraded verification source citations (CMS Open Payments 85% → 16%). Enriching the training pipeline---expanding system prompts, synthesising verification reasoning in thinking chains, and growing the dataset from 706 to 1,235 examples---restored verification citations and shifted the dominant improvement axis to training data quality. The final Qwen3.5-9B fine-tuned model achieved binary F1 of 0.924, recall of 0.950, and per-dimension F1 of 0.70--0.84 across all five domains, with 100% chain-of-thought reasoning. The unfine-tuned gpt-oss:20b MoE model's strong baseline performance (F1 0.918, verification 0.591) identifies it as a high-priority fine-tuning candidate whose MoE architecture may be particularly well-suited to multi-domain bias detection. Severity calibration remained the primary limitation across all configurations (κ = 0.12--0.29), driven by class imbalance in the training data producing systematic over-prediction of moderate severity.

**Conclusions:** BiasBuster demonstrates that a 9B-parameter model fine-tuned on 1,235 verification-focused examples can detect five dimensions of bias in clinical trial abstracts with F1 above 0.70 per dimension and 95% recall, while producing actionable verification recommendations citing specific databases. The strong zero-shot performance of gpt-oss:20b (F1 0.918) suggests that Mixture-of-Experts architectures may be inherently well-suited to multi-domain bias detection, and that fine-tuning such models could yield further gains. Training data quality---not model size or hyperparameter tuning---was the dominant factor across four iterative runs. The pipeline's modular design enables community extension to additional data sources, bias domains, and model architectures.

**Keywords:** bias detection, risk of bias, research integrity, large language models, training data, systematic review, spin, conflict of interest, outcome reporting

---

## Introduction

Bias in biomedical research undermines the reliability of evidence that guides clinical decision-making. Despite decades of methodological guidance and the development of structured assessment tools such as Cochrane's Risk of Bias 2.0 (RoB 2) instrument,^1^ bias remains prevalent across clinical trial literature.^2,3^ Statistical reporting bias---including selective emphasis on relative risk reductions without absolute measures---affects interpretation of treatment effects.^4^ Spin in abstracts misrepresents study findings.^5,6^ Outcome switching between trial registration and publication obscures negative results.^7^ Undisclosed conflicts of interest compromise objectivity.^8^

Manual bias assessment is labour-intensive and requires domain expertise, limiting its application to systematic reviews and guideline development. The emergence of large language models (LLMs) offers an opportunity to automate bias screening at scale, but this requires high-quality training data with structured, multi-dimensional annotations---a resource that does not currently exist in the public domain.

We present BiasBuster, an open-source pipeline that constructs training datasets for fine-tuning LLMs to detect bias in biomedical abstracts. The system makes three contributions:

1. **Multi-source ground truth construction.** By combining retracted papers (known positives), Cochrane RoB assessments (expert judgments), and heuristically screened RCTs (diverse signal), the pipeline provides training examples spanning the full spectrum of bias severity.

2. **Verification-focused annotations.** Unlike binary bias labels, our annotation schema teaches models *where to look*---citing specific databases (CMS Open Payments, ClinicalTrials.gov, ORCID) and providing actionable verification steps for each bias domain.

3. **Reproducible multi-model evaluation.** The pipeline supports annotation by multiple LLMs using identical prompts, enabling direct comparison with statistical tests and facilitating consensus-based labelling.

---

## Methods

### Overview

BiasBuster is implemented in Python as a five-stage asynchronous pipeline: (1) collection of abstracts from external APIs, (2) heuristic enrichment to stratify abstracts by suspicion level, (3) structured annotation by one or more LLMs, (4) human review and validation, and (5) export to fine-tuning formats. All data is stored in a single SQLite database with schema-enforced uniqueness and write-ahead logging for concurrent access. The full source code is available at [repository URL].

### Data Sources and Collection

#### Retracted Papers (Known Positives)

Retracted papers serve as high-confidence positive examples of biased research. We query the Crossref API using the Retraction Watch database, filtering for papers with the "retraction" update type. For each retraction, we retrieve the original paper's abstract via PubMed, capturing the research content as it appeared before retraction. Retraction reasons (e.g., data fabrication, manipulation of results, unreliable findings) are preserved as metadata.

Critically, we distinguish between *retraction notices*---brief editorial statements announcing the retraction---and *original papers that were later retracted*. Retraction notices contain no assessable scientific content and are filtered using pattern matching (title contains "Retracted:" or "Withdrawal:" and abstract length < 200 characters). Original papers with full abstracts are retained as valuable training examples where the abstract itself exhibits the biases that ultimately led to retraction.

#### Cochrane Risk of Bias Assessments (Expert Ground Truth)

Cochrane systematic reviews represent the gold standard for bias assessment in clinical research. We search Europe PMC for Cochrane reviews and parse their full-text XML to extract included study references and corresponding RoB 2 domain ratings. For each included study, we retrieve the original abstract from PubMed and associate it with the Cochrane reviewers' assessments across five RoB 2 domains: randomization process, deviations from intended interventions, missing outcome data, outcome measurement, and selection of reported results. Each domain is rated as low risk, some concerns, or high risk, with an overall judgment.

This source provides expert-validated examples at both ends of the bias spectrum: studies judged as "low risk" across all domains serve as negative examples, while those rated "high risk" serve as positive examples with domain-specific explanations.

#### PubMed Randomized Controlled Trials

To ensure adequate representation of common clinical domains, we search PubMed for recent randomized controlled trials (2020--present) across seven Medical Subject Heading (MeSH) focus domains: cardiovascular diseases, neoplasms, diabetes mellitus, mental disorders, musculoskeletal diseases, respiratory tract diseases, and anti-infective agents. Abstracts are retrieved in batches using the NCBI E-utilities API with appropriate rate limiting.

#### ClinicalTrials.gov Registry Data

For abstracts flagged as high-suspicion by heuristic enrichment, we query the ClinicalTrials.gov v2 API to detect outcome switching---changes to primary outcomes between trial registration and publication. We extract registered outcomes, sponsors, funding sources, and protocol amendment histories. Outcome matching between registered and published endpoints uses text similarity scoring with confidence grading (low, medium, high). Binary flags are generated for primary outcome switching, outcome omission, and retrospective registration.

#### Conflict of Interest Verification Sources

The pipeline integrates data from CMS Open Payments (physician payment disclosures), ORCID (author affiliations and employment history), and Europe PMC (funder metadata). These sources provide independent verification of conflict of interest disclosures in abstracts and are referenced in annotation verification steps.

### Heuristic Enrichment

Before LLM annotation, abstracts undergo automated pre-screening to stratify them by suspicion level, enabling efficient allocation of annotation resources.

#### Effect Size Auditing

The effect size auditor evaluates statistical reporting patterns using regular expression-based detection of relative measures (hazard ratio, odds ratio, relative risk, relative risk reduction, percentage reductions) and absolute measures (absolute risk reduction, absolute risk difference, number needed to treat/harm, baseline risk, raw event counts in both arms).

A composite reporting bias score (0--1) is computed from weighted indicators:

- Sole reliance on relative measures without absolute context: +0.4
- Emphasis on relative measures with minimal absolute reporting: +0.2
- Percentage reductions stated without baseline context: +0.15
- Relative effect sizes in title: +0.1
- Absence of number needed to treat, baseline risk, or confidence intervals: +0.1 each

Abstracts scoring >= 0.3 are classified as "high suspicion," those < 0.1 as "low suspicion," and those in between as "medium suspicion."

#### Funding Classification

The funding checker classifies funding sources as industry, public, mixed, not reported, or unclear, using a curated database of over 50 pharmaceutical company name variants and major public funding bodies (NIH, NSF, MRC, Wellcome Trust, NHMRC, and others). Classification is based on text matching against abstract content and structured metadata fields.

### Annotation

#### Bias Taxonomy

We developed a structured annotation schema covering five bias domains, informed by the Cochrane RoB 2 framework,^1^ the Boutron spin classification,^5,6^ and ICMJE disclosure requirements:^9^

1. **Statistical reporting bias.** Evaluates selective use of effect size measures: relative-only reporting (RRR, OR, HR without ARR, NNT, or baseline risk), selective p-value reporting, confidence interval completeness, and subgroup emphasis.

2. **Spin** (Boutron classification). Assesses discordance between results and conclusions: causal language from observational designs, focus on secondary outcomes when primary outcomes are non-significant, inappropriate extrapolation, and title spin. Spin is graded as none, low, moderate, or high following the Boutron taxonomy.

3. **Outcome reporting.** Evaluates outcome type (patient-centred vs. surrogate vs. composite), outcome switching relative to trial registration, use of unvalidated surrogate endpoints, and disaggregation of composite endpoints.

4. **Conflict of interest.** Assesses funding type, disclosure completeness, industry author affiliations, ghost authorship indicators, and cross-references with payment databases. Verification sources (CMS Open Payments, ORCID, EFPIA disclosure databases) are specified for each flagged concern.

5. **Methodological red flags.** Evaluates comparator appropriateness (placebo vs. active vs. standard of care), enrichment design, per-protocol-only analysis, premature stopping, blinding adequacy, and follow-up duration relative to domain-specific thresholds (e.g., < 12 months for chronic disease, < 4 weeks for acute conditions, < 30 days for surgical outcomes).

Each domain produces a severity rating (none, low, moderate, high, critical), evidence quotes from the abstract, and an overall bias probability (0.0--1.0) with free-text reasoning.

#### Operational Definitions

To reduce inter-annotator (and inter-model) disagreement, the annotation prompt includes nine explicit operational definitions addressing common ambiguities:

- **Absolute vs. relative measures:** Raw event counts in both treatment arms or percentages reported for both arms constitute absolute measures. Relative-only is flagged only when hazard ratios, odds ratios, or relative risk reductions are the *sole* effect size measures reported.
- **Surrogate vs. patient-centred outcomes:** Patient-centred outcomes include mortality, quality of life, and functional status. Surrogate outcomes include laboratory values, imaging findings, and dose modifications.
- **Follow-up adequacy:** Domain-specific thresholds define inadequate follow-up (e.g., < 12 months for chronic conditions, < 30 days for surgical outcomes).
- **COI disclosure:** Funding source identification alone does not constitute COI disclosure; an explicit author-level conflict of interest statement is required.

#### Multi-Model Annotation

The pipeline supports annotation by multiple LLMs using identical system prompts and input formatting to ensure comparability. Two backends are implemented:

1. **Anthropic Claude** (via the Anthropic async SDK), using the `claude-sonnet-4-6` model with a structured system prompt (~2,900 words) specifying the full bias taxonomy, operational definitions, and output JSON schema.

2. **OpenAI-compatible APIs** (via HTTP), supporting DeepSeek, vLLM, SGLang, and other compatible endpoints. Temperature is set to 0.1 to reduce response variability.

Both backends share identical prompt construction (`build_user_message`), JSON output parsing (with repair for common formatting errors), and incremental persistence---each annotation is saved to the database immediately upon completion, enabling checkpoint/resume for long-running annotation batches.

### Human Review

Between automated annotation and export, a human reviewer validates annotations using a web-based review tool built with NiceGUI. The interface provides an editable grid displaying PMIDs, titles, severity ratings, bias probabilities, and reasoning summaries. Reviewers can mark annotations as validated, override severity ratings, and add notes. All review decisions are stored in a dedicated `human_reviews` table linked to the annotation by composite key (PMID, model name).

### Training Data Export

Validated annotations are exported to three fine-tuning formats:

1. **Alpaca format** (instruction-tuning with `<think>` reasoning chains): The system prompt defines the model's role as a bias detection assistant. The instruction contains the abstract and metadata. The output includes a `<think>` block with step-by-step reasoning (extracted from the annotation's reasoning field or synthesized from domain assessments), followed by the structured JSON assessment. This format is compatible with Unsloth, TRL SFTTrainer, and similar frameworks.

2. **ShareGPT format** (multi-turn conversation): System, human, and assistant messages.

3. **OpenAI chat format** (role-based messages): System, user, and assistant roles.

Data is split deterministically (seed = 42) into training (80%), validation (10%), and test (10%) sets. The `<think>` reasoning chains are a key feature---they teach the fine-tuned model to reason step-by-step before producing structured assessments, following recent advances in chain-of-thought training.^10,11^

### Evaluation Framework

#### Metrics

Model performance is evaluated across three levels:

**Binary classification** (any bias concern vs. none): precision, recall, F1 score, and accuracy, computed per dimension and overall.

**Ordinal severity agreement:** Cohen's weighted kappa for the five-level severity scale (none, low, moderate, high, critical), mean absolute error, exact match rate, and within-one agreement rate.

**Flag-level accuracy:** Per-flag binary accuracy for specific indicators (e.g., relative_only, baseline_risk_reported, spin_level, funding_type, outcome_switching_detected).

**Calibration:** Expected calibration error comparing predicted bias probabilities against observed rates.

**Verification quality:** Coverage of recommended verification sources in model outputs.

#### Statistical Comparison

For head-to-head model comparison, we employ:

- **McNemar's test** for binary classification disagreements (exact binomial for n < 25 discordant pairs, chi-squared approximation otherwise).
- **Wilcoxon signed-rank test** for ordinal severity differences.
- **Effect sizes** computed as differences in F1, kappa, and calibration error.

Per-dimension comparison identifies which model excels at which bias type, generating radar-plot-ready data and a structured Markdown report with significance markers.

---

## Results

### Pipeline Implementation

The complete pipeline is implemented in approximately 8,000 lines of Python across 25 modules, organized into five packages (collectors, enrichers, annotators, evaluation, utilities). The system uses asynchronous I/O throughout the collection and annotation stages with configurable rate limiting and concurrency controls. All pipeline state is persisted in a single SQLite database with four tables (papers, enrichments, annotations, human_reviews) linked by PubMed ID.

### Data Source Characteristics

Table 1 summarizes the characteristics and intended roles of each data source.

**Table 1. Data sources and their roles in the training dataset.**

| Source | API | Target Papers | Role | Expected Bias Signal |
|--------|-----|---------------|------|---------------------|
| Retraction Watch / Crossref | Crossref REST API | Up to 2,000 | Known positives | Confirmed research integrity failures |
| Cochrane RoB 2 | Europe PMC | Up to 1,000 studies from 50 reviews | Expert ground truth | Domain-specific severity ratings |
| PubMed RCTs | NCBI E-utilities | Up to 5,000 | General population | Full spectrum (enrichment-stratified) |
| ClinicalTrials.gov | CT.gov v2 API | High-suspicion subset | Outcome switching enrichment | Registry--publication discrepancies |
| CMS Open Payments / ORCID | CMS, ORCID APIs | Flagged authors | COI verification | Payment and affiliation data |

### Annotation Schema Coverage

The five-domain annotation schema produces structured assessments with per-domain severity ratings, evidence quotes, and verification steps. Each annotation includes:

- 5 domain-level severity ratings (5-point ordinal scale)
- 15+ binary flags across domains (e.g., relative_only, spin_level, outcome_switching_detected)
- Free-text reasoning suitable for `<think>` chain-of-thought training
- Actionable verification steps citing specific databases
- Overall bias probability (continuous, 0.0--1.0)

### Enrichment Pre-Screening

The effect size auditor stratifies abstracts into three suspicion levels based on statistical reporting patterns. In preliminary analysis of PubMed RCTs, approximately 30--40% of abstracts score >= 0.3 (high suspicion), reflecting the well-documented prevalence of relative-only reporting in clinical trial abstracts.^4^ This pre-screening directs annotation resources toward abstracts most likely to exhibit reporting bias while ensuring representation of low-suspicion examples as negative training data.

### Zero-Shot Baseline Evaluation

To establish baseline performance before fine-tuning, we evaluated four open-weight models in zero-shot mode: Qwen 3.5-27B (q8_0 quantization, n = 89), OLMo-3.1-32B-Instruct (q8_0 quantization, n = 89), OpenAI gpt-oss:20b (n = 157), and IBM granite3.3:8b (n = 101). The first two were served via Ollama on a DGX Spark; the latter two on an Apple M3 Mac with 128 GB unified memory.

**Table 2. Zero-shot baseline: overall performance.**

| Metric | granite3.3 8B | Qwen 3.5-27B | OLMo-3.1-32B | gpt-oss:20b (MoE) |
|--------|:---:|:---:|:---:|:---:|
| Binary F1 | 0.022 | 0.989 | 0.989 | 0.918 |
| Precision | 1.000 | 0.978 | 0.978 | 0.895 |
| Recall | 0.011 | 1.000 | 1.000 | 0.941 |
| Severity κ (weighted) | 0.004 | 0.021 | 0.066 | **0.158** |
| Calibration error | 0.557 | 0.404 | 0.670 | 0.866 |
| Verification score | 0.435 | 0.539 | 0.528 | **0.591** |
| Parse failures | 0 | 0 | 0 | 0 |
| n (test examples) | 101 | 89 | 89 | 157 |

Baseline performance varied dramatically by model. Granite3.3:8b failed catastrophically, predicting NONE for nearly all dimensions (recall = 1.1%), indicating insufficient pretraining exposure to biomedical bias assessment concepts. Qwen 3.5-27B and OLMo-3.1-32B achieved identical binary classification (F1 = 0.989) with perfect recall, but their severity calibration was poor (κ = 0.021 and 0.066). OpenAI's gpt-oss:20b---a Mixture-of-Experts model with 21B total parameters but only 3.6B active per token---achieved the strongest severity calibration of any baseline (κ = 0.158) and the highest verification source citation rate (0.591), despite being evaluated on the largest test set (157 examples). Its lower binary F1 (0.918 vs. 0.989) likely reflects the harder, larger test set rather than weaker detection capability.

**Table 3. Zero-shot baseline: per-dimension binary F1 scores.**

| Dimension | granite3.3 8B | Qwen 3.5-27B | OLMo-3.1-32B | gpt-oss:20b |
|-----------|:---:|:---:|:---:|:---:|
| Statistical reporting | 0.027 | 0.853 | 0.846 | 0.805 |
| Spin (Boutron) | 0.000 | 0.921 | 0.896 | 0.748 |
| Outcome reporting | 0.000 | 0.950 | 0.940 | 0.752 |
| Conflict of interest | 0.000 | 0.928 | 0.667 | 0.751 |
| Methodology | 0.028 | 0.863 | 0.852 | 0.793 |

The only statistically significant difference among the viable models was in conflict of interest detection, where Qwen substantially outperformed OLMo (F1 0.928 vs. 0.667, p < 0.05). OLMo's low COI F1 was driven by poor recall (0.500 vs. 0.917). gpt-oss:20b showed balanced performance across all dimensions (0.75--0.81), with no single dimension falling below 0.70.

**Table 4. Zero-shot baseline: ordinal severity agreement.**

| Metric | granite3.3 8B | Qwen 3.5-27B | OLMo-3.1-32B | gpt-oss:20b |
|--------|:---:|:---:|:---:|:---:|
| Mean absolute error | 1.762 | 1.281 | 0.584 | **0.949** |
| Exact match | 11.9% | 14.6% | 51.7% | 32.5% |
| Within-one agreement | 28.7% | 64.0% | 89.9% | 80.3% |
| Weighted kappa | 0.004 | 0.021 | 0.066 | **0.158** |

OLMo showed the best ordinal calibration by MAE and within-one agreement, but gpt-oss:20b achieved the highest weighted kappa (0.158), indicating its severity ratings, while less tightly clustered around ground truth, captured ordinal structure more reliably. Granite3.3's near-zero kappa (0.004) confirms its complete failure to engage with the severity scale.

**Table 5. Zero-shot baseline: verification source knowledge.**

| Source | granite3.3 8B | Qwen 3.5-27B | OLMo-3.1-32B | gpt-oss:20b |
|--------|:---:|:---:|:---:|:---:|
| CMS Open Payments | 72% | 98% | 85% | **96%** |
| ClinicalTrials.gov | 94% | 98% | 99% | 94% |
| ORCID | 76% | 88% | 93% | **97%** |
| Retraction Watch | 73% | 100% | 96% | 96% |
| Europe PMC | 75% | 100% | 100% | **97%** |

gpt-oss:20b demonstrated the most balanced verification source knowledge, with all five databases above 94%. It notably outperformed OLMo on CMS Open Payments (96% vs. 85%) and ORCID (97% vs. 93%). None of the baseline models generated extended reasoning chains (`<think>` blocks) in zero-shot mode, confirming the need for chain-of-thought fine-tuning.

Inference latency on the DGX Spark averaged 150.8 seconds per abstract for Qwen and 87.5 seconds for OLMo, at comparable throughput (~6.5 tokens/second). On the M3 Mac, gpt-oss:20b averaged 76.7 seconds at 31.8 tokens/second---its MoE architecture (3.6B active parameters) gives it dense-4B-class inference speed despite accessing 21B of learned knowledge. All models achieved a 0% error rate across their respective test sets.

### Fine-Tuning: Iterative Improvement Across Four Runs

We conducted four LoRA fine-tuning runs over three days, each informed by the failures of the previous run. All training used the same hardware (DGX Spark), framework (TRL SFTTrainer), and base LoRA configuration (target modules: q, k, v, o, gate, up, down; bf16 precision; max sequence length 4,096 tokens; cosine learning rate schedule with 10% warmup). Table 6 summarises the configuration changes across runs.

**Table 6. Fine-tuning configuration across four runs.**

| Parameter | Run 1 (OLMo-32B) | Run 2 (Qwen-9B) | Run 3 (Qwen-9B) | Run 4 (Qwen-9B) |
|-----------|:-:|:-:|:-:|:-:|
| Base model | OLMo-3.1-32B | Qwen3.5-9B | Qwen3.5-9B | Qwen3.5-9B |
| Training examples | 706 | 706 | 1,235 | 1,235 |
| LoRA rank / alpha | 16 / 32 | 16 / 32 | 32 / 64 | 32 / 64 |
| Learning rate | 2×10⁻⁴ | 2×10⁻⁴ | 4×10⁻⁴ | 2×10⁻⁴ |
| Epochs | 3 | 3 | 5 | 3 |
| Effective batch size | 4 | 4 | 2 | 4 |
| LoRA dropout | 0.05 | 0.05 | 0.08 | 0.08 |
| Weight decay | 0.0 | 0.0 | 0.02 | 0.02 |
| Label smoothing | 0.0 | 0.0 | 0.05 | 0.05 |
| Total steps | 531 | 690 | 3,090 | 927 |
| Training time | ~4.5 h | ~2.5 h | ~8 h | ~2.5 h |

#### Run 1: OLMo-3.1-32B (Baseline Fine-Tune)

The initial run fine-tuned OLMo-3.1-32B-Instruct with a minimal training system prompt (~320 tokens) and incomplete thinking chains covering only 3 of 5 bias domains. Training loss decreased from 2.28 to 0.55 over 531 steps with peak GPU memory of 69.5 GiB.

**Table 7. Run 1: OLMo-3.1-32B fine-tuned vs. zero-shot baseline (n = 89).**

| Metric | Baseline | Fine-Tuned | Δ |
|--------|:---:|:---:|:---:|
| Binary F1 | 0.989 | 0.952 | -0.037 |
| Precision | 0.978 | 0.988 | +0.010 |
| Recall | 1.000 | 0.920 | -0.080 |
| Ordinal κ | 0.066 | **0.285** | **+0.219** |
| COI F1 | 0.667 | **0.927** | **+0.260** |
| Verification score | 0.528 | 0.368 | -0.160 |
| Thinking chains | 0% | 100% | — |

Fine-tuning dramatically improved severity grading (κ 0.066 → 0.285) and COI detection (F1 0.667 → 0.927) while introducing 100% chain-of-thought reasoning. However, binary F1 decreased slightly (-0.037), and verification source citations regressed severely: CMS Open Payments dropped from 85% to 16%, and Retraction Watch from 96% to 43%. Root cause analysis identified three deficiencies in the training pipeline: (1) the training system prompt lacked operational definitions present in the annotation prompt, (2) thinking chains were incomplete (3 of 5 domains), and (3) verification steps were passed through from annotations without synthesis, leaving many examples with sparse or missing database references.

A concurrent experiment revealed a critical finding: running the smaller Qwen3.5-9B model (no fine-tuning) with an enriched system prompt containing operational definitions produced binary F1 of 0.866 on the full test set---demonstrating that prompt engineering, not model size, was the dominant factor for coarse detection.

#### Run 2: Qwen3.5-9B with Enriched Training Data

Based on Run 1's findings, we enriched the training pipeline: (1) expanded the training system prompt to ~800 tokens with operational definitions and verification database criteria, (2) extended thinking chains to all 5 domains with database selection reasoning, and (3) synthesised missing verification steps from annotation metadata. The 9B model was fine-tuned with identical hyperparameters to Run 1 (controlling for data quality improvements).

**Table 8. Run 2: Qwen3.5-9B fine-tuned vs. enriched prompt baseline vs. Run 1 32B (n = 115).**

| Metric | 9B Fine-Tuned | 9B Enriched Prompt | 32B Fine-Tuned (Run 1) |
|--------|:---:|:---:|:---:|
| Binary F1 | 0.804 | 0.866 | 0.952 |
| Recall | 0.679 | 0.793 | 0.920 |
| Per-dim F1 (avg) | **0.70** | 0.44 | 0.83 |
| Ordinal κ | **0.159** | 0.118 | 0.285 |
| Verification score | **0.541** | 0.495 | 0.368 |
| Thinking chains | 99% | 0% | 100% |

Two findings emerged. First, the enriched training data decisively fixed verification citations: CMS Open Payments recovered from 16% (Run 1) to 57%, and Retraction Watch from 43% to 95%. Second, fine-tuning and prompt engineering solved different problems: the enriched prompt achieved higher overall recall (0.793 vs. 0.679) but its per-dimension F1 scores were 0.39--0.50 (near chance), while the fine-tuned model's per-dimension F1 of 0.64--0.76 reflected genuine understanding of the five bias domains, not blanket flagging. The fine-tuned 9B model's recall of 0.679, however, was unacceptably low for a screening tool.

#### Run 3: Failed Aggressive Hyperparameters

Simultaneously with expanding the training dataset from 706 to 1,235 examples (+75%), we tested 9B-optimised hyperparameters: higher learning rate (4×10⁻⁴), more epochs (5), smaller effective batch (2), and higher LoRA rank (32). The dataset expansion included three format changes: all five domains always emitted (including explicit NONE assessments with substantive reasoning), and rare severity classes (HIGH/CRITICAL) oversampled to ~5% of the training set.

Training curves revealed early saturation: loss dropped from 7.0 to 2.5 in the first 100 of 3,090 steps, then plateaued at 1.5--2.0 for the remaining 90% of training. The 4×10⁻⁴ learning rate combined with the small effective batch drove the model to a loss basin from which additional epochs could not escape. This run was not evaluated, as the training dynamics indicated no useful learning beyond step ~300.

#### Run 4: Conservative Hyperparameters with Expanded Data

We reverted learning dynamics to the Run 1/2 defaults (2×10⁻⁴ LR, 3 epochs, effective batch 4) while retaining the 9B-specific LoRA capacity and regularisation from Run 3 (rank 32, dropout 0.08, weight decay 0.02, label smoothing 0.05). Training loss declined gradually through all 927 steps with no saturation. Eval loss improved steadily from 1.267 to 1.101 and was still declining at completion.

**Table 9. Run 4 evaluation results vs. success criteria and prior runs (n = 144).**

| Metric | Target | Run 4 (9B) | Run 2 (9B) | Run 1 (32B) |
|--------|:---:|:---:|:---:|:---:|
| Binary F1 | > 0.90 | **0.924** ✓ | 0.804 | 0.952 |
| Recall | > 0.85 | **0.950** ✓ | 0.679 | 0.920 |
| Ordinal κ | > 0.20 | 0.124 ✗ | 0.159 | 0.285 |
| Verification | > 0.50 | 0.495 (marginal) | 0.541 | 0.368 |

**Table 10. Run 4: per-dimension binary F1 (all improved over Run 2).**

| Dimension | Run 2 (9B) | Run 4 (9B) | Δ |
|-----------|:---:|:---:|:---:|
| Statistical reporting | 0.730 | **0.806** | +0.076 |
| Spin | 0.727 | **0.826** | +0.099 |
| Outcome reporting | 0.755 | **0.839** | +0.084 |
| COI | 0.639 | **0.698** | +0.059 |
| Methodology | 0.656 | **0.737** | +0.081 |

The Run 2 recall problem was solved (0.679 → 0.950), and the 9B model now nearly matched the 32B on binary detection (F1 gap narrowed from 0.148 to 0.028), while exceeding it on recall (0.950 vs. 0.920) and verification quality (0.495 vs. 0.368).

### Cross-Model Summary

Table 11 summarises all evaluated configurations across four fine-tuning runs and expanded baseline comparison.

**Table 11. Cross-model performance summary.**

| Configuration | Type | Size | n | Binary F1 | Recall | Ordinal κ | Verification | Thinking |
|--------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| granite3.3:8b baseline | Baseline | 8B | 101 | 0.022 | 0.011 | 0.004 | 0.435 | 0% |
| Qwen3.5-27B baseline | Baseline | 27B | 89 | 0.989 | 1.000 | 0.021 | 0.539 | 0% |
| OLMo-3.1-32B baseline | Baseline | 32B | 89 | 0.989 | 1.000 | 0.066 | 0.528 | 0% |
| **gpt-oss:20b baseline** | **Baseline** | **20B MoE** | **157** | **0.918** | **0.941** | **0.158** | **0.591** | **0%** |
| Qwen3.5-9B enriched prompt | Prompt eng. | 9B | 115 | 0.866 | 0.793 | 0.118 | 0.495 | 0% |
| Qwen3.5-9B fine-tuned (Run 2) | Fine-tuned | 9B | 115 | 0.804 | 0.679 | 0.159 | **0.541** | 99% |
| OLMo-3.1-32B fine-tuned (Run 1) | Fine-tuned | 32B | 89 | 0.952 | 0.920 | **0.285** | 0.368 | 100% |
| **Qwen3.5-9B fine-tuned (Run 4)** | **Fine-tuned** | **9B** | **144** | **0.924** | **0.950** | 0.124 | 0.495 | **100%** |

The inclusion of gpt-oss:20b reveals a striking finding: this unfine-tuned MoE model outperforms all baselines on severity calibration (κ = 0.158) and verification quality (0.591), and achieves binary F1 (0.918) competitive with the fine-tuned models---without any domain-specific training. Its 32-expert architecture with top-4 routing may be particularly well-suited to multi-domain bias detection, where different bias domains could activate different expert subnetworks. This identifies gpt-oss:20b as a high-priority fine-tuning candidate.

### Severity Calibration Analysis

Ordinal kappa remained in the 0.12--0.29 range across all four runs regardless of model size or hyperparameters. Confusion matrix analysis revealed a systematic "moderate collapse": when the model detected bias, it defaulted to predicting MODERATE severity because MODERATE was the modal non-NONE class in the training data (50% of Statistical Reporting labels, 36% of COI, 35% of Outcome Reporting). For example, in Statistical Reporting, only 2 of 36 true-LOW examples were predicted correctly---the remainder were predicted as MODERATE (24) or HIGH (7).

**Table 12. Training data severity distribution (1,235 examples, Runs 3--4).**

| Dimension | NONE | LOW | MODERATE | HIGH | CRITICAL |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Statistical Reporting | 36% | 9% | 50% | 4% | 1% |
| Spin | 41% | 42% | 13% | 4% | 0% |
| Outcome Reporting | 39% | 22% | 35% | 3% | 2% |
| COI | 29% | 30% | 36% | 4% | 1% |
| Methodology | 49% | 25% | 21% | 3% | 2% |

HIGH and CRITICAL classes had only ~11 examples per dimension per class after oversampling---insufficient for reliable ordinal boundary learning.

### Verification Source Citation Patterns

Verification source knowledge showed a complex trajectory across runs. The enriched training pipeline (Runs 2--4) restored citations that Run 1 had degraded, but CMS Open Payments regressed in Run 4.

**Table 13. Verification source citation rates across runs.**

| Source | Baselines | Run 1 (32B) | Run 2 (9B) | Run 4 (9B) |
|--------|:---:|:---:|:---:|:---:|
| ClinicalTrials.gov | 98--99% | 89% | 99% | 99% |
| ORCID | 88--93% | 87% | 94% | 100% |
| Retraction Watch | 96--100% | 43% | 95% | 100% |
| Europe PMC | 100% | 97% | 98% | 100% |
| CMS Open Payments | 85--98% | **16%** | 57% | **22%** |

CMS Open Payments collapsed in Run 4 (57% → 22%) despite improving elsewhere. Training data analysis revealed the cause: Open Payments was cited in only 29.8% of training examples, with a steep skew by COI severity---100% citation rate for HIGH COI but only 16% for LOW and 13% for NONE. The model learned the strong HIGH-COI association and discarded the weaker signals.

### Training Data Format

The export module produces training examples in three formats. The Alpaca format with `<think>` chains is illustrated below:

```json
{
  "system": "You are a biomedical research bias detection assistant...",
  "instruction": "Analyze the following abstract for potential bias...\n\nTitle: ...\nAbstract: ...\nJournal: ...\nAuthors: ...",
  "output": "<think>\nThe abstract reports a 52% relative risk reduction in the primary endpoint but does not provide the absolute risk reduction or baseline event rate. Without knowing the control group event rate, the clinical significance cannot be assessed. The number needed to treat is not reported. Funding is from [Pharma Company] but no author-level COI statement appears in the abstract...\n</think>\n{\"statistical_reporting\": {\"severity\": \"moderate\", ...}, ...}"
}
```

This format trains the model to reason through each bias domain before producing a structured assessment, and to recommend specific verification actions (e.g., "Check CMS Open Payments for payments to lead author," "Compare registered primary outcome on ClinicalTrials.gov NCT01234567 with published endpoint").

---

## Discussion

### Strengths

BiasBuster addresses several limitations of existing approaches to bias detection in biomedical literature:

**Multi-dimensional assessment.** Unlike tools that provide binary bias labels or focus on a single domain, our five-domain taxonomy captures the heterogeneous nature of bias in clinical research. Statistical reporting bias, spin, outcome reporting, conflicts of interest, and methodological concerns are assessed independently, enabling nuanced risk profiles. The fine-tuned 9B model achieves per-dimension F1 of 0.70--0.84, confirming that domain-level granularity is learnable with modest training data.

**Verification-focused training.** A distinguishing feature is the inclusion of actionable verification steps in training data. Rather than training models to simply classify bias, the pipeline teaches them *where to verify*---citing specific databases (CMS Open Payments for author payments, ClinicalTrials.gov for outcome switching, ORCID for affiliations). This approach is motivated by the observation that bias detection is most useful when it enables verification, not when it produces unsubstantiated labels. Four of five verification sources reach 99--100% citation rates in the final model.

**Multi-source ground truth.** By combining retracted papers (high-confidence positives confirmed by editorial action), Cochrane RoB assessments (expert consensus), and heuristically screened RCTs (diverse clinical domains), the training dataset avoids the biases inherent in any single source. Retracted papers provide unambiguous positive examples but are not representative of typical bias; Cochrane assessments provide calibrated severity ratings but cover a limited set of trials; PubMed RCTs provide breadth but require automated or human annotation.

**Reproducibility.** The pipeline is fully deterministic given the same configuration: collection parameters, annotation prompts, and data splits are version-controlled. The use of identical prompts across multiple LLM backends enables direct comparison and consensus labelling.

### Training Data Quality Dominates Model Size and Hyperparameters

The most significant finding across four runs is that training data quality---not model size or hyperparameter tuning---is the dominant factor in fine-tuned model performance. The jump from 706 old-format examples to 1,235 new-format examples (with all five domains always emitted, substantive NONE reasoning, and severity oversampling) produced a +0.120 F1 improvement (Run 2 → Run 4). By comparison, the hyperparameter revision between Runs 3 and 4 produced a +0.046 improvement in eval loss, and using a 32B model instead of 9B produced a +0.028 F1 advantage that the improved training data nearly eliminated.

Three specific training data changes drove the largest gains:

1. **Always emitting all five domains**, including explicit NONE assessments with substantive reasoning. Previously, NONE domains were omitted or given empty reasoning. This taught the model that "no bias detected" is an active judgment requiring evidence, not a default.

2. **Enriched system prompts and thinking chains.** Expanding the training prompt from 320 to 800 tokens with operational definitions and extending thinking chains to cover all five domains with database selection reasoning fixed the verification citation collapse observed in Run 1.

3. **Oversampling rare severity classes.** Boosting HIGH and CRITICAL examples to ~5% of the training set improved the model's exposure to extreme cases, though the class imbalance at ordinal boundaries remained the primary bottleneck.

### Prompt Engineering and Fine-Tuning Are Complementary

The Run 1 prompt experiment demonstrated that an enriched system prompt alone transformed a 9B model from unusable (F1 = 0.455) to competitive with 32B baselines (F1 = 0.866) on coarse binary detection. However, the enriched prompt's per-dimension F1 scores were 0.39--0.50---barely above chance---indicating it detected bias as an undifferentiated mass without identifying *which* domains were affected. Fine-tuning lifted per-dimension F1 to 0.64--0.84, representing genuine understanding of the five bias domains.

This suggests a practical production architecture: an enriched-prompt model for high-recall screening, with a fine-tuned model for per-dimension analysis and severity grading. The two approaches are complementary, not competing.

### Learning Dynamics Are Model-Size-Agnostic for LoRA

Run 3 tested the hypothesis that smaller models require more aggressive hyperparameters (higher learning rate, more epochs, smaller batch). The 4×10⁻⁴ learning rate with effective batch size 2 caused early saturation: the model converged by step 300 of 3,090 and then sat idle, wasting ~90% of compute. Reverting to the 27B defaults (2×10⁻⁴ LR, effective batch 4, 3 epochs) in Run 4 produced better eval loss in 927 steps than the aggressive config managed in 3,090.

This result suggests that for LoRA fine-tuning on structured extraction tasks, the optimal learning dynamics are determined by the task and dataset, not model size. The 9B model's differentiation from 32B should be in LoRA capacity (higher rank) and regularisation (dropout, weight decay, label smoothing), not learning rate or epoch count.

### The Severity Calibration Bottleneck

Ordinal severity grading (κ = 0.12--0.29 across all runs) remains the primary unsolved problem. The "moderate collapse"---systematic over-prediction of MODERATE severity for any non-NONE case---is driven by class imbalance: MODERATE is the modal class in 3 of 5 dimensions, while HIGH and CRITICAL together constitute only 3--5% of examples. With ~11 examples per dimension per extreme class, the model lacks sufficient training signal to learn ordinal boundaries.

This is fundamentally a data problem, not a modelling problem. Four runs with different hyperparameters, learning rates, epoch counts, and LoRA ranks have not moved κ beyond 0.29. Three approaches may address this: (1) targeted annotation of 200+ boundary cases illustrating the LOW-MODERATE distinction, (2) ordinal-aware loss functions (e.g., CORN or cumulative link models) that penalise adjacent-class errors less than distant errors, and (3) post-hoc calibration (temperature or Platt scaling) on a held-out set.

### Verification Source Citation Fragility

Verification source knowledge showed a complex trajectory. Run 1 demonstrated that fine-tuning can *destroy* citation patterns learned during pretraining (CMS Open Payments 85% → 16%). Runs 2--4 showed that explicitly teaching database selection reasoning in thinking chains restores most citations. However, CMS Open Payments regressed again in Run 4 (57% → 22%), revealing that citation patterns are fragile when the training signal is unevenly distributed across severity levels.

The root cause---Open Payments cited in only 29.8% of training examples, concentrated at HIGH COI severity---suggests that the export pipeline should broaden citation triggers to any non-NONE COI severity, increasing training signal to ~70% of examples.

### Operational Definitions and Inter-Model Agreement

A key challenge in multi-model annotation is inconsistent interpretation of ambiguous cases. Early analysis of inter-model agreement between Claude and DeepSeek revealed substantial disagreement (~55% discordance on overall severity), driven by differing interpretations of what constitutes "absolute" reporting, adequate follow-up, and COI disclosure. We addressed this through nine explicit operational definitions embedded in the shared annotation prompt, covering:

- The distinction between raw event counts (absolute) and relative risk reductions (relative-only)
- Domain-specific follow-up adequacy thresholds
- The requirement for author-level COI statements beyond funding source identification
- Classification of patient-centred vs. surrogate outcomes
- Handling of retraction notices vs. original retracted papers

These definitions transform subjective judgments into reproducible classifications, reducing the need for post-hoc adjudication and improving training data consistency.

### Limitations

**Abstract-level analysis.** The current pipeline assesses bias based on abstracts and metadata alone, not full-text papers. While abstracts contain sufficient information for many bias indicators (statistical reporting, spin, disclosure), some assessments---particularly methodology and outcome switching---would benefit from full-text access. Europe PMC provides full-text access for open-access papers, and future versions could incorporate this.

**LLM annotation quality.** Despite structured prompts and operational definitions, LLM annotations are not equivalent to expert human judgment. The human review step is essential, and the pipeline is designed to support human-in-the-loop validation rather than replace it. The multi-model approach, where agreement between independently prompted models increases confidence, partially mitigates this limitation.

**Severity class imbalance.** The training data has a severe imbalance at ordinal boundaries (MODERATE dominates; HIGH/CRITICAL are rare), limiting severity calibration. This is partly inherent to the domain---most bias in clinical literature is moderate rather than extreme---and partly a data generation artefact that targeted annotation could address.

**CMS Open Payments coverage.** The pipeline's Open Payments citation rate of 22% in the final model is below the 57% achieved in Run 2, indicating that training data composition changes can inadvertently degrade specific capabilities. This requires explicit attention in the export pipeline.

**English-language bias.** All data sources and annotation prompts are English-language, limiting applicability to non-English biomedical literature.

**Temporal scope.** PubMed RCT collection is limited to 2020--present, which may underrepresent bias patterns from earlier publication eras.

**Domain coverage.** The seven MeSH focus domains, while covering major clinical areas, do not capture all biomedical research. Extension to additional domains requires only configuration changes.

### Mixture-of-Experts Architecture and Multi-Domain Tasks

The strong zero-shot performance of gpt-oss:20b---a 21B MoE model with 32 experts and top-4 routing---is a notable finding. Without any fine-tuning, it achieved the best severity calibration (κ = 0.158) and verification source coverage (0.591) of any model evaluated. Its per-dimension F1 scores (0.75--0.81) showed balanced performance across all five bias domains, suggesting that the MoE routing mechanism may naturally distribute multi-domain tasks across different expert subnetworks.

This has practical implications: MoE models activate only a fraction of their parameters per token (3.6B of 21B for gpt-oss:20b), giving them inference speeds comparable to much smaller dense models while accessing a larger parameter space. gpt-oss:20b processed abstracts at 31.8 tokens/second on an M3 Mac---2.6× faster than the 9B dense Qwen model at 12.4 tokens/second. A fine-tuned MoE model could potentially combine the accuracy gains from domain-specific training with the efficiency and capacity of the MoE architecture, using attention-only LoRA (targeting q/k/v/o projections while leaving expert FFN weights and the router frozen) to maintain training stability.

### Comparison with Existing Tools

Several tools support bias assessment in systematic reviews, including RobotReviewer,^12^ which uses machine learning for RoB assessment, and the Cochrane RoB 2 tool itself.^1^ BiasBuster differs in scope and purpose: rather than performing bias assessment directly, it constructs *training datasets* for fine-tuning domain-specific models and demonstrates that small (9B) fine-tuned models can approach expert-tool-level detection (F1 0.924) with the added benefit of verification-focused outputs. The verification-focused annotation schema---teaching models where to look rather than what to conclude---is, to our knowledge, novel.

Recent work on LLM-based bias assessment^13^ has demonstrated promising zero-shot performance, but our results show that zero-shot performance is highly prompt-dependent (F1 ranged from 0.455 to 0.967 for the same 9B model with different prompts) and that fine-tuning adds genuine domain-level understanding that prompting alone cannot achieve (per-dimension F1 improvement of +0.15 to +0.31). Fine-tuned models trained on domain-specific data consistently outperform general-purpose models on structured extraction tasks.^14^

### Future Directions

Planned extensions include: (1) LoRA fine-tuning of gpt-oss:20b using attention-only targeting to leverage its strong baseline and MoE efficiency, (2) targeted annotation of LOW-MODERATE boundary cases to address the "moderate collapse" in severity grading, (3) ordinal-aware loss functions as an alternative to cross-entropy for severity prediction, (4) broadening CMS Open Payments citation triggers in the export pipeline, (5) full-text analysis for papers available through Europe PMC, (6) active learning to prioritize human review of maximally informative examples, and (7) public release of a validated training dataset and evaluation benchmark as a community resource.

---

## Conclusions

BiasBuster provides a reproducible, modular pipeline for constructing multi-dimensional bias detection training datasets from diverse biomedical data sources, and demonstrates that the resulting training data can produce performant small models. Four iterative fine-tuning runs and expanded baseline comparison established several findings:

1. **A 9B-parameter model can approach 32B performance on bias detection.** The final Qwen3.5-9B model achieved binary F1 of 0.924 (vs. 0.952 for the 32B), recall of 0.950 (exceeding the 32B's 0.920), and per-dimension F1 of 0.70--0.84 across all five bias domains, while producing 100% chain-of-thought reasoning and actionable verification recommendations.

2. **Mixture-of-Experts architecture shows exceptional promise.** The unfine-tuned gpt-oss:20b (21B MoE, 3.6B active) achieved binary F1 of 0.918, severity κ of 0.158, and verification score of 0.591---outperforming all baselines on severity calibration and verification quality, and rivalling fine-tuned models on binary detection, while running 2.6× faster than the 9B dense model. This identifies MoE models as high-priority fine-tuning candidates for multi-domain tasks.

3. **Training data quality is the dominant lever.** Across four runs, improvements to training data format and content consistently produced larger gains than changes to model size, learning rate, epoch count, or LoRA rank.

4. **Prompt engineering and fine-tuning solve complementary problems.** Enriched prompts achieve high recall on coarse binary detection; fine-tuning adds per-dimension granularity, severity calibration, and structured reasoning chains.

5. **Severity calibration remains the primary bottleneck**, driven by class imbalance in the training data rather than model capacity or hyperparameter choice. Addressing this requires targeted annotation of boundary cases or ordinal-aware loss functions.

6. **Verification source knowledge is teachable but fragile.** Explicit database selection reasoning in training chains restores citation patterns that naïve fine-tuning destroys, but the training signal must be distributed evenly across severity levels.

7. **Not all small models are viable.** Granite3.3:8b's catastrophic failure (F1 = 0.022) demonstrates that baseline model capability varies dramatically---sufficient pretraining on biomedical and methodological literature is a prerequisite for both zero-shot and fine-tuned bias detection.

The open-source pipeline, shared annotation prompts, evaluation harness, and fine-tuned model weights support transparent, reproducible research in automated bias detection for biomedical literature.

---

## Data Availability

The BiasBuster pipeline source code is available at [repository URL]. The SQLite database schema and configuration templates are included. Raw data is sourced from publicly accessible APIs (Crossref, PubMed, Europe PMC, ClinicalTrials.gov, CMS Open Payments) and can be reproduced by running the collection pipeline.

## Funding

[TBD]

## Conflicts of Interest

[TBD]

## Author Contributions

[TBD - CRediT taxonomy recommended]

---

## References

1. Sterne JAC, Savovic J, Page MJ, et al. RoB 2: a revised tool for assessing risk of bias in randomised trials. *BMJ*. 2019;366:l4898. doi:10.1136/bmj.l4898

2. Ioannidis JPA. Why most published research findings are false. *PLoS Med*. 2005;2(8):e124. doi:10.1371/journal.pmed.0020124

3. Lundh A, Lexchin J, Mintzes B, Schroll JB, Bero L. Industry sponsorship and research outcome. *Cochrane Database Syst Rev*. 2017;2:MR000033. doi:10.1002/14651858.MR000033.pub3

4. Nuovo J, Melnikow J, Chang D. Reporting number needed to treat and absolute risk reduction in randomized controlled trials. *JAMA*. 2002;287(21):2813-2814. doi:10.1001/jama.287.21.2813

5. Boutron I, Dutton S, Ravaud P, Altman DG. Reporting and interpretation of randomized controlled trials with statistically nonsignificant results for primary outcomes. *JAMA*. 2010;303(20):2058-2064. doi:10.1001/jama.2010.651

6. Boutron I, Altman DG, Hopewell S, Vera-Badillo F, Tannock I, Ravaud P. Impact of spin in the abstracts of articles reporting results of randomized controlled trials in the field of cancer: the SPIIN randomized controlled trial. *J Clin Oncol*. 2014;32(36):4120-4126. doi:10.1200/JCO.2014.56.7503

7. Mathieu S, Boutron I, Moher D, Altman DG, Ravaud P. Comparison of registered and published primary outcomes in randomized controlled trials. *JAMA*. 2009;302(9):977-984. doi:10.1001/jama.2009.1242

8. Bekelman JE, Li Y, Gross CP. Scope and impact of financial conflicts of interest in biomedical research: a systematic review. *JAMA*. 2003;289(4):454-465. doi:10.1001/jama.289.4.454

9. International Committee of Medical Journal Editors. Recommendations for the conduct, reporting, editing, and publication of scholarly work in medical journals. Updated 2024.

10. Wei J, Wang X, Schuurmans D, et al. Chain-of-thought prompting elicits reasoning in large language models. *Adv Neural Inf Process Syst*. 2022;35:24824-24837.

11. Mukherjee S, Mitra A, Jawahar G, Agarwal S, Palangi H, Awadallah A. Orca: progressive learning from complex explanation traces of GPT-4. *arXiv preprint*. 2023;arXiv:2306.02707.

12. Marshall IJ, Kuber R, Wallace BC. RobotReviewer: evaluation of a system for automatically assessing bias in clinical trials. *J Am Med Inform Assoc*. 2016;23(1):193-201. doi:10.1093/jamia/ocv044

13. Tang R, Chuang YN, Hu X. Does synthetic data generation of LLMs help clinical text mining? *arXiv preprint*. 2023;arXiv:2303.04360.

14. Gutiérrez BJ, McNeal N, Washington C, et al. Thinking about GPT-3 in-context learning for biomedical IE? Think again. *Findings of EMNLP*. 2022:4497-4512.

---

## Supplementary Material

### Supplementary Table S1. Annotation System Prompt Summary

The full annotation system prompt (~2,900 words) specifies the bias taxonomy, output JSON schema, and nine operational definitions. Key sections include:

| Section | Content |
|---------|---------|
| Bias domains | 5 domains with field-level specifications |
| Spin classification | Boutron taxonomy: none / low / moderate / high |
| Verification sources | 8 databases with URLs and query guidance |
| Operational definitions | 9 principles resolving annotation ambiguities |
| Calibration guidance | "Not every industry-funded study is biased" |
| Output schema | Nested JSON with severity, evidence_quotes, verification_sources |

### Supplementary Table S2. Heuristic Enrichment Scoring Weights

| Indicator | Weight | Description |
|-----------|--------|-------------|
| Relative-only reporting | +0.40 | RRR/OR/HR without ARR/NNT/baseline risk |
| Relative emphasis | +0.20 | Relative measures present with minimal absolute |
| Percent reduction without context | +0.15 | "50% reduction" without baseline |
| Title relative effect | +0.10 | Effect sizes in title |
| Missing NNT | +0.10 | Number needed to treat absent |
| Missing baseline risk | +0.10 | Control group event rate absent |
| Missing confidence intervals | +0.10 | No CIs reported |

**Thresholds:** High suspicion >= 0.30; Low suspicion < 0.10

### Supplementary Table S3. Evaluation Metrics by Level

| Level | Metric | Application |
|-------|--------|-------------|
| Binary | Precision, Recall, F1, Accuracy | Any concern vs. none, per domain |
| Ordinal | Cohen's weighted kappa, MAE, Exact match, Within-one | 5-level severity scale |
| Flag | Binary accuracy | Per-flag (relative_only, spin_level, etc.) |
| Calibration | Expected calibration error | Predicted probability vs. observed rate |
| Verification | Source coverage | Proportion of relevant databases cited |
| Comparison | McNemar's test, Wilcoxon signed-rank | Head-to-head model evaluation |

### Supplementary Figure S1. Pipeline Architecture

```
[Crossref/Retraction Watch] ──┐
[PubMed E-utilities]          ├──→ [SQLite: papers] ──→ [Enrichment] ──→ [SQLite: enrichments]
[Europe PMC / Cochrane]       │                                                    │
[ClinicalTrials.gov]     ────┘                                                    ▼
                                                                          [LLM Annotation]
                                                                          (Claude / DeepSeek)
                                                                                   │
                                                                                   ▼
                                                                          [SQLite: annotations]
                                                                                   │
                                                                                   ▼
                                                                          [Human Review (NiceGUI)]
                                                                                   │
                                                                                   ▼
                                                                          [SQLite: human_reviews]
                                                                                   │
                                                                      ┌────────────┼────────────┐
                                                                      ▼            ▼            ▼
                                                                  [Alpaca]    [ShareGPT]   [OpenAI]
                                                                  + <think>    multi-turn    chat
                                                                      │            │            │
                                                                      └────────────┼────────────┘
                                                                                   ▼
                                                                          [80/10/10 splits]
                                                                          train / val / test
```

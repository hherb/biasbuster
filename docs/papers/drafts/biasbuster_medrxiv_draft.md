# BiasBuster: A Multi-Source Pipeline for Building Training Datasets to Detect Bias in Biomedical Research Abstracts

**Authors:** Herb, Horst; Brinkmann, Bernd; Herb, Frithjof; Herb, Hagen; Claude(Opus 4.6)

**Corresponding author:** Horst Herb <hherb@consensus-ai.org>

**Word count:** [TBD - target 3,500 for main text]

---

## Abstract

**Background:** Bias in biomedical research remains a pervasive threat to evidence-based medicine. While tools exist for expert-guided risk-of-bias assessment, automated screening at scale has been limited by the absence of large, high-quality training datasets with multi-dimensional bias annotations.

**Objective:** We describe BiasBuster, an open-source pipeline for constructing curated training datasets to fine-tune large language models (LLMs) for detecting bias in biomedical abstracts across five domains: statistical reporting, spin, outcome reporting, conflict of interest, and methodology.

**Methods:** The pipeline integrates five data sources: retracted papers via Crossref/Retraction Watch (known-biased positives), Cochrane Risk of Bias 2.0 assessments via Europe PMC (expert ground truth), PubMed randomized controlled trials filtered by MeSH domain (general population), ClinicalTrials.gov registry data (outcome switching detection), and CMS Open Payments/ORCID (conflict of interest verification). Abstracts undergo heuristic enrichment (effect size auditing, funding classification) before structured annotation by multiple LLMs using identical prompts. Inter-model agreement is evaluated with Cohen's kappa, McNemar's test, and Wilcoxon signed-rank tests. Training data is exported in three fine-tuning formats with chain-of-thought reasoning and actionable verification steps.

**Results:** We present the system architecture, annotation taxonomy, evaluation framework, and preliminary inter-model agreement analysis. The pipeline produces annotations across five bias domains with structured severity ratings, evidence quotes, and database-specific verification guidance (e.g., "check CMS Open Payments for author X," "compare registered vs. published primary outcome on ClinicalTrials.gov").

**Conclusions:** BiasBuster addresses a critical gap in bias detection tooling by providing reproducible, multi-source training data with verification-focused annotations. The pipeline's modular design enables community extension to additional data sources and bias domains.

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

**Multi-dimensional assessment.** Unlike tools that provide binary bias labels or focus on a single domain, our five-domain taxonomy captures the heterogeneous nature of bias in clinical research. Statistical reporting bias, spin, outcome reporting, conflicts of interest, and methodological concerns are assessed independently, enabling nuanced risk profiles.

**Verification-focused training.** A distinguishing feature is the inclusion of actionable verification steps in training data. Rather than training models to simply classify bias, the pipeline teaches them *where to verify*---citing specific databases (CMS Open Payments for author payments, ClinicalTrials.gov for outcome switching, ORCID for affiliations). This approach is motivated by the observation that bias detection is most useful when it enables verification, not when it produces unsubstantiated labels.

**Multi-source ground truth.** By combining retracted papers (high-confidence positives confirmed by editorial action), Cochrane RoB assessments (expert consensus), and heuristically screened RCTs (diverse clinical domains), the training dataset avoids the biases inherent in any single source. Retracted papers provide unambiguous positive examples but are not representative of typical bias; Cochrane assessments provide calibrated severity ratings but cover a limited set of trials; PubMed RCTs provide breadth but require automated or human annotation.

**Reproducibility.** The pipeline is fully deterministic given the same configuration: collection parameters, annotation prompts, and data splits are version-controlled. The use of identical prompts across multiple LLM backends enables direct comparison and consensus labelling.

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

**English-language bias.** All data sources and annotation prompts are English-language, limiting applicability to non-English biomedical literature.

**Temporal scope.** PubMed RCT collection is limited to 2020--present, which may underrepresent bias patterns from earlier publication eras.

**Domain coverage.** The seven MeSH focus domains, while covering major clinical areas, do not capture all biomedical research. Extension to additional domains requires only configuration changes.

### Comparison with Existing Tools

Several tools support bias assessment in systematic reviews, including RobotReviewer,^12^ which uses machine learning for RoB assessment, and the Cochrane RoB 2 tool itself.^1^ BMLibrarian differs in scope and purpose: rather than performing bias assessment directly, it constructs *training datasets* for fine-tuning domain-specific models. The verification-focused annotation schema---teaching models where to look rather than what to conclude---is, to our knowledge, novel.

Recent work on LLM-based bias assessment^13^ has demonstrated promising zero-shot performance, but fine-tuned models trained on domain-specific data consistently outperform general-purpose models on structured extraction tasks.^14^ BMLibrarian provides the training infrastructure to enable such fine-tuning.

### Future Directions

Planned extensions include: (1) full-text analysis for papers available through Europe PMC, (2) integration of additional LLM backends for broader consensus labelling, (3) active learning to prioritize human review of maximally informative examples, (4) expansion of the bias taxonomy to include publication bias indicators and selective reporting at the review level, and (5) public release of a validated training dataset as a community resource.

---

## Conclusions

BiasBuster provides a reproducible, modular pipeline for constructing multi-dimensional bias detection training datasets from diverse biomedical data sources. By combining retracted papers, expert Cochrane assessments, and heuristically screened RCTs with structured LLM annotations and human validation, the system produces training data that teaches models not only to identify bias but to recommend specific verification steps. The open-source implementation, shared annotation prompts, and statistical evaluation framework support transparent, reproducible research in automated bias detection.

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

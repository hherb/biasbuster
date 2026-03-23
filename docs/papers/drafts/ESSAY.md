# Teaching a Small Language Model to Detect Bias in Medical Research

## How we built BiasBuster: an open-source pipeline that fine-tunes 9B-parameter models to catch statistical spin, undisclosed conflicts, and outcome switching in clinical trial abstracts

---

Every year, clinicians, policymakers, and patients make decisions based on published medical research. Most of that research is honest. But a troubling fraction is not -- and the biases it contains are often invisible to readers without statistical training.

A pharmaceutical company funds a trial for its new drug. The drug reduces the *relative risk* of heart attack by 50% -- an impressive-sounding number that appears in the abstract's title, conclusion, and press release. Buried in the supplementary tables: the *absolute risk* dropped from 2% to 1%. The Number Needed to Treat is 100 -- you'd need to give the drug to 100 people for one to benefit. The trial was stopped early, after a planned interim analysis showed the relative measure looked good. The registered primary endpoint on ClinicalTrials.gov was overall survival; the published primary endpoint was "major adverse cardiac events," a composite that was never pre-specified.

None of this is fabrication. Every number is real. But the framing is designed to make a marginal benefit look transformative. This is bias -- not the blatant kind, but the subtle, structural kind that shapes treatment guidelines and costs healthcare systems billions.

We built BiasBuster to catch it.

## The Problem: Bias Hides in Plain Sight

Systematic reviews and meta-analyses are supposed to filter biased studies. But they depend on human reviewers using tools like Cochrane's Risk of Bias framework -- a process that takes hours per paper and still suffers from inter-rater disagreement. Meanwhile, thousands of new clinical trials are published every month.

The biases we target fall into five domains that research integrity experts have identified as the most consequential:

**Statistical Reporting Bias.** Reporting only relative risk reduction (RRR, OR, HR) without absolute measures (ARR, NNT, baseline risk). When a drug reduces relative risk by 50% but absolute risk by 0.5%, the framing matters enormously. Yet many abstracts report only the impressive-sounding relative number.

**Spin.** Conclusions that don't match the actual results. The Boutron taxonomy classifies this from "none" to "high" -- ranging from accurate summaries to conclusions that recommend clinical adoption despite non-significant primary outcomes. A surprisingly common pattern: when the primary endpoint fails, authors pivot to a secondary subgroup analysis in the conclusion.

**Outcome Reporting.** Using surrogate endpoints (lab values, imaging markers) instead of patient-centred outcomes (survival, quality of life). Worse still: switching the primary outcome between registration and publication. ClinicalTrials.gov keeps a record of what was promised; the published abstract reveals what was delivered.

**Conflict of Interest.** Industry funding without disclosure. Author affiliations with the trial sponsor. Payments from pharmaceutical companies to investigators -- publicly available in the US through CMS Open Payments, but rarely cross-referenced.

**Methodological Red Flags.** Inappropriate comparators (placebo when an active treatment exists), enrichment designs (run-in periods that exclude non-responders), per-protocol analysis without intention-to-treat, premature stopping.

## The Approach: Teach a Model WHERE to Look, Not Just What to Flag

Most AI-assisted bias detection treats the problem as classification: is this study biased or not? We think that's insufficient. A clinician doesn't just want a red flag -- they want to know *where to verify the claim*.

BiasBuster's core insight is verification-focused training. The model doesn't just learn to detect bias; it learns to recommend specific databases where each claim can be checked:

- **ClinicalTrials.gov** for registered outcomes vs. published outcomes (outcome switching)
- **CMS Open Payments** for undisclosed industry payments to investigators
- **ORCID** for author employment histories that reveal pharma affiliations
- **Europe PMC** for full-text funding disclosures not visible in the abstract
- **Retraction Watch / Crossref** for post-publication corrections and retractions

This matters because a bias flag without a verification path is just an opinion. A bias flag with "check CMS Open Payments for payments from Pfizer to the lead author" is actionable.

## Building the Training Data

You can't fine-tune a model for bias detection without high-quality training data, and no such dataset existed. We built one.

### Multi-Source Ground Truth

Rather than relying on a single source of "biased" and "unbiased" labels, we triangulated across multiple evidence streams:

**Retracted papers** serve as strong positive examples. These papers had flaws serious enough to warrant retraction -- not all for bias, but the overlap is substantial. We collected abstracts via the Crossref API's Retraction Watch integration, following the `update-to` relationship from retraction notices back to the original paper's DOI, then fetching the original abstract from PubMed. Importantly, we filtered out bare retraction notices ("This article has been retracted") and kept only the original research content.

**Cochrane Risk of Bias assessments** provide expert-level ground truth. Cochrane systematic reviews include structured RoB 2 judgments (low / some concerns / high) for every study they include, often with per-domain ratings across five RoB 2 domains (randomization, deviations from intended interventions, missing outcome data, outcome measurement, and selective reporting). We collected 260 expert-assessed papers from Europe PMC full-text XML, extracting the risk-of-bias tables via regex with LLM fallback (chunk & map-reduce, never truncation) and resolving "Author Year" study identifiers back to PubMed IDs through a five-layer resolution strategy. Oversized documents (e.g. entire books indexed in PMC) are automatically skipped when they exceed 2.4 MB to avoid wasting LLM tokens on non-review content. Per-domain ratings are stored alongside the overall judgment and passed to the annotation LLM as calibration context. All Cochrane data uses an upsert persistence pattern that preserves PubMed-fetched titles and abstracts while always updating domain ratings and review metadata -- a design driven by the principle that expensively generated data must never be silently discarded.

**PubMed RCTs by clinical domain** were screened using heuristic enrichers -- an effect-size auditor that scores abstracts on a 0-1 scale based on relative vs. absolute reporting patterns, and a funding classifier that identifies industry sponsorship from abstract text, PubMed grant metadata, and author affiliations. High-scoring abstracts (strong relative-only reporting, industry funding, no COI disclosure) became positive candidates; low-scoring ones became negative controls.

**ClinicalTrials.gov** provided outcome switching detection. For each RCT with a discoverable NCT ID, we compared registered primary outcomes against what was actually reported in the published abstract.

### LLM Annotation with Operational Definitions

With candidate abstracts collected and enriched, we used Claude to generate structured 5-domain bias assessments with severity ratings, evidence quotes, and verification step recommendations. But early annotation revealed a problem: inter-model disagreement was ~55% on 898 shared abstracts between Claude and DeepSeek. (The dataset has since grown to 1,521 annotated examples through continued pipeline runs.)

The fix was operational definitions -- nine explicit principles that resolved the ambiguities:

- Raw event rates in both arms (e.g., "84% vs 36%") count as absolute measures, even without ARR or NNT
- Process measures (dose modifications, lab values) are surrogates; mortality, quality of life, and functional status are patient-centred
- Naming a funding source alone is not COI disclosure; requires explicit author-level conflict statements
- Domain-specific follow-up thresholds: <12 months for chronic disease = short; <4 weeks for acute = short

These definitions were embedded in both the annotation prompt and the training system prompt.

### Training Format: Thinking Chains + Verification Steps

Each training example follows the Alpaca format with a `<think>` reasoning chain:

```
System: You are a biomedical research integrity analyst...

Instruction: Assess the following clinical trial abstract for potential bias...

Output:
<think>
Statistical reporting: The abstract reports HR 0.67 (95% CI 0.52-0.86)
without absolute risk reduction or NNT. No baseline event rates are provided...

Spin: The conclusion states the drug "significantly improved outcomes" but
the primary endpoint (overall survival) was not significant (p=0.12)...

Verification databases: ClinicalTrials.gov should be checked for NCT01234567
to compare registered vs published primary outcomes. CMS Open Payments should
be searched for the lead author given industry sponsorship...
</think>

## Statistical Reporting: MODERATE
- Reports HR without absolute measures...

## Recommended Verification Steps
- Search ClinicalTrials.gov for NCT01234567...
- Check CMS Open Payments for Dr. Smith...
```

The thinking chain teaches the model to reason step-by-step through each domain *and* to select which verification databases are relevant based on the study's characteristics. This is the key training signal -- not just "this study is biased" but "here's why, and here's where to check."

### Why Natural Language, Not Function Calls?

An obvious question: if the model is recommending database lookups, why not train it to emit structured tool calls -- JSON function signatures that an agent can execute directly?

We considered this. Modern tool-use formats (OpenAI function calling, Qwen's ReACT format, Anthropic's tool_use blocks) would let the model output something like:

```json
{"tool": "search_open_payments", "args": {"physician_last": "Smith", "physician_first": "John"}}
```

We chose natural language verification steps instead, for three reasons:

**1. The model serves two audiences.** A human reviewer reading the assessment needs to understand *what to check and why* -- "Check CMS Open Payments for author payment records from Pfizer, given that this is an industry-funded cardiovascular trial" is immediately actionable for a human. A JSON function call is not. Natural language serves both the human reader and an automated agent that can parse it downstream.

**2. Tool-use training requires multi-turn examples.** Function-calling models are trained on sequences where the model emits a tool call, receives a tool result, then continues reasoning. Building these multi-turn training examples requires actually executing the tools during data generation -- calling ClinicalTrials.gov, ORCID, and Open Payments for every training abstract, then incorporating the results into the annotation. We had 920 training examples. Running 5-6 API calls per example, handling failures and rate limits, and generating coherent multi-turn reasoning chains would have multiplied the data generation effort by an order of magnitude. Natural language verification steps can be synthesised from the annotation metadata alone.

**3. The set of verification databases is small and stable.** There are exactly six databases the model recommends: ClinicalTrials.gov, CMS Open Payments, ORCID, Europe PMC, Retraction Watch, and Medicines Australia/EFPIA. Pattern matching ("ClinicalTrials.gov" in step text) is 100% reliable for routing to the correct API. If we had dozens of tools with overlapping capabilities, structured function calls would be worth the training cost. With six well-separated databases, keyword routing is simpler, cheaper, and just as accurate.

The practical result: our verification agent wrapper parses natural language steps with simple regex patterns, extracts parameters (NCT IDs, author names, PMIDs) from the text, and dispatches to the existing API clients. It works because the model was trained to cite specific database names in specific contexts -- consistently enough that keyword matching never fails.

This is a deliberately pragmatic choice. If we later need the model to chain multiple tool calls iteratively (e.g., "search ORCID, find an industry affiliation, then check Open Payments for that company"), we would need structured tool-use training. But for the current single-pass verification architecture, natural language is the right tradeoff.

## The Hardware: Desktop-Class Training and Inference

Training runs on an NVIDIA DGX Spark -- a desktop-class machine with 128 GB of unified memory and a GB10 Blackwell GPU. Evaluation and inference also run on an Apple M3 Mac with 128 GB of unified memory. Neither is a datacenter; both sit on a desk. The constraint shaped our architecture: SGLang and vLLM don't yet support the ARM/Blackwell combination, so all local inference runs through Ollama.

LoRA fine-tuning of a 9B model takes ~2.5 hours. A 32B model takes ~4.5 hours. Both fit comfortably in memory without quantisation during training (bf16). Inference on the fine-tuned 9B model runs at ~11 tokens/second on the DGX Spark and ~12.4 tokens/second on the M3 Mac.

## Results: What We Learned in Four Fine-Tuning Runs -- and an Unexpected Baseline

Before discussing the fine-tuning runs, two additional baselines deserve mention. We evaluated OpenAI's gpt-oss:20b -- their first open-weight model, a Mixture-of-Experts architecture with 32 experts and top-4 routing (21B total parameters, ~3.6B active per token) -- and IBM's granite3.3:8b on a 157-example test set.

Granite3.3 was a catastrophic failure. It predicted NONE for nearly every dimension -- recall of 1.1%, F1 of 0.022. The model simply lacks sufficient pretraining exposure to biomedical bias assessment concepts to engage with the task at all. Not every 8B model is created equal.

gpt-oss:20b was a revelation. Without any fine-tuning, it achieved binary F1 of 0.918, recall of 0.941, and -- crucially -- the best severity calibration of any baseline we've tested (weighted κ = 0.158, vs 0.021-0.066 for the Qwen/OLMo baselines). Its verification source citation rates were above 94% for all five databases, with a mean verification score of 0.591 -- exceeding even our fine-tuned models. And it did this at 31.8 tokens/second on the M3 Mac, 2.6× faster than the 9B Qwen model, because its MoE architecture activates only 3.6B of its 21B parameters per token. It's a 20B model with the speed of a 4B model.

The MoE architecture may be particularly well-suited to multi-domain bias detection. With 32 experts and top-4 routing, the model has the capacity to develop specialised subnetworks for different task aspects -- statistical reporting analysis might activate different experts than COI assessment or methodology evaluation. This would explain its balanced per-dimension F1 scores (0.75-0.81 across all five domains) without any domain-specific training.

This result reframed our fine-tuning strategy. We'll return to its implications after discussing the four runs.

### First Run: OLMo-3.1-32B

We started with OLMo-3.1-32B-Instruct -- chosen for its full transparency (open training data, weights, and intermediate checkpoints) and its heavy academic pretraining corpus (238 million academic PDFs via Dolma).

The results were mixed. Fine-tuning dramatically improved severity grading (ordinal kappa 0.066 to 0.285) and conflict of interest detection (F1 0.667 to 0.927). But it *hurt* verification source citations -- CMS Open Payments mentions dropped from 85% to 16%, Retraction Watch from 96% to 43%. The training data wasn't teaching database selection consistently enough.

The most surprising finding came from a side experiment: running the smaller Qwen3.5-9B model with an enriched system prompt (containing the operational definitions) and no fine-tuning at all. The 9B model's binary F1 jumped from 0.455 to 0.967 -- matching the 32B baselines. The enriched prompt alone was doing most of the work for coarse detection.

### Second Run: Qwen3.5-9B with Enriched Training Data

Armed with this insight, we fixed the training pipeline:

1. Expanded the training system prompt from 320 tokens to 800 tokens with operational definitions and verification database criteria
2. Extended thinking chains to cover all 5 dimensions with database selection reasoning
3. Synthesised missing verification steps from annotation flags (e.g., if funding_type = "industry", ensure CMS Open Payments is mentioned)

Then we fine-tuned the 9B model with the same LoRA configuration as the 32B run -- identical hyperparameters, as a controlled experiment.

**The enriched training data fixed verification citations decisively.** CMS Open Payments went from 16% (32B fine-tuned) to 57% (9B fine-tuned). Retraction Watch went from 43% to 95%. The model learned where to look.

**But the comparison with the enriched-prompt baseline told a more nuanced story.** On the same 115-example test set:

| What | Enriched Prompt (no FT) | Fine-Tuned 9B |
|------|------------------------|---------------|
| Overall binary F1 | **0.866** | 0.804 |
| Recall | **0.793** | 0.679 |
| Per-dimension F1 (avg) | 0.44 | **0.70** |
| Ordinal kappa | 0.118 | **0.159** |
| Severity MAE | 1.209 | **0.878** |
| Thinking chains | 0% | **99%** |
| Verification score | 0.495 | **0.541** |

The enriched prompt is better at the coarse question -- "is this study biased at all?" -- because it has higher recall. But its per-dimension F1 scores are 0.39-0.50, meaning it can't reliably tell you *which* of the five domains is biased. The fine-tuned model's 0.64-0.76 per-dimension F1 represents genuine understanding of the bias taxonomy, not just blanket flagging.

And the fine-tuned model produces reasoning chains, which is essential for a tool that needs to explain *why* it flagged something.

### Third Run: The Aggressive Hyperparameters That Failed

The Second Run had proposed 9B-optimised hyperparameters: a higher learning rate (4e-4 vs 2e-4), more epochs (5 vs 3), and a smaller effective batch size (2 vs 4). The reasoning was sound -- smaller models should tolerate more aggressive updates. The data was wrong.

Simultaneously, we overhauled the training data. The dataset grew from 706 to 1,235 training examples (+75%), and the format changed substantially: all five bias domains are now always emitted (including explicit NONE assessments with substantive reasoning), and rare severity classes (HIGH/CRITICAL) were oversampled to ~5% of the training set.

The training curves told the story immediately. Loss dropped from ~7.0 to ~2.5 in the first 100 steps, then plateaued at ~1.5-2.0 for the remaining 2,900 steps. The model learned everything it could in the first 10% of training and sat idle for the other 90%. The 4e-4 learning rate combined with a small batch size drove the model into a loss basin it couldn't escape. Five epochs and 3,090 steps were mostly wasted compute.

### Fourth Run: Conservative Hyperparameters Win

We reverted the learning dynamics to the 27B defaults (2e-4 LR, 3 epochs, effective batch 4) while keeping the 9B-specific LoRA capacity and regularisation settings (rank 32, dropout 0.08, weight decay 0.02, label smoothing 0.05). Same training data as the Third Run.

The difference was stark. Training loss declined gradually through all 927 steps with no saturation plateau. Eval loss improved steadily from 1.267 to 1.101 -- still declining at completion, suggesting the model would benefit from a fourth epoch. The conservative config achieved better eval loss in 927 steps than the aggressive config managed in 3,090.

**Evaluation confirmed the improvement across the board:**

| What | Second Run (9B, 706 ex.) | Fourth Run (9B, 1,235 ex.) |
|------|--------------------------|----------------------------|
| Overall binary F1 | 0.804 | **0.924** |
| Recall | 0.679 | **0.950** |
| Precision | 0.986 | 0.898 |
| Per-dimension F1 (avg) | 0.70 | **0.78** |
| Ordinal kappa | 0.159 | 0.124 |
| Verification score | 0.541 | 0.495 |
| Thinking chains | 99% | **100%** |

The recall problem from the Second Run is solved. The model is no longer too conservative -- it catches 95% of biased studies, up from 68%. Every per-dimension F1 improved: statistical reporting (0.73 to 0.81), spin (0.73 to 0.83), outcome reporting (0.76 to 0.84), COI (0.64 to 0.70), methodology (0.66 to 0.74).

**Closing the gap to 32B.** The 9B model now nearly matches the 32B on binary detection (F1 0.924 vs 0.952, a gap of 0.028 -- down from 0.148). It actually *beats* the 32B on recall (0.950 vs 0.920) and verification quality (0.495 vs 0.368). The remaining gap is in severity calibration.

### What Regressed -- and Why It Matters

**Severity grading got worse, not better.** Ordinal kappa dropped from 0.159 to 0.124, despite better binary detection. The confusion matrices reveal a systematic "moderate collapse": when the model detects bias, it defaults to predicting MODERATE severity because MODERATE is the dominant non-NONE class in the training data (50% of Statistical Reporting labels, 36% of COI, 35% of Outcome Reporting). Only 2 of 36 true-LOW Statistical Reporting examples were predicted correctly -- the rest were called MODERATE or HIGH. The model has learned "biased means moderate" as a strong prior, with no reliable signal for distinguishing LOW from MODERATE.

This is fundamentally a class imbalance problem. The training data contains enough examples (1,235) to learn the binary boundary between "biased" and "not biased," but far too few per-class examples to learn the ordinal boundaries between severity levels. HIGH and CRITICAL have only ~11 examples per dimension per class after oversampling.

**CMS Open Payments collapsed from 57% to 22%.** Analysis of the training data shows Open Payments is mentioned in only 29.8% of examples, with a steep skew: 100% citation rate for HIGH COI severity, but only 16% for LOW and 13% for NONE. The model learned the strong HIGH-COI association and discarded the weaker signals. Since most test examples have LOW-MODERATE COI, the model defaults to not citing Open Payments.

### Fifth Run: Fine-Tuning the MoE Model

The gpt-oss:20b results were too promising to leave on the table. We fine-tuned it.

This required solving several infrastructure problems first. GPT-OSS stores its expert weights in MXFP4 (4-bit floating point), and the backward pass isn't implemented for MXFP4 in PyTorch. We dequantised to BF16 on load via `Mxfp4Config(dequantize=True)`, which expanded the model to ~42 GB -- still comfortable within the DGX Spark's 128 GB. The model also requires `attn_implementation="eager"` instead of PyTorch SDPA, per the OpenAI cookbook.

The LoRA configuration was deliberately conservative: attention-only targets (q/k/v/o projections), rank 16, learning rate 5×10⁻⁶ (50× lower than dense model defaults), 1 epoch, and higher dropout (0.1) to combat memorisation. Only 7.96M of 20.9B parameters (0.04%) were trainable. The initial training run with 3 epochs and 1×10⁻⁵ LR showed rapid saturation -- loss collapsed to near-zero within 200 steps, with the remaining 1,000 steps producing essentially zero gradients. The revised configuration (1 epoch, 5×10⁻⁶ LR) was more conservative still.

Export also required a new approach. Standard adapter merging fails because MXFP4→BF16 reverse conversion isn't implemented, and dequantising to BF16 then re-quantising to GGUF would be wasteful when Ollama supports MXFP4 natively at ~14 GB. Instead, we used Ollama's `ADAPTER` directive to overlay the LoRA adapter on the MXFP4 base model at load time.

**The results were a mixed bag:**

| What | gpt-oss:20b (baseline) | gpt-oss:20b (fine-tuned) |
|------|:---:|:---:|
| Overall binary F1 | 0.918 | **0.938** |
| Recall | 0.941 | **1.000** |
| Precision | **0.895** | 0.883 |
| Per-dimension F1 (avg) | 0.77 | **0.80** |
| Ordinal kappa | **0.158** | 0.042 |
| Verification score | 0.591 | **0.624** |
| Thinking chains | 0% | 0% |
| Mean latency | **76.7s** | 123.6s |

Fine-tuning achieved the **best binary detection of any model we've tested** -- F1 0.938 with perfect recall (catches every biased paper). COI detection saw the largest per-dimension gain (+0.077 F1, driven by recall jumping from 0.734 to 0.973). Spin and outcome reporting also improved.

But severity calibration **collapsed**. Kappa dropped from 0.158 to 0.042 -- the worst of any fine-tuned model. Three of five ordinal tests reached statistical significance, all favouring the baseline (methodology p<0.001, COI p=0.017, statistical reporting p=0.049). The fine-tuned model learned "something is biased here" extremely well, but lost the ability to grade *how* biased.

And the biggest disappointment: **no thinking chains**. Despite the training data containing `<think>` reasoning blocks, the fine-tuned model never produces them. The attention-only LoRA -- which can redirect the model's attention to bias-relevant features -- apparently lacks the capacity to teach a new output structure. The frozen expert FFN weights control output formatting, and 0.04% trainable parameters can't override them.

This tells us something important about attention-only LoRA on MoE models: it can sharpen detection (what the attention layers are already doing) but can't teach new behaviours (severity grading, thinking chains) that depend on the frozen expert weights. The adapter is too thin.

### The Key Takeaways

**Mixture-of-Experts models are exceptionally strong baselines -- and respond differently to fine-tuning than dense models.** gpt-oss:20b's attention-only LoRA improved binary detection to the best we've seen (F1 0.938, recall 1.000) but degraded severity calibration (κ 0.158 → 0.042) and failed to produce thinking chains. The 0.04% trainable parameter fraction can redirect attention but cannot override frozen expert knowledge. Dense model fine-tuning (Runs 1-4) successfully taught both new capabilities and output formatting; MoE fine-tuning with conservative LoRA only sharpened existing capabilities.

**Not all small models are equal.** Granite3.3:8b's catastrophic failure (F1 0.022, recall 1.1%) versus Qwen3.5-9B's strong performance (F1 0.924 fine-tuned) shows that baseline capability depends on pretraining corpus composition. Sufficient exposure to biomedical methodology literature is a prerequisite.

**Prompt engineering and fine-tuning solve different problems.** An enriched prompt handles coarse detection; fine-tuning adds granular domain-level analysis, severity calibration, and structured reasoning. They're complementary, not competing.

**Verification source knowledge can be taught -- and lost.** The First Run showed that fine-tuning can *destroy* database citation patterns. The Second Run showed that explicitly teaching database selection reasoning in the training data fixes this. The Fourth Run showed that even with good training, citation patterns are fragile when the training signal is unevenly distributed across severity levels. The fine-tuned gpt-oss:20b showed a nuanced version of this: individual source citation rates dropped (e.g. ORCID 97% → 85%, Retraction Watch 96% → 78%), but the mean verification *score* improved (0.591 → 0.624), suggesting citations became more targeted even if less frequent.

**Training data quality dominates hyperparameters.** Five runs have shown that the single biggest lever is the training data format -- not learning rate, not epoch count, not LoRA rank. The jump from 706 old-format examples to 1,235 new-format examples (with all five domains, NONE reasoning, and severity oversampling) produced a +0.120 F1 improvement. The hyperparameter change between the Third and Fourth Runs produced a +0.046 improvement in eval loss. Data quality wins by a wide margin.

**Learning dynamics are model-size-agnostic for LoRA -- but MoE models are a different beast.** The 27B defaults (2e-4 LR, 3 epochs, effective batch 4) work equally well for both 9B and 32B dense models on this task. But MoE models require dramatically more conservative settings: attention-only targeting, 50× lower learning rate (5e-6 vs 2e-4), higher dropout, and fewer epochs. Even then, attention-only LoRA at 0.04% trainable parameters can sharpen detection but cannot teach new output behaviours like thinking chains.

**Severity calibration is a data problem, not a modelling problem.** Five training runs with different hyperparameters, model architectures, and LoRA configurations have left ordinal kappa stuck in the 0.04-0.29 range. The "moderate collapse" is driven by class imbalance in the training data, not by model capacity or learning dynamics. The fine-tuned gpt-oss:20b made this worse (κ 0.158 → 0.042), suggesting that attention-only LoRA actively harms calibration by amplifying detection sensitivity without the capacity to encode ordinal gradations. Fixing this requires either hundreds more boundary-case annotations, an ordinal-aware loss function, or post-hoc calibration.

**The right architecture depends on the use case.** Five runs have revealed a clear trade-off landscape: the fine-tuned gpt-oss:20b is the best *screening* model (F1 0.938, recall 1.000 -- misses nothing), the unfine-tuned gpt-oss:20b is the best *calibrated* assessor (κ 0.158, balanced severity ratings), and the fine-tuned Qwen3.5-9B is the best *explainer* (100% thinking chains, per-dimension F1 0.70-0.84). A production system might use all three in sequence: fine-tuned MoE for high-recall screening, unfine-tuned MoE for severity assessment, and fine-tuned 9B for detailed explanations on flagged papers.

## From Assessment to Action: The Verification Agent

The fine-tuned model tells you what to check. But can it actually check?

We built a proof-of-concept agent wrapper that closes the loop. The agent:

1. Sends an abstract to the fine-tuned model for bias assessment
2. Parses the verification recommendations from the model's output
3. Dispatches actual API calls -- fetching trial registrations from ClinicalTrials.gov, searching ORCID for author affiliations, querying Europe PMC for funding metadata
4. Feeds the verification results back to the model for a refined assessment

The model recommends "Check ClinicalTrials.gov for NCT01234567"; the agent *actually checks* and returns: "Registered primary outcome: overall survival at 24 months. Published primary outcome appears different. Sponsor: Pfizer (INDUSTRY). Possible outcome switching detected."

This is where the verification-focused training pays off. Because the model was trained to cite specific databases in specific contexts, its recommendations map cleanly to automated tool calls. A model that just says "check for conflicts of interest" isn't actionable. A model that says "search CMS Open Payments for Dr. John Smith, given that this is an industry-funded cardiovascular trial" is.

## What's Next

**Unlocking thinking chains on gpt-oss:20b.** The Fifth Run showed that attention-only LoRA can't teach the MoE model to produce `<think>` reasoning. Three approaches are on the table: (1) expanding LoRA targets to include non-expert MLP layers (gate/up/down projections, skipping expert FFNs and the router) to increase trainable parameters while preserving routing stability, (2) increasing LoRA rank from 16 to 64 to give the adapter more capacity, and (3) training for 2-3 epochs at the same conservative LR to give the model more exposure to the thinking chain format. If any of these produces reasoning chains while maintaining the 0.938 F1, the gpt-oss:20b fine-tune becomes the clear production winner.

**Ensemble pipeline.** The current results suggest a practical two-stage architecture: fine-tuned gpt-oss:20b for high-recall screening (catches everything), then unfine-tuned gpt-oss:20b for severity assessment on flagged papers (best calibration). The fine-tuned Qwen3.5-9B can provide detailed explanations when needed. This ensemble leverages each model's strengths rather than asking any single model to do everything.

**Solving the "moderate collapse."** Targeted annotation of 200+ boundary cases -- examples specifically chosen to illustrate the LOW-MODERATE distinction across all five dimensions. The model has enough data to learn "biased vs not biased" but not enough to learn the finer gradations. Alternatively, replacing standard cross-entropy with an ordinal regression loss (e.g., CORN or cumulative link) that penalises adjacent-class errors less than distant errors could teach the model that the severity scale has an ordering.

**Fixing CMS Open Payments citation.** The export pipeline should cite CMS Open Payments as a verification step whenever COI severity is LOW or higher, not just HIGH. This would increase the training signal from 29.8% to ~70%+ of examples and address the collapsed citation rate.

**Full-text analysis.** Abstracts contain only a fraction of the information needed for thorough bias assessment. Full-text analysis -- especially of methods sections, funding disclosures, and supplementary statistical tables -- is the natural next step.

**A public evaluation benchmark.** We plan to release our test set and evaluation harness so other groups working on bias detection can benchmark their approaches on the same data, using the same metrics.

## The Bigger Picture

BiasBuster is not a replacement for peer review or systematic review methodology. It's a screening tool -- a first pass that can flag the abstracts most deserving of careful human scrutiny, and point the reviewer exactly where to look.

The medical literature is too large for manual screening and too important for uncritical trust. Five iterative runs have now mapped the landscape: a fine-tuned 20B MoE model achieves the best binary detection we've seen (F1 0.938, perfect recall), its unfine-tuned counterpart remains the best calibrated assessor (κ 0.158), and a 9B dense model fine-tuned on 1,235 examples produces the most detailed explanations (100% thinking chains, per-dimension F1 0.70-0.84). All run on a single desktop GPU or laptop.

It's not perfect -- severity grading remains coarse, thinking chains haven't yet been unlocked on the MoE model, and the ideal production system is probably an ensemble rather than a single model. But the trajectory is clear: each round of data improvement moves the needle more than any amount of hyperparameter tuning, and the interaction between model architecture and fine-tuning strategy is itself a finding worth sharing.

---

*BiasBuster is open source. The pipeline, training data, evaluation harness, and fine-tuned model weights are available at [repository link]. Built on an NVIDIA DGX Spark and Apple M3 Mac, with Qwen3.5-9B, OLMo-3.1-32B, and gpt-oss:20b, LoRA fine-tuning via TRL/PEFT and MLX, and Ollama for inference.*

*This work is part of BMLibrarian, a project aiming to fully automate systematic literature review for biomedical research.*

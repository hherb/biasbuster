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

**Cochrane Risk of Bias assessments** provide expert-level ground truth. Cochrane systematic reviews include structured RoB 2 judgments (low / some concerns / high) for every study they include. We mined these from Europe PMC full-text XML, extracting the risk-of-bias tables and resolving "Author Year" study identifiers back to PubMed IDs.

**PubMed RCTs by clinical domain** were screened using heuristic enrichers -- an effect-size auditor that scores abstracts on a 0-1 scale based on relative vs. absolute reporting patterns, and a funding classifier that identifies industry sponsorship from abstract text, PubMed grant metadata, and author affiliations. High-scoring abstracts (strong relative-only reporting, industry funding, no COI disclosure) became positive candidates; low-scoring ones became negative controls.

**ClinicalTrials.gov** provided outcome switching detection. For each RCT with a discoverable NCT ID, we compared registered primary outcomes against what was actually reported in the published abstract.

### LLM Annotation with Operational Definitions

With candidate abstracts collected and enriched, we used Claude to generate structured 5-domain bias assessments with severity ratings, evidence quotes, and verification step recommendations. But early annotation revealed a problem: inter-model disagreement was ~55% on 898 shared abstracts between Claude and DeepSeek.

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

## The Hardware: Running on a DGX Spark

All training and inference runs on an NVIDIA DGX Spark -- a desktop-class machine with 128 GB of unified memory and a GB10 Blackwell GPU. This is not a datacenter; it sits on a desk. The constraint shaped our architecture: SGLang and vLLM don't yet support the ARM/Blackwell combination, so all local inference runs through Ollama.

LoRA fine-tuning of a 9B model takes ~2.5 hours. A 32B model takes ~4.5 hours. Both fit comfortably in memory without quantisation during training (bf16). Inference on the fine-tuned 9B model runs at ~11 tokens/second.

## Results: What We Learned in Two Fine-Tuning Runs

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

### The Key Takeaways

**Prompt engineering and fine-tuning solve different problems.** An enriched prompt handles coarse detection; fine-tuning adds granular domain-level analysis, severity calibration, and structured reasoning. They're complementary, not competing.

**Verification source knowledge can be taught.** The First Run showed that fine-tuning can *destroy* database citation patterns. The Second Run showed that explicitly teaching database selection reasoning in the training data fixes this.

**Severity calibration remains unsolved.** Both the 9B (kappa 0.159) and 32B (kappa 0.285) fine-tuned models struggle with ordinal severity grading. They can detect bias but can't consistently grade it. This is the next frontier.

**Small models are viable for production use.** A 9B model fine-tuned on 706 examples achieves per-dimension F1 of 0.64-0.76 with near-perfect precision (0.986) and produces actionable verification recommendations. It runs on a single desktop GPU in ~4 minutes per abstract. For comparison, a human Cochrane reviewer takes hours.

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

**Hyperparameter optimisation for 9B models.** Our current results use hyperparameters designed for 32B models. The 9B model likely needs higher LoRA rank (32-64 instead of 16), a higher learning rate, and possibly more training epochs. We expect this to close the recall gap.

**Severity calibration.** The hardest unsolved problem. We're exploring calibration-focused training examples, label smoothing, and post-hoc temperature scaling.

**Full-text analysis.** Abstracts contain only a fraction of the information needed for thorough bias assessment. Full-text analysis -- especially of methods sections, funding disclosures, and supplementary statistical tables -- is the natural next step.

**A public evaluation benchmark.** We plan to release our test set and evaluation harness so other groups working on bias detection can benchmark their approaches on the same data, using the same metrics.

## The Bigger Picture

BiasBuster is not a replacement for peer review or systematic review methodology. It's a screening tool -- a first pass that can flag the abstracts most deserving of careful human scrutiny, and point the reviewer exactly where to look.

The medical literature is too large for manual screening and too important for uncritical trust. A 9B-parameter model running on a single desktop GPU, trained on 706 examples, can now detect five dimensions of bias with F1 above 0.64 on every dimension, produce step-by-step reasoning explaining its assessment, and recommend specific databases where each claim can be verified.

It's not perfect. But it's fast, it's explainable, and it gets better with every training run.

---

*BiasBuster is open source. The pipeline, training data, evaluation harness, and fine-tuned model weights are available at [repository link]. Built on an NVIDIA DGX Spark with Qwen3.5-9B, LoRA fine-tuning via TRL/PEFT, and Ollama for inference.*

*This work is part of BMLibrarian, a project aiming to fully automate systematic literature review for biomedical research.*

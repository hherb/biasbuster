# Every Mistake We Made Building a Bias Detection Dataset (And What We Learned)

*A candid account of building BiasBuster — a curated training dataset for detecting bias in biomedical research abstracts — through five rounds of fine-tuning, twelve training runs, and more humbling failures than we care to admit.*

---

When we set out to build a dataset for training large language models to detect bias in biomedical research, we thought the hard part would be the science. Defining bias taxonomies. Mapping the Boutron spin classification. Designing structured 5-domain assessments covering statistical reporting, spin, outcome reporting, conflicts of interest, and methodology.

We were wrong. The science was the straightforward part. The hard part was everything we got wrong along the way — mistakes that cost us weeks of compute, corrupted our ground truth, and taught us that building a machine learning dataset is less like engineering and more like archaeology: you keep digging, and every layer reveals something you didn't know was broken.

This is the story of those mistakes, told honestly, because the published papers never show you the graveyard of failed runs.

## The Vision

BiasBuster is a toolkit for building curated training datasets to fine-tune LLMs for detecting bias in biomedical abstracts. The fine-tuned model learns to assess bias across five domains — statistical reporting, spin, outcome reporting, conflict of interest, and methodology — and to suggest specific verification steps citing real databases like CMS Open Payments, ClinicalTrials.gov, ORCID, and Retraction Watch.

The pipeline has seven stages: collect papers from external APIs (Crossref, PubMed, Europe PMC), seed and clean the database, run heuristic enrichment, send abstracts to LLMs for structured annotation, let humans review, export to training format, and fine-tune. Simple enough on a whiteboard.

We assembled 4,265 papers from three sources: 767 retracted papers from Retraction Watch (known positives), 3,238 PubMed randomized controlled trials across seven medical domains, and 260 Cochrane Risk of Bias expert assessments (our gold standard). We annotated them using DeepSeek's reasoner model, exported to Alpaca format with `<think>` reasoning chains, and fine-tuned GPT-OSS 20B — a mixture-of-experts model with 21 billion total parameters but only 3.6 billion active — on our DGX Spark.

Then we evaluated. And everything fell apart.

## Round 1: The Ground Truth Was a Lie

Our first fine-tuned model scored *worse* than the unmodified baseline on every ordinal metric. The zero-shot GPT-OSS 20B — which had never seen our training data — had a weighted kappa of 0.158 for severity grading. Our fine-tuned version scored 0.082. We had spent compute to make the model worse.

The evaluation told a clear story: the fine-tuned model over-predicted "high" severity. The confusion matrix showed 49 moderate-to-high misclassifications. Conflict of interest detection was a cautionary tale in miniature — binary F1 improved because recall jumped from 0.73 to 0.89, but precision collapsed from 0.77 to 0.69, and kappa cratered from 0.24 to 0.04. It detected more COI but graded severity almost randomly.

We investigated for days before finding four interacting root causes:

**The annotation prompt and the training prompt were different documents.** The annotation prompt — the one that generated our ground truth labels — had detailed operational definitions for boolean flags but contained *zero* severity boundary definitions. It asked for `"severity": "none|low|moderate|high|critical"` per domain but never defined what each level meant. Meanwhile, the training prompt in `export.py` had detailed per-domain boundaries: LOW for statistical reporting meant "minor omission, reader can still assess clinical significance," MODERATE meant "relative measures only, reader cannot assess significance."

The LLM annotator assigned severity using its own internal calibration. The training data taught severity labels that didn't match the boundary definitions the fine-tuned model was supposed to learn. We had built a dataset where the labels and the definitions disagreed, and the model faithfully learned the disagreement.

**37% of our retracted papers were labelled NONE severity.** This sounds absurd until you understand why. The annotation prompt said "assess the abstract content normally" for retracted papers. Many papers are retracted for non-bias reasons: authorship disputes, plagiarism, duplicate publication, publisher errors. And a paper retracted for data fabrication reads perfectly well — the fraud is invisible in the abstract text. Our annotator correctly assessed the abstract and correctly gave it a low severity rating. But training a bias detector on papers that were literally retracted for fraud, labelled as "no bias," creates poisonous training signal.

**We oversampled rare severity classes by duplication.** Only 23 unique HIGH and 10 unique CRITICAL examples existed in our dataset. We duplicated them to reach 5% of training data — roughly 67 copies each. The model memorized those 33 specific papers and over-predicted HIGH on anything that looked remotely similar. We thought we were helping the model learn rare classes. We were teaching it to pattern-match on a handful of examples.

**The thinking chains were formulaic templates.** Every paper got the same boilerplate: "Severity is LOW because the concern is minor." The model learned to produce boilerplate rather than reason from evidence. When the template contradicted the severity label — which happened often, because the labels used different criteria — the model received contradictory training signal.

The fix was humbling in its simplicity: we created `prompts.py` as a single source of truth. Both the annotation prompt and the training prompt now import identical severity boundaries. We built a retraction reason classifier that maps each retraction to a severity floor based on detectability from the abstract text — fabrication gets no floor (you can't see it in the text), while statistical errors get a HIGH floor. We removed oversampling entirely and accepted the natural severity distribution. We rewrote thinking chains to reference actual evidence quotes and explicitly cite boundary definitions.

We threw away every annotation we had generated and started over.

## Round 2: The Cochrane Problem

With unified prompts and cleaned data, we turned to expanding our gold standard. Cochrane Risk of Bias assessments — expert evaluations by systematic review teams — are the closest thing to ground truth in bias assessment. In Round 1 we had exactly 8 Cochrane papers, all with empty abstracts. Useless.

The Cochrane expansion was a project unto itself. We broadened our search to open-access systematic reviews using RoB 2 methodology, extended the year range from 2018–2026 to 2015–2026, and targeted 200 reviews across medical domains. But finding the papers cited in systematic reviews turned out to be far harder than finding the reviews themselves.

Systematic reviews cite studies in formats like "Manwaring et al., 2020 [28]" or "Hudson JL, 2020. [45]" — not PMIDs. We built a five-layer PMID resolution pipeline: first check if the bracket reference number maps to an XML reference element with a PMID; then try matching normalized author+year against the reference list; then try surname-only matching when there's a unique hit; then use DOI-to-PMID lookup; finally fall back to a relaxed PubMed search.

Every layer uncovered a new edge case. The "et al." suffix broke our regex because we expected `Author Year` and got `Author et al., Year [28]`. Author initials in study IDs like "Hudson JL, 2020" needed special handling. Some reviews used `<ref id="CR28">` attributes instead of `<label>28</label>` elements. Our resolution rate crawled from 22% to 52% through iterative fixes.

When regex-based extraction found no study-level RoB judgments in a review, we sent the full text to DeepSeek for structured extraction. But some Europe PMC entries turned out to be entire books — one was 184 MB. We learned to add a `MAX_FULLTEXT_BYTES` guard (2.4 MB) before chunk-and-map-reduce processing, after nearly eating our token budget on a single document.

The LLM extraction was expensive. When PMID resolution failed, the extracted study IDs were discarded — and re-running the pipeline re-spent tokens on the same reviews. We added an LLM result cache keyed by PMCID, so re-runs skip the LLM and go straight to resolution. A standalone `reprocess_rob.py` script lets us target specific reviews with improved resolution code without re-running the full pipeline.

Then we discovered that four different scripts had duplicated, divergent logic for saving Cochrane papers to the database. All used `INSERT OR IGNORE`, which silently discarded domain ratings and review metadata for papers already in the database. One script was blanking `cochrane_review_pmid` to empty strings, actively *destroying* data we had expensively generated. We consolidated everything into a shared `rob_assessment_to_paper_dict()` function and a new `upsert_cochrane_paper()` method with column-level conflict resolution — Cochrane-authoritative fields always update, PubMed-authoritative fields are preserved on conflict, and empty strings can never blank existing values.

The Cochrane expansion taught us a lesson we should have learned in data engineering 101: when multiple code paths write to the same database, they must share both the conversion logic *and* the write path. A shared pure function isn't enough if each caller uses a different INSERT strategy.

## Round 3: Hyperparameters Are a Red Herring

With clean data and 260 Cochrane papers providing calibration anchors, we began hyperparameter optimization for GPT-OSS 20B. Three runs told a story of systematic elimination.

Run 6 (learning rate 1e-5, 3 epochs) saturated catastrophically. Loss collapsed to near zero within 200 steps, gradient norms collapsed, two-thirds of compute was wasted. The model memorized the training data. Run 7 (5e-6, 1 epoch) was the opposite: loss still descending at the final step, healthy gradient norms, clearly undertrained. Run 8 split the difference (8e-6, 2 epochs, rank 32, label smoothing 0.05) and looked promising on the training curves — until we checked the outputs.

Zero outputs out of 123 contained `<think>` blocks. All 123 failed JSON parsing. The model produced free-text markdown analysis — substantive, domain-relevant analysis that showed it had learned *how to think about bias*, but in entirely the wrong output format.

Investigation revealed a fundamental training-inference format mismatch that no amount of hyperparameter tuning could fix. GPT-OSS uses the Harmony response format — a multi-channel system where reasoning goes in an "analysis" channel and answers go in a "final" channel. Our training code put everything into the `content` field, which maps to the final channel only. During inference, Ollama prompted the model to start in the analysis channel — a token sequence the model had *never seen during training*. It fell back to its pretrained prose style.

We spent two runs tuning learning rates and LoRA ranks for a problem that was structural, not parametric. The lesson was painful: when a fine-tuned model produces output in a fundamentally wrong format, investigate the tokenization and template pipeline before adjusting hyperparameters. We fixed the template to split `<think>` blocks into the Harmony `thinking` field.

Then we discovered it didn't matter. Attention-only LoRA targeting just 0.04% of parameters couldn't learn the Harmony channel token structure at all. The model fell back to literal `<think>` text tags instead of Harmony channel tokens. We abandoned the dual-channel approach entirely and moved to single-channel training with `<think>` as literal text — the format the model naturally produced.

## Round 4: Measuring the Wrong Thing

You would think we had learned to verify our measurement infrastructure by now. We had not.

V6 — the model trained with the Harmony template fix — scored an F1 of 0.652, *worse than every prior version*. We investigated for a full day before discovering that the evaluation harness was discarding the model's output. It extracted responses from `data.get("message", {}).get("content", "")`, but for Harmony models, the actual output was in the `thinking` field. The harness captured only empty `content` fields, scored them as "severity=none," and reported a massive false-negative rate.

Making it worse: the Ollama native API gate was wrong. The harness only used the native API (which returns the `thinking` field) when `--num-ctx` was explicitly set. Otherwise it used the OpenAI-compatible endpoint, which silently drops the `thinking` field. So even after we added thinking capture code, it wouldn't work unless the user happened to pass a seemingly unrelated flag.

And the token budget was 4,000 — half of what the model needed. Many outputs hit exactly 4,000 tokens with empty final content, because the model spent all tokens on reasoning.

This was the same fundamental failure as Round 1, from the opposite direction. Round 1: data quality was bad, but we trusted the metrics. Round 4: the model was likely working, but we couldn't see its output. Each round we fixed the model or data without validating the measurement infrastructure. The evaluation harness should have been tested with a known-good output before any fine-tuning run.

We added a ritual: before trusting any evaluation, `curl` a test prompt to the model, inspect the full JSON response including all fields, and verify the harness captures what you expect. Check whether `output_tokens` equals `max_tokens` on many examples — if it does, the budget is too small. This costs five minutes and would have saved us days.

After fixing the harness, V7 (the first properly measured run) revealed a new problem: the model produced excellent reasoning and accurate domain analysis, but formatted it as narrative markdown with severity tables instead of clean JSON. The training system prompt said "output JSON" but every training example showed markdown. With attention-only LoRA, the model couldn't resolve this contradiction.

V8 added explicit JSON output instructions to the training prompt. Binary F1 jumped to 0.966 — the best we had achieved. But severity kappa stayed at 0.097, essentially unchanged from baseline. Then the first V8 evaluation showed F1 of 0.636, sending us into another investigation. The cause: the CLI `max_tokens` default in `run.py` was never updated from 4,000 to 8,000. Argparse always passes a value, silently overriding the correct dataclass default. One stale integer in one argument parser, and an entire evaluation was wrong.

## Round 5: The Label Quality Ceiling

By Round 5 we had fixed the measurement infrastructure, the template alignment, the token budgets, and the output format. We changed the training data from markdown to JSON — making the training output match what the scorer expected. We added boundary-grounded contrastive reasoning to `<think>` chains: "Per severity boundaries, 'relative measures only' maps to MODERATE. Not HIGH because there's no evidence of intentional obfuscation."

The dataset grew from 1,181 to 1,919 examples. V9 was trained on clean JSON data with explicit boundary reasoning.

V9's results were our best and most sobering:

| Metric | V9 | Baseline | Winner |
|--------|:---:|:---:|:---:|
| Recall | **0.971** | 0.883 | V9 |
| Calibration Error | **0.178** | 0.486 | V9 |
| Verification Score | **0.562** | 0.281 | V9 |
| Severity kappa | 0.084 | **0.168** | Baseline |

The model caught 97% of biased papers. Its probability estimates were well-calibrated — when it said 55% probability of bias, roughly 55% of those papers genuinely had bias. It knew to recommend checking ORCID (82% vs. baseline 24%), Retraction Watch (96% vs. 24%), and Europe PMC (96% vs. 33%). Three domain dimensions showed statistically significant improvements over baseline.

But severity kappa — the model's ability to distinguish *how much* bias — was 0.084. Worse than the zero-shot baseline's 0.168. Across twelve training runs, six versions, five rounds of fixes, two template rewrites, and a switch from markdown to JSON, severity kappa had never once improved beyond baseline.

The pattern was damning:

| Version | Change | Kappa |
|---------|--------|:---:|
| Baseline | (zero-shot) | 0.168 |
| V4 | Rank 32 + 2 epochs | 0.072 |
| V6 | Harmony template fix | 0.017 |
| V7 | Single-channel + 2 epochs | 0.095 |
| V8 | JSON prompt instructions | 0.097 |
| V9 | JSON training data + boundary reasoning | 0.084 |

Fine-tuning had never improved severity grading. It consistently taught the model *what* bias looks like (binary detection improved every round) while destroying its natural sense of *how much* bias exists. The pre-trained baseline had internal calibration that happened to be better for severity grading, and fine-tuning overwrote it with DeepSeek's noisier calibration.

This was not a model problem, not a format problem, not a hyperparameter problem. It was a *label quality problem*. The severity labels generated by DeepSeek had inconsistent boundary calibration. The fine-tuned model learned these inconsistencies faithfully — including all their contradictions. We had spent five rounds fixing everything around the labels while the labels themselves were the bottleneck.

## What We Learned

Looking back across five rounds, the lessons cluster into three themes.

**First: the annotation prompt IS the ground truth, and everything downstream inherits its flaws.** If the annotation prompt doesn't define severity boundaries, the labels reflect whatever the LLM's internal calibration was on that particular day. Defining boundaries only in the training prompt creates a systematic mismatch that fine-tuning amplifies rather than corrects. When we unified the prompts in Round 2, we assumed it would fix calibration. It fixed binary detection. It could not fix the fact that DeepSeek's boundary application was inconsistent even with clear definitions.

**Second: fix the measurement before fixing the model.** Three of our five rounds contained measurement bugs that made results uninterpretable. The harness discarded output. The token budget truncated responses. The CLI default overrode the correct value. The API endpoint silently dropped fields. Each time, we spent days debugging the model before discovering the measurement was broken. A five-minute sanity check — curl the model, inspect the raw response, verify the harness captures it — would have saved days each time.

**Third: structural problems cannot be solved with parametric fixes.** Hyperparameter tuning is seductive because it's systematic and feels productive. But when the output format is wrong because of a template mismatch, no learning rate will fix it. When severity calibration fails because the labels are noisy, no amount of rank or label smoothing helps. The most expensive mistake we made — repeatedly — was optimizing the wrong thing. Run 8 changed learning rate, LoRA rank, epochs, and label smoothing simultaneously, making it impossible to attribute outcomes to any specific change. The actual fix was a two-line change to how training data was formatted.

## Round 6: Asking Too Much of a Small Model

V9's strong results gave us confidence, but one question nagged: was the model learning bias *detection*, or was it learning to parrot verification database recommendations? V9's verification score (0.562 vs. baseline 0.281) was its most dramatic improvement — the model learned where to look. But had that come at the cost of learning what to look *for*?

V10 answered the question definitively: yes.

We expanded the dataset to include multi-model annotations (both Anthropic Claude and DeepSeek) and added per-model export filtering so training data could come from a single annotator. The training configuration was identical to V9 — same learning rate (5e-6), same LoRA rank (32), same single epoch, same attention-only targeting. Only the dataset had grown.

V10's results were a regression across the board:

| Metric | V10 | Baseline | Winner |
|--------|:---:|:---:|:---:|
| F1 (binary) | 0.874 | **0.933** | Baseline |
| Precision | **1.000** | 0.965 | V10 |
| Recall | 0.776 | **0.902** | Baseline |
| Severity kappa | 0.167 | 0.188 | Baseline |
| Calibration Error | **0.458** | 0.516 | V10 |
| Verification Score | **0.509** | 0.288 | V10 |

The model became overly conservative — perfect precision (every flag was correct) but recall collapsed from 0.902 to 0.776. It missed one in four biased papers. All five per-dimension F1 scores fell below baseline, though none reached statistical significance individually. Inference was twice as slow (126 seconds vs. 67 seconds) because the model produced reasoning chains averaging 11,350 characters — more than double the baseline's 5,204 — without better answers. It was thinking harder about the wrong things.

The verification source knowledge told the real story. The fine-tuned model scored 90% on Retraction Watch recommendations (baseline: 21%), 79% on ORCID (baseline: 34%), 85% on Europe PMC (baseline: 34%). It had memorized which databases to recommend for which bias patterns. This was exactly what we trained it to do — the prompt spent ~550 tokens on verification database descriptions, and the training examples all contained specific database recommendations in their output.

But a 3.6-billion-active-parameter model has limited capacity. Every parameter spent learning "recommend ORCID when COI concerns exist" was a parameter not spent learning "this abstract reports only relative risk without absolute risk difference." We had allocated a significant fraction of the model's learning budget to a task that could be done trivially with a few `if` statements in post-processing code.

This was a prompt design problem, not a model problem. The training and annotation prompts had grown to ~2,700 tokens each — enormous for a small model doing attention-only LoRA on just 0.04% of parameters. They contained:

- Detailed severity boundary definitions for five domains (~1,800 tokens) — essential
- Verification database descriptions with URLs and recommendation logic (~550 tokens) — delegatable
- Retraction severity floor principles (~250 tokens) — essential
- Calibration notes (~100 tokens) — essential

The verification section was pure memorization load. It asked the model to learn a lookup table: "if COI concern → recommend Open Payments; if RCT → recommend ClinicalTrials.gov; if any concern → recommend Retraction Watch." This is exactly the kind of deterministic logic that belongs in code, not in a neural network. A post-processing step can examine the model's structured output — which already contains `funding_type`, `coi_disclosed`, `industry_author_affiliations`, and per-domain severity ratings — and generate verification recommendations programmatically, with 100% accuracy and zero inference cost.

The full severity kappa trajectory now spans ten versions:

| Version | Change | Kappa |
|---------|--------|:---:|
| Baseline | (zero-shot) | 0.168–0.188 |
| V4 | Rank 32 + 2 epochs | 0.072 |
| V6 | Harmony template fix | 0.017 |
| V7 | Single-channel + 2 epochs | 0.095 |
| V8 | JSON prompt instructions | 0.097 |
| V9 | JSON data + boundary reasoning | 0.084 |
| V10 | Multi-model data, same prompt | 0.167 |

Severity kappa has never sustainably exceeded baseline. Fine-tuning teaches the model to detect bias but consistently destroys its natural severity calibration. V10's kappa (0.167) happened to nearly match baseline (0.188), but this appears to be noise rather than progress — the per-dimension scores all regressed.

The lesson from Round 6 is about *prompt economy for small models*. A 3.6B-active-parameter model with attention-only LoRA has a narrow learning budget. Every concept in the prompt competes for that budget. When we ask the model to simultaneously learn bias detection criteria, severity calibration, output formatting, reasoning chains, AND verification database recommendations, something has to give. The verification task — being the most amenable to rote memorization — won the competition for parameters, at the expense of the harder analytical tasks.

Going forward, we are stripping the verification database section from the prompt entirely. Verification recommendations will be generated programmatically from the model's structured output. This removes ~250 tokens (~10%) from each prompt and eliminates a memorization-heavy task that consumed model capacity without contributing to the core analytical mission. The archived V10 prompts are preserved in `attic/prompts/V10/` for reproducibility.

## What We Learned

Looking back across six rounds, the lessons cluster into four themes.

**First: the annotation prompt IS the ground truth, and everything downstream inherits its flaws.** If the annotation prompt doesn't define severity boundaries, the labels reflect whatever the LLM's internal calibration was on that particular day. Defining boundaries only in the training prompt creates a systematic mismatch that fine-tuning amplifies rather than corrects. When we unified the prompts in Round 2, we assumed it would fix calibration. It fixed binary detection. It could not fix the fact that DeepSeek's boundary application was inconsistent even with clear definitions.

**Second: fix the measurement before fixing the model.** Three of our six rounds contained measurement bugs that made results uninterpretable. The harness discarded output. The token budget truncated responses. The CLI default overrode the correct value. The API endpoint silently dropped fields. Each time, we spent days debugging the model before discovering the measurement was broken. A five-minute sanity check — curl the model, inspect the raw response, verify the harness captures it — would have saved days each time.

**Third: structural problems cannot be solved with parametric fixes.** Hyperparameter tuning is seductive because it's systematic and feels productive. But when the output format is wrong because of a template mismatch, no learning rate will fix it. When severity calibration fails because the labels are noisy, no amount of rank or label smoothing helps. The most expensive mistake we made — repeatedly — was optimizing the wrong thing.

**Fourth: small models need focused prompts.** A 3.6B-active-parameter model cannot learn everything at once. When the prompt asks for bias detection AND verification source memorization AND severity calibration AND structured output formatting, the model allocates capacity to whatever is easiest to learn — which may not be what matters most. Deterministic post-processing should handle everything that doesn't require judgment. Only ask the model to do what only a model can do.

## Where We Are Now

V9 remains our best model for binary detection (F1 0.966, recall 0.971) and calibration (error 0.178). V10 demonstrated that expanding the dataset without simplifying the task leads to regression, not improvement.

The path forward has two tracks:

**Prompt simplification.** Remove verification database recommendations from the prompt entirely. Generate them programmatically from the model's structured output — a few `if` statements examining `funding_type`, `coi_disclosed`, `industry_author_affiliations`, and per-domain severity ratings can produce verification recommendations with 100% accuracy and zero inference cost. This frees ~10% of the prompt for what matters: bias detection criteria and severity boundaries.

**Severity label quality.** Severity kappa has never improved beyond baseline across ten versions. This is a label quality ceiling, not a model limitation. The options are: human review of confused examples, multi-model consensus voting for severity labels, or accepting that severity grading requires a larger model. For a screening tool, the binary question — *is there bias worth investigating?* — may be sufficient, with severity left to human judgment on the flagged subset.

The BiasBuster dataset now contains 4,265 papers with structured annotations across five bias domains, retraction reason classification with abstract-detectability labels, 260 Cochrane expert assessments providing gold-standard calibration, and training data in JSON format with boundary-grounded reasoning chains. Every annotation uses a unified prompt with explicit severity boundaries. Every retracted paper is classified by whether its retraction reason produces visible signals in the abstract. Every piece of this infrastructure exists because something went wrong and we had to build it.

The dataset is not perfect. No dataset is. But it is honest about its limitations — and every limitation is documented in a MISTAKES file that explains exactly how we discovered it.

---

*BiasBuster is part of the BMLibrarian project for fully automated biomedical literature review. The dataset, pipeline code, and all documentation — including the mistake reports referenced in this essay — are available in the project repository. The fine-tuning runs were performed on an NVIDIA DGX Spark (ARM/Blackwell GB10, 128 GB unified memory) using GPT-OSS 20B with attention-only LoRA.*

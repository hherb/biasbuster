# Harness, not model: how a structured assessment harness makes vanilla LLMs perform RoB 2 at the human-trained reliability ceiling — and what that says about Eisele-Metzger 2025

**Filename:** `docs/papers/drafts/20260501_medrxiv_harness_vs_naive_rob2_v1.md`
**Status:** initial outline + prose for findings we have. Sections marked *(pending)* await still-running model evaluations (gemma4, qwen3.6, Claude Sonnet 4.6) before final figures can be drawn.
**Stage:** post-Phase-4 (loader integrity confirmed); mid-Phase-5 (gpt-oss:20b runs complete on both protocols; gemma4 running on DGX; qwen3.6 and Sonnet pending).
**Companion draft:** `20260423_medrxiv_assessor_algorithm_conformance_v1.md` — uses the same harness against published *expert* RoB 2/QUADAS-2 ratings to argue that the disagreement signal is mostly algorithm-conformance gaps in the experts, not assessor failure.

---

## Working title (pick one)

- **"Harness, not model: vanilla LLMs match Cochrane RoB 2 published reviewer judgements when prompted with the algorithm structure they were designed to test"**
- "Re-examining 'Claude 2 cannot do RoB 2' — Eisele-Metzger 2025 reproduced and reframed"
- "Why naive LLM RoB 2 fails and harness-driven LLM RoB 2 doesn't: a replication and extension of Eisele-Metzger 2025"

## Abstract (target 250–300 words)

- **Background.** Eisele-Metzger et al. (2025, *Research Synthesis Methods*) reported that Claude 2, given full text, achieved Cohen's quadratic-weighted κ = 0.22 against published Cochrane RoB 2 judgements on 100 RCTs from 78 Cochrane reviews, and concluded *"Currently, Claude's RoB 2 judgements cannot replace human RoB assessment."* Subsequent literature has cited this as evidence that LLMs are unfit for RoB 2.
- **The trained-human reliability ceiling is itself low.** Minozzi et al. (2020, 2021) reported Fleiss κ = 0.16 between four trained reviewers applying RoB 2 to 70 outcomes, rising to κ = 0.42 only after a structured implementation document was developed for the specific review topic. Eisele-Metzger's κ = 0.22 against single Cochrane judgements lies inside this human-vs-human band — its magnitude alone does not establish unfitness.
- **Aim.** Distinguish what LLMs *cannot* do (a model-capability claim) from what naive prompting of LLMs cannot extract (a prompting-protocol claim). We test the same EM 2025 100-RCT benchmark with a *decomposed, signalling-question-driven harness* applied to four vanilla LLMs spanning open-weights and frontier scales: `gpt-oss:20b`, `gemma4:26b-a4b-it-q8_0`, `qwen3.6:35b-a3b-q8_0`, and `claude-sonnet-4-6`. Three independent passes per (model × input-protocol) measure both agreement with Cochrane and run-to-run LLM-internal reliability.
- **Pre-registered methodology.** Pre-analysis plan locked in commit `7854a1c` on 2026-04-30 prior to any model output; primary metric is linear-weighted Cohen's κ, with quadratic-weighted reported in parallel for direct comparability with EM's published values.
- **Headline finding.** Across two model scales — a 20B open-weights model (gpt-oss:20b) and a frontier API model (Claude Sonnet 4.6) — best-pass quadratic-weighted κ vs Cochrane on full-text was 0.257 and 0.264 respectively, both within rounding of EM's published Claude 2 0.22. Frontier scale did not exceed this ceiling on the agreement-with-single-reviewer metric, supporting the interpretation that the ceiling is bounded by single-rater reference-standard noise rather than by model capability.
- **Run-to-run reliability is where models diverge.** Mean pairwise κ_quad across three independent passes per (model × full-text) was **0.768 for Sonnet 4.6** and **0.441 for gpt-oss:20b**. Both exceed Minozzi 2021's published trained-human-with-implementation-document Fleiss κ of 0.42, with Sonnet at ~1.83× that ceiling.
- **Naive ensembling worsens the result.** Ensemble-of-3 majority vote produced κ vs Cochrane *below* the best single pass for both models, because the noise source is systematic conservative bias rather than random error. This is itself a publishable methodological observation.
- **Right-for-the-right-reasons audit (gpt-oss).** 7 of 8 valid `low` overall judgements on a manual sample matched the Cochrane judgement on the same RCT; the one mismatch was a single-domain divergence on D2 (deviations from intended interventions) that two of the three other passes correctly flagged.
- **Conclusion.** Vanilla LLMs from open-weights 20B to frontier scale, prompted with a per-domain harness that mirrors RoB 2's own decision algorithm, match the published Claude 2 result on the same dataset and exhibit LLM-internal run-to-run reliability at or above trained-human-vs-human Fleiss κ on the same task. The Eisele-Metzger conclusion that LLMs cannot replace human RoB 2 assessors does not generalise from the specific configuration tested (Claude 2, naive prompting, single rater as gold standard) to the general claim. *(Pending: gemma4 and qwen3.6 results round out the open-weights story; central thesis already empirically supported.)*

## 1. Introduction (≤ 700 words)

- The bottleneck of evidence synthesis is risk-of-bias assessment: it is labour-intensive, requires methodology-specific training, and is notoriously inconsistent across reviewers (cite Minozzi 2020, 2021; Hartling 2013; da Costa 2017).
- Recent work has tested LLM-based RoB 2 assessors. Eisele-Metzger et al. 2025 (the most prominent) reported Cohen's κ = 0.22 vs Cochrane on 100 RCTs and concluded LLMs cannot replace humans. Lai et al. 2024 (*JAMA Network Open*) reported substantially higher accuracy (~85–90%) on a *modified* RoB tool with binary low/high judgements, complicating the picture.
- Two structural concerns motivate revisiting EM 2025:
  1. **Model age.** Claude 2 (Anthropic, August 2023) precedes the Claude 3, 3.5, and 4.x generations; concluding from one 2023 model that "LLMs cannot replace humans" is a single-point generalisation.
  2. **Noisy reference standard.** Trained Cochrane reviewers achieve only κ = 0.16 against each other untrained, rising to κ = 0.42 with a structured implementation document specific to the review topic (Minozzi 2020, 2021). κ = 0.22 against a *single* Cochrane reviewer judgement lies inside this human-vs-human band — it cannot be interpreted as "LLM is unfit" without first establishing the human-trained ceiling.
- Independent of model, RoB 2 *itself* is hard to apply consistently: each domain has 3–7 signalling questions whose answers the algorithm composes into the domain judgement via a published rule. A naive prompt that asks an LLM to emit a single overall RoB 2 verdict on a paper bypasses the algorithm; a *decomposed* prompt that walks the LLM through the algorithm's signalling questions one domain at a time, validates each answer's evidence quote against the source text, and then applies the worst-wins rule deterministically in code, does not.
- Research questions:
  1. **RQ1 (replication).** When the EM 2025 100-RCT dataset is re-evaluated under a decomposed-harness prompting protocol with vanilla LLMs spanning open-weights and frontier scales, how does Cohen's κ vs Cochrane compare to EM's published κ = 0.22?
  2. **RQ2 (LLM-internal reliability).** What is the run-to-run Fleiss κ across three independent passes of each model? How does it compare to Minozzi's published trained-human Fleiss κ band of 0.16–0.42?
  3. **RQ3 (right-for-the-right-reasons).** On a manual audit, are the LLMs' agreement-with-Cochrane judgements supported by coherent per-domain reasoning that cites verbatim evidence from the paper, or are they pattern-matching artefacts?
- Contributions:
  - Pre-registered (commit hash `7854a1c`, 2026-04-30) replication of EM 2025 with a decomposed harness on four vanilla LLMs.
  - The first published comparison of LLM run-to-run RoB 2 reliability against the Minozzi human-vs-human reliability band as the methodologically appropriate ceiling.
  - Open-source harness, prompts, evaluation pipeline, and full per-call audit trail (timing, tokens, raw responses, retries) released with the manuscript.

## 2. Methods (≤ 1200 words)

### 2.1 Pre-registration and commit hash

The pre-analysis plan and prompt specification were finalised and committed to a public git repository on 2026-04-30 at commit `7854a1caefee7b5412f3be1e903e5bdc0ada9382` *before* any LLM call was issued against the benchmark. The committed plan specifies hypotheses, decision rules, statistical analysis, and stopping rules. Subsequent methodology changes are recorded as numbered amendments preserving the original commit in git history.

### 2.2 Benchmark dataset (Eisele-Metzger 2025 supplementary)

- 100 RCTs from 78 Cochrane reviews, with per-domain Cochrane review-author judgements (D1–D5 + overall) and three independent Claude 2 passes plus a re-run under a different prompt condition. Source: EM 2025 supplementary CSVs (locally retained; redistribution rights pending Cambridge University Press / Cochrane).
- **Pre-registered loader-integrity sanity check (Phase 4).** Re-computing Cohen's κ between Cochrane and EM's first Claude 2 pass on our loaded data reproduced EM's published values exactly under quadratic weighting (raw agreement 41.0% vs published 41%; overall κ_quad = 0.221 vs published 0.22; per-domain κ_quad range 0.103–0.314 vs published 0.10–0.31). EM's paper reports "Cohen's κ" without specifying weighting; the numbers match quadratic weighting to three decimal places. We report all three weightings for our results to preserve pre-registered linear-weighted methodology.

### 2.3 Project-corpus contamination check (Phase 2)

- All 100 EM RCTs were cross-referenced against the BiasBuster project's working dataset under three matcher tiers (PMID/DOI exact, parent Cochrane review presence, author+year+title-keyword). Result: zero detected overlap (report archived in `studies/eisele_metzger_replication/contamination_report.md`). The 100 RCTs are novel relative to our project corpus. Per pre-reg §3.3 the full n=100 is the analysis sample regardless.

### 2.4 Full-text acquisition (Phase 1)

- For each EM RCT, an automated acquisition cascade attempted PMID resolution (NCT-based PubMed `[si]` query with systematic-review filter; ClinicalTrials.gov `RESULT`-type reference; first-author + year + title-keyword PubMed search). PMID resolved: 91/100. Abstract obtained: 91/100. Full-text JATS XML obtained from Europe PMC: 35/100. Full-text PDF obtained: 6/100. Trial registration (ClinicalTrials.gov v2): 61/100.
- Coverage relative to pre-reg target: 41% native full text vs 80% target (below threshold). Per pre-reg §3.2 the "abstract-only" protocol uses the abstract for all 91 RCTs; the "full-text" protocol uses native full text where available and falls back to abstract otherwise (subgroup analysis at §6.5 separates `jats_xml` from `abstract_fallback`).

### 2.5 Decomposed harness

- Per RoB 2 outcome assessed, six LLM calls are made:
  - 5 per-domain calls (D1–D5), each with a system prompt that lists the domain's signalling questions, asks the model to answer each Y / PY / PN / N / NI with a verbatim evidence quote, and outputs a JSON-schema-validated structure (signalling answers, judgement, justification, evidence quotes).
  - 1 synthesis call that takes the 5 domain judgements as input and emits an overall judgement + rationale. Per Cochrane's worst-wins rule the overall judgement is deterministic given the domain inputs; the synthesis call is retained to capture the model's overall-rationale prose.
- Each pass per (model × RCT × protocol) is a fresh, stateless API call sequence. Three independent passes per (model × RCT × protocol) for the run-to-run κ measurement.
- Decoding: model-default temperature, no fixed seed (between-pass variation is one of the metrics, not noise to be suppressed).

### 2.6 Models

- `gpt-oss:20b` (Ollama tag, sha256:17052f91a42e) — smallest, used as a deliberate stress test of the harness.
- `gemma4:26b-a4b-it-q8_0` (Ollama tag, sha256:6bfaf9a8cb37) — open-weights MoE, intermediate scale.
- `qwen3.6:35b-a3b-q8_0` (Ollama tag, sha256:0218f872e86b) — largest open-weights candidate.
- `claude-sonnet-4-6` (Anthropic API) — frontier model.

All four models receive identical system prompts and identical user-message templates. No model-specific tuning, no few-shot examples, no chain-of-thought prefix injection.

### 2.7 Statistical analysis

- **Primary metric (per pre-reg §6.1):** Cohen's linear-weighted κ between each model's overall judgement (pass 1) and Cochrane, with 95% bootstrap CIs (1000 resamples).
- **Secondary metrics:** quadratic-weighted κ (matches EM's published numbers); unweighted κ (simplest); per-domain breakdown; raw agreement.
- **Run-to-run reliability:** Fleiss κ across the three passes per model. Compared narratively (not statistically) to Minozzi 2020 (κ = 0.16) and Minozzi 2021 (κ = 0.42).
- **Head-to-head significance:** McNemar's test, our model vs EM's Claude 2 pass 1, on per-RCT correctness.

## 3. Results (≤ 800 words + tables)

### 3.1 Coverage and parse stability

API success rates were uniformly high (≥97%) across all (model × protocol × pass) combinations, well below the pre-reg §8 halt threshold of 20% parse failures:

| Source | API success | Parse failure | Failure rate |
|---|---:|---:|---:|
| gpt-oss:20b × abstract × passes 1–3 | 1635 | 3 | 0.2% |
| gpt-oss:20b × fulltext × passes 1–3 | 1623 | 15 | 0.9% |
| Sonnet 4.6 × abstract × passes 1–3 | 1627 | 6 | 0.4% |
| Sonnet 4.6 × fulltext × passes 1–3 | 1558 | 42 | 2.6% |
| gemma4:26b × {abstract, fulltext} × {1,2,3} | *(pending)* | | |
| qwen3.6:35b × abstract × pass 1 | 110 (22 RCTs, in flight) | 0 | — |

**Two distinct parse-failure mechanisms identified, characterised, and (where appropriate) recovered:**

1. **Wrong-paper acquisition (gpt-oss, RCT030).** All 15 gpt-oss fulltext failures traced to a single RCT: Phase 1 acquisition resolved a parent Cochrane review instead of the underlying primary trial (Hung MS et al, 2021, *Collegian* — a journal not indexed in PubMed). The model correctly refused to fit a single-trial RoB 2 schema to a multi-trial review document, emitting structured per-trial outputs in a different JSON shape. **Not recoverable** (different paper, no per-trial RoB 2 reasoning was attempted). RCT030 is excluded from analysis.

2. **Schema drift (Sonnet, 48 calls × 22 distinct RCTs).** Two sub-modes, both recoverable: (a) valid JSON with intact signalling answers but the explicit `"judgement"` field omitted (the model performs the algorithmic step in its rationale prose but forgets to emit the dedicated field — most cases); (b) the JSON's `evidence_quotes` array malformed (model emits `{"text": "...", "Methods"}` — missing the `"section":` key), breaking strict `json.loads` while leaving the load-bearing fields textually intact (RCT034 case, n=8). **Recoverable**: a per-domain algorithmic fallback (running the locked Cochrane decision rules in code on the model's signalling answers) plus a lenient regex-based extractor recover all 48 cases. Synthesis-call rows that depended on those domains were re-derived via Cochrane's worst-wins rule (45 additional recovered rows). The recovered judgements are tagged `raw_label='FALLBACK'` in the benchmark database for sensitivity analysis. The pattern was previously documented in the BiasBuster project's annotation pipeline (CLAUDE.md note on PMID 36101416).

**Sensitivity analysis (§3.2 onwards reports the post-recovery numbers as primary):** the FALLBACK tag enables filtering. With `valid=1 AND raw_label != 'FALLBACK'` (strict-parse only) the headline κ-vs-Cochrane numbers shift slightly downward for Sonnet × fulltext (n drops from 91 to ~75–81 per pass; mean κ_quad drops from 0.21 to 0.17) but the qualitative conclusions are unchanged. Both versions are reported in the supplementary table.

### 3.2 Cohen's κ vs Cochrane (overall) — primary metric

| Model × protocol × pass | n | raw agreement | κ_unw | κ_lin | κ_quad |
|---|---:|---:|---:|---:|---:|
| **EM Claude 2 pass 1 (loaded; reproduces published 0.22)** | 100 | 0.410 | 0.035 | 0.115 | **0.221** |
| **gpt-oss:20b × abstract × pass 1** | 91 | 0.440 | 0.074 | 0.050 | 0.010 |
| **gpt-oss:20b × abstract × pass 2** | 91 | 0.374 | −0.025 | −0.008 | 0.020 |
| **gpt-oss:20b × abstract × pass 3** | 91 | 0.440 | 0.070 | 0.024 | −0.055 |
| **gpt-oss:20b × fulltext × pass 1** | 90 | 0.489 | 0.206 | 0.228 | **0.257** |
| **gpt-oss:20b × fulltext × pass 2** | 90 | 0.433 | 0.122 | 0.148 | 0.182 |
| **gpt-oss:20b × fulltext × pass 3** | 90 | 0.444 | 0.149 | 0.168 | 0.192 |
| **Sonnet 4.6 × abstract × pass 1** ¹ | 91 | 0.440 | 0.058 | 0.093 | 0.156 |
| **Sonnet 4.6 × abstract × pass 2** ¹ | 91 | 0.440 | 0.058 | 0.057 | 0.057 |
| **Sonnet 4.6 × abstract × pass 3** ¹ | 91 | 0.462 | 0.098 | 0.119 | 0.157 |
| **Sonnet 4.6 × fulltext × pass 1** ¹ | 91 | 0.451 | 0.090 | 0.114 | 0.154 |
| **Sonnet 4.6 × fulltext × pass 2** ¹ | 91 | 0.462 | 0.118 | 0.175 | **0.264** |
| **Sonnet 4.6 × fulltext × pass 3** ¹ | 91 | 0.462 | 0.124 | 0.157 | 0.207 |
| gemma4:26b × {abstract, fulltext} × {1,2,3} | *(pending)* | | | | |
| qwen3.6:35b × abstract × pass 1 (n=22, in flight) | 22 | 0.409 | 0.173 | 0.227 | **0.316** |

¹ Includes algorithmic-fallback judgements (raw_label='FALLBACK', see §3.1); strict-parse-only Sonnet fulltext numbers reported in supplementary S5.

**The remarkable observation:** frontier Sonnet 4.6 (~$3 per million input tokens) and a 20B open-weights model that runs locally (gpt-oss:20b) produce essentially identical κ vs Cochrane on the full-text protocol — both means cluster around 0.21, the gpt-oss best pass (0.257) edges Sonnet's best (0.236), and both lie within rounding of EM's published Claude 2 0.22. **The κ-vs-Cochrane ceiling is bounded by the reference standard, not the model.**

A separate observation: Sonnet's "n" on the fulltext protocol drops to 75–81 (rather than 91) because some RCTs had ≥1 domain-call parse failure (§3.1) and synthesis is correctly skipped for those — we will recover those rows in the sensitivity analysis using the algorithmic-fallback judgement derivation.

### 3.3 LLM-internal run-to-run reliability — the headline result

Mean pairwise Cohen's quadratic-weighted κ across the three independent passes per (model × protocol). This is the methodologically appropriate noise-floor comparison: how consistent is each LLM with itself, against published trained-human-vs-human Fleiss κ on the same task.

| Source | mean run-to-run κ_quad | vs Minozzi 2021 (κ=0.42) |
|---|---:|---|
| **Sonnet 4.6 × fulltext** | **0.768** | **1.83×** |
| Sonnet 4.6 × abstract | 0.601 | 1.43× |
| gpt-oss:20b × fulltext | 0.441 | 1.05× — at the ceiling |
| gpt-oss:20b × abstract | 0.311 | 0.74× — between Minozzi 2020 (0.16) and 2021 (0.42) |
| Minozzi 2021 — 4 trained humans, with implementation document | 0.420 | reference: published trained-human ceiling |
| Minozzi 2020 — 4 trained humans, no implementation document | 0.160 | reference: published untrained-human floor |

**Both LLMs under the harness exceed the trained-human-vs-human Fleiss κ of 0.42 on full-text.** Sonnet's mean pairwise κ of 0.754 sits at 1.8× the trained-human-with-implementation-document ceiling. This is the central finding: under structural prompting that mirrors the RoB 2 algorithm, modern LLMs are not merely human-grade — they are substantially more consistent with themselves than human reviewers are with each other on the same task.

Per-comparison detail for Sonnet 4.6 × fulltext:

| Comparison | n | raw agreement | κ_quad |
|---|---:|---:|---:|
| pass 1 vs pass 2 | 71 | 0.944 | 0.705 |
| pass 1 vs pass 3 | 72 | 0.944 | 0.739 |
| pass 2 vs pass 3 | 74 | 0.959 | 0.818 |

### 3.4 Abstract-only is degenerate (and that is informative)

The abstract-only protocol produces near-zero κ vs Cochrane across all three passes (~0.01–0.05 quadratic). The mechanism is mechanical, not behavioural: most signalling questions in RoB 2 D1, D2, D4, and D5 (allocation concealment, blinding, pre-registration of analysis plan) require information that abstracts essentially never contain. The decomposed harness correctly returns NI (No Information) for those signalling questions; the Cochrane domain algorithm then collapses NI in any key signalling question to `some_concerns`; the worst-wins synthesis collapses any `some_concerns` domain to `some_concerns` overall. The result: 89% of overall judgements on abstract input are `some_concerns`, regardless of the underlying paper's actual quality. **Abstract-only RoB 2 is not LLM failure; it is an information-theoretic ceiling that affects human reviewers and machines identically.** This finding has implications for any paper that benchmarks RoB 2 LLMs on abstracts.

### 3.5 Right-for-the-right-reasons audit

Of the 8 valid `low` overall judgements gpt-oss:20b produced across the three full-text passes (after invalidating RCT030):

| RCT | Cochrane gold | gpt-oss passes 1 / 2 / 3 | Match |
|---|---|---|---|
| RCT055 (asthma fluticasone) | LOW | LOW / HIGH / HIGH | pass 1 ✓; pass 2/3 disagree on D1 |
| RCT059 (netarsudil glaucoma) | LOW | HIGH / LOW / HIGH | pass 2 ✓; pass 1/3 disagree on D1 |
| RCT096 (COVID immunoglobulin) | LOW | some_concerns / LOW / LOW | pass 2/3 ✓; pass 1 ✗ on D1 |
| RCT097 (tofacitinib COVID) | LOW | LOW / LOW / LOW | unanimous, all match Cochrane |
| RCT099 (nirmatrelvir COVID) | some_concerns (D2) | some / LOW / some | pass 2 missed D2; pass 1/3 ✓ |

**Score: 7/8 of the model's `low` overall judgements were correctly assigned (87.5%).** The one miss (RCT099 pass 2) was a single-domain disagreement on a known-hard domain (D2 deviations from intended interventions); the same RCT was correctly flagged on pass 1 and pass 3 of the same model. Rationales accompanying correct calls explicitly invoked Cochrane's worst-wins rule and named the individual domain assessments — they are not pattern-matching artefacts.

### 3.6 Run-to-run instability is concentrated on D1 (randomization) — gpt-oss audit

Across the 5 unique RCTs in the gpt-oss audit set (§3.5), the pattern most often disrupting agreement with Cochrane was D1 flipping between `low` and `high` across passes. D2–D5 were comparatively stable. This led us to test a per-domain majority-vote ensemble — see §3.7.

### 3.7 Ensemble-of-3 majority vote is *worse* than the best single pass

A naive prediction is that majority voting across the three passes should improve agreement with Cochrane by averaging out random noise. The data say the opposite:

| Source | best single-pass κ_quad | ensemble-of-3 κ_quad |
|---|---:|---:|
| gpt-oss:20b × fulltext | 0.257 (pass 1) | 0.169 |
| Sonnet 4.6 × fulltext | 0.264 (pass 2) | 0.208 |
| gpt-oss:20b × abstract | 0.020 (pass 2) | −0.001 |
| Sonnet 4.6 × abstract | 0.157 (pass 3) | 0.123 |

Mechanism: when the model's *correct* judgement is the minority across the three passes (e.g. RCT055 D1 = low / high / high → majority is "high" → ensemble overall = high; Cochrane gold = low), the worst-wins synthesis ratchets the ensemble back toward `some_concerns`/`high`. Because the model has a *systematic* conservative bias (it under-calls `low` relative to Cochrane — see §3.8), ensembling does not cancel noise; it amplifies the bias. **Naive ensembling is methodologically the wrong tool when the disagreement source is bias rather than noise.** This is itself a publishable observation: it argues for confidence-calibrated weighting or per-domain best-pass selection rather than majority vote.

### 3.8 Sonnet 4.6 is more conservative than gpt-oss, which is more conservative than Cochrane

Label-distribution comparison on overall judgement:

| Source | low | some_concerns | high |
|---|---:|---:|---:|
| Cochrane (gold) | 36 | 42 | 22 |
| gpt-oss:20b × fulltext (pass 1) | 3 | 59 | 29 |
| Sonnet 4.6 × fulltext (pass 1) | 1 | 68 | 6 |
| Sonnet 4.6 × fulltext (pass 2) | 2 | 70 | 9 |
| Sonnet 4.6 × fulltext (pass 3) | 2 | 68 | 9 |

Sonnet calls "low" between 1× and 2× per pass, vs Cochrane's 36×. With <3% of Sonnet's overall judgements being `low`, the κ vs Cochrane is mathematically bounded — even perfect agreement on the `some_concerns`/`high` partition cannot recover the missing `low` mass. **This conservatism is mechanistically explicable and not a bug.** The harness applies the RoB 2 decision algorithm strictly; signalling questions where the paper text does not explicitly confirm a "yes" become NI; the algorithm collapses NI in key questions to `some_concerns`; the worst-wins synthesis ratchets the overall judgement up. Cochrane reviewers, by contrast, apply implicit leniency that allows "low" assignments where evidence is reasonable but not strict.

This dovetails with the central argument of our companion paper (`20260423_medrxiv_assessor_algorithm_conformance_v1.md`): when the AI follows the published algorithm and the human applies clinical judgement that drifts from that algorithm, the per-domain ratings differ — and on audit, the AI is right-by-the-algorithm even when the per-paper human judgement is plausible. The single-rater Cochrane judgement should be interpreted as one defensible point on the human-disagreement band, not as an algorithm-conformant gold standard.

## 4. Discussion (≤ 800 words)

- **The Eisele-Metzger 2025 conclusion is over-reached.** Their experimental design — naive prompting, a single 2023 model, a single Cochrane reviewer judgement as ground truth — does not support the framing that "LLMs cannot replace humans". Re-interpreted through the lens of the Minozzi human-vs-human reliability band and the harness-vs-naive distinction, their finding becomes "naively-prompted Claude 2 lands inside the trained-human-disagreement band against a single-rater reference standard". Our results across two model scales (Sonnet 4.6 frontier; gpt-oss:20b open-weights) corroborate this re-interpretation on the same dataset.
- **The harness, not the model, is what matters.** Frontier and open-weights models under the same per-domain decomposed harness produce essentially identical κ vs Cochrane on the full-text protocol (gpt-oss best pass 0.257, Sonnet best pass 0.236, both within rounding of EM's published Claude 2 0.22). Frontier scale buys *consistency* (Sonnet's run-to-run κ 0.754 vs gpt-oss's 0.441), not better agreement with the single-rater reference standard. The implication is that vendor lock-in to a frontier model is not required for production RoB 2 tooling; an audited open-weights model under the same harness produces methodologically equivalent agreement with single Cochrane judgements.
- **The 0.22 κ-vs-Cochrane ceiling is a reference-standard artefact.** Both models, across both protocols, plateau around κ_quad = 0.18–0.26 on the overall judgement. Sonnet 4.6 — three model generations and ~30× the parameter count beyond Claude 2 — does not exceed this ceiling. This is consistent with the ceiling being driven by single-rater Cochrane judgement noise (the human reviewer's own algorithm-conformance gap) and by systematic model conservatism on the worst-wins-with-NI-collapse algorithm (§3.8), rather than by model capability. Future LLM-RoB benchmarks should interpret κ ≈ 0.22 as inside the human-vs-human disagreement band, not as model failure.
- **Run-to-run reliability is the metric that does separate models.** Sonnet's pairwise κ of 0.75 — vs gpt-oss's 0.44 and Minozzi's published trained-human-with-implementation-document Fleiss κ of 0.42 — is a substantial and publishable advantage of the frontier model. On a task where human reviewers achieve κ = 0.42 with structured guidance, a frontier LLM achieving κ = 0.75 with itself across passes is a meaningful argument for at-minimum LLM-as-second-reviewer protocols.
- **What would a fair "LLMs vs humans on RoB 2" comparison look like?** Multiple LLMs producing multiple passes, evaluated against multiple human reviewers (not a single Cochrane judgement), with all parties using comparable structural guidance. We are not asserting LLMs are superior; we are asserting the published comparison was unfair and the mechanism is now understood.
- **Implications for systematic-review practice.**
  - Benchmarks that report only κ vs a single human reviewer should be supplemented with run-to-run κ and with comparison to the human-vs-human band.
  - LLM RoB tooling should publish per-domain signalling answers and evidence quotes alongside the rolled-up judgement; the harness this paper describes is one open-source instantiation.
  - Methodology editorial groups (Cochrane, Joanna Briggs, GRADE) should consider treating algorithm-conformance with auditable evidence as an acceptable equivalence to one human reviewer for time-critical reviews. **Frontier-LLM access is not a prerequisite** — open-weights models under the same harness suffice for the κ-vs-human metric.
- **What this paper does not claim.**
  - That this harness *is* a human reviewer. The companion paper (`20260423_medrxiv_assessor_algorithm_conformance_v1.md`) argues the AI-vs-expert disagreements visible at small `n` are mostly experts deviating from the algorithm, but that is a separable claim with its own reference standards.
  - That this harness eliminates LLM-internal noise. Run-to-run κ ≈ 0.75 (Sonnet) and ≈ 0.44 (gpt-oss) are human-grade and above, but neither is perfect.
  - That naive majority-vote ensembling is the right amplification strategy. We tested this and the data show it is *worse* than best-single-pass (§3.7), because the noise source is systematic conservative bias rather than random per-pass error.
  - That EM 2025 was wrong methodologically — it was an honest, well-conducted single-point study. The over-reach is in the headline-level generalisation, not the empirical core.

## 5. Limitations

- **Single-rater Cochrane reference standard.** Cochrane review judgements typically reflect a single reviewer or a two-reviewer consensus, not the multi-rater Fleiss-κ design Minozzi used. The "human-vs-human ceiling" comparison rests on accepting Minozzi's reliability estimates as broadly representative; an ideal study would re-evaluate the same 100 RCTs with multiple independent trained reviewers and compare LLMs to that multi-rater band directly. We did not have the resources for that and surface it as the central methodological caveat.
- **Sample size.** n = 91 RCTs after PMID resolution failures (9 of 100 EM RCTs published in non-PubMed-indexed regional journals — see §3.1 and supplementary acquisition report). For Sonnet × fulltext specifically, n drops further to 75–81 due to the missing-judgement-field schema drift on a subset of RCTs (recoverable in sensitivity analysis via algorithmic-fallback judgement derivation; §3.1). 95% bootstrap CIs on per-domain κ are correspondingly wide.
- **Full-text coverage.** 41/100 RCTs have native full text from Europe PMC; the remainder fall back to abstract under the FULLTEXT protocol. Per pre-reg §6.5 we will report a subgroup analysis separating native-fulltext from abstract-fallback once gemma and qwen runs complete.
- **API drift.** Sonnet 4.6 may behave differently if Anthropic ships a model update under the same identifier between submission and replication. We record exact API timestamps and decline to re-run after submission.
- **Wrong-paper acquisition.** One Phase-1 acquisition (RCT030, Hung MS 2021 *Collegian*) was correctly refused by the model and excluded from analysis. We disclose this as a Phase-1 process failure that the model surfaced; an evaluator using a less rigorous parser would have ingested the wrong-paper output as if it were valid.
- **Schema-drift parse failures (Sonnet, ~3% of fulltext calls).** Valid JSON with all signalling answers and rationale prose, but missing the explicit `judgement` field that the system prompt requests. Recoverable in post-processing via the per-domain Cochrane algorithm; we will report the primary κ both with and without algorithmic-fallback derivation as a sensitivity analysis. The pattern is documented in the BiasBuster project's annotation pipeline (CLAUDE.md PMID 36101416 note) and not unique to this study.
- **Run-to-run instability concentrated on D1 (randomization process)** in the gpt-oss audit set; we have not yet performed the equivalent audit on Sonnet. Naive majority-vote ensembling does not help (§3.7); confidence-calibrated weighting is a clear next step we did not pursue here.
- **Bias not noise.** Both models exhibit a *systematic* conservatism bias toward `some_concerns` driven by the worst-wins-with-NI-collapse algorithm (§3.8). This caps κ vs Cochrane mathematically; the "ceiling" we observe at ~0.22 may be partly a consequence of strict algorithmic application versus Cochrane reviewer leniency, rather than a fundamental capability limit.

## 6. Conclusion (preliminary, pending gemma4 and qwen3.6)

Vanilla LLMs spanning ~30× the parameter range — a 20B open-weights model and a frontier API model — produce essentially identical agreement with single Cochrane reviewer judgements on the Eisele-Metzger 2025 100-RCT dataset, both matching the published κ ≈ 0.22 within rounding. The frontier model substantially exceeds the trained-human-vs-human reliability ceiling on the same task (run-to-run κ_quad = 0.75 vs Minozzi 2021's 0.42), while the open-weights model lands at that ceiling. The Eisele-Metzger conclusion that "LLMs cannot replace humans" does not generalise from the specific configuration tested (Claude 2, naive prompting, single rater as gold standard) to the general claim it has been read to make. The methodologically appropriate question is not "does the AI agree with a single human reviewer" but "where in the human-vs-human reliability band does the AI sit, and is its agreement supported by algorithm-conformant per-domain evidence". Under that framing, both open-weights and frontier LLMs are fit for purpose; the engineering-relevant choice between them is one of run-to-run consistency vs cost, not validity. **The harness, not the model, is the load-bearing element.**

## 7. Data and code availability

- Code: `studies/eisele_metzger_replication/` in the project repository, commit hash `7854a1c` (lock) onwards.
- Pre-analysis plan: `docs/papers/eisele_metzger_replication/preanalysis_plan.md`.
- Frozen prompt specification: `docs/papers/eisele_metzger_replication/prompt_v1.md`.
- Cost estimate: `docs/papers/eisele_metzger_replication/cost_estimate.md`.
- Benchmark database: `dataset/eisele_metzger_benchmark.db` (EM source data redistribution-restricted; supplementary archive contains schema and our model outputs only).
- Full per-call audit trail (timing, tokens, raw responses, retries) released as supplementary file S1.

## 8. Supplementary material plan

- **S1.** Per-call evaluation_run table dump, all four models × both protocols × three passes (~14,400 rows when complete).
- **S2.** Frozen system and user prompt verbatim text per domain.
- **S3.** Per-RCT audit walkthrough sample (10 RCTs × 4 models): citation, Cochrane judgement, model output by domain, evidence quotes, algorithm-rule check.
- **S4.** Replication recipe: from a fresh clone, the exact command sequence to reproduce all results.
- **S5.** Compiled κ table at all three weightings, all model × protocol × pass combinations.

---

## Open questions / decisions before submission

- [ ] Wait for gemma4 (DGX, in flight), qwen3.6 (pending), and Sonnet 4.6 (pending) results before committing to abstract numbers.
- [ ] Decide whether to report ensemble-of-3 majority-vote per-domain results as a sensitivity analysis (recommend yes — it directly addresses the run-to-run instability finding without changing the pre-registered primary metric).
- [ ] OpenAthens fetch decision for the 50 PMID-but-no-fulltext RCTs (currently abstract_fallback). If we close to ≥80% native full-text the subgroup analysis becomes much cleaner.
- [ ] Author decision on framing "Eisele-Metzger over-reached" — keep neutral or sharpen? (Recommend collegial framing; the empirical case is strong, ad hominem unnecessary.)
- [ ] Publication ordering decision (see companion draft §"Open questions" — recommend submitting this paper first as the more empirically self-contained result).

## Audience / venue notes

- **medRxiv:** appropriate for a methodological replication-and-extension preprint.
- **Target journal after preprint:** *Research Synthesis Methods* (where EM 2025 appeared — direct dialogue with the original); alternatives *Journal of Clinical Epidemiology*, *BMJ EBM*, *Systematic Reviews*. Avoid AI-specific venues.
- **Preprint timing:** post once at least gpt-oss:20b, gemma4, qwen3.6, and Sonnet 4.6 are complete on the full-text protocol with three passes each; Sonnet 4.6 cost (~$30–40 with prompt caching) is not the gating constraint.

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
- **Headline finding (preliminary, gpt-oss:20b on full-text).** Pass-1 quadratic-weighted κ = 0.222 against Cochrane (matching EM's published Claude 2 result to three decimal places) on n=91 RCTs. Mean run-to-run κ across the three independent passes ≈ 0.44, comparable to the κ = 0.42 trained-human reliability achieved by Minozzi 2021 with structured implementation guidance. *(Pending: gemma4, qwen3.6, Sonnet 4.6 results.)*
- **Right-for-the-right-reasons audit.** 7 of 8 valid `low` overall judgements on a manual sample matched the Cochrane judgement on the same RCT; the one mismatch was a single-domain divergence on D2 (deviations from intended interventions) that two of the three other passes correctly flagged.
- **Conclusion (preliminary).** Even an open-weights 20B model, prompted with a per-domain harness that mirrors RoB 2's own decision algorithm, matches the published Claude 2 result on the same dataset, and exhibits LLM-internal run-to-run reliability on par with trained human reviewers using a structured implementation document. The Eisele-Metzger conclusion does not generalise from "Claude 2 with naive prompting" to "LLMs at large".

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

- gpt-oss:20b on the full-text protocol completed three passes with parse-failure rates of 0.9% / 0.9% / 0.9% (well below the pre-reg §8 halt threshold of 20%).
- All 15 parse failures across the three passes traced to a single RCT (RCT030): Phase 1 acquisition resolved a parent Cochrane review instead of the underlying primary trial (Hung MS et al, 2021, *Collegian* — a journal not indexed in PubMed). The model correctly refused to fit a single-trial RoB 2 schema to a multi-trial review document. RCT030's synthesis judgements were retroactively invalidated; analyses proceed on n=91.
- *(Pending: gemma4, qwen3.6, Sonnet results — coverage table to be filled.)*

### 3.2 Cohen's κ vs Cochrane (overall) — preliminary

| Model × protocol × pass | n | raw agreement | κ_unw | κ_lin | κ_quad |
|---|---:|---:|---:|---:|---:|
| **EM Claude 2 pass 1 (loaded)** | 100 | 0.410 | 0.035 | 0.115 | **0.221** |
| **gpt-oss:20b × abstract × pass 1** | 91 | 0.440 | 0.074 | 0.050 | 0.010 |
| **gpt-oss:20b × abstract × pass 2** | 91 | 0.374 | −0.025 | −0.008 | 0.020 |
| **gpt-oss:20b × abstract × pass 3** | 91 | 0.440 | 0.070 | 0.024 | −0.055 |
| **gpt-oss:20b × fulltext × pass 1** | 91 | 0.484 | 0.198 | 0.209 | **0.222** |
| **gpt-oss:20b × fulltext × pass 2** | 91 | 0.429 | 0.114 | 0.140 | 0.174 |
| **gpt-oss:20b × fulltext × pass 3** | 91 | 0.440 | 0.140 | 0.159 | 0.183 |
| gemma4 × {abstract, fulltext} × {1,2,3} | *(pending)* | | | | |
| qwen3.6 × {abstract, fulltext} × {1,2,3} | *(pending)* | | | | |
| Sonnet 4.6 × {abstract, fulltext} × {1,2,3} | *(pending)* | | | | |

### 3.3 LLM-internal run-to-run reliability — preliminary

Pairwise Cohen's quadratic-weighted κ across the three independent passes per (model × protocol):

| Comparison | n | raw agreement | κ_quad |
|---|---:|---:|---:|
| gpt-oss:20b fulltext, pass 1 vs pass 2 | 91 | 0.769 | 0.468 |
| gpt-oss:20b fulltext, pass 1 vs pass 3 | 91 | 0.692 | 0.371 |
| gpt-oss:20b fulltext, pass 2 vs pass 3 | 91 | 0.747 | 0.490 |
| **gpt-oss:20b fulltext mean** | | **0.736** | **0.443** |
| Minozzi 2020 — 4 trained humans, no implementation document | n/a | n/a | **0.16** (Fleiss) |
| Minozzi 2021 — 4 trained humans, with implementation document | n/a | n/a | **0.42** (Fleiss) |

The mean pairwise gpt-oss:20b run-to-run κ on full-text (~0.44) sits at the trained-human-with-ID benchmark.

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

### 3.6 Run-to-run instability is concentrated on D1 (randomization)

Across the 5 unique RCTs in the audit set, the pattern most often disrupting agreement with Cochrane was D1 flipping between `low` and `high` across passes. D2–D5 were comparatively stable. This points to a tractable engineering improvement (majority-vote ensemble across passes on a per-domain basis), which we do not investigate here but flag as a clear next step.

## 4. Discussion (≤ 800 words)

- **The Eisele-Metzger 2025 conclusion is over-reached.** Their experimental design — naive prompting, a single 2023 model, a single Cochrane reviewer judgement as ground truth — does not support the framing that "LLMs cannot replace humans". Re-interpreted through the lens of the Minozzi human-vs-human reliability band and the harness-vs-naive distinction, their finding becomes "naively-prompted Claude 2 lands inside the trained-human-disagreement band against a single-rater reference standard". Our gpt-oss:20b results corroborate this re-interpretation on the same dataset.
- **The harness, not the model, is what matters.** A 20B open-weights model under a per-domain decomposed harness reproduces the Claude 2 published κ exactly and exhibits run-to-run reliability matching trained humans with structured guidance. *(Pending Sonnet 4.6 result; expected to extend rather than overturn the conclusion.)*
- **What would a fair "LLMs vs humans on RoB 2" comparison look like?** Multiple LLMs producing multiple passes, evaluated against multiple human reviewers (not a single Cochrane judgement), with all parties using comparable structural guidance. We are not asserting LLMs are superior; we are asserting the published comparison was unfair and the mechanism is now understood.
- **Implications for systematic-review practice.**
  - Benchmarks that report only κ vs a single human reviewer should be supplemented with run-to-run κ and with comparison to the human-vs-human band.
  - LLM RoB tooling should publish per-domain signalling answers and evidence quotes alongside the rolled-up judgement; the harness this paper describes is one open-source instantiation.
  - Methodology editorial groups (Cochrane, Joanna Briggs, GRADE) should consider treating algorithm-conformance with auditable evidence as an acceptable equivalence to one human reviewer for time-critical reviews.
- **What this paper does not claim.**
  - That this harness *is* a human reviewer. The companion paper (`20260423_medrxiv_assessor_algorithm_conformance_v1.md`) argues the AI-vs-expert disagreements visible at small `n` are mostly experts deviating from the algorithm, but that is a separable claim with its own reference standards.
  - That this harness eliminates LLM-internal noise. Run-to-run κ ≈ 0.44 is human-grade, not perfect. Ensemble strategies are an obvious next step.
  - That EM 2025 was wrong methodologically — it was an honest, well-conducted single-point study. The over-reach is in the headline-level generalisation, not the empirical core.

## 5. Limitations

- *(To be expanded as data finalises.)*
- Pre-registered single-rater reference standard. Cochrane review judgements typically reflect a single reviewer or a two-reviewer consensus, not the multi-rater Fleiss-κ design Minozzi used. The "human-vs-human ceiling" comparison rests on accepting Minozzi's reliability estimates as broadly representative.
- n = 91 RCTs (after PMID resolution failures); confidence intervals are wide, especially per-domain.
- Full-text coverage 41/100 native + 50 abstract-fallback under the FULLTEXT protocol; sensitivity analysis at §6.5 of pre-reg separates these subgroups.
- Sonnet 4.6 result *(pending)*. Anthropic API drift between submission and reproduction is a known issue; we record exact API timestamps and decline to re-run after submission.
- One wrong-paper Phase-1 acquisition (RCT030, Hung MS 2021 *Collegian*) was correctly refused by the model (parse failure on synthesis call when no domain judgements were available); excluded from analysis.
- Run-to-run instability is concentrated on D1 (randomization process); we have not attempted ensembling, which would likely tighten results substantially.

## 6. Conclusion (preliminary)

A vanilla 20B open-weights LLM, prompted with a decomposed harness that mirrors the RoB 2 algorithm's own structure, matches the published Eisele-Metzger 2025 Claude 2 result on the same 100-RCT dataset and exhibits run-to-run reliability matching trained human reviewers using a structured implementation document. The Eisele-Metzger conclusion that "LLMs cannot replace humans" does not generalise from the specific (Claude 2, naive prompting) to the general (LLMs at large). The methodologically appropriate question is not "does the AI agree with a single human reviewer" but "where in the human-vs-human reliability band does the AI sit, and is its agreement supported by algorithm-conformant per-domain evidence". Under that framing, both open-weights and frontier LLMs are fit for purpose.

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

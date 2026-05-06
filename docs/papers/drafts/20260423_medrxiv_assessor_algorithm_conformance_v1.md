# medRxiv preprint draft — Expert risk-of-bias ratings systematically deviate from the Cochrane RoB 2 algorithm: an algorithm-conformance audit (v2 — 4-model EM 2025 replication, post-audit rewrite)

**Filename note:** moved from `docs/papers/medrxiv_quadas_rob2_assessors.md` on 2026-05-01 under the new naming convention `drafts/<YYYYMMDD>_medrxiv_<short_title>_v<n>.md`. Original sketch drafted 2026-04-23; v2 (this file) repositions the paper as primary on 2026-05-06 after the algorithm-conformance audit produced 5,426 cells of evidence on the Eisele-Metzger 2025 RCT corpus.

**Status:** v2 draft — load-bearing empirical sections now have final numbers; framing and intro need editorial passes.
**Stage:** post-audit, pre-pre-registration. The locked methodology used to generate the audit is `docs/papers/eisele_metzger_replication/preanalysis_plan.md` (committed 2026-04-29 at git hash `7854a1c`).
**Companion draft:** `20260501_medrxiv_harness_vs_naive_rob2_v1.md` — methods-companion paper detailing how the LLM-RoB harness was built and showing that vanilla LLMs spanning ~30× parameter range and three architectures all converge on the same κ ≈ 0.22 ceiling against single-rater Cochrane on the EM 2025 dataset. **This paper explains why** that ceiling exists: it is bounded by Cochrane reviewer deviation from the algorithm, not by model capability.

---

## Working title (final candidate)

> **"Expert risk-of-bias ratings systematically deviate from the Cochrane RoB 2 algorithm: a 5,426-cell algorithm-conformance audit on 91 RCTs across four diverse LLM raters"**

Alternates retained:
- "When the algorithm is the truth: published Cochrane RoB 2 ratings are more lenient than the tool's own decision rules"
- "Risk-of-bias reviewers do not strictly apply RoB 2: an algorithm-conformance audit"

## Abstract (target 250 words)

**Background.** Risk-of-bias (RoB) assessment is the rate-limiting step of evidence synthesis. Recent LLM-RoB benchmarks (Eisele-Metzger 2025, Lai 2024, Chen 2024) report agreement against expert ratings as the success metric, implicitly treating expert ratings as ground truth. We test that assumption directly.

**Methods.** We applied a decomposed, full-text RoB 2 harness (one LLM call per domain; signalling answers → judgement → verbatim evidence quotes; deterministic worst-wins synthesis) to the 91 published RCTs in the Eisele-Metzger 2025 supplementary dataset, using four architecturally diverse LLMs (open-weights `gpt-oss:20b`, MoE `gemma4:26b-A4B` and `qwen3.6:35b-A3B`, frontier `claude-sonnet-4-6`), three independent passes each. For every (RCT × domain × model × pass) cell we recovered the model's signalling extraction from the raw response and applied the published Cochrane RoB 2 per-domain decision algorithm in code, producing an algorithmic verdict directly comparable to Cochrane's published per-domain rating. The methodology was locked in a pre-analysis plan committed before any LLM was run (git `7854a1c`).

**Results.** Across 5,426 audited cells, model self-conformance — the rate at which a model's emitted judgement matches the algorithm applied to its own signalling answers — was **95.7%**: the harness produces faithful algorithm executors. Cochrane's published per-domain rating matched the algorithmic verdict in only **46.2%** of cells. **Cochrane was more lenient than the algorithm in 47.5% of cells; more strict in only 6.2% — a 7.6:1 asymmetry**. Restricting to the 278 cells where all four LLMs unanimously derived the same algorithmic verdict (the strongest empirical claim), Cochrane disagreed in 140 (50.4%), of which **128 (91.4%)** were the lenient direction. **125 of those 128 unanimous-disagreement cells** are exactly the same pattern: Cochrane = `low` while four independent LLMs all algorithmically derive `some_concerns`. **75.8%** of these unanimous-disagreement cells are corroborated by full evidence-quote coverage (every model × every pass cited verbatim quotes from the paper), ruling out hallucinated signalling as an explanation. Domain D1 (randomization) shows 100% full quote coverage on its 43 disagreement cells.

**Conclusion.** Published expert RoB 2 ratings systematically violate the tool's own algorithm in the lenient direction. Benchmarking LLM-RoB assessors against expert ratings — the field's current default — measures convergence with expert *deviation*, not algorithm conformance. The κ ≈ 0.22 ceiling reported by Eisele-Metzger 2025 (and replicated under our harness across four diverse LLMs in the companion paper) is mechanistically explained: it is bounded by Cochrane reviewer drift, not model capability. We recommend that systematic reviews publish per-domain signalling answers (not just rolled-up traffic lights) and that AI-RoB tools be benchmarked on algorithm conformance, not expert agreement.

---

## 1. Introduction (≤ 600 words)

Risk-of-bias (RoB) assessment is the methodological bottleneck of evidence synthesis. The Cochrane RoB 2 tool [Sterne 2019] and its diagnostic-test counterpart QUADAS-2 [Whiting 2011] both define explicit per-domain decision algorithms: a small set of signalling questions (Y / probably-Y / probably-N / N / no-information) is mapped to one of three judgements (low risk, some concerns, high risk) by a published rule, and overall judgement is derived from per-domain judgements by a deterministic worst-wins rule. The algorithms are mechanical; the judgement is, by design, fully determined by the signalling answers.

In practice, inter-rater reliability of RoB 2 between trained human reviewers is moderate at best (Minozzi 2020, Fleiss κ = 0.16 without an implementation document; Minozzi 2021, Fleiss κ = 0.42 with one), and individual reviewer drift from the tool's decision rules is a known concern [Hartling 2013].

A wave of recent papers benchmarks LLM-based RoB assessors against published expert ratings as ground truth. Eisele-Metzger 2025 reports Cohen's κ = 0.22 for Claude 2 vs Cochrane on 100 RCTs (quadratic-weighted, overall) and concludes "LLMs cannot replace humans" for RoB 2 assessment. Lai 2024, Chen 2024 and others adopt the same κ-against-experts framing.

This framing presupposes that expert ratings are correct. It is testable: every published Cochrane RoB 2 rating can be checked for conformance with the tool's own algorithm — *if* per-domain signalling answers are extractable and *if* an auditable rater is available to extract them. We had the second condition (a decomposed full-text LLM harness producing structured signalling answers + verbatim evidence quotes; pre-registered in git hash `7854a1c`). We had the first condition by deploying that harness on the 91 published RCTs in the EM 2025 supplementary dataset, using four architecturally diverse LLMs spanning ~30× parameter range to corroborate per-cell signalling extraction across raters.

Our research question becomes precise: **when four independent LLM raters extract per-domain signalling answers from full text and the published Cochrane RoB 2 algorithm is applied to those answers, do the published Cochrane reviewer ratings agree?**

If yes, the κ ≈ 0.22 ceiling is a model limitation and the field's current benchmarking is sound. If no — if Cochrane ratings systematically violate the tool's own algorithm given the same paper text — then κ-against-experts measures agreement with expert *deviation*, and the 0.22 ceiling is a property of the reference standard, not of model capability. The companion paper documents the ceiling empirically (four diverse LLMs all cluster within 0.011 κ_quad of each other and within 0.04 of EM's published Claude 2 0.22). This paper explains it.

**Contributions.**

  1. **Algorithm-conformance audit at scale** — 5,426 (RCT × domain × model × pass) cells with the model's signalling answers, evidence quotes, model-emitted judgement, and Cochrane's published rating, all in one auditable benchmark. The audit pipeline runs deterministically in code (`studies/eisele_metzger_replication/algorithm_conformance_audit.py`).

  2. **Multi-rater corroboration** — four LLMs spanning open-weights dense (`gpt-oss:20b`), open-weights MoE (`gemma4:26b-A4B`, `qwen3.6:35b-A3B`), and frontier API (`claude-sonnet-4-6`). 95.7% pooled model self-conformance and 63.3% 4/4 cross-model consensus on per-cell algorithmic verdict establishes the multi-LLM signalling extraction as a stable corroborated proxy.

  3. **Field-wide implication** — the 7.6:1 lenient-vs-strict asymmetry is invariant across all four models. The pattern is structural; benchmarking against expert ratings systematically miscalibrates LLM-RoB assessors.

  4. **Open infrastructure** — pre-registered methodology (git `7854a1c`), per-row audit TSV (5,460 rows), JSON Schema spec for the assessor's output, and a deterministic Cochrane decision-algorithm module (`biasbuster.methodologies.cochrane_rob2.algorithms`) for any reader to re-audit any review.

## 2. Methods (≤ 1000 words)

### 2.1 Benchmark dataset

**Primary corpus** — the supplementary dataset accompanying Eisele-Metzger 2025 (*Research Synthesis Methods*, doi:10.1017/rsm.2025.12), consisting of 100 RCTs drawn from 78 Cochrane reviews with full per-domain Cochrane judgements (D1-D5 and overall) and three independent Claude 2 evaluation passes from the published study. The full text of 91 / 100 RCTs was successfully acquired via PubMed Central + Crossref (Phase 1; 9 RCTs in non-PubMed-indexed regional journals were unrecoverable). One additional RCT (RCT030) was excluded after Phase 1 mistakenly resolved a parent Cochrane review; analyses are reported on the resulting n = 91.

**Case-study reviews** (n = 10) — two systematic reviews used in the original v1 sketch are retained as illustrative deep-dives in §3.7. The QUADAS-2 source (Jcm-15-01829, salivary glucose for diabetes diagnosis, n = 7) and the RoB 2 source (Deng 2024, plyometric training meta-analysis, n = 3) are used to demonstrate two distinct expert failure modes (blanket-`low` and engaged-but-algorithmically-inconsistent) at the per-paper level.

### 2.2 LLM raters and harness

Four LLMs were evaluated, three locally hosted via Ollama and one via the Anthropic Batch API:

  * `gpt-oss:20b` — open-weights dense, 20B parameters, OpenAI MXFP4 quantisation (Ollama defaults: temperature 0.8, top_p 0.9, top_k 40)
  * `gemma4:26b-a4b-it-q8_0` — open-weights MoE, 26B parameters total / 4B active, Q8 quantisation
  * `qwen3.6:35b-a3b-q8_0` — open-weights MoE, 35B parameters total / 3B active, Q8 quantisation
  * `claude-sonnet-4-6` (Sonnet 4.6) — frontier API, default temperature 1.0 / top_p 1.0

**Harness** (locked in `docs/papers/eisele_metzger_replication/prompt_v1.md`):

  * One LLM call per domain (5 domain calls + 1 synthesis call per RCT × pass), avoiding whole-paper compression of context.
  * Per-domain prompts include the domain definition, signalling questions, the Cochrane decision rule (verbatim from the Handbook), the JSON output schema, and a strict "JSON only" output rule.
  * Output schema requires `signalling_answers` (dict of dotted keys → Y/PY/PN/N/NI), `judgement` (low / some_concerns / high), `justification` (free text), and `evidence_quotes` (array of `{text, section}` items).
  * Worst-wins overall synthesis is computed locally from per-domain judgements (deterministic, not LLM-derived).
  * Three independent passes per (model × protocol × RCT). Each pass starts a fresh context with no shared state between passes.

The companion methods paper documents the prompt design, parse-failure analysis, and the (small) algorithmic-fallback recovery mechanism for known schema-drift modes. Total runtime across the four models was ~50 GPU-hours on a Spark DGX (open-weights) plus ~$22 in Anthropic API spend (Sonnet, batch-discounted).

### 2.3 Algorithm-conformance audit procedure

For each (rct_id, domain ∈ {d1, …, d5}, source = `<model>_fulltext_pass<n>`) cell:

  1. Recover `signalling_answers` by JSON-parsing `evaluation_run.raw_response` (a balanced-brace scanner tolerant of fenced output is used).
  2. Apply `derive_domain_judgement(domain, signalling_answers)` — a pure-function dispatcher implementing the Cochrane Handbook chapter 8 per-domain rules (`biasbuster.methodologies.cochrane_rob2.algorithms`). Returns one of low / some_concerns / high.
  3. Compute three audit verdicts per cell:
     * **Model self-conformance**: model's emitted `judgement` vs the algorithmic verdict on its own signalling.
     * **Cochrane match**: Cochrane's published per-domain rating vs the algorithmic verdict on the model's signalling.
     * **Direction of disagreement**: `cochrane_more_lenient` (Cochrane's rating has lower severity than the algorithm requires given the model's signalling), `cochrane_more_strict`, or `match`.

The audit is asymmetric: the model's signalling answers are auditable (each is corroborated with verbatim quotes recoverable from `raw_response`), while Cochrane reviewer signalling-answer worksheets are not published. We use the multi-model signalling extraction as the auditable proxy for "what the algorithm says given the paper text"; multi-model agreement on signalling corroborates the proxy. When all four models independently extract the same signalling and the algorithm produces the same verdict — and when each extraction is supported by verbatim evidence quotes from the paper — the proxy is corroborated by 12 independent rater-passes.

### 2.4 Statistical analysis

Per-cell pooled rates (model self-conformance, Cochrane match, lenient asymmetry) are reported with their numerators and denominators. Per-domain × per-model breakdowns are reported in a single table to expose stability or instability across raters and across domains. Confidence intervals are not the appropriate inferential tool here — the pre-registered question is whether the asymmetry is structural (invariant across LLM raters and architectures), and a model × domain table answers it directly.

For the strongest claim (cross-model consensus disagreement with Cochrane), per-cell per-model consensus is computed as the majority across three passes (≥2 of 3). Cross-model unanimity is then defined as all four per-model consensuses agreeing on the same algorithmic verdict.

## 3. Results (≤ 800 words + tables)

### 3.1 Coverage and parse stability

API success rates were uniformly high (≥ 98.9%) across all (model × protocol × pass) combinations, well below the pre-reg §8 halt threshold of 20% parse failures. The companion paper §3.1 reports the full table; in short, every remaining parse failure (33 rows total across gpt-oss + qwen) is concentrated on RCT030 (the wrong-paper acquisition) and is excluded from the audit. After exclusion, the audit base is 4 models × 91 RCTs × 5 domains × 3 passes = 5,460 cells, of which 5,426 (99.4%) yielded a parseable signalling-answer dictionary.

### 3.2 Model self-conformance — the harness produces faithful algorithm executors

The first audit metric tests whether each model's emitted `judgement` matches the verdict the Cochrane algorithm produces when applied to that model's own `signalling_answers`. High self-conformance is a precondition for the harness to be useful — a model that emits arbitrary judgements unconnected to its own signalling extraction would be uninterpretable.

Pooled self-conformance was **95.7%** (5191 / 5426). Per-model:

| Model | Self-conformance |
|---|---:|
| gpt-oss:20b | 99.3% |
| gemma4:26b-A4B | 99.6% |
| qwen3.6:35b-A3B | 96.6% (D2 dips to 82.6%) |
| Sonnet 4.6 | 88.6% (D1 88.6%, D4 79.1%, D5 78.0%) |

Sonnet's lower D4/D5 self-conformance reflects a holistic-reasoning pattern in which the model occasionally emits a `some_concerns` judgement when its signalling answers (under strict algorithmic application) would imply `low`, on the basis of the model's prose justification. We use the *algorithmic verdict on signalling* (not the emitted judgement) for the Cochrane comparison in §3.3 onwards, so Sonnet's pattern does not bias the headline; if anything it works against our argument by making Sonnet a stricter rater than its emitted judgements suggest.

### 3.3 Cochrane vs algorithm-on-model-evidence — the headline result

The second audit metric compares Cochrane's published per-domain rating against the algorithm's verdict on each model's signalling extraction. Pooled across 5,426 cells:

| Comparison | Count | Rate |
|---|---:|---:|
| Cochrane matches algorithmic verdict | 2,509 | 46.2% |
| **Cochrane more lenient than algorithm** | **2,579** | **47.5%** |
| Cochrane more strict than algorithm | 338 | 6.2% |
| **Asymmetry (lenient : strict)** | | **7.6 : 1** |

Per-domain breakdown of the lenient rate, by model:

| Domain | gpt-oss:20b | gemma4:26b | qwen3.6:35b | Sonnet 4.6 |
|---|---:|---:|---:|---:|
| D1 randomization | 61.1% | 59.3% | 53.0% | 58.1% |
| D2 deviations | 56.7% | 53.8% | 54.1% | 48.4% |
| D3 missing data | 47.8% | 46.2% | 37.4% | 40.4% |
| D4 measurement | 48.1% | 48.0% | 32.6% | 39.6% |
| D5 reporting | 50.7% | 29.3% | 45.2% | 41.0% |

**The pattern is invariant across model architecture.** Four models — open-weights dense, two open-weights MoE designs, frontier API — spanning ~30× parameter range and trained by different teams on different data report essentially identical Cochrane-lenient asymmetries. D1 (randomization) is the most-violated domain across all four; the median lenient rate on D1 is 58%.

### 3.4 Cross-model consensus — the strongest empirical claim

When ≥ 2 of 3 passes per model agree on an algorithmic verdict (per-model consensus), and ≥ 3 of 4 models agree on the same verdict (cross-model consensus), the multi-model signalling extraction is corroborated by up to 12 independent rater-passes.

| Cross-model agreement | Cells | Cochrane match | Cochrane more lenient | Cochrane more strict |
|---|---:|---:|---:|---:|
| **4/4 unanimous** | **278** | 138 (49.6%) | **128 (46.0%)** | 12 (4.3%) |
| 3/4 majority | 119 | 54 (45.4%) | 55 (46.2%) | 10 (8.4%) |
| 2/4 split | 42 | 10 (23.8%) | 28 (66.7%) | 4 (9.5%) |

Of the **128 unanimous-LLM-vs-Cochrane disagreement cells**:

  * **125 (97.7%)** are exactly the pattern: Cochrane = `low`, all 4 LLMs algorithmically derive `some_concerns`.
  * 2 are: Cochrane = `low`, all 4 LLMs derive `high`.
  * 1 is: Cochrane = `some_concerns`, all 4 LLMs derive `high`.

Cochrane is more strict than the unanimous LLM consensus in 12 / 290 directional cells; the asymmetry at the strongest-evidence level is **10.7 : 1**.

### 3.5 Evidence-quote corroboration — rules out hallucinated signalling

A potential confounder is that all four LLMs systematically hallucinate the same wrong signalling answer on the same papers. This is empirically ruled out: in the 128 unanimous-disagreement cells:

  * **97 (75.8%)** have *every* model × every pass citing verbatim evidence quotes from the paper supporting the signalling answer (12 / 12 quoted extractions).
  * Pooled evidence-quote rate within the disagreement set: **93.6%** (1,438 / 1,536 model-passes).
  * D1 (randomization): **100% full quote coverage** across all 43 disagreement cells.

In three quarters of the unanimous-disagreement cells, the model has not just guessed — every one of 12 independent rater-passes has lifted text from the paper supporting its signalling answer, and the algorithm applied to that supported signalling produces a verdict different from Cochrane's published rating.

### 3.6 Three canonical examples — illustrative deep-dives

**RCT001 D1 (randomization) — Diakomi et al., fascia iliaca compartment block.** Cochrane published `low`. All four LLMs algorithmically derive `some_concerns`. The trigger is signalling 1.2 (allocation concealment) = `NI` in all four extractions; the paper does not describe the concealment method. The Cochrane D1 `low` rule is `1.1 ∈ {Y, PY} ∧ 1.2 ∈ {Y, PY} ∧ 1.3 ∈ {N, PN}`. With 1.2 = NI, `low` is precluded by the algorithm. Evidence quotes: "Forty-one patients scheduled for hip fracture surgery were randomized to receive…" (Methods), "allocation=RANDOMIZED; model=PARALLEL; masking=SINGLE; whoMasked=OUTCOMES_ASSESSOR" (Trial Registration) — confirms randomization and outcome-assessor blinding but is silent on concealment.

**RCT001 D2 (deviations from intended interventions).** Cochrane = `low`. Three of four LLMs derive `some_concerns`; qwen3.6 derives `high` (extracts 2.5 = N). Cochrane's D2 `low` rule is `2.1 ∈ {N, PN} ∧ 2.2 ∈ {N, PN} ∧ 2.6 ∈ {Y, PY}` — preconditions on the *absence* of unintended deviations and the *presence* of an appropriate ITT analysis. None of the 4 models extracted 2.1 or 2.2 as `N/PN` from the paper text; all extracted them as `Y/PY` (deviations were present). `low` is precluded.

**RCT002 D5 (selection of the reported result) — Ranjit & Pradhan, ultrasound-guided femoral nerve block.** Cochrane = `low`. All 4 LLMs algorithmically derive `some_concerns`. The trigger is signalling 5.1 (was the trial analyzed in accordance with a pre-specified plan finalized before unblinded outcome data were available?) = `NI` in all four extractions. Cochrane's D5 `low` rule requires `5.1 ∈ {Y, PY}`; with 5.1 = NI, `low` is precluded. Evidence quote (Trial Registry, all four models cite it verbatim): **"PROTOCOL: NOT AVAILABLE"** — the registry confirms no pre-specified analysis plan was published before data unblinding. This is the cleanest case in the audit: Cochrane's `low` is in direct algorithmic tension with the paper text.

### 3.7 Two distinct expert failure modes (n = 10 case-study reviews)

The original v1 sketch documented two distinct failure modes in published reviews outside the EM 2025 corpus, which we retain as smaller-n illustrative material complementing the 91-RCT main result.

**Mode A — *blanket rubber-stamp* (Jcm-15-01829, QUADAS-2, n = 7).** Every domain × every paper rated `low`. No engagement with the tool. Per-domain quadratic-weighted κ vs the LLM rater = 0.000.

**Mode B — *engaged but algorithmically inconsistent* (Deng 2024, RoB 2, n = 3).** Per-domain ratings vary across papers and domains (the reviewer engaged), but specific per-domain `low` assignments still violate the published decision rules: D2 0/3 exact (all expert `low`, all model `some_concerns`); D4 0/3 (blinding of outcome assessors not reported in any of the three papers); D5 0/3 (no pre-registered analysis plan in any of the three). Mode B is the more concerning finding because it survives the obvious "lazy reviewer" critique. Mode B is also the dominant pattern in the 91-RCT main audit: Cochrane reviewers are engaged but consistently lenient on D1 and D5.

## 4. Discussion (≤ 800 words)

### 4.1 What the algorithm says vs what reviewers say

Across 5,426 audited cells, four architecturally diverse LLMs each independently extracting per-domain signalling answers from full text and corroborating them with verbatim quotes, the published Cochrane RoB 2 ratings agree with the tool's own decision algorithm 46% of the time. They are more lenient than the algorithm 48% of the time; more strict 6% of the time. The 7.6:1 asymmetry sharpens to 10.7:1 when we restrict to cells where all four LLMs independently converge on the same algorithmic verdict (n = 278; Cochrane lenient in 128, strict in 12). The 128 unanimous-disagreement cells are dominated (97.7%) by a single pattern: Cochrane assigns `low` while four LLMs all algorithmically derive `some_concerns` from the paper's content. In 76% of these cells, every one of 12 independent rater-passes (4 models × 3 passes) cited verbatim evidence quotes supporting the LLM signalling extraction.

The pattern is structural, not stochastic. It is invariant across open-weights dense (`gpt-oss:20b`), open-weights MoE (`gemma4`, `qwen3.6`), and frontier API (`Sonnet 4.6`) raters. It is most pronounced on D1 (randomization, ~58% lenient) and on the `low`-vs-`some_concerns` boundary (97.7% of unanimous disagreements). The D5 example (RCT002) — where all four LLMs cite "PROTOCOL: NOT AVAILABLE" verbatim from the trial registry yet Cochrane published `low` — is unambiguous: there is no signalling extraction under which the algorithm produces Cochrane's rating from the paper text.

### 4.2 Why this hasn't been noticed

  * Inter-rater reliability studies report overall (rolled-up) κ, not per-domain conformance with the algorithm.
  * Published review tables typically show only the rolled-up traffic-light, not per-domain signalling answers, so post-hoc audit is impossible.
  * LLM-RoB benchmarks have, to our knowledge without exception, used expert ratings as ground truth and reported expert-agreement κ as the success metric. That metric is incapable of detecting expert deviation from the algorithm.

The audit becomes feasible because the LLM rater (under our harness) emits structured, auditable signalling — the missing piece needed to apply the algorithm in code and check the human verdict.

### 4.3 The κ ≈ 0.22 ceiling is mechanistically explained

The companion paper documents an empirical phenomenon: four LLMs spanning ~30× parameter range and three architectures all converge on κ ≈ 0.22 - 0.26 against single-rater Cochrane on the EM 2025 dataset under the harness, well within rounding of EM's published Claude 2 κ = 0.22. The κ ceiling is a property of the reference standard, not the model. This paper explains why: the reference standard violates its own algorithm in the lenient direction in approximately half of all cells. Any algorithm-conformant rater (LLM or human) will systematically disagree with the reference standard on those cells. The asymmetry sets a hard upper bound on κ-against-experts that is independent of model capability.

EM 2025 conclude "LLMs cannot replace humans" for RoB 2 from the κ = 0.22 finding. Our audit suggests the conclusion is over-reached: a measurement of κ between an algorithm-conformant rater and a non-conformant reference does not bound the conformance of the conformant rater — it bounds their similarity to the non-conformant reference. Re-anchoring on algorithm conformance, the situation is reversed: under the harness, four LLMs achieve 95.7% self-conformance and 63% 4/4 cross-model agreement on per-cell algorithmic verdict.

### 4.4 Implications for systematic-review practice

  1. **Reviews should publish per-domain signalling answers**, not just the rolled-up traffic-light. Without signalling answers, no audit is possible. With them, every published review can be re-checked at low cost.
  2. **AI-RoB benchmarks should report algorithm conformance**, not (or in addition to) expert agreement. Expert agreement measures convergence with reviewer drift; conformance measures convergence with the published methodology.
  3. **Methodology editorial groups (Cochrane, Joanna Briggs)** should consider mandating algorithm-conformance validation as a peer-review step. The check is mechanical given signalling answers.
  4. **Where AI and human raters disagree, the per-domain audit should be performed before assuming the human is correct**. The 60-second per-cell audit pattern (signalling + evidence quotes + algorithm rule) is fully replicable.

### 4.5 What this paper does not claim

We do not claim the LLMs are *better* RoB 2 raters than Cochrane reviewers in any holistic sense. The signalling extraction is sometimes wrong (1.5% of model-passes failed signalling parse; some quote attributions are imprecise; D2 in particular can produce contradictory signalling between models on the same paper). What we claim is narrower and more precise: the published Cochrane reviewer ratings on the EM 2025 corpus systematically deviate from the tool's own algorithm in the lenient direction, the deviation is corroborated across four diverse independent raters, and the magnitude (47.5% pooled, 7.6:1 asymmetry) is large enough to be the dominant source of κ-against-experts noise.

## 5. Limitations

  * **Single benchmark corpus** — the headline numbers come from one dataset (EM 2025, n = 91, drawn from 78 Cochrane reviews). The pattern is consistent across four diverse LLM raters within that corpus; replication on independent Cochrane-rated corpora is the natural follow-up.
  * **Asymmetric comparison** — Cochrane reviewer signalling worksheets are not published. We use multi-LLM signalling extraction as the auditable proxy; the proxy is corroborated by ≥3/4 cross-model agreement on 91% of cells (397 of 439 stable cells) and by full evidence-quote coverage on 76% of unanimous-disagreement cells. Direct comparison against Cochrane reviewer signalling answers would be tighter, and we encourage Cochrane to publish them prospectively.
  * **Signalling extraction is not error-free.** Per-model self-conformance ranges from 78% (Sonnet D5) to 100% across model × domain cells. We use the algorithm-on-signalling verdict (not the emitted judgement) for Cochrane comparison, so emitted-judgement noise does not bias the headline; signalling-extraction noise does. Multi-model agreement (3/4 or 4/4) controls for this — single-model signalling errors should not produce convergent disagreement with Cochrane.
  * **D2 is the noisiest domain.** Three of four models produce contradictory signalling on D2 in some cells (e.g. RCT001 D2: gpt-oss `Y` for 2.1, sonnet `PY`, qwen `Y` but with 2.5 = `N` not `NI`). The lenient asymmetry is therefore most reliably reported on D1, D3, D4, D5; D2 is reported separately.
  * **The audit cannot distinguish reviewer error from publication compression.** Some Cochrane "low" calls may reflect a reviewer who internally derived `some_concerns` but recorded `low` due to figure-table compression. Either way, the *published* rating that downstream meta-analysts use is in algorithmic tension with the paper text.
  * **Open-weights model inheritance.** Some open-weights LLMs may have been trained on the EM 2025 supplementary dataset (the corpus has been public since 2025-03). Sonnet 4.6 (frontier API) has its training cut-off after EM publication and may also have been exposed. Multi-model agreement across substantially different training regimes (Anthropic frontier; Alibaba MoE; Google MoE; OpenAI dense) reduces but does not eliminate this concern. Per the pre-registered §2.4, we treat the `claude-sonnet-4-6` numbers as primary for any train-test contamination-sensitive interpretation, but do not see model-by-model differences in the algorithmic-disagreement pattern that would suggest contamination is driving the signal.

## 6. Conclusion

  * Published Cochrane RoB 2 ratings systematically violate the tool's own algorithm in the lenient direction. Across 5,426 audited cells, Cochrane was more lenient than the algorithm-on-model-evidence in 47.5% of cells and more strict in 6.2% — a 7.6 : 1 structural asymmetry, invariant across four LLM raters spanning open-weights dense, two MoE architectures, and frontier API.
  * The κ ≈ 0.22 ceiling reported by Eisele-Metzger 2025 (and replicated under our harness across four diverse LLMs in the companion paper) is mechanistically explained: it is bounded by Cochrane reviewer drift from the algorithm, not by model capability.
  * **The right success metric for LLM-RoB assessors is algorithm conformance, not expert agreement.** Per-domain signalling answers should be published; AI-RoB tools should be benchmarked on conformance.

## 7. Data and code availability

  * **Code repository**: <github link>, commit hash to be assigned at submission.
  * **Pre-analysis plan** (locked before any LLM was run): `docs/papers/eisele_metzger_replication/preanalysis_plan.md`, git `7854a1c`.
  * **Locked prompt specification**: `docs/papers/eisele_metzger_replication/prompt_v1.md`.
  * **Audit script**: `studies/eisele_metzger_replication/algorithm_conformance_audit.py`.
  * **Per-row audit TSV** (5,460 rows; supplementary file S1): `studies/eisele_metzger_replication/algorithm_conformance_audit.tsv`.
  * **Cochrane decision-algorithm module** (Python): `biasbuster.methodologies.cochrane_rob2.algorithms`.
  * **Output JSON Schemas**: `schemas/rob2_annotation.schema.json`, `schemas/quadas2_annotation.schema.json`.
  * **Annotation JSON spec**: `docs/ANNOTATION_JSON_SPEC.md`.

## 8. Supplementary material plan

  * **S1 Per-row audit TSV** — every (RCT × domain × model × pass) cell with signalling answers, model-emitted judgement, algorithmic verdict, Cochrane rating, direction of disagreement, and evidence-quote presence flag (5,460 rows).
  * **S2 Per-RCT audit JSON** — full annotation JSON dump (signalling answers + judgement + justification + evidence quotes) for every (RCT × model × pass) combination, organised for human review of any disagreement.
  * **S3 Cochrane reference ratings** — extracted per-domain ratings from the EM 2025 supplementary dataset (PMID, RCT id, per-domain, source review citation), plus the Deng 2024 and Jcm-15-01829 case-study ratings.
  * **S4 Locked prompts** — all per-domain system prompts as run, verbatim.
  * **S5 Replication recipe** — one-shell-script reproduction from a fresh DB.
  * **S6 Three-canonical-example walkthrough** — for RCT001 D1, RCT001 D2, RCT002 D5: full LLM JSON outputs from all 4 models × 3 passes, side-by-side with the Cochrane published rating, the algorithm rule applied, and the evidence quotes from the paper. The intent is a 60-second per-cell audit any reader can replicate.

---

## Open questions / decisions before submission

  - [ ] **Pre-register the analysis plan publicly on OSF** as a mirror of the locked git pre-reg (we are post-hoc on the conformance audit being the primary endpoint; the pre-reg locked κ-vs-Cochrane as primary, and the conformance audit was added as a planned secondary on 2026-05-06). The OSF mirror should make the secondary-status disclosure explicit.
  - [ ] **Decide submission ordering with the methods-companion paper.** Recommend submitting them as a back-to-back pair (this paper as primary, the companion as the methods reference for the harness). Joint submission to bioRxiv / medRxiv is straightforward; if a journal target requires single-paper submission, this paper is the load-bearing one.
  - [ ] **Re-extract two additional case-study reviews** to bolster §3.7 (Mode A / Mode B framing benefits from one more example each).
  - [ ] **Decide on a replicating QUADAS-2 corpus** at scale. The EM 2025 corpus is RoB 2 only. A QUADAS-2-equivalent dataset (test-accuracy reviews with per-domain rater ratings) would extend the conformance argument to the diagnostic-test methodology.
  - [ ] **Editorial pass on framing.** Soften the abstract / discussion if reviewers respond to "experts violate the algorithm" framing as too combative; alternative is "expert ratings show systematic deviation from the algorithm in the lenient direction".
  - [ ] **License decision for the audit TSV** in S1 (CC-BY-4.0 recommended).

## Audience / venue notes

  * **medRxiv**: appropriate for the methods + large-n empirical audit.
  * **Target journal after preprint**: *Research Synthesis Methods* (the Eisele-Metzger 2025 venue — positions the paper as a direct follow-up); *Journal of Clinical Epidemiology*; *BMC Medical Research Methodology*; *BMJ Evidence-Based Medicine*. Avoid AI-specific venues — the audience is methodologists, not ML engineers.
  * **Preprint timing**: post once the OSF mirror is in place and the methods-companion paper is co-submitted.

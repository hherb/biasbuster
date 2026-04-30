# Pre-Analysis Plan — LLM-based RoB 2 Assessment, Replication and Extension of Eisele-Metzger 2025

**Status:** DRAFT, awaiting sign-off. **Will be locked by a dated git commit before any results are inspected.**

**Authors:** Horst Herb (corresponding).
**AI assistance disclosure:** see §11.
**Date drafted:** 2026-04-30.
**Lock date:** *to be filled at commit time.*
**Lock commit hash:** *to be filled at commit time.*

---

## 1. Background

Eisele-Metzger et al. (2025, *Res Synth Methods*, doi:10.1017/rsm.2025.12) reported that Claude 2 achieved Cohen's κ = 0.22 against published Cochrane RoB 2 judgments on 100 RCTs and concluded *"Currently, Claude's RoB 2 judgements cannot replace human RoB assessment."* Their conclusion is interpreted by some readers as a statement about LLMs as a class.

Two structural concerns motivate this replication:

1. **Model age.** Claude 2 (Aug 2023) precedes multiple subsequent generations. Generalizing from a single 2023 model to "LLMs" is unsupported.
2. **Noisy ceiling.** Trained human raters applying RoB 2 to the same trials achieve overall Fleiss κ in the 0.16–0.42 range (Minozzi et al., *J Clin Epidemiol* 2020 and 2021). κ = 0.22 against a single Cochrane judgment lies inside this human-vs-human reliability band — its magnitude alone does not establish "LLMs cannot replace humans."

## 2. Research questions and pre-specified hypotheses

**RQ1 (primary).** Does a current-generation vanilla LLM (Claude Sonnet 4.6) achieve a higher Cohen's κ vs Cochrane RoB 2 judgments than the κ = 0.22 reported by Eisele-Metzger for Claude 2, on the same 100 RCTs?

> **H1 (primary).** Claude Sonnet 4.6 achieves overall Cohen's κ vs Cochrane judgments ≥ 0.35 (a clinically meaningful improvement over Claude 2's 0.22, and reaching the lower bound of "fair" agreement).
>
> **H0 (primary).** Sonnet 4.6 overall κ ≤ 0.22 (no improvement over Claude 2).

**RQ2 (secondary).** Does *any* of the three vanilla open-weights local candidates (`gpt-oss:20b`, `gemma4:26b-a4b-it-q8_0`, `qwen3.6:35b-a3b-q8_0`) reach Sonnet 4.6's κ within a tolerance of 0.05?

**RQ3 (descriptive).** What is the run-to-run Fleiss κ for each model across three independent passes? Is LLM-internal reliability higher than the published human Fleiss κ of 0.16 (Minozzi 2020, untrained) or 0.42 (Minozzi 2021, with implementation document)?

**RQ4 (descriptive).** Does performance differ between abstract-only and full-text inputs?

## 3. Data

### 3.1 Benchmark dataset

Eisele-Metzger 2025 supplementary CSVs, locally available at `DATA/20240318_Data_for_analysis_full/` (gitignored — redistribution rights with the original publisher). Contains 100 RCTs from 78 Cochrane reviews, with per-domain Cochrane judgments and three independent Claude 2 passes per RCT.

Schema notes: `DATA/20240318_Data_for_analysis_full/STRUCTURE.md`.

### 3.2 Full-text sources (Phase 1 acquisition)

Acquisition cascade: local cochrane_rob cache → Europe PMC → Crossref/Unpaywall → publisher OA → JCU + Queensland Health OpenAthens (manual). Target ≥ 80% full-text coverage; remainder evaluated on abstract-only.

### 3.3 Overlap with the BiasBuster project corpus (reporting only)

Before any model is run, every RCT in the Eisele-Metzger benchmark will be cross-referenced against `dataset/biasbuster.db` (`papers`, `human_reviews` tables). The full overlap composition will be reported in the methods section.

**No exclusion is performed for vanilla-model evaluation.** The four evaluated models (Claude Sonnet 4.6, gpt-oss:20b, gemma4:26b-a4b-it-q8_0, qwen3.6:35b-a3b-q8_0) were trained by their respective providers on data we do not control; whether the foundation models encountered any of the EM-100 during their pre-training is unknowable and unrelated to whether those RCTs appear in our project's working dataset. Reducing the evaluation sample below n=100 would degrade statistical power without addressing any actual bias in vanilla-model performance.

**The full n=100 is the primary analysis sample regardless of overlap.** Overlap is reported transparently as a methods detail.

**Out of scope for this paper:** any fine-tuning of any of the evaluated models. A separate future paper, if conducted, will require strict exclusion of the EM-100 from any training set used by that work; that future paper will pre-specify its own contamination thresholds. This pre-registration covers the vanilla-evaluation paper only.

## 4. Models evaluated

| Slot | Model | Identifier (recorded for reproducibility) | Role |
|---|---|---|---|
| 1 | Claude Sonnet 4.6 | `claude-sonnet-4-6` (API) | Primary, performance ceiling |
| 2 | gpt-oss 20B | `gpt-oss:20b` (Ollama, sha256:17052f91a42e) | Open-weights candidate |
| 3 | Gemma 4 26B-A4B-IT-Q8 | `gemma4:26b-a4b-it-q8_0` (Ollama, sha256:6bfaf9a8cb37) | Open-weights candidate |
| 4 | Qwen 3.6 35B-A3B-Q8 | `qwen3.6:35b-a3b-q8_0` (Ollama, sha256:0218f872e86b) | Open-weights candidate |
| (5) | Claude 2 (published) | per Eisele-Metzger 2025 | Cited baseline (no re-run; deprecated by Anthropic) |

All vanilla — no fine-tuned variants are in scope for the primary analysis.

## 5. Experimental procedure

### 5.1 Prompting

A single fixed prompt template will be used for all models, derived from BiasBuster's existing `prompts.py` ANNOTATION_SYSTEM_PROMPT but adapted to RoB 2's five domains and ordinal scale (low / some concerns / high). The exact prompt text is committed alongside this plan as `prompt_v1.md` and frozen.

### 5.2 Inputs per RCT

Two input protocols, evaluated separately:
- **Abstract-only:** `rct_ref` citation + abstract retrieved by DOI/PMID.
- **Full-text:** main RCT PDF parsed to text, plus protocol and trial registration document where available.

### 5.3 Passes

**Three independent passes per model per RCT per protocol.** Sampling temperature is fixed at the model's default; pass independence is achieved by separate API/Ollama calls with no conversation context shared.

### 5.4 Outputs

For each pass: per-domain judgment (low / some concerns / high) + free-text rationale, plus an overall judgment derived per Cochrane RoB 2 algorithm. Stored in `dataset/eisele_metzger_benchmark.db` (gitignored).

## 6. Statistical analysis

### 6.1 Primary analysis

**Cohen's κ (linear-weighted)** between each model's overall judgment (pass 1) and Cochrane's overall judgment, with 95% bootstrap CIs (1000 resamples, BCa method).

H1 supported if Sonnet 4.6 lower 95% CI > 0.22 (Claude 2's published value). H1 rejected otherwise.

### 6.2 Per-domain analysis

Cohen's κ per domain (D1–D5) for each model. Reported as a 5×4 matrix with CIs. No multiple-comparison correction applied to per-domain κ — these are descriptive.

### 6.3 Run-to-run reliability

**Fleiss κ** across the three passes per model. Compared narratively (not statistically) to Minozzi 2020 (κ = 0.16) and Minozzi 2021 (κ = 0.42).

### 6.4 Head-to-head significance

**McNemar's test** on the 2×2 contingency of per-RCT correctness (collapsing RoB 2 judgment to "matches Cochrane" / "does not match"), Sonnet 4.6 vs Claude 2 (using Claude 2's `claude1_overall` from the supplementary data). Two-sided, α = 0.05.

### 6.5 Subgroup sensitivity

Repeat 6.1 separately on the full-text-available and abstract-only subsets. If the κ delta between subsets exceeds 0.10, flag as a substantive moderator.

## 7. Local-model selection (descriptive)

**Best-local rule:** the local candidate with the highest mean κ across {overall, D1, D2, D3, D4, D5} on pass 1 is declared "best local." Tie-break: higher run-to-run Fleiss κ.

The "best local" model is identified for descriptive comparison against Sonnet 4.6 and as a candidate for any future fine-tuning study, but **no fine-tuning is performed in this paper**. All four models are reported on equal footing in the primary results; the "best local" label is a post-hoc reading aid, not a privileged comparison.

## 8. Stopping rules

- If primary analysis (Sonnet 4.6) shows κ < 0.10 vs Cochrane → halt for code/prompt review before drafting the preprint. A negative finding worse than the published Claude 2 baseline is a signal that something is wrong with our pipeline, not a valid result.
- If any model produces structurally invalid output (e.g. fails JSON parsing on > 20% of RCTs) → halt and revise prompts before counting that model's pass.

## 9. What constitutes a publishable result

A medRxiv preprint will be drafted only if **at least one of**:
- **H1 is supported:** Sonnet 4.6 lower-95%-CI κ > 0.22 (Claude 2's published value).
- **Run-to-run finding:** any vanilla model's three-pass Fleiss κ exceeds the Minozzi 2020 human Fleiss κ of 0.16 *and* that model's κ vs Cochrane lies within the published human-vs-human reliability band (0.16–0.42), demonstrating LLMs achieve at least human-equivalent reliability on this task.

If neither holds, results are reported internally only and the work does not appear publicly. There is no professional or financial obligation to publish; the disseminate-only-if-meaningful principle governs.

## 10. Limitations acknowledged in advance

1. **Single-rater Cochrane gold standard.** "Cochrane judgments" in the dataset reflect the original Cochrane review process (typically two reviewers reaching consensus, but not multi-rater Fleiss). κ vs a single judgment line is an upper bound on apparent disagreement.
2. **Sample size.** n = 100 is fixed by the dataset; CIs will be wide, especially per-domain.
3. **Model API drift.** Sonnet 4.6 outputs may shift if Anthropic updates the model under the same ID. We will record exact API timestamps and decline to re-run after submission.
4. **Outcome clustering.** Multiple outcomes per RCT may appear; primary analysis treats each row as an independent observation. Sensitivity analysis will collapse to per-RCT and report.
5. **Domain heterogeneity.** Some RoB 2 domains have low base-rate variation (e.g. randomization usually low risk), inflating apparent agreement irrespective of model skill. Per-domain κ is sensitive to this prevalence effect.

## 11. Disclosures

**Conflicts of interest.** None declared. BiasBuster is a research project of the corresponding author, unaffiliated with Anthropic, OpenAI, Google, Alibaba, Cochrane, or any of the model providers cited.

**Use of generative AI.** Per [medRxiv FAQ policy](https://www.medrxiv.org/about/FAQ) (retrieved 2026-04-30): *"AI tools and large language models (LLMs) do not meet [the accountability] requirement and should not be listed as authors on articles."* Authorship is therefore restricted to the human author(s). However, this study uses Anthropic's Claude (model identifiers `claude-opus-4-7` and `claude-sonnet-4-6`) extensively as a computational methodology. The use will be detailed in the manuscript's Methods section, including: literature retrieval and triage, code generation for data ingestion and statistical analysis, drafting of this pre-analysis plan, and manuscript prose drafting. The corresponding author retains accountability for the accuracy, integrity, and originality of all content, having reviewed every AI-generated artefact before inclusion. Note also that the same Claude model family is itself one of the four systems evaluated against the benchmark (§4) — this dual role (analytical assistant *and* evaluated system) is non-circular because the evaluated outputs are a separate set of API calls operating on the benchmark RCTs, with no access to this manuscript or its analysis. Full prompt text, generated code, and an audit trail of conversation transcripts will be archived in the supplementary materials.

**Cost-control measures.** Anthropic prompt caching is enabled on the system prompt and the user-message materials block. This is a billing optimization with no methodological impact (caching changes how input tokens are *billed*, not how the model *responds*); we note it only for transparency and reproducibility. See `cost_estimate.md` in this folder for the full token and dollar estimate.

## 12. Lock-in protocol

Before any LLM call is issued against the benchmark dataset:
1. This document is finalized.
2. Prompt template `prompt_v1.md` is finalized in this folder.
3. Both files are committed to `main`. The commit hash is recorded above as the lock hash.
4. Any subsequent change to methodology requires a dated amendment in this file and a new commit, *with the original commit preserved in history*.

---

**Sign-off requested before commit.**

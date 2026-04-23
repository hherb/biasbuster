# medRxiv preprint draft (sketch)

**Status:** rough outline; expand each bullet into prose before submission.
**Stage:** pre-data-finalization — `n` will grow as additional reviews are added.

---

## Working title (pick one)

- "AI risk-of-bias assessors that strictly apply published methodology algorithms expose systematic algorithm-violations in expert-extracted ratings: a benchmark on Cochrane RoB 2 and QUADAS-2"
- "When the algorithm is the truth: comparing strict-application AI risk-of-bias assessors to published expert ratings"
- "An auditable AI workflow for Cochrane RoB 2 and QUADAS-2 reveals systematic expert-rater drift from the tools' own decision rules"

## Abstract (target 250 words)

- Risk-of-bias (RoB) assessment is the bottleneck of evidence synthesis; expert ratings are assumed authoritative.
- We built a decomposed, full-text, LLM-driven assessor (Cochrane RoB 2, QUADAS-2) that emits per-domain signalling answers, judgements, justifications, and verbatim evidence quotes — JSON-Schema-validated for independent audit.
- Compared AI assessor against published expert ratings from two systematic reviews (one QUADAS-2, one RoB 2), `n` papers each.
- **Headline finding**: per-domain agreement was poor (weighted kappa near 0 on 3-4 of 4-5 domains) — but on audit, the AI applied the published Cochrane / Whiting decision rules correctly while the experts assigned ratings that *contradict the tools' own algorithms* given the paper's content.
- Two distinct expert failure modes observed: blanket-`low` rubber-stamping (one source) and per-domain `low` assignments incompatible with the signalling facts (another source).
- Overall worst-wins judgements still agreed (kappa=1.0 in the engaged-expert source) because both AI and human correctly identified the dominant concerning domain — the disagreement is in the *granular* ratings reviewers and meta-analysts actually rely on.
- **Implication**: AI is not the weak link; *strict application of published methodology* is. We provide the assessor, JSON spec, and audit workflow as open infrastructure.

## 1. Introduction (≤ 600 words)

- RoB assessment is rate-limiting and inconsistent across reviewers (cite McKenzie/Higgins on inter-rater reliability of RoB 2; cite Hartling on reviewer drift).
- LLM-based RoB assessors are emerging — most published work reports *agreement* against expert ratings as the success metric (cite Lai 2024, Chen 2024, etc.). This implicitly treats expert ratings as ground truth.
- We argue this framing is wrong: published expert ratings are themselves error-prone, and the methodologies (Cochrane RoB 2, QUADAS-2) define algorithms that *can be checked*. The right metric is "does the assessor follow the algorithm?", not "does the assessor agree with this reviewer?".
- Research question: when an AI assessor that emits auditable per-domain reasoning *strictly* applies the published RoB 2 / QUADAS-2 algorithms, where do disagreements with expert ratings actually come from?
- Contributions:
  - **Decomposed full-text assessor** (one LLM call per domain; signalling answers → judgement → evidence quotes) for both RoB 2 and QUADAS-2.
  - **JSON Schema spec** for the assessor's output, enabling 60-second per-disagreement audit by any reviewer with the tool's decision rules in hand.
  - **Empirical comparison** against two published expert review sources, with full per-domain breakdown rather than overall-only kappa.
  - **Open-source pipeline** (link to repo).

## 2. Methods (≤ 1000 words)

### 2.1 AI assessor architecture

- One LLM call per domain (5 for RoB 2, 4 for QUADAS-2). Avoids whole-paper compression of context.
- Per-domain prompt structure: domain definition + signalling questions + algorithm + JSON output schema + "CRITICAL OUTPUT RULE: respond with ONLY the JSON object below".
- Backend: `anthropic/claude-sonnet-4-6` via Anthropic API. Identical decomposed orchestration for both methodologies.
- Full-text source: Europe PMC `fullTextXML` (JATS); abstract-only papers refused (`requires_full_text=True`).
- Worst-wins rollup applied locally from per-domain judgements (deterministic, not LLM-derived).

### 2.2 Output JSON contract (citable supplement)

- Each assessment stores: per-domain signalling answers (Y/PY/PN/N/NI for RoB 2; yes/no/unclear for QUADAS-2), judgement, justification, and verbatim evidence quotes from the paper.
- Validated against published JSON Schema (`schemas/rob2_annotation.schema.json`, `schemas/quadas2_annotation.schema.json`) at insert-time. Malformed annotations cannot enter the dataset.
- See [`docs/ANNOTATION_JSON_SPEC.md`](../ANNOTATION_JSON_SPEC.md) for the formal spec and worked audit example.

### 2.3 Engineering hardening (worth a short paragraph — most LLM-methodology papers gloss this)

- Prose-vs-JSON: tightened "JSON only" prompts + brace-balanced fallback parser.
- British/American spelling: accept `judgement` and `judgment` as equivalent keys.
- Per-domain LLM hedging: when the LLM emits valid signalling answers but omits the judgement field (observed for RoB 2 D2 deviations on PMID 36101416 across 3 retries), default to `some_concerns` per the Cochrane algorithm's documented "everything else" rule rather than aborting the paper.
- All these decisions are documented and tested; the failure modes are real and worth flagging because most LLM-RoB papers do not describe them.

### 2.4 Expert-rating ground-truth sources

- **QUADAS-2** source: salivary glucose for diabetes mellitus systematic review (Jcm-15-01829), per-domain ratings extracted from JATS Table 2.
- **RoB 2** source: Deng et al. 2024, plyometric training meta-analysis (`10.1038/s41598-024-61905-7`), per-domain ratings extracted from Figure 2 traffic-light plot.
- (Pending) one additional source per methodology — see Limitations.

### 2.5 Statistical analysis

- Per-domain confusion matrices: model rating × expert rating.
- Weighted Cohen's kappa per domain (treating ordinal scale low<unclear/some_concerns<high).
- Mean absolute error on the ordinal scale.
- Overall worst-wins agreement separately reported.
- **Critical methodological choice**: every disagreement is paired with the model's signalling answers and the algorithm in the original tool's prompt, so each disagreement is independently re-checkable by any reader.

## 3. Results (≤ 800 words + tables)

### 3.1 QUADAS-2 (jcm-15-01829, n=7 papers, salivary glucose for diabetes)

- All 7 expert ratings were `low` across all domains. AI ratings: patient_selection 7/7 `high`; index_test 4 `high` + 3 `unclear`; reference_standard 7/7 `unclear`; flow_and_timing 5 `low` + 2 `unclear`.
- Weighted kappa = 0.000 on the rolled-up overall.
- Audit (representative): every paper structurally compares cases (diagnosed diabetics) vs. controls (healthy volunteers). QUADAS-2 Q1.2 ("Was a case-control design avoided?") = no → bias is "almost certainly high" per Whiting 2011. Expert's `low` is incompatible with the tool's text.
- Reference standard: every paper used HbA1c or FPG (universally accepted reference standards). Model hedged to `unclear` due to missing blinding statement; expert assigned `low` without justification. Both defensible; a prompt tweak would resolve this specific case.

### 3.2 RoB 2 (Deng 2024, n=3 papers with full text in cache, plyometric training)

- Overall worst-wins agreement: 3/3 exact, weighted kappa = 1.000.
- Per-domain agreement: D1 randomization 1/3 exact; **D2 deviations 0/3** (all expert `low`, all model `some_concerns`); D3 missing outcome 2/3 exact (kappa=0.667 — the *only* domain with non-trivial agreement); **D4 measurement 0/3** (all expert `low`, all model `some_concerns`); **D5 reporting 0/3** (all expert `low`, all model `some_concerns`).
- **D2 audit**: signalling answers were identical across all 3 papers — `2.1=Y, 2.2=Y, 2.6=PN`. Cochrane D2 rule: `low` requires `2.1/2.2 both N/PN AND 2.6 Y/PY`. Three preconditions, all violated. Expert's `low` cannot be reconciled with the algorithm.
- **D4 audit**: blinding of outcome assessors not reported in any of the 3 papers. Cochrane D4: `low` requires `4.3=Y/PY` (assessment method unaffected by knowledge of intervention); when not stated, `some_concerns` is the documented default.
- **D5 audit**: no pre-registered analysis plan in any of the 3 papers. Cochrane D5: `low` requires `5.1=Y/PY` (analyses match a pre-specified plan); without registration this is at minimum `some_concerns`.

### 3.3 Two failure modes for "expert ground truth"

- Mode A — *blanket rubber-stamp* (jcm-15-01829): every domain × every paper = `low`. No engagement with the tool.
- Mode B — *engaged but algorithmically inconsistent* (Deng 2024): per-domain ratings vary across papers and domains (the reviewer engaged), but specific per-domain `low` assignments still violate the published decision rules.
- Mode B is the more concerning finding: it survives the obvious "lazy reviewer" critique.

### 3.4 Where the model and engaged-expert agree

- D3 (missing outcome data) in RoB 2 — the only per-domain finding where both AI and engaged expert reliably distinguished `low`/`high`. Both correctly identified Van Roie 2020 as `high` (25% dropout) and Radwan 2021 as `low`.
- The overall worst-wins rollup (the single number readers extract from a published RoB 2 figure) agreed 3/3 in the engaged-expert dataset.
- Translation: the AI doesn't disagree about the *bottom line* — it disagrees about the *route*, and on audit its route is more rigorous.

## 4. Discussion (≤ 800 words)

- Calling the AI "wrong" because it disagrees with the published reviewer is precisely backwards when the reviewer's rating is itself algorithmically inconsistent.
- The right benchmark for an AI RoB assessor is: *given the paper's content, does the AI's per-domain rating follow the methodology's decision rule?* — not "does it agree with this reviewer?".
- **Prior work that benchmarks LLM RoB assessors against expert ratings as ground truth is interpreting their results upside-down**: a high agreement number may indicate the AI has learned to replicate expert *errors*, not that it has learned the methodology.
- Our model's signalling-answer + algorithm-rule audit trail makes it possible to distinguish "AI is wrong" from "expert is wrong" *paper by paper, domain by domain*, in 60 seconds — which neither published expert ratings nor closed-output AI assessors permit.
- Implications for systematic-review practice:
  - Reviewers should publish per-domain signalling-question answers, not just the rolled-up traffic-light.
  - Cochrane / Joanna Briggs / methodology editorial groups should consider mandating algorithm-conformant validation as a peer-review step.
  - Funders / journals should accept AI-assisted RoB with full audit trail as at-minimum equivalent to one human reviewer.
- Why hasn't this been noticed before? — overall kappa is what's reported in inter-rater reliability studies; per-domain breakdown with algorithm-conformance check is rare. The AI assessor surfaces it because the JSON output is structured.

## 5. Limitations

- Small sample (`n` = 7 + 3 + pending). Expanding to more reviews per methodology is the obvious next step.
- Two domains, two journal sources. Generalizability across clinical specialties untested.
- Sonnet 4.6 only. Other LLM backends may produce different per-domain calibrations.
- Expert ratings extracted from published figures/tables, not from the original reviewers' raw notes. Some expert "low" calls may reflect transcription compression rather than the reviewer's true judgement.
- The model's `some_concerns` default on missing-judgement (per Cochrane's documented "everything else" rule) introduces a small per-domain bias; we report this explicitly.
- The "engaged expert" comparison (Deng 2024) is `n=3` with full text; the abstract-only papers remain unevaluated. PDF fallback would expand `n`.

## 6. Conclusion

- LLM RoB assessors that strictly apply published methodology algorithms produce per-domain ratings that **differ systematically from published expert ratings — and on audit the AI is right**.
- The right success metric for LLM RoB assessors is algorithm-conformance, not expert-agreement.
- Open spec, schemas, and codebase: <link>.

## 7. Data and code availability

- Code: <github link to biasbuster repo, commit hash>.
- Schemas: `schemas/rob2_annotation.schema.json`, `schemas/quadas2_annotation.schema.json`.
- Annotation JSON spec: `docs/ANNOTATION_JSON_SPEC.md`.
- All AI assessor outputs (annotations.* table dump): supplementary file S1.
- Expert ratings dataset: supplementary file S2.
- Per-paper audit trail (60-second walkthrough for every disagreement): supplementary file S3.

## 8. Supplementary material plan

- S1: full annotation JSON dump per paper × per methodology (model output).
- S2: extracted expert ratings (PMID, methodology, per-domain, source review citation).
- S3: per-disagreement audit table (PMID × domain × expert rating × model rating × signalling answers × algorithm rule check).
- S4: prompts (per-domain system prompts as run, verbatim).
- S5: replication recipe (one-shell-script reproduction from a fresh DB).

---

## Open questions / decisions before submission

- [ ] Pick the second QUADAS-2 review and second RoB 2 review (in progress).
- [ ] Decide: report n separately per source, or pooled across sources? (Recommend per-source, with pooled as a sensitivity analysis.)
- [ ] Pre-register the analysis plan on OSF before adding sources 3-4.
- [ ] Recruit one independent methodologist to do a parallel manual RoB 2 / QUADAS-2 on 5 papers as a *third* ground truth — would close the "are you sure the experts are wrong?" critique.
- [ ] Editorial note: consider whether to soften "experts are wrong" framing in abstract — alternative framing is "AI assessor reveals algorithm-conformance gaps in published expert ratings" (less aggressive, same finding).
- [ ] License decision for the AI annotations in supplement (CC-BY-4.0 recommended).

## Audience / venue notes

- **medRxiv**: appropriate for the methods + small-n empirical case study.
- **Target journal after preprint**: Research Synthesis Methods, Journal of Clinical Epidemiology, BMC Medical Research Methodology, BMJ EBM. Avoid AI-specific venues — the audience is methodologists, not ML engineers.
- **Preprint timing**: post once both methodologies have ≥2 source reviews and pre-registration is in place.

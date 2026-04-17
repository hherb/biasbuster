# Assessing Risk of Bias in Medical Literature

A practical guideline for choosing and applying risk-of-bias (RoB) tools
when appraising biomedical studies.  This document is both a reference
for human annotators working on the BiasBuster dataset and a source of
truth the annotation prompt can draw from when suggesting verification
steps.

The core principle: **tool selection is driven by study design, not by
clinical specialty**.  A cardiology RCT and a cardiology diagnostic
accuracy study need different instruments, even though the topic is
identical.

---

## 1. Quick selection map

| If the paper is a… | Use this tool | What it assesses |
|---|---|---|
| Parallel-group randomized controlled trial | **RoB 2** | Risk of bias in the effect estimate of a randomized intervention study |
| Cluster-randomized trial | **RoB 2 for CRTs** | RoB 2 extended for cluster allocation (adds recruitment bias domain) |
| Crossover trial | **RoB 2 for crossover trials** | RoB 2 extended for period/carry-over effects |
| Non-randomized study of an intervention (cohort, interrupted time-series, before/after) | **ROBINS-I** | Bias relative to a hypothetical target trial |
| Non-randomized study of an exposure (aetiology, harm from environmental/occupational exposure) | **ROBINS-E** | Bias relative to a hypothetical target trial of the exposure |
| Diagnostic accuracy study (index test vs. reference standard) | **QUADAS-2** (or **QUADAS-C** for comparative accuracy) | Patient selection, index test, reference standard, flow & timing |
| Clinical prediction model (development, validation, or updating) | **PROBAST** | Participants, predictors, outcome, analysis |
| AI/ML prediction model | **PROBAST+AI** | PROBAST extended with AI-specific items (data leakage, hyper-parameter tuning, fairness) |
| Systematic review as the unit being appraised | **ROBIS** (review-level RoB); **AMSTAR 2** (methodological quality, complementary) | Review process: eligibility, identification, data collection, synthesis |
| Meta-analysis potentially distorted by missing studies | **ROB-ME** | Bias due to missing evidence in the synthesis |
| Animal intervention study (preclinical) | **SYRCLE RoB** | RoB 2 analogue adapted for animal experiments |

**Fast decision rule**

1. What is the paper estimating? Intervention effect, test accuracy, or predictive performance?
2. **Intervention effect** → randomized: **RoB 2**.  Non-randomized: **ROBINS-I**.
3. **Test accuracy** → **QUADAS-2**.
4. **Prediction model** → **PROBAST** (or **PROBAST+AI** for ML/AI).
5. **Systematic review / evidence synthesis** → **ROBIS** (+ **ROB-ME** if missing-evidence bias is plausible).

Mnemonic: *trials = RoB 2, observational interventions = ROBINS-I,
tests = QUADAS-2, models = PROBAST, reviews = ROBIS, missing evidence
= ROB-ME.*

---

## 2. Tool reference

Each section summarises the tool's scope, domains, signalling-question
structure, and overall judgement.  Citations point to the primary
methodology paper and the maintained online resource.

### 2.1 RoB 2 — Randomized trials

**Scope.** Parallel-group RCTs, with design-specific variants for
cluster and crossover trials.  Replaces the original Cochrane "RoB 1"
tool (2011) which used a single study-level judgement.  RoB 2 operates
**per outcome, per result** — a single trial can be low-RoB for its
primary outcome and high-RoB for a harm outcome.

**Five domains.**

1. **Bias arising from the randomization process** — sequence
   generation, allocation concealment, baseline imbalance suggestive of
   failure.
2. **Bias due to deviations from intended interventions** — separately
   considers effect of *assignment* (intention-to-treat-like) and
   effect of *adhering* (per-protocol-like) to intervention.  Reviewers
   must declare which is of interest.
3. **Bias due to missing outcome data** — whether outcome data were
   available for nearly all participants, and whether missingness
   depends on the true value.
4. **Bias in measurement of the outcome** — who measured, blinding of
   assessors, appropriateness of method.
5. **Bias in selection of the reported result** — selective reporting
   from multiple outcome measurements or analyses.

**Workflow.** Each domain has ~5 signalling questions answered *Yes /
Probably Yes / Probably No / No / No Information*; an algorithm maps
those answers to a domain judgement of **Low / Some concerns / High
risk of bias**.  Overall judgement defaults to the worst domain (any
"High" → overall High; any "Some concerns" → at least Some concerns).

**Primary reference.**  Sterne JAC, Savović J, Page MJ, et al. *RoB 2:
a revised tool for assessing risk of bias in randomised trials.* BMJ
2019;366:l4898. <https://doi.org/10.1136/bmj.l4898>

**Online tool and variants.**
<https://www.riskofbias.info/welcome/rob-2-0-tool>
(cluster and crossover variants linked from the same page).

**Practical notes for annotators.**
- RoB 2 is explicit that "unclear" is *not* a valid final answer — the
  algorithm must run even with "No Information" inputs.
- For industry-sponsored trials with sponsor-employed authors, RoB 2
  does **not** classify this as a source of bias by itself; it is
  treated as a threat only if it manifests in one of the five domains
  (e.g. result-selection bias).  BiasBuster's annotation policy
  deliberately diverges here and treats industry + sponsor-employed
  authors as a hard-HIGH trigger on the conflict-of-interest domain
  (see `prompts.py` and `feedback_risk_not_proof` in project memory).

### 2.2 ROBINS-I — Non-randomized studies of interventions

**Scope.** Comparative observational or quasi-experimental studies
estimating the effect of an intervention: cohort studies, controlled
before-after studies, interrupted time-series, registry-based
comparative effectiveness.  *Not* for case reports, case series, or
purely descriptive studies.

**Conceptual anchor.** ROBINS-I is built around a **target trial** — a
hypothetical pragmatic RCT that would answer the same clinical
question.  Bias is judged by comparing the observed study to that
ideal trial.

**Seven domains** (three pre-intervention, one at-intervention, three
post-intervention):

1. **Bias due to confounding** (pre)
2. **Bias in selection of participants into the study** (pre)
3. **Bias in classification of interventions** (pre / at)
4. **Bias due to deviations from intended interventions** (post)
5. **Bias due to missing data** (post)
6. **Bias in measurement of outcomes** (post)
7. **Bias in selection of the reported result** (post)

**Judgements.** Low / Moderate / Serious / Critical / No information.
"Low" in ROBINS-I means equivalent to a well-conducted RCT — a
demanding bar that observational studies rarely meet.  "Moderate" is
the more common best-case outcome.

**Primary reference.**  Sterne JA, Hernán MA, Reeves BC, et al.
*ROBINS-I: a tool for assessing risk of bias in non-randomised studies
of interventions.* BMJ 2016;355:i4919.
<https://doi.org/10.1136/bmj.i4919>

**Online tool.** <https://www.riskofbias.info/welcome/home/current-version-of-robins-i>

**Practical notes.**
- Confounding is usually the dominant domain.  If the authors cannot
  list the key confounders they adjusted for, the domain is at least
  Serious.
- A "target trial" framing forces reviewers to name the contrast of
  interest (e.g. "start metformin vs. start sulfonylurea at diabetes
  diagnosis") rather than vague exposure comparisons.

### 2.3 ROBINS-E — Non-randomized studies of exposures

**Scope.** Studies of the causal effect of an **exposure** on a health
outcome (nutrition, occupational hazard, pollutant, lifestyle factor).
Structurally parallel to ROBINS-I but adapted for settings where
"intervention" is not a deliberate act.

**Primary reference.**  Higgins JPT, Morgan RL, Rooney AA, et al. *A
tool to assess risk of bias in non-randomized follow-up studies of
exposure effects (ROBINS-E).* Environ Int 2024;186:108602.
<https://doi.org/10.1016/j.envint.2024.108602>

**Online tool.** <https://www.riskofbias.info/welcome/robins-e-tool>

### 2.4 QUADAS-2 — Diagnostic accuracy studies

**Scope.** Primary studies reporting sensitivity, specificity,
likelihood ratios, or other measures of diagnostic or screening test
performance against a reference standard.

**Four domains**, each rated for both **risk of bias** *and*
**applicability concerns**:

1. **Patient selection** — how were participants recruited? Case-control
   designs, inappropriate exclusions, and convenience sampling raise
   concern.
2. **Index test** — was it conducted/interpreted without knowledge of
   the reference standard? Was the threshold pre-specified?
3. **Reference standard** — is it a valid gold standard? Was it
   interpreted without knowledge of the index test?
4. **Flow and timing** — interval between tests, whether all patients
   received the same reference standard, handling of withdrawals.

**Judgements.** Low / High / Unclear for each domain × (bias,
applicability).

**Primary reference.**  Whiting PF, Rutjes AWS, Westwood ME, et al.
*QUADAS-2: a revised tool for the quality assessment of diagnostic
accuracy studies.* Ann Intern Med 2011;155(8):529–36.
<https://doi.org/10.7326/0003-4819-155-8-201110180-00009>

**Comparative accuracy extension.**  Yang B, Mallett S, Takwoingi Y,
et al. *QUADAS-C: a tool for assessing risk of bias in comparative
diagnostic accuracy studies.* Ann Intern Med 2021;174(11):1592–9.
<https://doi.org/10.7326/M21-2234>

**Practical notes.**
- "Differential verification" (index-positive and index-negative
  patients get different reference standards) and "partial
  verification" (some patients never get the reference standard) are
  the two most common flow-and-timing failures.
- Case-control designs inflate apparent accuracy and should almost
  always be rated high-risk on patient selection.

### 2.5 PROBAST and PROBAST+AI — Prediction-model studies

**Scope.**  Studies developing, validating, or updating a **diagnostic
or prognostic prediction model** (risk score, nomogram, classifier)
for individual patients.  PROBAST covers classical statistical models;
**PROBAST+AI** (2025) is the updated instrument for AI/ML models.

**Four domains.**

1. **Participants** — data source, eligibility, representativeness.
2. **Predictors** — definition, assessment, blinding to outcome.
3. **Outcome** — definition, blinding to predictors, time horizon.
4. **Analysis** — sample size, handling of missing data, selection of
   predictors, performance measures, **calibration** *and*
   discrimination, optimism/overfitting adjustment, internal and
   external validation.

**Judgements.** Low / High / Unclear risk of bias per domain, plus
separate **applicability** ratings for the first three domains.

**Primary references.**
- Wolff RF, Moons KGM, Riley RD, et al. *PROBAST: a tool to assess the
  risk of bias and applicability of prediction model studies.* Ann
  Intern Med 2019;170(1):51–8. <https://doi.org/10.7326/M18-1376>
- Moons KGM, Wolff RF, Riley RD, et al. *PROBAST: a tool to assess
  risk of bias and applicability of prediction model studies:
  explanation and elaboration.* Ann Intern Med 2019;170(1):W1–33.
  <https://doi.org/10.7326/M18-1377>
- Moons KGM, Damen JAA, Kaul T, et al. *PROBAST+AI: an updated
  quality, risk of bias, and applicability assessment tool for
  prediction models using regression or artificial intelligence
  methods.* BMJ 2025;388:e082505.
  <https://doi.org/10.1136/bmj-2024-082505>

**AI-specific concerns added in PROBAST+AI.**
- **Data leakage** (e.g. using patient IDs or acquisition-site cues as
  predictors, splitting by image rather than by patient).
- **Hyper-parameter tuning without an independent test set.**
- **Fairness and subgroup performance.**
- **Transparency and reproducibility** (code, weights, data availability).
- **Deployment-shift considerations.**

### 2.6 ROBIS — Systematic reviews

**Scope.** Risk-of-bias appraisal **of a systematic review itself**,
as an input to an overview/umbrella review or to inform guideline
recommendations.  Complements AMSTAR 2 (which focuses on methodological
quality rather than bias per se).

**Phases and domains.**
- **Phase 1** — assess relevance of the review to the overview's
  question (optional).
- **Phase 2 — four domains:**
  1. Study eligibility criteria
  2. Identification and selection of studies
  3. Data collection and study appraisal
  4. Synthesis and findings
- **Phase 3** — overall judgement of risk of bias in the review's
  conclusions (Low / High / Unclear).

**Primary reference.**  Whiting P, Savović J, Higgins JPT, et al.
*ROBIS: a new tool to assess risk of bias in systematic reviews was
developed.* J Clin Epidemiol 2016;69:225–34.
<https://doi.org/10.1016/j.jclinepi.2015.06.005>

### 2.7 AMSTAR 2 — Methodological quality of systematic reviews

**Scope.** Quality appraisal of systematic reviews that include
randomized or non-randomized studies of interventions.  Often cited
alongside ROBIS; they are complementary rather than interchangeable.
AMSTAR 2 is more prescriptive (16 items, each Yes/Partial Yes/No) and
defines seven **critical items** whose failure collapses confidence in
the review.

**The seven critical items** (paraphrased):
1. Protocol registered before commencement of the review.
2. Adequacy of the literature search.
3. Justification for excluding individual studies.
4. Risk of bias from individual studies being included.
5. Appropriateness of meta-analytical methods.
6. Consideration of RoB when interpreting results.
7. Assessment of publication bias.

**Overall rating.** High / Moderate / Low / Critically Low, determined
by the number of critical and non-critical weaknesses.

**Primary reference.**  Shea BJ, Reeves BC, Wells G, et al. *AMSTAR 2:
a critical appraisal tool for systematic reviews that include
randomised or non-randomised studies of healthcare interventions, or
both.* BMJ 2017;358:j4008. <https://doi.org/10.1136/bmj.j4008>

### 2.8 ROB-ME — Missing evidence in meta-analyses

**Scope.** Used **once a meta-analysis exists**, to assess whether the
synthesis is distorted by studies that were conducted but are not
contributing results (publication bias, selective non-reporting of
results, grey-literature exclusion).  Complements — does not replace —
within-study RoB tools.

**Five signalling questions** covering the likelihood of unpublished
studies, selective non-reporting, and whether statistical/graphical
methods (funnel plot, Egger's test, selection models) are appropriate
and were used.  Judgement: Low / Some concerns / High / No information.

**Primary reference.**  Page MJ, Sterne JAC, Boutron I, et al. *ROB-ME:
a tool for assessing risk of bias due to missing evidence in systematic
reviews with meta-analysis.* BMJ 2023;383:e076754.
<https://doi.org/10.1136/bmj-2023-076754>

**Online tool.** <https://www.riskofbias.info/welcome/rob-me-tool>

### 2.9 SYRCLE RoB — Animal intervention studies

**Scope.** Risk-of-bias assessment for controlled animal experiments
testing an intervention.  Adapted from the original Cochrane RoB tool
with items specific to pre-clinical work (randomization of housing,
blinding of caretakers, reporting of animal exclusions).

**Primary reference.**  Hooijmans CR, Rovers MM, de Vries RBM, et al.
*SYRCLE's risk of bias tool for animal studies.* BMC Med Res Methodol
2014;14:43. <https://doi.org/10.1186/1471-2288-14-43>

---

## 3. Edge cases and awkward designs

Not every paper fits one of the instruments above.  For these, either
use a specialised tool if one exists or document explicitly that no
single instrument is appropriate.

| Design | Recommended approach |
|---|---|
| Case report / case series | **JBI Critical Appraisal Checklist** for case reports/series, or **IHE** quality appraisal for case series.  No SOTA RoB tool exists — document this limitation. |
| Qualitative research | **CASP Qualitative Checklist** (not a RoB tool, but the closest analogue). |
| Mixed-methods study | **MMAT** (Mixed Methods Appraisal Tool). |
| Economic evaluation | **Drummond 10-item checklist** or **CHEERS reporting statement** (reporting rather than bias). |
| Laboratory / mechanistic in vitro study | No dominant tool.  Domain-specific checklists exist (e.g. **ToxRTool** for toxicology).  See Tran L et al., BMC Med Res Methodol 2021;21:101 for a review of the landscape. <https://doi.org/10.1186/s12874-021-01295-w> |
| Genetic association study | **Q-Genie**. |
| Narrative review | Not eligible for RoB appraisal in the conventional sense — use **SANRA** for quality if needed. |

If a paper does not fit any instrument, annotators should say so
explicitly (in both the human-review UI and the exported annotation)
rather than forcing an inappropriate tool — a forced rating is worse
than a documented gap.

---

## 4. How this relates to BiasBuster annotations

The BiasBuster annotation schema (see [ANNOTATED_DATA_SET.md](ANNOTATED_DATA_SET.md))
rates each abstract across **five bias domains**: statistical reporting,
spin, outcome reporting, conflict of interest, and methodology.  These
domains are **not** identical to any single RoB tool's domains — they
are a distillation chosen because they are detectable from an abstract
alone and map across multiple study designs.

When assessing a paper during human review, the full-text RoB tool for
that design (RoB 2, ROBINS-I, QUADAS-2, etc.) is the reference
standard.  The abstract-level annotation is a *prediction* of what a
full-text RoB assessment would conclude.  Annotators should:

1. **Identify the study design first** using the quick map in §1.
2. **Mentally map the five BiasBuster domains onto the relevant tool's
   domains.** For example, for an RCT:
   - BiasBuster *methodology* ≈ RoB 2 domains 1 (randomization) + 2
     (deviations).
   - BiasBuster *outcome reporting* ≈ RoB 2 domain 5 (selection of
     reported result).
   - BiasBuster *statistical reporting* and *spin* cut across RoB 2
     domains 4 and 5 plus reporting-quality concerns not formally in
     RoB 2.
   - BiasBuster *conflict of interest* is **not** in RoB 2 at all —
     this is a deliberate BiasBuster extension grounded in the
     evidence that industry sponsorship independently predicts
     favourable conclusions (see
     [feedback_risk_not_proof](../../.claude/projects/-Users-hherb-src-biasbuster/memory/feedback_risk_not_proof.md)).
3. **Consult the Cochrane RoB 2 domain ratings** when the paper is a
   Cochrane-reviewed RCT, since these are stored in the database
   (`cochrane_*` fields — see `database.py`) and represent expert
   consensus.

BiasBuster is a **risk-of-bias predictor**, not a proof-of-bias system.
A "high" rating on a BiasBuster domain indicates that a domain expert
performing a full-text RoB assessment would likely be concerned — not
that the study's conclusions are wrong.

---

## 5. Citations and further reading

### Core methodology papers (freely available)

- **RoB 2:** Sterne JAC et al. BMJ 2019;366:l4898. <https://doi.org/10.1136/bmj.l4898>
- **ROBINS-I:** Sterne JA et al. BMJ 2016;355:i4919. <https://doi.org/10.1136/bmj.i4919>
- **ROBINS-E:** Higgins JPT et al. Environ Int 2024;186:108602. <https://doi.org/10.1016/j.envint.2024.108602>
- **QUADAS-2:** Whiting PF et al. Ann Intern Med 2011;155:529–36. <https://doi.org/10.7326/0003-4819-155-8-201110180-00009>
- **QUADAS-C:** Yang B et al. Ann Intern Med 2021;174:1592–9. <https://doi.org/10.7326/M21-2234>
- **PROBAST:** Wolff RF et al. Ann Intern Med 2019;170:51–8. <https://doi.org/10.7326/M18-1376>
- **PROBAST+AI:** Moons KGM et al. BMJ 2025;388:e082505. <https://doi.org/10.1136/bmj-2024-082505>
- **ROBIS:** Whiting P et al. J Clin Epidemiol 2016;69:225–34. <https://doi.org/10.1016/j.jclinepi.2015.06.005>
- **AMSTAR 2:** Shea BJ et al. BMJ 2017;358:j4008. <https://doi.org/10.1136/bmj.j4008>
- **ROB-ME:** Page MJ et al. BMJ 2023;383:e076754. <https://doi.org/10.1136/bmj-2023-076754>
- **SYRCLE RoB:** Hooijmans CR et al. BMC Med Res Methodol 2014;14:43. <https://doi.org/10.1186/1471-2288-14-43>

### Maintained online resources

- Cochrane Methods — Bias: <https://methods.cochrane.org/bias/>
- riskofbias.info (RoB 2, ROBINS-I, ROBINS-E, ROB-ME tools and forms): <https://www.riskofbias.info>
- Cochrane Handbook (chapters 7, 8, 25 on bias): <https://training.cochrane.org/handbook>
- EQUATOR Network (reporting guidelines, complementary to RoB): <https://www.equator-network.org>

### Comparative and critical literature

- Perry R, Whitmarsh A, Leach V, Davies P. *A comparison of two
  assessment tools used in overviews of systematic reviews: ROBIS
  versus AMSTAR-2.* Syst Rev 2021;10:273.
  <https://doi.org/10.1186/s13643-021-01819-x>
- Tran L, Tam DNH, Elshafay A, et al. *Quality assessment tools used
  in systematic reviews of in vitro studies: a systematic review.*
  BMC Med Res Methodol 2021;21:101.
  <https://doi.org/10.1186/s12874-021-01295-w>
- Minozzi S et al. *The revised Cochrane risk of bias tool for
  randomized trials (RoB 2) showed low interrater reliability and
  challenges in its application.* J Clin Epidemiol 2020;126:37–44.
  <https://doi.org/10.1016/j.jclinepi.2020.06.015>

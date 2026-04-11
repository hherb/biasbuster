# Design Rationale — Why BiasBuster's COI Assessment Diverges from Cochrane RoB 2

**Status:** normative design decision, validated 2026-04-11.
**Owner:** BiasBuster project.
**Scope:** the Conflict of Interest (COI) domain and its mechanical HIGH
triggers in `biasbuster/prompts_v3.py`.

---

## 1. Purpose of this document

BiasBuster's v3 pipeline includes Conflict of Interest as one of five
assessment domains. On papers with industry funding and sponsor-employed
authors, the pipeline can rate Conflict of Interest as HIGH — and
therefore the overall severity as HIGH — even when the methodology is
well-conducted and Cochrane's Risk of Bias 2 (RoB 2) tool has rated
the same paper LOW.

This is an **intentional design choice**, not a calibration error.
This document exists to:

1. Record the rationale in one durable place, so future contributors
   don't mistake the divergence for a bug and propose weakening the
   trigger.
2. Help users of BiasBuster interpret the pipeline's output correctly
   — particularly when the rating disagrees with an external tool.
3. Give reviewers, auditors, and downstream consumers a clear
   statement of what BiasBuster is and isn't measuring.

## 2. The core framing — risk of bias, not proof of bias

Every rating the pipeline emits is a statement about **risk**, not a
forensic finding. The output is not "this paper is biased" — it is
"this paper carries a level of risk that a reader should factor in".

This distinction matters because the thing we can assess from the
paper text is always upstream of the thing we actually care about:

| What the paper text shows | What it implies about bias |
|---|---|
| Who funded the study | A structural incentive that increases the probability of bias |
| Who is employed by the sponsor | A structural channel through which bias can enter — manuscript drafting, data interpretation, framing of conclusions |
| Whether methodology is visibly flawed | Direct evidence of bias in the work we can see |
| Whether the abstract uses hedged language | A stylistic signal — weak evidence about bias |

A paper can have structural COI present **and** impeccable
methodology. A reader who accepts the paper's conclusions uncritically
is still accepting them from a source with structural incentives to
favour the sponsor's product. The pipeline's job is to surface this
so the reader can verify independently — not to adjudicate whether
the structural incentive *actually* affected the results in this
specific case.

The categorical severity + probability combination is the mechanism
for expressing this separation: a rating of `high / 0.68` means
"categorical HIGH (risk level that warrants independent verification)
with probability 0.68 (at the top of moderate, signalling that the
methodology itself is acceptable — the HIGH category is driven by
the structural signal, not by direct evidence of damage)".

## 3. What Cochrane RoB 2 actually covers

Cochrane's Risk of Bias 2 tool is authoritative for assessing the
internal methodological quality of randomised trials. Its five
domains are:

1. Bias arising from the randomisation process
2. Bias due to deviations from intended interventions
3. Bias due to missing outcome data
4. Bias in measurement of the outcome
5. Bias in selection of the reported result

These are strictly about whether the **methodology** of the trial
is sound. Cochrane RoB 2 deliberately **excludes**:

- Funding source
- Author affiliations and conflicts of interest
- Structural relationships between authors and sponsors
- Adequacy of sample size (a design-time question, not a
  methodological one under RoB 2's framing)
- Spin in the conclusions and title

This is a deliberate design choice by Cochrane. RoB 2 answers one
specific question very well — "are the reported results a
methodologically-valid estimate of the effect?" — and explicitly
leaves other sources of bias risk to separate assessments (e.g.
funding analyses, GRADE evaluations, narrative reviews).

**This means a paper can legitimately rate LOW under Cochrane RoB 2
while carrying substantial COI risk that RoB 2 cannot and does not
measure.**

## 4. What BiasBuster covers

BiasBuster assesses five domains, only some of which overlap with
Cochrane RoB 2:

| BiasBuster domain | Overlaps RoB 2? | Notes |
|---|---|---|
| Statistical Reporting | Partial | Overlaps with "selection of the reported result" |
| Spin | No | Framing/conclusion language — outside RoB 2 scope |
| Outcome Reporting | Partial | Overlaps with "selection of the reported result" |
| Conflict of Interest | **No** | Cochrane RoB 2 explicitly excludes COI |
| Methodology | Yes | Overlaps with randomisation, deviations, missing data |

The two tools are **not measuring the same thing**. A "disagreement"
between BiasBuster and Cochrane RoB 2 on a paper is not automatically
a signal that one of them is wrong — it may simply be a signal that
they are measuring different properties of the paper.

Specifically:

- **BiasBuster Methodology domain vs Cochrane Methodology domains**
  should track closely. Systematic disagreement here is a real
  calibration problem.
- **BiasBuster Statistical Reporting and Outcome Reporting domains
  vs Cochrane "selection of the reported result"** should track
  closely. Systematic disagreement here is a real calibration problem.
- **BiasBuster Spin and Conflict of Interest domains vs anything in
  Cochrane** have no equivalent. Disagreement here is expected and
  not a calibration problem — the pipeline is adding information that
  Cochrane does not provide.

## 5. The specific decision — COI as a first-class domain with mechanical HIGH triggers

BiasBuster treats Conflict of Interest as a full domain on par with
Methodology, not as a moderating factor applied to other domains.
The prompt at `biasbuster/prompts_v3.py` (Round 10, commit `c34885a`)
defines four mechanical HIGH triggers for this domain:

1. `sponsor_controls_analysis = True AND sponsor_controls_manuscript = True`
2. Industry funding AND undisclosed COI AND industry author affiliations
3. `sponsor_controls_analysis = True` in a trial with surrogate primary outcomes
4. **`funding_type ∈ {industry, mixed} AND authors_with_industry_affiliation
   contains ≥1 entry with role ∈ {employee, shareholder}`**

Trigger (d) is the mechanism responsible for most of BiasBuster's
COI-driven disagreement with Cochrane. It fires on any industry-funded
or mixed-funded paper where at least one author is employed by, or
holds shares (including stock options) in, the sponsor. When it fires,
severity MUST be HIGH regardless of transparency, independent
statistical oversight, or "sponsor had no role" disclaimers.

### Why this specific rule

1. **Authorship is participation in manuscript drafting by
   definition.** An author writes, revises, and approves the paper
   as part of their role. When the author is a sponsor employee,
   the sponsor is drafting the manuscript via that author —
   regardless of whether the paper includes a "sponsor had no role"
   statement. That statement is about the institutional sponsor
   entity, not about the specific employees who are authors.

2. **Disclaimer language is extractable but untestable from the
   paper alone.** A paper can say the sponsor had no role in
   analysis or manuscript drafting, but the reader cannot verify
   that claim from inside the paper. The structural signal —
   sponsor employees are authors — is directly observable in the
   author list and is harder to fake.

3. **Structural COI has produced real harm, including fabricated
   results.** The project owner has first-hand experience of
   pharmaceutical-company studies where the structural incentives
   manifested in ways that would not have been visible from a
   methodology-only assessment. A risk assessment that ignores
   structural incentives is insensitive to an entire category of
   failure.

4. **The cost of a false-positive HIGH rating is much lower than
   the cost of a false-negative.** A HIGH rating asks the reader
   to verify the findings via external sources; it does not claim
   the paper is wrong. A MODERATE rating does not trigger the same
   verification behaviour. Asymmetric costs argue for being
   conservative on the risk-flagging side.

### What trigger (d) does NOT claim

- It does not claim the specific paper under assessment is biased.
- It does not claim sponsor-employed authors acted improperly.
- It does not claim the methodology is flawed.
- It does not override evidence that the methodology IS sound.
  The probability score carries that information — a paper with
  trigger (d) firing and otherwise-clean methodology will end up
  at the low end of the HIGH probability range (~0.65-0.75),
  signalling "structural risk present, methodology otherwise
  acceptable".

## 6. Empirical validation — the tapinarof case study

On 2026-04-11 we ran Claude's full-text pipeline on
PMID `39777610` — a Dermavant-sponsored phase 3 atopic dermatitis
trial (Brown et al., *Dermatology and Therapy*, 2025) that Cochrane
had rated as LOW risk of bias (Round 10 calibration test, see
`INITIAL_FINDINGS_V3.md` §6.2). Claude rated it `high / 0.68` with
this signature:

| Domain | Severity |
|---|---|
| statistical_reporting | moderate (subgroup emphasis, unclear multiplicity) |
| spin | moderate ("can be used without restrictions on duration of use, extent of BSA treated, or sites of application" — extrapolation beyond 8-week trial data) |
| outcome_reporting | moderate (secondary PRO endpoints with unclear pre-specification) |
| **conflict_of_interest** | **high** (trigger (d): Dermavant-funded, 9 industry-affiliated authors including 4 Dermavant employees) |
| methodology | moderate (post-hoc pooled analyses) |

Claude is the strongest model in our test set and was used as
reference ground truth for the Round 10 reliability verification
(§3.12). Its independent application of the v3 prompts on this paper
produced:

- Four moderates and one high
- `overall_bias_probability = 0.68` (top of the MODERATE anchor
  range 0.40–0.65, bottom of HIGH 0.70–0.85)
- A reasoning trace that explicitly identifies the spin concern
  ("without restrictions" extrapolation) as corroborating the COI
  risk signal

This tells us:

1. **The pipeline, when executed faithfully, produces the
   risk-focused divergence from Cochrane RoB 2 that we intended.**
   The disagreement with Cochrane on this paper is not a random
   artefact or a local-model failure — it is the prompt doing its
   job.
2. **The categorical + probability mechanism works as intended.**
   Claude reached HIGH through the trigger mechanism but placed the
   probability at the lower edge of HIGH, correctly expressing
   "structural risk level, methodology fine". A reader seeing
   `high / 0.68` knows to verify the COI disclosures and the
   post-hoc pooled analyses without concluding the paper is wrong.
3. **The spin concern is real and corroborates the COI signal.**
   The authors chose to frame the 8-week results as supporting
   unrestricted clinical use — a framing choice more consistent with
   sponsor-aligned marketing language than with the evidentiary weight
   of an 8-week trial. This would not trigger Cochrane RoB 2 (spin
   is outside its scope) but is exactly the class of signal that
   COI risk is meant to flag for verification.

The empirical evidence is consistent with the design rationale. The
divergence on this paper is the feature.

## 7. Implications for interpretation

When BiasBuster rates a paper HIGH on COI and LOW or MODERATE on the
methodological domains, the correct interpretation is:

- The trial was methodologically competent (same as Cochrane would say)
- The structural COI profile is such that independent verification
  is warranted before accepting the conclusions
- The specific things to verify are: CMS Open Payments records for
  named authors, ClinicalTrials.gov for registered outcomes vs
  reported, ORCID for undisclosed affiliations, and the published
  protocol for any outcome switching

The recommended verification steps in the output should be treated
as the actionable part of the rating. A HIGH-COI rating without
follow-through on those verification steps is incomplete use of
the pipeline.

## 8. When disagreement with Cochrane IS a bug

The design rationale above covers the **expected** divergence on
COI. The following disagreement patterns would still be bugs and
should be investigated:

- **Methodology domain disagreement** — if BiasBuster systematically
  rates Cochrane-LOW papers as HIGH on the methodology domain
  (outside the COI channel), something is miscalibrated. The
  methodology domain is supposed to track Cochrane.
- **LOW on a Cochrane-HIGH paper** — the pipeline should not
  under-call methodologically flawed papers. Round 4 found that
  single-call full-text mode (`f1`) collapses to LOW on everything;
  two-call full-text mode (`f2`) does not.
- **Trigger (d) firing without industry funding** — the rule is
  gated on `funding_type ∈ {industry, mixed}`. If a public-funded
  paper's COI comes back HIGH via trigger (d), the extraction is
  mislabelling funding.
- **Overall severity below the maximum domain severity** — the max
  rule means `overall ≥ max(domain severities)`. If overall is lower
  than any domain severity, the assessment prompt is broken.

If any of these appear, they are real bugs and should be fixed.

## 9. Implications for training data

BiasBuster generates fine-tuning data by exporting its own
annotations (see `biasbuster/export.py` and the main project
README). This means models fine-tuned on BiasBuster output will
**learn the same COI-aggressive framing** as the source pipeline.

This is intentional:

- The fine-tuned model is meant to replicate BiasBuster's risk
  judgement, not Cochrane RoB 2's methodology judgement.
- Downstream users of the fine-tuned model should be told, via
  model card / README, that the model is a risk-of-bias assessor
  that is intentionally more aggressive than RoB 2 on COI and that
  the recommended verification steps are part of the intended
  output.
- Comparisons of the fine-tuned model against Cochrane RoB 2 as
  an aggregate metric will show systematic disagreement on
  industry-funded papers — this is a validated feature, not a
  regression.

## 10. Change log

- **2026-04-11** — first version written during the Round 10
  calibration test discussion. Prompted by the tapinarof case
  (PMID 39777610) where Claude's pipeline rated a Cochrane-LOW
  industry phase 3 trial as `high / 0.68`, and the user explicitly
  validated the framing as "risk of bias, not proof of bias" —
  with reference to first-hand pharmaceutical-industry experience
  including fabricated results.

## Related documents

- [`INITIAL_FINDINGS_V3.md`](./INITIAL_FINDINGS_V3.md) — full
  empirical history, including §3.11 (Round 10 prompt edits),
  §3.12 (Round 10 verification across all three local model
  families), and §6.2 (calibration paper test plan).
- [`architecture_guide.md`](./architecture_guide.md) — the two-call
  architecture that this rationale presupposes.
- [`MERGE_STRATEGY.md`](./MERGE_STRATEGY.md) — how per-section
  extractions are merged, including the COI fields that feed
  trigger (d).
- [`CONTEXT_FOR_CLAUDE_CODE.md`](./CONTEXT_FOR_CLAUDE_CODE.md) —
  the original Seed Health probiotic failure case that motivated
  the v3 rebuild and exposed the COI under-call gap.
- `biasbuster/prompts_v3.py` — the authoritative source of the
  current trigger definitions.

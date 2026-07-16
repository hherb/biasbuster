# Bias Assessment Report

**Paper:** Multi-Species Synbiotic Supplementation After Antibiotics Promotes Recovery of Microbial Diversity and Function, and Increases Gut Barrier Integrity: A Randomized, Placebo-Controlled Trial
**PMID:** 12937403
**DOI:** 10.3390/antibiotics15020138
**Model:** anthropic:claude-sonnet-4-6
**Content:** fulltext_jats
**Date:** 2026-04-05T11:08:53.871871+00:00

## Overall Assessment

| Metric | Value |
|--------|-------|
| Severity | **HIGH** |
| Bias Probability | 82% |
| Confidence | HIGH |

**Reasoning:** This trial exhibits multiple converging high-severity bias signals across all five domains. (1) Statistical reporting: Effect sizes are reported almost exclusively as relative measures (percentage changes, fold-changes) without absolute values in either arm, making clinical significance unassessable. A post-hoc subgroup analysis for butyrate in 'low baseline' participants is presented without pre-specification, and no multiplicity correction is applied across numerous simultaneous endpoints. (2) Spin: The conclusions assert broad clinical relevance and use language such as 'compelling evidence' and 'distinct advantages' for a proof-of-mechanism study in healthy volunteers with exclusively surrogate endpoints and no patient-reported outcomes. The title itself implies clinical benefit ('promotes recovery') beyond what the data support. The selective benchmarking against Suez et al. without acknowledging design differences further inflates perceived superiority. (3) Outcome reporting: All primary endpoints are unvalidated surrogates (microbiome diversity, SCFAs, urolithin A, p-cresol sulfate, lactulose permeability). No patient-centred outcomes were collected. The authors themselves acknowledge the trial was not designed to define clinically meaningful thresholds, yet clinical benefit language pervades the conclusions. (4) Conflict of interest: The study is entirely funded by Seed Health, Inc., the commercial manufacturer of DS-01. Two authors are employees and shareholders of the sponsor, and two are paid advisors. The data analyst and lead manuscript author (B.A.N.) is a sponsor employee and shareholder — a high-risk combination. While COI is formally disclosed, the sponsor's influence over data analysis and manuscript preparation is not mitigated by the disclaimer that clinical operations were independent. (5) Methodology: The most serious concern is the 34.4% overall attrition rate with asymmetric dropout (7 in the active arm vs. 4 in placebo) and exclusive per-protocol analysis with no ITT or sensitivity analysis reported. This combination can substantially bias results in favour of the intervention. The small sample (n=21 completers) with multiple primary endpoints and no multiplicity correction further inflates false-positive risk. The artificial antibiotic challenge in healthy volunteers limits generalisability to real-world clinical populations. Together, these concerns — particularly the sponsor-employed data analyst, per-protocol-only analysis with asymmetric dropout, exclusive surrogate endpoints, and pervasive spin — constitute a pattern consistent with high overall bias risk.

## Statistical Reporting

**Severity:** HIGH

| Flag | Value |
|------|-------|
| Relative Only | Yes |
| Absolute Reported | No |
| Nnt Reported | No |
| Baseline Risk Reported | No |
| Selective P Values | No |
| Subgroup Emphasis | Yes |

**Evidence:**
> fecal butyrate (119%, p < 0.05), fecal acetate (62%, p < 0.01), and UroA (13,008%, p < 0.05), whereas detrimental metabolite pCS decreased (68%, p < 0.05) compared to placebo

> the multi-species synbiotic significantly increased the abundance of synbiotic strains in the stool compared to placebo immediately following cessation of antibiotics (Day 7, 2587-fold-change, p < 0.0001)

> participants in the present trial exhibited up to 305% improvement in intestinal permeability, an accepted functional surrogate for gut barrier function, compared to placebo.

> Significance was defined a priori at p < 0.05.

> it has been documented in previous clinical trials that specific synbiotics are most effective in increasing fecal butyrate production in individuals with low baseline butyrate levels [33]. Thus, we evaluated the recovery of butyrate production after antibiotics in a population with low-baseline fecal butyrate.

## Spin

**Severity:** HIGH

| Flag | Value |
|------|-------|
| Spin Level | high |
| Conclusion Matches Results | No |
| Causal Language From Observational | No |
| Focus On Secondary When Primary Ns | No |
| Inappropriate Extrapolation | Yes |
| Title Spin | Yes |

**Evidence:**
> this multi-species synbiotic promotes recovery of gut microbiome diversity and native beneficial microbes, microbiome metabolite recovery, and gut barrier function, all of which underpin antibiotic-associated gastrointestinal symptoms.

> the data from this trial provide compelling evidence that a 24-strain synbiotic supports the preservation and recovery of microbiome community structure and key microbiome functions while promoting gut barrier function following antibiotics... this trial suggests that this synbiotic may offer distinct advantages.

> administration of a multi-species synbiotic during and following broad-spectrum antibiotic exposure accelerated restoration of microbiome diversity, promoted recovery of key microbial metabolic outputs, and improved indices of gut barrier function, outcomes mechanistically linked to post-antibiotic recovery.

> This trial focused on mechanistic endpoints and did not include patient-reported GI symptom outcomes. While the observed improvements in microbiome composition, metabolite recovery, and gut barrier integrity provide strong biological plausibility for clinical benefit, future trials should incorporate validated symptom-based endpoints to directly link these mechanistic effects to clinical outcomes.

> In contrast to previous studies that question the utility and raised concerns about delays in microbiome recovery due to probiotic use, this trial suggests that this synbiotic may offer distinct advantages.

## Outcome Reporting

**Severity:** MODERATE

| Flag | Value |
|------|-------|
| Primary Outcome Type | surrogate |
| Surrogate Without Validation | Yes |
| Composite Not Disaggregated | No |

**Evidence:**
> Endpoints included fecal microbiome composition, fecal acetate and butyrate levels, urinary Urolithin A (UroA), serum p-cresol sulfate (pCS), gut barrier integrity, and safety.

> The study endpoints included microbiome compositional dynamics assessed by whole-genome shotgun sequencing, fecal SCFAs, urinary UroA production, serum pCS and TMAO, gut barrier integrity assessed by a lactulose-based test, and safety parameters (vital signs, clinical chemistry, and hematology).

> This trial focused on mechanistic endpoints and did not include patient-reported GI symptom outcomes.

> this proof-of-mechanism trial was not designed to define clinically meaningful thresholds

## Conflict of Interest

**Severity:** HIGH

| Flag | Value |
|------|-------|
| Funding Type | industry |
| Funding Disclosed In Abstract | No |
| Industry Author Affiliations | Yes |
| Coi Disclosed | Yes |

**Evidence:**
> B.A.N. and Z.K. are employees and shareholders of Seed Health, Inc. G.R. and R.J. serve as advisors to Seed Health, Inc... data analysis B.A.N.; writing—preparation of the original draft, B.A.N., Z.K. and R.J.

> The sponsor (Seed Health, Inc.) had no role in clinical trial operations, which were performed independently by KGK Science, Inc.

## Methodology

**Severity:** HIGH

| Flag | Value |
|------|-------|
| Inappropriate Comparator | No |
| Enrichment Design | No |
| Per Protocol Only | Yes |
| Premature Stopping | No |
| Short Follow Up | No |

**Evidence:**
> Eleven participants were lost to follow-up (Placebo: n = 4; DS-01: n = 7), leaving 21 participants who completed the trial and were included in the statistical analysis.

> Of the 86 individuals who responded to recruitment and underwent eligibility screening, 32 met the inclusion criteria... leaving 21 participants who completed the trial and were included in the statistical analysis.

> To induce a standardized dysbiosis challenge, all participants received broad-spectrum antibiotics during the first 7 days of the intervention period... All participants were also administered a 7-day oral antibiotic course consisting of ciprofloxacin (500 mg, twice daily) and metronidazole (500 mg, three times daily).

> Longitudinal changes in microbiome, metabolite, and gut permeability endpoints were analyzed using mixed modeling frameworks incorporating fixed effects for treatment group, time, and their interaction, with random intercepts for participants to account for repeated measures.

## Recommended Verification Steps

- Check ClinicalTrials.gov registration (NCT number) to verify pre-specified primary endpoints, sample size justification, and whether the low-baseline butyrate subgroup analysis was pre-registered.
- Request individual participant data or full statistical analysis plan to verify whether ITT or mITT analyses were conducted and suppressed.
- Obtain absolute values (means and SDs) for both arms at each timepoint for all primary endpoints to enable independent assessment of clinical significance.
- Verify whether multiplicity correction was pre-specified or applied in the statistical analysis plan.
- Examine the financial relationships between all authors and Seed Health, Inc. beyond what is disclosed, including consulting fees, grants, and equity valuations.
- Assess whether the asymmetric dropout pattern (7 vs. 4) is explained by differential adverse events or tolerability issues in the active arm, which could indicate safety signals.
- Evaluate whether the lactulose permeability test used has established validated thresholds for clinical meaningfulness in post-antibiotic populations.
- Seek independent replication of key findings (particularly the 13,008% UroA increase and 305% permeability improvement) in a trial without sponsor-employed data analysts.
- Review the ethics committee approval for administering broad-spectrum antibiotics to healthy volunteers solely for experimental dysbiosis induction.

---
*Generated by BiasBuster CLI*
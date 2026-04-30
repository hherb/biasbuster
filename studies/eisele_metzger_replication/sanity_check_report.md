# Phase 4 Sanity Check: Reproducing EM Published κ ≈ 0.22

**Method:** Cohen's κ between source labels `cochrane` and
`em_claude2_run1` (the first of the three Claude 2 passes
EM ran), loaded from `dataset/eisele_metzger_benchmark.db`.

**Three weightings reported:**
- **Unweighted** (classical Cohen's κ): exact-match agreement only.
- **Linear-weighted** (pre-registered primary, §6.1): credits partial
  agreement linearly with ordinal distance.
- **Quadratic-weighted**: credits more for small disagreements,
  penalises large ones more sharply. *EM's published numbers match
  this weighting.*

**95% CIs:** percentile bootstrap, 1000 resamples, seed=42.

## Headline result — does the loader reproduce EM's 0.22?

**Raw agreement (overall judgment):** 41.0% (EM published: 41% → exact match)

| Weighting | κ | 95% CI | vs EM 0.22 |
|---|---:|---|---|
| none | **0.035** | [-0.093, 0.175] |  |
| linear | **0.115** | [-0.020, 0.245] |  |
| quadratic | **0.221** | [0.056, 0.362] | ✅ matches |

**Verdict:** ✅ PASS — quadratic-weighted overall κ = 0.221 matches EM's published 0.22 within ±0.04. Raw agreement is also an exact match (41% vs 41% published). Loader integrity confirmed.

## Per-domain breakdown

EM published per-domain κ range: **0.10–0.31** (slight to fair).
Our quadratic-weighted reproduction landing inside this range across all 5 domains is the main loader-integrity check.

| Domain | unweighted | linear (primary) | quadratic (matches EM) |
|---|---:|---:|---:|
| d1 | 0.032 | 0.063 | 0.109 |
| d2 | 0.112 | 0.114 | 0.117 |
| d3 | 0.186 | 0.242 | 0.314 |
| d4 | 0.175 | 0.163 | 0.147 |
| d5 | 0.111 | 0.108 | 0.103 |

## Confusion matrix (overall judgment, n=100)

Rows = Cochrane reviewer judgment, Cols = EM Claude 2 run 1.

|  | claude:low | claude:some_concerns | claude:high |
|---|---:|---:|---:|
| **cochrane:low** | 18 | 18 | 0 |
| **cochrane:some_concerns** | 17 | 22 | 3 |
| **cochrane:high** | 4 | 17 | 1 |

## Methodological note for the manuscript

EM's paper reports "Cohen's κ = 0.22" without specifying weighting.
Our reproduction reveals their reported value matches **quadratic-
weighted** Cohen's κ exactly (overall 0.221 vs published 0.22;
per-domain range 0.103–0.314 vs published 0.10–0.31). We will note
this in the manuscript's Methods section.

Per pre-reg §6.1, our primary analysis uses **linear-weighted** κ,
which is more conservative on ordinal data (smaller credit for
partial agreement). All three weightings (unweighted, linear,
quadratic) will be reported in the manuscript's main results table
for transparency, with linear-weighted as the headline number to
preserve pre-registered methodology.
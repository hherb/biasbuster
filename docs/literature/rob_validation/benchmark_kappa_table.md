# RoB 2 Inter-Rater Reliability Benchmark Table

Compiled 2026-04-30 from primary sources held in this folder. All κ values are extracted directly from published abstracts/papers; verify against full text before citing.

## Headline result

> Eisele-Metzger et al. 2025 reported Claude 2 vs Cochrane RoB 2: overall κ = 0.22.
> Trained human reviewers vs each other on RoB 2 (Minozzi et al. 2020): overall κ = 0.16.
>
> The LLM finding lies *inside* the human-vs-human reliability band.

## Human-vs-human RoB 2 IRR

### Minozzi 2020 — baseline RoB 2 reliability (no extra training)
- Source: doi:10.1016/j.jclinepi.2020.06.015 (PMID 32562833)
- Setup: 4 raters with substantial systematic-review expertise, 70 outcomes from 70 RCTs, Fleiss' κ (multi-rater).

| RoB 2 domain | Fleiss' κ | 95% CI | Verbal |
|---|---:|---|---|
| **Overall judgment** | **0.16** | 0.08 – 0.24 | slight |
| D1: Randomization process | 0.45 | 0.37 – 0.53 | moderate |
| D2: Deviations (effect of *assignment*) | 0.04 | -0.06 – 0.14 | none |
| D2: Deviations (effect of *adherence*) | 0.21 | 0.11 – 0.31 | fair |
| D3: Missing outcome data | 0.22 | 0.14 – 0.30 | fair |
| D4: Measurement of outcome | (fair, range 0.22–0.30) | — | fair |
| D5: Selection of reported result | 0.30 | 0.22 – 0.38 | fair |

Mean time per outcome: 28 min (SD 13.4). Authors' own conclusion: *"RoB 2 is detailed and comprehensive but difficult and demanding, even for raters with substantial expertise."*

### Minozzi 2021 — RoB 2 reliability with implementation document (ID)
- Source: doi:10.1016/j.jclinepi.2021.09.021 (PMID 34537386)
- Setup: 4 raters × 80 results from 16 RCTs (cannabinoids/MS Cochrane review). Calibration on 5 studies → ID developed → measured before/after.

| Phase | Overall κ | Verbal |
|---|---:|---|
| Calibration (no ID) | **−0.15** | worse than chance |
| First 5 studies post-ID | 0.11 | slight |
| Remaining 11 studies post-ID | **0.42** | moderate |

Time to develop ID: ~40 hours. Per-study application time: 168 → 41 min after ID.

### Minozzi 2019 — ROBINS-I (non-randomized) for context
- Source: doi:10.1016/j.jclinepi.2019.04.001 (PMID 30981833)
- 5 raters, 31 studies. Overall κ = 0.06 (slight), domain κ = 0.04–0.18.

## LLM-vs-expert RoB studies

### Eisele-Metzger 2025 — Claude 2 vs Cochrane RoB 2
- Source: doi:10.1017/rsm.2025.12 (PMID 41626932)
- Setup: Claude 2 (off-the-shelf, prompt-engineered) vs Cochrane authors' published RoB 2 judgments. 100 RCTs.

| Metric | Value |
|---|---:|
| Raw agreement (overall) | 41% |
| Overall Cohen's κ | **0.22** |
| Domain-level κ range | 0.10 – 0.31 |

Conclusion (verbatim): *"Currently, Claude's RoB 2 judgements cannot replace human RoB assessment."*

### Lai 2024 — ChatGPT + Claude vs 3 experts, modified Cochrane RoB
- Source: doi:10.1001/jamanetworkopen.2024.12687 (PMID 38776081)
- **Important caveat:** uses CLARITY group's *modified* Cochrane RoB tool (binary low/high per domain), not the full RoB 2 with signaling questions. Easier task.
- Setup: 30 RCTs, two passes per LLM, 3 expert reviewers as criterion standard.

| Model | Mean correct rate | 95% CI | Inter-pass consistency |
|---|---:|---|---:|
| ChatGPT (LLM 1) | 84.5% | 81.5–87.3 | 84.0% |
| Claude (LLM 2) | 89.5% | 87.0–91.8 | 87.3% |

Inter-pass Cohen's κ > 0.80 on 7/8 domains (LLM 1) and 8/8 (LLM 2). Mean assessment time: 53–77 sec/study.

## Synthesis for the preprint

The takeaway is not "Claude 2 was good." It's that the Eisele-Metzger conclusion is over-reached because:

1. **Claude 2 (κ = 0.22) is in the same band as untrained human raters on the same tool (κ = 0.16, Minozzi 2020).** The headline κ comparison is being made against a noisy single-Cochrane-reviewer judgment, not against a stable human gold standard.
2. **With a structured implementation document, trained humans reach κ ≈ 0.42 (Minozzi 2021).** This is the effect of providing structure. The biasbuster training pipeline (curated dataset, severity boundaries unified between annotation and training prompts, verification chains) is the LLM analogue.
3. **When the rubric is simpler, LLMs already match well (Lai 2024).** The CLARITY-modified RoB version got LLMs to ~85–90% accuracy and inter-pass κ > 0.80. RoB 2's complexity is the problem, not LLMs.
4. **Claude 2 is two model generations behind anything we'd field today.** Eisele-Metzger's experimental setup was reasonable; the *generalization* in their conclusion is what's broken.

## Open data status

- Eisele-Metzger 2025 supplementary data: **available as XLS from authors** (the user has obtained this). Once converted to CSV/JSON, this enables direct re-running of their 100-RCT comparison with our fine-tuned model — the empirical refutation path.

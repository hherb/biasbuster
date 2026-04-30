# RoB 2 Validation Literature

Curated reference set assembled while preparing the BiasBuster preprint. Two purposes:

1. **Refute over-reached "LLMs cannot do RoB" claims** — primarily Eisele-Metzger et al. 2025 (Claude 2 vs Cochrane RoB 2).
2. **Establish the human-vs-human RoB 2 inter-rater reliability floor** — the κ values trained reviewers achieve against each other, which is the correct ceiling for any LLM-vs-human comparison.

## Files

| File | Citation | Role |
|---|---|---|
| `eisele_metzger_2025_claude2_rob2.pdf` | Eisele-Metzger A, et al. *Res Synth Methods.* 2025;16(3):491–508. doi:10.1017/rsm.2025.12. PMID 41626932. | Target paper. Claude 2 vs Cochrane RoB 2 on 100 RCTs. κ overall = 0.22. Concludes LLMs unfit for RoB. |
| `lai_2024_llm_rob_jamanetworkopen.pdf` | Lai H, et al. *JAMA Netw Open.* 2024;7(5):e2412687. doi:10.1001/jamanetworkopen.2024.12687. PMID 38776081. | Counter-evidence. ChatGPT and Claude vs 3 expert reviewers using CLARITY-modified Cochrane RoB on 30 RCTs. 84.5%–89.5% correct rates; κ > 0.80 on most domains. |
| `trevino_juarez_2024_chatgpt4_rob2.pdf` | Treviño-Juarez AS. *Med Sci Educ.* 2024. doi:10.1007/s40670-024-02034-8. PMID 38887420. | ChatGPT-4 + Cochrane RoB 2 — overview piece. |
| `nashwan_2023_llm_rob_editorial_cureus.pdf` | Nashwan AJ, Jaradat JH. *Cureus.* 2023. doi:10.7759/cureus.43023. PMID 37674957. | Editorial framing LLM use in systematic reviews. |
| `jorgensen_2016_rob1_evaluation.pdf` | Jørgensen L, et al. *Syst Rev.* 2016;5:80. doi:10.1186/s13643-016-0259-8. PMID 27160280. | Evaluation of original Cochrane RoB tool (RoB 1). Useful for historical context. |
| `kalaycioglu_2023_irr_nonrandomized.pdf` | Kalaycioglu I, et al. *Syst Rev.* 2023. doi:10.1186/s13643-023-02389-w. PMID 38057883. | IRR comparison of six RoB tools for non-randomized studies. |
| `sallam_2023_chatgpt_healthcare.pdf` | Sallam M. *Healthcare (Basel).* 2023;11(6):887. doi:10.3390/healthcare11060887. PMID 36981544. | Broader systematic review of ChatGPT in healthcare. |

## Abstract-only (no open-access PDF located)

| Citation | Why we keep it | DOI |
|---|---|---|
| **Minozzi S, Cinquini M, Gianola S, Gonzalez-Lorenzo M, Banzi R.** "The revised Cochrane risk of bias tool for randomized trials (RoB 2) showed low interrater reliability and challenges in its application." *J Clin Epidemiol.* 2020;126:37–44. PMID 32562833. | **Foundational RoB 2 IRR paper.** 4 raters × 70 RCTs. Overall κ = 0.16 (slight). Establishes the human-vs-human RoB 2 reliability floor. | doi:10.1016/j.jclinepi.2020.06.015 |
| **Minozzi S, Dwan K, Borrelli F, Filippini G.** "Reliability of the revised Cochrane risk-of-bias tool for randomised trials (RoB2) improved with the use of implementation instruction." *J Clin Epidemiol.* 2021;141:99–105. PMID 34537386. | Follow-up showing IRR improves with structured guidance. After implementation document: overall κ rises from −0.15 → 0.42. | doi:10.1016/j.jclinepi.2021.09.021 |
| **Minozzi S, Cinquini M, Gianola S, Castellini G, Gerardi C, Banzi R.** "Risk of bias in nonrandomized studies of interventions showed low inter-rater reliability and challenges in its application." *J Clin Epidemiol.* 2019;112:28–35. PMID 30981833. | Same group, ROBINS-I (non-randomized). Overall κ = 0.06. Confirms the IRR pattern is structural, not RoB 2-specific. | doi:10.1016/j.jclinepi.2019.04.001 |

The Minozzi papers are paywalled (Elsevier). Abstracts captured in `metadata.json`.

## Triage notes

`triage_pubmed_search.txt` — first pass over 60+ PubMed hits across 3 search strategies. Most were systematic reviews that *used* RoB 2 but did not measure its IRR. Filtered down to the papers above plus the Minozzi trio.

## Reproducing the searches

```
PubMed query 1: ("RoB 2" OR "RoB-2" OR "ROB2") AND (reliability OR agreement OR "inter-rater" OR interrater OR kappa)
PubMed query 2: "risk of bias" AND ("inter-rater reliability" OR "interrater reliability" OR "inter-rater agreement") AND (Cochrane OR randomized) AND tool
PubMed query 3: ("large language model" OR "LLM" OR "GPT" OR "Claude" OR "ChatGPT") AND "risk of bias"
```

Source: PubMed E-utilities, 2026-04-30.

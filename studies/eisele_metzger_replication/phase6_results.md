# Phase 6 Cross-Model Comparison

**Generated:** by `studies/eisele_metzger_replication/compute_phase6_kappa.py`
**Output companions:** `phase6_results.csv` (raw rows) and `phase6_forest_data.csv` (forest-plot input).

Coverage of the table fills in as Phase 5 evaluation runs complete. Empty model rows = data not yet in the DB.

## 1. Single-pass κ vs Cochrane (overall judgment)

| Source | n | raw agr | κ_unw | κ_lin (95% CI) | κ_quad |
|---|---:|---:|---:|---|---:|
| gpt_oss_20b_abstract_pass1 | 91 | 0.440 | 0.074 | 0.050 [-0.053, 0.142] | 0.010 |
| gpt_oss_20b_abstract_pass2 | 91 | 0.374 | -0.025 | -0.008 [-0.101, 0.083] | 0.020 |
| gpt_oss_20b_abstract_pass3 | 91 | 0.440 | 0.070 | 0.024 [-0.063, 0.125] | -0.055 |
| gpt_oss_20b_fulltext_pass1 | 90 | 0.489 | 0.206 | 0.228 [0.111, 0.370] | 0.257 |
| gpt_oss_20b_fulltext_pass2 | 90 | 0.433 | 0.122 | 0.148 [0.027, 0.288] | 0.182 |
| gpt_oss_20b_fulltext_pass3 | 90 | 0.444 | 0.149 | 0.168 [0.043, 0.296] | 0.192 |
| qwen3_6_35b_abstract_pass1 | 22 | 0.409 | 0.173 | 0.227 [0.000, 0.431] | 0.316 |
| sonnet_4_6_abstract_pass1 | 91 | 0.440 | 0.058 | 0.093 [0.003, 0.204] | 0.156 |
| sonnet_4_6_abstract_pass2 | 91 | 0.440 | 0.058 | 0.057 [-0.022, 0.145] | 0.057 |
| sonnet_4_6_abstract_pass3 | 91 | 0.462 | 0.098 | 0.119 [0.016, 0.220] | 0.157 |
| sonnet_4_6_fulltext_pass1 | 91 | 0.451 | 0.090 | 0.114 [-0.010, 0.230] | 0.154 |
| sonnet_4_6_fulltext_pass2 | 91 | 0.462 | 0.118 | 0.175 [0.045, 0.291] | 0.264 |
| sonnet_4_6_fulltext_pass3 | 91 | 0.462 | 0.124 | 0.157 [0.024, 0.277] | 0.207 |

*Reference:* EM Claude 2 published κ_quad ≈ 0.22.

## 2. Run-to-run κ across the 3 passes (LLM-internal noise)

| Model × protocol | n_pairs | mean κ_unw | mean κ_lin | mean κ_quad |
|---|---:|---:|---:|---:|
| gpt-oss 20B × abstract | 3 | 0.311 | 0.311 | 0.311 |
| gpt-oss 20B × fulltext | 3 | 0.465 | 0.457 | 0.441 |
| Claude Sonnet 4.6 × abstract | 3 | 0.601 | 0.601 | 0.601 |
| Claude Sonnet 4.6 × fulltext | 3 | 0.756 | 0.760 | 0.768 |

*References:* Minozzi 2020 trained-human Fleiss κ = 0.16; Minozzi 2021 with implementation document = 0.42.

## 3. Ensemble-of-3 majority vote vs Cochrane (overall judgment)

Per-domain majority vote across the three passes, then worst-wins synthesis.

| Source | n | raw agr | κ_unw | κ_lin (95% CI) | κ_quad |
|---|---:|---:|---:|---|---:|
| gpt_oss_20b_abstract_ensemble | 91 | 0.440 | 0.064 | 0.040 [-0.047, 0.147] | -0.001 |
| gpt_oss_20b_fulltext_ensemble | 90 | 0.456 | 0.157 | 0.162 [0.035, 0.289] | 0.169 |
| sonnet_4_6_abstract_ensemble | 91 | 0.451 | 0.076 | 0.093 [0.000, 0.197] | 0.123 |
| sonnet_4_6_fulltext_ensemble | 91 | 0.462 | 0.115 | 0.151 [0.032, 0.272] | 0.208 |

## 4. Per-domain κ_quad across all sources

| Source | d1 | d2 | d3 | d4 | d5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| gpt_oss_20b_abstract_ensemble | 0.000 | 0.063 | 0.134 | 0.136 | 0.036 | -0.001 |
| gpt_oss_20b_abstract_pass1 | 0.018 | 0.028 | 0.172 | 0.134 | 0.005 | 0.010 |
| gpt_oss_20b_abstract_pass2 | -0.010 | 0.074 | 0.124 | 0.158 | 0.024 | 0.020 |
| gpt_oss_20b_abstract_pass3 | 0.000 | 0.019 | 0.109 | 0.128 | 0.074 | -0.055 |
| gpt_oss_20b_fulltext_ensemble | 0.083 | 0.123 | 0.233 | 0.335 | 0.188 | 0.169 |
| gpt_oss_20b_fulltext_pass1 | 0.129 | 0.176 | 0.237 | 0.385 | 0.175 | 0.257 |
| gpt_oss_20b_fulltext_pass2 | 0.125 | 0.106 | 0.222 | 0.282 | 0.227 | 0.182 |
| gpt_oss_20b_fulltext_pass3 | 0.148 | 0.087 | 0.257 | 0.274 | 0.195 | 0.192 |
| qwen3_6_35b_abstract_pass1 | 0.000 | 0.156 | 0.388 | 0.072 | 0.210 | 0.316 |
| sonnet_4_6_abstract_ensemble | 0.032 | 0.152 | 0.225 | 0.213 | 0.046 | 0.123 |
| sonnet_4_6_abstract_pass1 | 0.032 | 0.152 | 0.232 | 0.176 | 0.034 | 0.156 |
| sonnet_4_6_abstract_pass2 | 0.043 | 0.165 | 0.218 | 0.198 | 0.059 | 0.057 |
| sonnet_4_6_abstract_pass3 | 0.022 | 0.152 | 0.232 | 0.225 | 0.046 | 0.157 |
| sonnet_4_6_fulltext_ensemble | 0.196 | 0.110 | 0.364 | 0.273 | 0.105 | 0.208 |
| sonnet_4_6_fulltext_pass1 | 0.208 | 0.110 | 0.363 | 0.265 | 0.092 | 0.154 |
| sonnet_4_6_fulltext_pass2 | 0.196 | 0.110 | 0.346 | 0.374 | 0.105 | 0.264 |
| sonnet_4_6_fulltext_pass3 | 0.196 | 0.097 | 0.339 | 0.256 | 0.092 | 0.207 |

## 5. Forest-plot data (for the manuscript figure)

| Series | κ_quad | κ_lin (95% CI) | n |
|---|---:|---|---:|
| gpt-oss 20B (abstract, pass 1) | 0.010 | 0.050 [-0.053, 0.142] | 91 |
| gpt-oss 20B (abstract, pass 2) | 0.020 | -0.008 [-0.101, 0.083] | 91 |
| gpt-oss 20B (abstract, pass 3) | -0.055 | 0.024 [-0.063, 0.125] | 91 |
| gpt-oss 20B (fulltext, pass 1) | 0.257 | 0.228 [0.111, 0.370] | 90 |
| gpt-oss 20B (fulltext, pass 2) | 0.182 | 0.148 [0.027, 0.288] | 90 |
| gpt-oss 20B (fulltext, pass 3) | 0.192 | 0.168 [0.043, 0.296] | 90 |
| Qwen 3.6 35B-A3B (abstract, pass 1) | 0.316 | 0.227 [0.000, 0.431] | 22 |
| Claude Sonnet 4.6 (abstract, pass 1) | 0.156 | 0.093 [0.003, 0.204] | 91 |
| Claude Sonnet 4.6 (abstract, pass 2) | 0.057 | 0.057 [-0.022, 0.145] | 91 |
| Claude Sonnet 4.6 (abstract, pass 3) | 0.157 | 0.119 [0.016, 0.220] | 91 |
| Claude Sonnet 4.6 (fulltext, pass 1) | 0.154 | 0.114 [-0.010, 0.230] | 91 |
| Claude Sonnet 4.6 (fulltext, pass 2) | 0.264 | 0.175 [0.045, 0.291] | 91 |
| Claude Sonnet 4.6 (fulltext, pass 3) | 0.207 | 0.157 [0.024, 0.277] | 91 |
| gpt-oss 20B (abstract, ensemble) | -0.001 | 0.040 [-0.047, 0.147] | 91 |
| gpt-oss 20B (fulltext, ensemble) | 0.169 | 0.162 [0.035, 0.289] | 90 |
| Claude Sonnet 4.6 (abstract, ensemble) | 0.123 | 0.093 [0.000, 0.197] | 91 |
| Claude Sonnet 4.6 (fulltext, ensemble) | 0.208 | 0.151 [0.032, 0.272] | 91 |
| EM Claude 2 (published, single pass) | 0.220 | — | — |
| Minozzi 2020 — trained humans, no ID | 0.160 | — | — |
| Minozzi 2021 — trained humans, with ID | 0.420 | — | — |

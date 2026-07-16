# Phase 6 Cross-Model Comparison

**Generated:** by `studies/eisele_metzger_replication/compute_phase6_kappa.py`
**Output companions:** `phase6_results.csv` (raw rows) and `phase6_forest_data.csv` (forest-plot input).

Coverage of the table fills in as Phase 5 evaluation runs complete. Empty model rows = data not yet in the DB.

> **STRICT MODE (`--exclude-fallback`):** algorithm-derived (`raw_label='FALLBACK'`) judgements are excluded — these numbers reproduce the pre-registered *model-emitted* primary metric.

## 1. Single-pass κ vs Cochrane (overall judgment)

| Source | n | raw agr | κ_unw | κ_lin (95% CI) | κ_quad |
|---|---:|---:|---:|---|---:|
| gpt_oss_20b_abstract_pass1 | 91 | 0.440 | 0.074 | 0.050 [-0.053, 0.142] | 0.010 |
| gpt_oss_20b_abstract_pass2 | 91 | 0.374 | -0.025 | -0.008 [-0.101, 0.083] | 0.020 |
| gpt_oss_20b_abstract_pass3 | 91 | 0.440 | 0.070 | 0.024 [-0.063, 0.125] | -0.055 |
| gpt_oss_20b_fulltext_pass1 | 90 | 0.489 | 0.206 | 0.228 [0.111, 0.370] | 0.257 |
| gpt_oss_20b_fulltext_pass2 | 90 | 0.433 | 0.122 | 0.148 [0.027, 0.288] | 0.182 |
| gpt_oss_20b_fulltext_pass3 | 90 | 0.444 | 0.149 | 0.168 [0.043, 0.296] | 0.192 |
| gemma4_26b_abstract_pass1 | 91 | 0.440 | 0.110 | 0.113 [-0.006, 0.229] | 0.117 |
| gemma4_26b_abstract_pass2 | 91 | 0.429 | 0.068 | 0.083 [-0.031, 0.189] | 0.106 |
| gemma4_26b_abstract_pass3 | 91 | 0.429 | 0.065 | 0.058 [-0.052, 0.177] | 0.048 |
| gemma4_26b_fulltext_pass1 | 91 | 0.451 | 0.169 | 0.190 [0.066, 0.326] | 0.216 |
| gemma4_26b_fulltext_pass2 | 91 | 0.462 | 0.182 | 0.195 [0.055, 0.334] | 0.212 |
| gemma4_26b_fulltext_pass3 | 91 | 0.451 | 0.172 | 0.210 [0.097, 0.337] | 0.254 |
| qwen3_6_35b_abstract_pass1 | 91 | 0.407 | 0.032 | 0.050 [-0.051, 0.172] | 0.079 |
| qwen3_6_35b_abstract_pass2 | 91 | 0.407 | 0.022 | 0.041 [-0.054, 0.140] | 0.072 |
| qwen3_6_35b_abstract_pass3 | 91 | 0.396 | 0.006 | 0.063 [-0.047, 0.172] | 0.155 |
| qwen3_6_35b_fulltext_pass1 | 91 | 0.516 | 0.235 | 0.235 [0.090, 0.380] | 0.234 |
| qwen3_6_35b_fulltext_pass2 | 91 | 0.473 | 0.164 | 0.202 [0.062, 0.360] | 0.253 |
| qwen3_6_35b_fulltext_pass3 | 91 | 0.462 | 0.158 | 0.182 [0.038, 0.323] | 0.214 |
| sonnet_4_6_abstract_pass1 | 89 | 0.427 | 0.037 | 0.069 [-0.013, 0.171] | 0.126 |
| sonnet_4_6_abstract_pass2 | 89 | 0.427 | 0.056 | 0.056 [-0.020, 0.146] | 0.058 |
| sonnet_4_6_abstract_pass3 | 90 | 0.456 | 0.097 | 0.119 [0.017, 0.224] | 0.157 |
| sonnet_4_6_fulltext_pass1 | 75 | 0.413 | 0.049 | 0.060 [-0.055, 0.168] | 0.079 |
| sonnet_4_6_fulltext_pass2 | 81 | 0.469 | 0.122 | 0.166 [0.042, 0.285] | 0.236 |
| sonnet_4_6_fulltext_pass3 | 79 | 0.456 | 0.133 | 0.160 [0.045, 0.289] | 0.203 |

*Reference:* EM Claude 2 published κ_quad ≈ 0.22.

## 2. Run-to-run κ across the 3 passes (LLM-internal noise)

| Model × protocol | n_pairs | mean κ_unw | mean κ_lin | mean κ_quad |
|---|---:|---:|---:|---:|
| gpt-oss 20B × abstract | 3 | 0.311 | 0.311 | 0.311 |
| gpt-oss 20B × fulltext | 3 | 0.465 | 0.457 | 0.441 |
| Gemma 4 26B-A4B × abstract | 3 | 0.623 | 0.623 | 0.623 |
| Gemma 4 26B-A4B × fulltext | 3 | 0.753 | 0.769 | 0.797 |
| Qwen 3.6 35B-A3B × abstract | 3 | 0.361 | 0.364 | 0.370 |
| Qwen 3.6 35B-A3B × fulltext | 3 | 0.593 | 0.620 | 0.665 |
| Claude Sonnet 4.6 × abstract | 3 | 0.594 | 0.594 | 0.594 |
| Claude Sonnet 4.6 × fulltext | 3 | 0.740 | 0.745 | 0.754 |

*References:* Minozzi 2020 trained-human Fleiss κ = 0.16; Minozzi 2021 with implementation document = 0.42.

## 3. Ensemble-of-3 majority vote vs Cochrane (overall judgment)

Each signalling domain (d1–d5) is a strict majority vote across the three passes; `overall` is then the worst of those five ensemble domains (RoB 2 worst-wins), not a direct majority vote of the passes' overall labels.

| Source | n | raw agr | κ_unw | κ_lin (95% CI) | κ_quad |
|---|---:|---:|---:|---|---:|
| gpt_oss_20b_abstract_ensemble | 91 | 0.429 | 0.042 | 0.014 [-0.051, 0.108] | -0.037 |
| gpt_oss_20b_fulltext_ensemble | 90 | 0.456 | 0.154 | 0.170 [0.046, 0.295] | 0.191 |
| gemma4_26b_abstract_ensemble | 91 | 0.462 | 0.122 | 0.116 [0.005, 0.231] | 0.106 |
| gemma4_26b_fulltext_ensemble | 91 | 0.451 | 0.167 | 0.198 [0.069, 0.333] | 0.236 |
| qwen3_6_35b_abstract_ensemble | 91 | 0.418 | 0.034 | 0.058 [-0.035, 0.162] | 0.098 |
| qwen3_6_35b_fulltext_ensemble | 91 | 0.495 | 0.191 | 0.212 [0.068, 0.379] | 0.241 |
| sonnet_4_6_abstract_ensemble | 91 | 0.451 | 0.076 | 0.093 [-0.000, 0.191] | 0.123 |
| sonnet_4_6_fulltext_ensemble | 91 | 0.462 | 0.115 | 0.151 [0.032, 0.285] | 0.208 |

## 4. Per-domain κ_quad across all sources

| Source | d1 | d2 | d3 | d4 | d5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| gemma4_26b_abstract_ensemble | 0.000 | 0.088 | 0.118 | 0.247 | 0.152 | 0.106 |
| gemma4_26b_abstract_pass1 | 0.000 | 0.043 | 0.115 | 0.207 | 0.181 | 0.117 |
| gemma4_26b_abstract_pass2 | 0.000 | 0.078 | 0.133 | 0.234 | 0.138 | 0.106 |
| gemma4_26b_abstract_pass3 | 0.000 | 0.061 | 0.121 | 0.220 | 0.202 | 0.048 |
| gemma4_26b_fulltext_ensemble | 0.198 | 0.258 | 0.208 | 0.299 | 0.243 | 0.236 |
| gemma4_26b_fulltext_pass1 | 0.209 | 0.225 | 0.099 | 0.294 | 0.262 | 0.216 |
| gemma4_26b_fulltext_pass2 | 0.209 | 0.197 | 0.157 | 0.319 | 0.237 | 0.212 |
| gemma4_26b_fulltext_pass3 | 0.169 | 0.198 | 0.254 | 0.298 | 0.290 | 0.254 |
| gpt_oss_20b_abstract_ensemble | 0.000 | 0.063 | 0.134 | 0.136 | 0.036 | -0.037 |
| gpt_oss_20b_abstract_pass1 | 0.018 | 0.028 | 0.172 | 0.134 | 0.005 | 0.010 |
| gpt_oss_20b_abstract_pass2 | -0.010 | 0.074 | 0.124 | 0.158 | 0.024 | 0.020 |
| gpt_oss_20b_abstract_pass3 | 0.000 | 0.019 | 0.109 | 0.128 | 0.074 | -0.055 |
| gpt_oss_20b_fulltext_ensemble | 0.083 | 0.123 | 0.233 | 0.335 | 0.188 | 0.191 |
| gpt_oss_20b_fulltext_pass1 | 0.129 | 0.176 | 0.237 | 0.385 | 0.175 | 0.257 |
| gpt_oss_20b_fulltext_pass2 | 0.125 | 0.106 | 0.222 | 0.282 | 0.227 | 0.182 |
| gpt_oss_20b_fulltext_pass3 | 0.148 | 0.087 | 0.257 | 0.274 | 0.195 | 0.192 |
| qwen3_6_35b_abstract_ensemble | 0.000 | 0.132 | 0.216 | 0.179 | 0.171 | 0.098 |
| qwen3_6_35b_abstract_pass1 | 0.022 | 0.197 | 0.253 | 0.200 | 0.114 | 0.079 |
| qwen3_6_35b_abstract_pass2 | 0.022 | 0.119 | 0.195 | 0.178 | 0.144 | 0.072 |
| qwen3_6_35b_abstract_pass3 | 0.011 | 0.153 | 0.187 | 0.326 | 0.151 | 0.155 |
| qwen3_6_35b_fulltext_ensemble | 0.180 | 0.149 | 0.248 | 0.467 | 0.195 | 0.241 |
| qwen3_6_35b_fulltext_pass1 | 0.123 | 0.159 | 0.260 | 0.381 | 0.255 | 0.234 |
| qwen3_6_35b_fulltext_pass2 | 0.169 | 0.134 | 0.306 | 0.415 | 0.208 | 0.253 |
| qwen3_6_35b_fulltext_pass3 | 0.258 | 0.129 | 0.239 | 0.391 | 0.111 | 0.214 |
| sonnet_4_6_abstract_ensemble | 0.032 | 0.152 | 0.225 | 0.213 | 0.046 | 0.123 |
| sonnet_4_6_abstract_pass1 | 0.032 | 0.157 | 0.237 | 0.176 | 0.034 | 0.126 |
| sonnet_4_6_abstract_pass2 | 0.043 | 0.170 | 0.222 | 0.198 | 0.059 | 0.058 |
| sonnet_4_6_abstract_pass3 | 0.021 | 0.152 | 0.232 | 0.225 | 0.046 | 0.157 |
| sonnet_4_6_fulltext_ensemble | 0.196 | 0.110 | 0.364 | 0.273 | 0.105 | 0.208 |
| sonnet_4_6_fulltext_pass1 | 0.208 | 0.092 | 0.331 | 0.305 | 0.092 | 0.079 |
| sonnet_4_6_fulltext_pass2 | 0.196 | 0.115 | 0.306 | 0.339 | 0.105 | 0.236 |
| sonnet_4_6_fulltext_pass3 | 0.194 | 0.121 | 0.348 | 0.285 | 0.092 | 0.203 |

## 5. Forest-plot data (for the manuscript figure)

| Series | κ_quad | κ_lin (95% CI) | n |
|---|---:|---|---:|
| gpt-oss 20B (abstract, pass 1) | 0.010 | 0.050 [-0.053, 0.142] | 91 |
| gpt-oss 20B (abstract, pass 2) | 0.020 | -0.008 [-0.101, 0.083] | 91 |
| gpt-oss 20B (abstract, pass 3) | -0.055 | 0.024 [-0.063, 0.125] | 91 |
| gpt-oss 20B (fulltext, pass 1) | 0.257 | 0.228 [0.111, 0.370] | 90 |
| gpt-oss 20B (fulltext, pass 2) | 0.182 | 0.148 [0.027, 0.288] | 90 |
| gpt-oss 20B (fulltext, pass 3) | 0.192 | 0.168 [0.043, 0.296] | 90 |
| Gemma 4 26B-A4B (abstract, pass 1) | 0.117 | 0.113 [-0.006, 0.229] | 91 |
| Gemma 4 26B-A4B (abstract, pass 2) | 0.106 | 0.083 [-0.031, 0.189] | 91 |
| Gemma 4 26B-A4B (abstract, pass 3) | 0.048 | 0.058 [-0.052, 0.177] | 91 |
| Gemma 4 26B-A4B (fulltext, pass 1) | 0.216 | 0.190 [0.066, 0.326] | 91 |
| Gemma 4 26B-A4B (fulltext, pass 2) | 0.212 | 0.195 [0.055, 0.334] | 91 |
| Gemma 4 26B-A4B (fulltext, pass 3) | 0.254 | 0.210 [0.097, 0.337] | 91 |
| Qwen 3.6 35B-A3B (abstract, pass 1) | 0.079 | 0.050 [-0.051, 0.172] | 91 |
| Qwen 3.6 35B-A3B (abstract, pass 2) | 0.072 | 0.041 [-0.054, 0.140] | 91 |
| Qwen 3.6 35B-A3B (abstract, pass 3) | 0.155 | 0.063 [-0.047, 0.172] | 91 |
| Qwen 3.6 35B-A3B (fulltext, pass 1) | 0.234 | 0.235 [0.090, 0.380] | 91 |
| Qwen 3.6 35B-A3B (fulltext, pass 2) | 0.253 | 0.202 [0.062, 0.360] | 91 |
| Qwen 3.6 35B-A3B (fulltext, pass 3) | 0.214 | 0.182 [0.038, 0.323] | 91 |
| Claude Sonnet 4.6 (abstract, pass 1) | 0.126 | 0.069 [-0.013, 0.171] | 89 |
| Claude Sonnet 4.6 (abstract, pass 2) | 0.058 | 0.056 [-0.020, 0.146] | 89 |
| Claude Sonnet 4.6 (abstract, pass 3) | 0.157 | 0.119 [0.017, 0.224] | 90 |
| Claude Sonnet 4.6 (fulltext, pass 1) | 0.079 | 0.060 [-0.055, 0.168] | 75 |
| Claude Sonnet 4.6 (fulltext, pass 2) | 0.236 | 0.166 [0.042, 0.285] | 81 |
| Claude Sonnet 4.6 (fulltext, pass 3) | 0.203 | 0.160 [0.045, 0.289] | 79 |
| gpt-oss 20B (abstract, ensemble) | -0.037 | 0.014 [-0.051, 0.108] | 91 |
| gpt-oss 20B (fulltext, ensemble) | 0.191 | 0.170 [0.046, 0.295] | 90 |
| Gemma 4 26B-A4B (abstract, ensemble) | 0.106 | 0.116 [0.005, 0.231] | 91 |
| Gemma 4 26B-A4B (fulltext, ensemble) | 0.236 | 0.198 [0.069, 0.333] | 91 |
| Qwen 3.6 35B-A3B (abstract, ensemble) | 0.098 | 0.058 [-0.035, 0.162] | 91 |
| Qwen 3.6 35B-A3B (fulltext, ensemble) | 0.241 | 0.212 [0.068, 0.379] | 91 |
| Claude Sonnet 4.6 (abstract, ensemble) | 0.123 | 0.093 [-0.000, 0.191] | 91 |
| Claude Sonnet 4.6 (fulltext, ensemble) | 0.208 | 0.151 [0.032, 0.285] | 91 |
| EM Claude 2 (published, single pass) | 0.220 | — | — |
| Minozzi 2020 — trained humans, no ID | 0.160 | — | — |
| Minozzi 2021 — trained humans, with ID | 0.420 | — | — |

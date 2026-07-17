# Phase 6 Cross-Model Comparison

**Generated:** by `studies/eisele_metzger_replication/compute_phase6_kappa.py`
**Output companions:** `phase6_results.csv` (raw rows) and `phase6_forest_data.csv` (forest-plot input).

Coverage of the table fills in as Phase 5 evaluation runs complete. Empty model rows = data not yet in the DB.

> **INCLUSIVE MODE (default):** algorithm-derived (`raw_label='FALLBACK'`) judgements are included. Re-run with `--exclude-fallback` for the pre-registered model-emitted primary metric.

## 1. Single-pass κ vs Cochrane (overall judgment)

| Source | n | raw agr | κ_unw | κ_lin (95% CI) | κ_quad |
|---|---:|---:|---:|---|---:|
| gpt_oss_20b_abstract_pass1 | 78 | 0.436 | 0.067 | 0.037 [-0.062, 0.144] | -0.011 |
| gpt_oss_20b_abstract_pass2 | 78 | 0.372 | -0.020 | 0.005 [-0.097, 0.110] | 0.042 |
| gpt_oss_20b_abstract_pass3 | 78 | 0.462 | 0.106 | 0.046 [-0.047, 0.159] | -0.053 |
| gpt_oss_20b_fulltext_pass1 | 78 | 0.487 | 0.212 | 0.260 [0.122, 0.402] | 0.321 |
| gpt_oss_20b_fulltext_pass2 | 78 | 0.436 | 0.142 | 0.185 [0.051, 0.322] | 0.235 |
| gpt_oss_20b_fulltext_pass3 | 78 | 0.423 | 0.132 | 0.154 [0.031, 0.289] | 0.179 |
| gemma4_26b_abstract_pass1 | 78 | 0.423 | 0.084 | 0.110 [-0.013, 0.236] | 0.147 |
| gemma4_26b_abstract_pass2 | 78 | 0.423 | 0.059 | 0.089 [-0.024, 0.208] | 0.134 |
| gemma4_26b_abstract_pass3 | 78 | 0.436 | 0.080 | 0.089 [-0.029, 0.207] | 0.102 |
| gemma4_26b_fulltext_pass1 | 78 | 0.436 | 0.153 | 0.160 [0.040, 0.295] | 0.169 |
| gemma4_26b_fulltext_pass2 | 78 | 0.462 | 0.182 | 0.204 [0.059, 0.346] | 0.230 |
| gemma4_26b_fulltext_pass3 | 78 | 0.436 | 0.150 | 0.200 [0.073, 0.340] | 0.259 |
| qwen3_6_35b_abstract_pass1 | 78 | 0.410 | 0.047 | 0.071 [-0.036, 0.198] | 0.108 |
| qwen3_6_35b_abstract_pass2 | 78 | 0.410 | 0.029 | 0.068 [-0.038, 0.183] | 0.128 |
| qwen3_6_35b_abstract_pass3 | 78 | 0.397 | 0.000 | 0.070 [-0.048, 0.196] | 0.185 |
| qwen3_6_35b_fulltext_pass1 | 78 | 0.500 | 0.208 | 0.213 [0.060, 0.364] | 0.219 |
| qwen3_6_35b_fulltext_pass2 | 78 | 0.487 | 0.195 | 0.229 [0.082, 0.368] | 0.273 |
| qwen3_6_35b_fulltext_pass3 | 78 | 0.449 | 0.148 | 0.173 [0.034, 0.321] | 0.203 |
| sonnet_4_6_abstract_pass1 | 78 | 0.462 | 0.089 | 0.124 [0.014, 0.247] | 0.187 |
| sonnet_4_6_abstract_pass2 | 78 | 0.462 | 0.089 | 0.082 [-0.010, 0.191] | 0.071 |
| sonnet_4_6_abstract_pass3 | 78 | 0.487 | 0.136 | 0.155 [0.047, 0.288] | 0.189 |
| sonnet_4_6_fulltext_pass1 | 78 | 0.462 | 0.111 | 0.140 [0.001, 0.254] | 0.186 |
| sonnet_4_6_fulltext_pass2 | 78 | 0.474 | 0.145 | 0.210 [0.067, 0.349] | 0.309 |
| sonnet_4_6_fulltext_pass3 | 78 | 0.474 | 0.152 | 0.191 [0.033, 0.338] | 0.246 |

*Reference:* EM Claude 2 published κ_quad ≈ 0.22.

## 2. Run-to-run κ across the 3 passes (LLM-internal noise)

| Model × protocol | n_pairs | mean κ_unw | mean κ_lin | mean κ_quad |
|---|---:|---:|---:|---:|
| gpt-oss 20B × abstract | 3 | 0.331 | 0.331 | 0.331 |
| gpt-oss 20B × fulltext | 3 | 0.474 | 0.466 | 0.452 |
| Gemma 4 26B-A4B × abstract | 3 | 0.643 | 0.643 | 0.643 |
| Gemma 4 26B-A4B × fulltext | 3 | 0.760 | 0.778 | 0.806 |
| Qwen 3.6 35B-A3B × abstract | 3 | 0.418 | 0.422 | 0.428 |
| Qwen 3.6 35B-A3B × fulltext | 3 | 0.613 | 0.642 | 0.688 |
| Claude Sonnet 4.6 × abstract | 3 | 0.526 | 0.526 | 0.526 |
| Claude Sonnet 4.6 × fulltext | 3 | 0.749 | 0.754 | 0.764 |

*References:* Minozzi 2020 trained-human Fleiss κ = 0.16; Minozzi 2021 with implementation document = 0.42.

## 3. Ensemble-of-3 majority vote vs Cochrane (overall judgment)

Each signalling domain (d1–d5) is a strict majority vote across the three passes; `overall` is then the worst of those five ensemble domains (RoB 2 worst-wins), not a direct majority vote of the passes' overall labels.

| Source | n | raw agr | κ_unw | κ_lin (95% CI) | κ_quad |
|---|---:|---:|---:|---|---:|
| gpt_oss_20b_abstract_ensemble | 78 | 0.436 | 0.054 | 0.023 [-0.052, 0.126] | -0.031 |
| gpt_oss_20b_fulltext_ensemble | 78 | 0.449 | 0.157 | 0.186 [0.067, 0.313] | 0.223 |
| gemma4_26b_abstract_ensemble | 78 | 0.462 | 0.122 | 0.127 [0.008, 0.255] | 0.134 |
| gemma4_26b_fulltext_ensemble | 78 | 0.436 | 0.146 | 0.176 [0.045, 0.315] | 0.212 |
| qwen3_6_35b_abstract_ensemble | 78 | 0.423 | 0.046 | 0.077 [-0.019, 0.178] | 0.126 |
| qwen3_6_35b_fulltext_ensemble | 78 | 0.500 | 0.207 | 0.227 [0.072, 0.373] | 0.253 |
| sonnet_4_6_abstract_ensemble | 78 | 0.474 | 0.110 | 0.124 [0.022, 0.241] | 0.148 |
| sonnet_4_6_fulltext_ensemble | 78 | 0.474 | 0.141 | 0.182 [0.041, 0.306] | 0.246 |

## 4. Per-domain κ_quad across all sources

| Source | d1 | d2 | d3 | d4 | d5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| gemma4_26b_abstract_ensemble | 0.000 | 0.090 | 0.133 | 0.243 | 0.144 | 0.134 |
| gemma4_26b_abstract_pass1 | 0.000 | 0.052 | 0.129 | 0.208 | 0.178 | 0.147 |
| gemma4_26b_abstract_pass2 | 0.000 | 0.078 | 0.152 | 0.227 | 0.127 | 0.134 |
| gemma4_26b_abstract_pass3 | 0.000 | 0.056 | 0.127 | 0.249 | 0.202 | 0.102 |
| gemma4_26b_fulltext_ensemble | 0.172 | 0.211 | 0.231 | 0.294 | 0.220 | 0.212 |
| gemma4_26b_fulltext_pass1 | 0.172 | 0.151 | 0.099 | 0.288 | 0.257 | 0.169 |
| gemma4_26b_fulltext_pass2 | 0.184 | 0.187 | 0.174 | 0.317 | 0.214 | 0.230 |
| gemma4_26b_fulltext_pass3 | 0.184 | 0.162 | 0.265 | 0.301 | 0.241 | 0.259 |
| gpt_oss_20b_abstract_ensemble | 0.000 | 0.063 | 0.143 | 0.163 | 0.044 | -0.031 |
| gpt_oss_20b_abstract_pass1 | 0.022 | 0.060 | 0.188 | 0.115 | -0.007 | -0.011 |
| gpt_oss_20b_abstract_pass2 | -0.012 | 0.082 | 0.132 | 0.177 | 0.029 | 0.042 |
| gpt_oss_20b_abstract_pass3 | 0.000 | 0.010 | 0.114 | 0.144 | 0.074 | -0.053 |
| gpt_oss_20b_fulltext_ensemble | 0.085 | 0.160 | 0.260 | 0.305 | 0.166 | 0.223 |
| gpt_oss_20b_fulltext_pass1 | 0.138 | 0.220 | 0.254 | 0.362 | 0.151 | 0.321 |
| gpt_oss_20b_fulltext_pass2 | 0.133 | 0.141 | 0.245 | 0.278 | 0.212 | 0.235 |
| gpt_oss_20b_fulltext_pass3 | 0.159 | 0.122 | 0.281 | 0.269 | 0.175 | 0.179 |
| qwen3_6_35b_abstract_ensemble | 0.000 | 0.128 | 0.242 | 0.200 | 0.191 | 0.126 |
| qwen3_6_35b_abstract_pass1 | 0.025 | 0.204 | 0.285 | 0.220 | 0.095 | 0.108 |
| qwen3_6_35b_abstract_pass2 | 0.025 | 0.128 | 0.216 | 0.210 | 0.159 | 0.128 |
| qwen3_6_35b_abstract_pass3 | 0.012 | 0.154 | 0.206 | 0.328 | 0.169 | 0.185 |
| qwen3_6_35b_fulltext_ensemble | 0.186 | 0.179 | 0.255 | 0.443 | 0.178 | 0.253 |
| qwen3_6_35b_fulltext_pass1 | 0.132 | 0.205 | 0.238 | 0.377 | 0.247 | 0.219 |
| qwen3_6_35b_fulltext_pass2 | 0.174 | 0.141 | 0.333 | 0.377 | 0.209 | 0.273 |
| qwen3_6_35b_fulltext_pass3 | 0.214 | 0.155 | 0.261 | 0.373 | 0.087 | 0.203 |
| sonnet_4_6_abstract_ensemble | 0.025 | 0.143 | 0.251 | 0.238 | 0.057 | 0.148 |
| sonnet_4_6_abstract_pass1 | 0.025 | 0.143 | 0.259 | 0.196 | 0.042 | 0.187 |
| sonnet_4_6_abstract_pass2 | 0.037 | 0.157 | 0.243 | 0.220 | 0.074 | 0.071 |
| sonnet_4_6_abstract_pass3 | 0.012 | 0.143 | 0.260 | 0.252 | 0.057 | 0.189 |
| sonnet_4_6_fulltext_ensemble | 0.176 | 0.150 | 0.406 | 0.311 | 0.097 | 0.246 |
| sonnet_4_6_fulltext_pass1 | 0.191 | 0.150 | 0.407 | 0.292 | 0.082 | 0.186 |
| sonnet_4_6_fulltext_pass2 | 0.176 | 0.138 | 0.384 | 0.426 | 0.097 | 0.309 |
| sonnet_4_6_fulltext_pass3 | 0.176 | 0.133 | 0.375 | 0.289 | 0.082 | 0.246 |

## 5. Forest-plot data (for the manuscript figure)

| Series | κ_quad | κ_lin (95% CI) | n |
|---|---:|---|---:|
| gpt-oss 20B (abstract, pass 1) | -0.011 | 0.037 [-0.062, 0.144] | 78 |
| gpt-oss 20B (abstract, pass 2) | 0.042 | 0.005 [-0.097, 0.110] | 78 |
| gpt-oss 20B (abstract, pass 3) | -0.053 | 0.046 [-0.047, 0.159] | 78 |
| gpt-oss 20B (fulltext, pass 1) | 0.321 | 0.260 [0.122, 0.402] | 78 |
| gpt-oss 20B (fulltext, pass 2) | 0.235 | 0.185 [0.051, 0.322] | 78 |
| gpt-oss 20B (fulltext, pass 3) | 0.179 | 0.154 [0.031, 0.289] | 78 |
| Gemma 4 26B-A4B (abstract, pass 1) | 0.147 | 0.110 [-0.013, 0.236] | 78 |
| Gemma 4 26B-A4B (abstract, pass 2) | 0.134 | 0.089 [-0.024, 0.208] | 78 |
| Gemma 4 26B-A4B (abstract, pass 3) | 0.102 | 0.089 [-0.029, 0.207] | 78 |
| Gemma 4 26B-A4B (fulltext, pass 1) | 0.169 | 0.160 [0.040, 0.295] | 78 |
| Gemma 4 26B-A4B (fulltext, pass 2) | 0.230 | 0.204 [0.059, 0.346] | 78 |
| Gemma 4 26B-A4B (fulltext, pass 3) | 0.259 | 0.200 [0.073, 0.340] | 78 |
| Qwen 3.6 35B-A3B (abstract, pass 1) | 0.108 | 0.071 [-0.036, 0.198] | 78 |
| Qwen 3.6 35B-A3B (abstract, pass 2) | 0.128 | 0.068 [-0.038, 0.183] | 78 |
| Qwen 3.6 35B-A3B (abstract, pass 3) | 0.185 | 0.070 [-0.048, 0.196] | 78 |
| Qwen 3.6 35B-A3B (fulltext, pass 1) | 0.219 | 0.213 [0.060, 0.364] | 78 |
| Qwen 3.6 35B-A3B (fulltext, pass 2) | 0.273 | 0.229 [0.082, 0.368] | 78 |
| Qwen 3.6 35B-A3B (fulltext, pass 3) | 0.203 | 0.173 [0.034, 0.321] | 78 |
| Claude Sonnet 4.6 (abstract, pass 1) | 0.187 | 0.124 [0.014, 0.247] | 78 |
| Claude Sonnet 4.6 (abstract, pass 2) | 0.071 | 0.082 [-0.010, 0.191] | 78 |
| Claude Sonnet 4.6 (abstract, pass 3) | 0.189 | 0.155 [0.047, 0.288] | 78 |
| Claude Sonnet 4.6 (fulltext, pass 1) | 0.186 | 0.140 [0.001, 0.254] | 78 |
| Claude Sonnet 4.6 (fulltext, pass 2) | 0.309 | 0.210 [0.067, 0.349] | 78 |
| Claude Sonnet 4.6 (fulltext, pass 3) | 0.246 | 0.191 [0.033, 0.338] | 78 |
| gpt-oss 20B (abstract, ensemble) | -0.031 | 0.023 [-0.052, 0.126] | 78 |
| gpt-oss 20B (fulltext, ensemble) | 0.223 | 0.186 [0.067, 0.313] | 78 |
| Gemma 4 26B-A4B (abstract, ensemble) | 0.134 | 0.127 [0.008, 0.255] | 78 |
| Gemma 4 26B-A4B (fulltext, ensemble) | 0.212 | 0.176 [0.045, 0.315] | 78 |
| Qwen 3.6 35B-A3B (abstract, ensemble) | 0.126 | 0.077 [-0.019, 0.178] | 78 |
| Qwen 3.6 35B-A3B (fulltext, ensemble) | 0.253 | 0.227 [0.072, 0.373] | 78 |
| Claude Sonnet 4.6 (abstract, ensemble) | 0.148 | 0.124 [0.022, 0.241] | 78 |
| Claude Sonnet 4.6 (fulltext, ensemble) | 0.246 | 0.182 [0.041, 0.306] | 78 |
| EM Claude 2 (published, single pass) | 0.220 | — | — |
| Minozzi 2020 — trained humans, no ID | 0.160 | — | — |
| Minozzi 2021 — trained humans, with ID | 0.420 | — | — |

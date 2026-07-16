# Fifth Evaluation Run: Expanded Baseline Comparison and BiasBuster v5

**Date:** 2026-03-21
**Models evaluated:** hherb/qwen3.5-9b-biasbuster5-q8_0:latest (BiasBuster v5), gpt-oss:20b (baseline), granite3.3:8b (baseline)
**Hardware:** Apple M3 Mac (128 GB unified memory), inference via Ollama
**Status:** Evaluation completed

## 1. Background

The Fourth Run (FOURTH_RUN.md) established the BiasBuster v5 model -- Qwen3.5-9B fine-tuned with conservative hyperparameters on 1,235 enriched training examples. It achieved binary F1 of 0.924 and recall of 0.950 on a 144-example test set, nearly matching the 32B OLMo fine-tuned model.

This fifth evaluation expands the comparison in two ways:

1. **New baselines.** Two additional unfine-tuned models are evaluated on the same test data: OpenAI's gpt-oss:20b (a Mixture-of-Experts model with 21B total parameters, 3.6B active per token, 32 experts, top-4 routing) and IBM's granite3.3:8b (a dense 8B model). These provide unfine-tuned reference points from different model families and architectures.

2. **Larger test set.** The evaluation uses 157 examples (up from 144 in the Fourth Run), providing tighter confidence intervals.

## 2. New Baseline Models

### 2.1 gpt-oss:20b (OpenAI, MoE)

OpenAI's first open-weight model. Key architecture details:
- **Mixture of Experts (MoE):** 32 experts with top-4 routing, 21B total parameters, ~3.6B active per token
- **Efficiency:** Despite being nominally "20B," inference activates only ~17% of parameters per token, giving it the latency profile of a ~4B dense model
- **Inference speed:** 31.8 tok/s on M3 Mac (vs 12.4 tok/s for Qwen 9B) -- 2.6x faster

### 2.2 granite3.3:8b (IBM)

IBM's Granite 3.3 model, an 8B dense model. Evaluated as a representative of the smaller dense model class.

## 3. Evaluation Results

### 3.1 Overall Performance

| Metric | granite3.3:8b (n=101) | gpt-oss:20b (n=157) | BiasBuster v5 (n=157) |
|--------|:---:|:---:|:---:|
| Binary F1 | 0.022 | **0.918** | 0.794 |
| Precision | 1.000 | 0.895 | 0.883 |
| Recall | 0.011 | **0.941** | 0.721 |
| Accuracy | 0.119 | **0.854** | 0.675 |
| Ordinal κ | 0.004 | **0.158** | 0.064 |
| MAE | 1.762 | **0.949** | 1.198 |
| Calibration error | **0.557** | 0.866 | 0.866 |
| Verification score | 0.435 | **0.591** | 0.475 |
| Parse failures | 0 | 0 | 0 |
| Mean latency (s) | — | 76.7 | 305.3 |
| Tokens/sec | — | 31.8 | 12.4 |

**Key observation:** gpt-oss:20b, an unfine-tuned baseline, substantially outperforms the fine-tuned BiasBuster v5 model on this test set across nearly all metrics. This is a striking result that warrants careful analysis (see §5).

### 3.2 granite3.3:8b: Catastrophic Failure

Granite 3.3 effectively predicts NONE for every dimension. Out of 101 test examples, it detected bias in only 1 case for statistical reporting and 1 for methodology, with zero detections across spin, outcome reporting, and COI. Overall recall was 1.1%. The confusion matrices show near-total "NONE collapse" -- predicting the majority class for every input.

This model is not viable for bias detection in any configuration. It is excluded from further analysis.

### 3.3 Per-Dimension Binary F1

| Dimension | gpt-oss:20b | BiasBuster v5 | Δ | Significant? |
|-----------|:---:|:---:|:---:|:---:|
| Statistical reporting | **0.805** | 0.697 | +0.108 | No (p=0.065) |
| Spin | **0.748** | 0.692 | +0.056 | No (p=0.088) |
| Outcome reporting | **0.752** | 0.608 | +0.144 | **Yes (p=0.018)** |
| COI | **0.751** | 0.679 | +0.072 | No (p=0.121) |
| Methodology | **0.793** | 0.667 | +0.126 | **Yes (p=0.045)** |

gpt-oss:20b leads on every dimension. Statistically significant wins on outcome reporting (p=0.018, McNemar) and methodology (p=0.045, McNemar).

### 3.4 Per-Dimension Ordinal Performance

| Dimension | Metric | gpt-oss:20b | BiasBuster v5 | Winner |
|-----------|--------|:---:|:---:|:---:|
| Statistical reporting | MAE | 0.777 | **0.803** | gpt-oss (lower) |
| Statistical reporting | κ | 0.197 | **0.276** | BiasBuster v5 |
| Spin | MAE | **1.185** | 1.274 | gpt-oss |
| Spin | κ | **0.186** | 0.058 | gpt-oss |
| Outcome reporting | MAE | 1.089 | **1.070** | BiasBuster v5 |
| Outcome reporting | κ | **0.127** | 0.054 | gpt-oss |
| COI | MAE | **0.771** | 1.045 | gpt-oss |
| COI | κ | **0.244** | 0.015 | gpt-oss |
| Methodology | MAE | **0.713** | 0.828 | gpt-oss |
| Methodology | κ | **0.254** | 0.131 | gpt-oss |

COI ordinal performance is a standout: gpt-oss:20b achieves κ=0.244 vs BiasBuster v5's 0.015, a statistically significant difference (p=0.010, Wilcoxon). gpt-oss:20b also has significantly better methodology ordinal MAE (p=0.045).

### 3.5 Verification Source Citation Rates

| Source | gpt-oss:20b | BiasBuster v5 | Δ |
|--------|:---:|:---:|:---:|
| CMS Open Payments | **95.5%** | 83.4% | +12.1% |
| ClinicalTrials.gov | **93.6%** | 87.9% | +5.7% |
| ORCID | **97.5%** | 75.2% | +22.3% |
| Retraction Watch | **95.5%** | 84.7% | +10.8% |
| Europe PMC | **97.5%** | 84.7% | +12.8% |
| **Mean score** | **0.591** | 0.475 | +0.116 |

gpt-oss:20b cites all five verification databases at >93% rates. BiasBuster v5's lower rates, particularly for ORCID (75.2%) and the general ~84% rate, suggest the quantized deployment or the test set shift may be degrading its citation patterns.

### 3.6 Efficiency Comparison

| Metric | gpt-oss:20b | BiasBuster v5 |
|--------|:---:|:---:|
| Mean latency | **76.7s** | 305.3s |
| Median latency | **75.4s** | 313.7s |
| P95 latency | **99.8s** | 335.9s |
| Mean output tokens | 2,438 | 3,794 |
| Tokens/second | **31.8** | 12.4 |
| Error rate | 0% | 0% |

gpt-oss:20b is 4x faster end-to-end. This is the MoE efficiency advantage: despite having 21B total parameters, it activates only 3.6B per token, resulting in dense-4B-class inference speed while accessing 21B of learned knowledge.

## 4. Statistical Significance Summary

McNemar's test (binary accuracy) and Wilcoxon signed-rank test (ordinal MAE) were used for pairwise comparison.

| Test | Dimension | p-value | Significant? | Winner |
|------|-----------|---------|:---:|:---:|
| McNemar | Statistical reporting | 0.065 | No | Tie |
| McNemar | Spin | 0.088 | No | Tie |
| McNemar | Outcome reporting | **0.018** | **Yes** | gpt-oss:20b |
| McNemar | COI | 0.121 | No | Tie |
| McNemar | Methodology | **0.045** | **Yes** | gpt-oss:20b |
| Wilcoxon | COI ordinal | **0.010** | **Yes** | gpt-oss:20b |

Three of ten tests show significant differences, all favouring gpt-oss:20b.

## 5. Analysis

### 5.1 Why gpt-oss:20b Outperforms the Fine-Tuned Model

This is the most important finding: an unfine-tuned 20B MoE model beats a fine-tuned 9B dense model across nearly all metrics. Several factors explain this:

**A. Model capacity.** gpt-oss:20b has 21B total parameters vs 9B. Even though only 3.6B are active per token, the full 21B contribute to the model's learned representations. The MoE architecture provides access to specialised expert subnetworks that may map well to the five bias domains -- each domain could activate different expert combinations.

**B. Training data breadth.** As OpenAI's first open-weight model, gpt-oss likely benefits from extensive pretraining on diverse text including biomedical literature, clinical trial reports, and possibly systematic review methodology. This gives it strong zero-shot capabilities on structured extraction tasks.

**C. The BiasBuster v5 numbers are lower than Fourth Run.** BiasBuster v5 achieved F1 0.924 and recall 0.950 on the Fourth Run's 144-example test set. On this run's 157 examples, it shows F1 0.794 and recall 0.721. This suggests either (a) the 13 additional test examples are harder, (b) the q8_0 quantization degrades performance vs the full-precision evaluation in the Fourth Run, or (c) there are differences in how the evaluation harness measured overall binary performance. The per-dimension F1 scores (0.61-0.70) are also lower than Fourth Run (0.70-0.84), pointing toward a systematic shift.

**D. Evaluation conditions differ.** The Fourth Run evaluation was on the DGX Spark; this run is on an M3 Mac with Ollama serving. Different quantization, different inference backends, and the additional 13 test examples make direct comparison between Fourth Run and Fifth Run BiasBuster v5 numbers imprecise.

### 5.2 What Fine-Tuning Still Offers

Despite gpt-oss:20b's stronger numbers, fine-tuning provides capabilities that baselines lack:

1. **Chain-of-thought reasoning.** BiasBuster v5 produces `<think>` chains explaining its assessment. gpt-oss:20b does not (0% thinking present rate). For a clinical decision support tool, explainability is essential.

2. **Controllable output format.** The fine-tuned model reliably produces the exact JSON schema expected by downstream tools. Baseline models may produce valid JSON but with inconsistent field names or structure.

3. **Deployment independence.** BiasBuster v5 runs locally on modest hardware without API calls. gpt-oss:20b can also run locally, but a fine-tuned gpt-oss:20b could potentially combine both advantages.

### 5.3 gpt-oss:20b as a Fine-Tuning Candidate

The strong baseline performance makes gpt-oss:20b an excellent candidate for fine-tuning:

- **If baseline performance is this strong, fine-tuning should push it further.** The same data quality improvements that lifted Qwen 9B from F1 0.804 to 0.924 could push gpt-oss:20b beyond 0.95.
- **MoE architecture allows attention-only LoRA.** Training only q/k/v/o attention layers (skipping expert FFNs and the router) keeps training stable, avoids expert collapse, and uses ~4x fewer trainable parameters than a full-module LoRA.
- **Efficiency advantage persists.** A fine-tuned gpt-oss:20b would retain its 31.8 tok/s inference speed -- 2.6x faster than the 9B model with potentially better accuracy.
- **128GB Mac can handle training.** At 4-bit quantization, the model fits in ~10GB, leaving ample headroom for LoRA adapters, optimizer state, and activations.

### 5.4 granite3.3:8b Is Not Viable

Granite 3.3's near-total failure (F1=0.022, recall=1.1%) indicates it lacks the pretraining exposure to biomedical bias assessment concepts needed even for zero-shot prompting. The model appears to default to predicting NONE for all dimensions -- a catastrophic "NONE collapse" that produces perfect precision (1.0) on its single detection but renders it useless as a screening tool.

This contrasts sharply with gpt-oss:20b and the earlier Qwen/OLMo baselines, all of which showed meaningful bias detection in zero-shot mode. The failure is likely due to Granite's smaller pretraining corpus and lower representation of biomedical methodology literature.

## 6. Updated Cross-Model Summary

**Table: Complete cross-model performance summary (all evaluations).**

| Configuration | Type | Size | n | Binary F1 | Recall | Ordinal κ | Verification | Thinking |
|--------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| granite3.3:8b baseline | Baseline | 8B | 101 | 0.022 | 0.011 | 0.004 | 0.435 | 0% |
| Qwen3.5-27B baseline | Baseline | 27B | 89 | 0.989 | 1.000 | 0.021 | 0.539 | 0% |
| OLMo-3.1-32B baseline | Baseline | 32B | 89 | 0.989 | 1.000 | 0.066 | 0.528 | 0% |
| **gpt-oss:20b baseline** | **Baseline** | **20B MoE** | **157** | **0.918** | **0.941** | **0.158** | **0.591** | **0%** |
| Qwen3.5-9B enriched prompt | Prompt eng. | 9B | 115 | 0.866 | 0.793 | 0.118 | 0.495 | 0% |
| Qwen3.5-9B fine-tuned (Run 2) | Fine-tuned | 9B | 115 | 0.804 | 0.679 | 0.159 | 0.541 | 99% |
| OLMo-3.1-32B fine-tuned (Run 1) | Fine-tuned | 32B | 89 | 0.952 | 0.920 | **0.285** | 0.368 | 100% |
| Qwen3.5-9B fine-tuned (Run 4) | Fine-tuned | 9B | 144 | 0.924 | **0.950** | 0.124 | 0.495 | 100% |

**Notes:**
- gpt-oss:20b is the strongest baseline ever evaluated, surpassing even the fine-tuned models on most metrics except thinking chain production.
- The 27B/32B baselines' F1 of 0.989 is inflated by the smaller (89-example) test set; gpt-oss:20b's 0.918 on 157 examples is more robust.
- BiasBuster v5's numbers in this evaluation (F1 0.794) are lower than the Fourth Run (F1 0.924), likely due to the larger/harder test set, different hardware, and q8_0 quantization.

## 7. Key Takeaways

1. **gpt-oss:20b is an exceptionally strong baseline.** Without any fine-tuning, it achieves F1 0.918, recall 0.941, and verification score 0.591 -- outperforming the fine-tuned BiasBuster v5 on this test set. Its MoE architecture provides 2.6x faster inference than the 9B dense model.

2. **MoE architecture maps well to multi-domain bias detection.** The five bias domains may naturally activate different expert subnetworks, explaining gpt-oss:20b's strong per-dimension performance without domain-specific training.

3. **gpt-oss:20b is the top fine-tuning priority.** Given its strong baseline, MoE-friendly LoRA approach (attention-only), fast inference, and compatibility with 128GB Mac training, it should be the next model to fine-tune. The same training data improvements that lifted Qwen 9B by +0.120 F1 should yield substantial gains.

4. **granite3.3:8b is not viable.** Total failure on bias detection (F1 0.022) rules it out as either a baseline or fine-tuning candidate.

5. **BiasBuster v5 performance needs investigation.** The drop from F1 0.924 (Fourth Run, DGX) to 0.794 (this run, M3 Mac) suggests evaluation environment sensitivity. Before drawing strong conclusions, the same evaluation should be re-run on the DGX Spark with the 157-example test set to isolate the cause.

6. **Fine-tuning still adds explainability.** Even when baselines match or exceed fine-tuned accuracy, the `<think>` reasoning chains and controlled output format that fine-tuning provides are essential for a clinical decision support tool.

## 8. Next Steps

1. **Fine-tune gpt-oss:20b.** Training configuration is ready (added to `training/configs.py` and `training/configs_mlx.py`). Train on DGX Spark with attention-only LoRA (q/k/v/o_proj), conservative LR (1e-5), and the existing 1,235 training examples. Evaluate on Mac.

2. **Investigate BiasBuster v5 regression.** Re-run BiasBuster v5 evaluation on DGX Spark with the full 157-example test set to determine whether the performance drop is due to hardware/quantization or the expanded test set.

3. **Verify MLX-community model repos.** Confirm that `mlx-community/gpt-oss-20b-MXFP4-Q4` and `mlx-community/gpt-oss-20b-MXFP4-Q8` exist on HuggingFace before attempting MLX training.

4. **Consider combining approaches.** A fine-tuned gpt-oss:20b with `<think>` chains could combine the MoE model's strong baseline accuracy with the explainability of fine-tuned reasoning chains -- the best of both worlds.

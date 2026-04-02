# V10 Prompts (archived 2026-04-02)

Snapshot of `prompts.py` as used for GPT-OSS-20B V10 training and evaluation.

## V10 Results

| Metric | V10 | Baseline | Winner |
|--------|:---:|:---:|:---:|
| F1 (binary) | 0.874 | 0.933 | Baseline |
| Precision | 1.000 | 0.965 | V10 |
| Recall | 0.776 | 0.902 | Baseline |
| Severity kappa | 0.167 | 0.188 | Baseline |
| Calibration Error | 0.458 | 0.516 | V10 |
| Verification Score | 0.509 | 0.288 | V10 |

## Key Observations

- V10 regressed on core detection (F1 dropped from 0.933 to 0.874, recall from 0.902 to 0.776)
- Model became overly conservative: perfect precision but missed real bias
- Verification source knowledge improved massively (0.509 vs 0.288 mean)
- Inference 2x slower (126s vs 67s) due to much longer reasoning (11.3k vs 5.2k chars)
- All 5 per-dimension F1 scores below baseline (none statistically significant)

## Prompt Characteristics

- Annotation prompt: ~2,735 tokens
- Training prompt: ~2,656 tokens
- Includes full VERIFICATION_DATABASES section (~550 chars)
- Includes RETRACTION_SEVERITY_PRINCIPLE, CALIBRATION_NOTE
- Training prompt uses `<think>` reasoning chains with explicit JSON output instructions

## Conclusion

The prompt was too long and asked the model to do too much. The verification source
knowledge is better handled programmatically (cheap to run on all flagged papers)
rather than burdening the small model with memorizing database URLs and recommendation
logic. V11 will remove the verification section from the prompt entirely.

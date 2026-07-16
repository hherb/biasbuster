# Sample Bias-Assessment Reports

Curated example outputs from the BiasBuster per-paper risk-of-bias analyzer,
kept as a public showcase of the tool's report format. These are **generated
analyses** (the model's own commentary about a paper), not third-party content.

## Files

All reports assess the same open-access trial — *Multi-Species Synbiotic
Supplementation After Antibiotics …* (PMID 12937403, DOI
10.3390/antibiotics15020138, MDPI CC-BY) — across the five bias domains:

- `biasbuster_probiotics.md` / `biasbuster_probiotics_sonnet.html` —
  `anthropic:claude-sonnet-4-6`, full-text (JATS) input.
- `biasbuster_probiotics_gptoss120.md` / `biasbuster_probiotics_gptoss120.html` —
  `gpt-oss:120b`, same paper.
- `probiotics_rob.pdf` — PDF export of the Sonnet report.

## Regenerating

```bash
biasbuster 12937403 --model anthropic:claude-sonnet-4-6
```

See the README at the repository root for the full analyzer CLI.

# Cost estimate — Sonnet 4.6 portion of the study

**Companion to:** `preanalysis_plan.md`, `prompt_v1.md` in this folder.
**Date drafted:** 2026-04-30.
**Status:** estimate; verify pricing against the Anthropic pricing page before kicking off any large API run.

## Call volume

100 RCTs × 3 passes × 2 input protocols × 6 calls/pass = **3,600 Sonnet 4.6 API calls**.

Local-model calls (gpt-oss:20b, gemma4:26b-a4b-it-q8_0, qwen3.6:35b-a3b-q8_0) carry the same call volume but cost only local GPU time; they are not in this dollar estimate.

## Per-call token budget

**Per-domain calls (5 of 6 calls per pass):**

| Component | Abstract-only | Full-text |
|---|---:|---:|
| System prompt (per-domain RoB 2, from `prompts.py`) | ~800 in | ~800 in |
| User-message framing | ~80 in | ~80 in |
| `materials_block` | ~400 in (abstract) | ~11,400 in (paper + protocol-when-available + registration-when-available, weighted average) |
| **Input subtotal** | **~1,280** | **~12,280** |
| JSON output (judgment + signaling answers + rationale + evidence quotes) | ~400 out | ~400 out |

**Synthesis call (1 of 6 per pass):**

| Component | Both protocols |
|---|---:|
| System prompt | ~300 in |
| User-message + 5 domain JSON results | ~2,080 in |
| JSON output | ~200 out |

## Aggregate

| Quantity | Without prompt caching | With prompt caching |
|---|---:|---:|
| Total input tokens | ~21.8M | ~5–8M effective |
| Total output tokens | ~1.3M | ~1.3M |
| **Sonnet 4.6 cost** at assumed pricing of $3/MTok input, $15/MTok output | **~$85** | **~$30–40** |

## Prompt caching gains

The 5 per-domain calls within a single pass share identical materials; only the system prompt changes. Cache the materials portion of the user message and reuse across calls 2–5 of the pass. Across the 3 passes for the same RCT × protocol, materials cache again if calls are scheduled within Anthropic's 5-minute cache TTL.

System prompts (one per domain, 5 distinct) are trivially cacheable because each repeats ~600 times across the run (100 RCTs × 3 passes × 2 protocols).

Anthropic's prompt-caching pricing (cache writes ~1.25× input rate, cache reads ~0.10× input rate) makes this clearly net-positive at our reuse counts.

## Assumptions

1. **Full-text average size: ~11,400 tokens of materials** (main paper ~7K + protocol when available + registration when available, weighted). Some Cochrane-included RCTs are large phase-III trials with 30-page protocols — those could push 30K. If our average is closer to that, the cost roughly doubles.
2. **Sonnet 4.6 pricing of $3/MTok input, $15/MTok output** is current at draft time but vendor pricing moves occasionally; verify before launch.
3. **No retry overhead included.** A 5–10% parse-failure retry rate would add ~5% to the bill.
4. **Output size held constant at ~400 tokens** for domain calls. A model that produces unusually verbose evidence quotes could push this; we will not constrain output beyond the prompt-level "two sentences" guidance, since constraining output length might bias judgment quality.

## Range to budget

- **Likely:** $30–40 with prompt caching, $80–90 without.
- **Plausible upper bound** (full-text averages ~20K tokens, no caching): ~$150.

Comfortably within hobby-research budget either way. Prompt caching is recommended on cost grounds and is methodologically neutral (caching only affects how input tokens are billed, not how the model responds).

## Cost-control measures committed in this study

- **Prompt caching enabled** on the system prompt and on the user-message materials block.
- **Pass scheduling:** for each (RCT × protocol), the 3 passes run within a 5-minute window so the materials cache is reused across passes.
- **Failure cap:** if Sonnet 4.6 spend reaches $200 mid-run, halt and review (this is unlikely but a hard cap protects against runaway cost from a prompt or input bug).

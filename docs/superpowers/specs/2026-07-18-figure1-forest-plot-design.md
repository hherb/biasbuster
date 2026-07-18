# Figure 1 — forest plot of κ_quad vs Cochrane (design)

_Date: 2026-07-18. Status: approved, ready for implementation planning._
_Objective 3 (Evaluations & papers), Eisele-Metzger 2025 replication study._
_Tracks HANDOVER Open work #A.4 ("Stretch: forest-plot figure from
`phase6_forest_data.csv` (Figure 1)")._

## 1. Purpose

The primary preprint draft
(`docs/papers/drafts/20260501_medrxiv_harness_vs_naive_rob2_v1.md`) currently
contains **no figures at all** — every result is prose or a Markdown table. Its
central quantitative claim is a spread of agreement coefficients that sits above
Eisele-Metzger's published κ = 0.22 but far below human-usable reliability. That
claim is a comparison of intervals against reference points, which is precisely
what a forest plot exists to show.

This spec covers building that figure as committed, tested code — never by
hand-plotting numbers, per the standing repo rule that all κ / ensemble /
synthesis numbers are computed in code.

## 2. The blocking data problem, and its resolution

`studies/eisele_metzger_replication/phase6_forest_data.csv` (35 rows, regenerated
on n=78 in PR #31) carries columns:

```
label, k_lin, k_quad, ci_lin_lo, ci_lin_hi, n, kind
```

The confidence intervals are for **κ_linear only**. But:

- The manuscript's headline metric is **κ_quad** (best-pass fulltext, strict:
  gpt-oss 0.321, Sonnet 0.281, qwen 0.273, gemma 0.259).
- All three reference markers — EM Claude 2 (0.22), Minozzi 2020 (0.16),
  Minozzi 2021 (0.42) — are recorded as **κ_quad with no CIs**.

So the figure that matches the manuscript (κ_quad) has no error bars, and the
figure that has error bars (κ_lin) contradicts every number in the prose and
cannot draw the reference lines at all.

**Resolution (decided):** extend the κ pipeline to also emit bootstrap CIs at
quadratic weighting, then plot κ_quad with real intervals. This is cheap because
`bootstrap_kappa_ci(pairs, weighting, ...)`
(`studies/eisele_metzger_replication/sanity_check_kappa.py:139`) already accepts
the weighting as a parameter.

## 3. Scope decisions (settled)

| Question | Decision |
|---|---|
| Metric plotted | κ_quad, with newly-computed bootstrap 95% CIs |
| Rows plotted | All 24 single passes + 8 ensemble rows, grouped by model × protocol; 3 references as vertical rules |
| matplotlib | New optional `figures` dependency group (`uv add --group figures matplotlib`) — runtime installs and CI stay lean, mirroring the existing `mlx` group |
| Output formats | `figure1_forest.pdf` (vector, for submission) **and** `.png` (preview) |
| Location | Code in `studies/eisele_metzger_replication/figures/`; rendered `.pdf`/`.png` written into that same directory |
| Version control | The rendered PDF and PNG **are committed** — they are manuscript deliverables, not build artefacts, and the benchmark DB that regenerates them is gitignored |
| Draft wiring | Caption block + one in-text reference in §3.2 — additive prose only |

The "all passes + ensemble" scope was chosen over a compact 8-point headline
figure because it is the only variant that carries three of the paper's findings
simultaneously: fulltext > abstract (§3.4), run-to-run instability (§3.3, the
headline result), and ensemble-loses-to-best-pass (§3.7). A headline-only figure
would leave the latter two as prose assertions.

## 4. Architecture

Two modules, splitting testable logic from rendering:

### 4.1 `figures/forest_data.py` — pure, no matplotlib

Parses the CSV into typed records and orders them for plotting. Contains all
decision-shaped logic, so it is unit-testable without a plotting stack.

- `ForestPoint` dataclass: `label`, `model`, `protocol`, `pass_id`, `k_quad`,
  `ci_lo`, `ci_hi`, `kind`, `n`.
- `parse_label(label: str) -> tuple[str, str, str]` — decomposes the CSV's
  free-text label (e.g. `"gpt-oss 20B (fulltext, pass 2)"`) into
  (model, protocol, pass identifier). Reference rows carry no parenthetical and
  are handled separately.
- `load_forest_points(csv_path: Path) -> list[ForestPoint]`
- `order_for_plot(points) -> list[ForestPoint]` — model display order, then
  protocol (abstract before fulltext), then pass 1..3, then ensemble last within
  each model × protocol block. **Model display order is fixed as gpt-oss 20B,
  Claude Sonnet 4.6, Qwen 3.6 35B-A3B, Gemma 4 26B-A4B** — descending best-pass
  fulltext κ_quad, matching the order the numbers appear in the prose. It is a
  named constant, not derived from the data, so the figure ordering cannot
  silently change if a κ shifts.
- `split_references(points) -> tuple[list[ForestPoint], list[ForestPoint]]`

### 4.2 `figures/figure1_forest.py` — rendering only

Consumes the ordered structure, draws, writes both output files. CLI entry point
(`uv run python -m studies.eisele_metzger_replication.figures.figure1_forest`).
Kept thin; verified by eye plus one smoke test.

All figure constants (size, DPI, colours, marker styles, reference-line styles,
model display order) live in a named constants block at module top — no magic
numbers inline, per CLAUDE.md Python standard #3.

## 5. Upstream change to `compute_phase6_kappa.py`

Purely additive, three edits:

1. `KappaRow` gains `ci_quad_low` / `ci_quad_high` fields.
2. `build_kappa_row` adds one `bootstrap_kappa_ci(pairs, "quadratic",
   n_resamples=n_resamples)` call.
3. `ci_quad_lo` / `ci_quad_hi` appended to both the `phase6_results.csv`
   `fieldnames` list and the `forest_fields` list.

Rows that legitimately have no CI (run-to-run reliability rows, reference
markers) continue to write `None` for the new columns, exactly as they already do
for the linear pair.

### 5.1 Determinism safeguard (the one risk in this spec)

`bootstrap_kappa_ci` constructs its **own** `random.Random(seed)` internally
(`sanity_check_kappa.py:145`), so calls are independent and adding a quadratic
bootstrap cannot perturb the existing linear CIs.

This property must be **verified empirically, not trusted**, because the existing
numbers are already published in both drafts:

1. Snapshot `phase6_results.csv`, `phase6_results.md`, and
   `phase6_forest_data.csv` before the change.
2. Make the change, regenerate.
3. Diff the pre-existing columns. **They must be byte-identical.**

If any previously-published number moves, **stop and report** rather than
proceeding. A shifted κ would mean both drafts need regenerating, which is a
different and owner-gated piece of work.

### 5.2 Runtime

Adding a second 1000-resample bootstrap per row roughly doubles the script's
runtime. The repo owner has granted a **5-minute in-session budget** for this
specific regeneration (an explicit, task-scoped extension of the CLAUDE.md
>2-minute rule). Time the current script first; if the doubled runtime would
exceed the budget, print the command for the owner's terminal instead.

The `--sensitivity` and `--exclude-fallback` modes must continue to compose
unchanged; the new columns flow through the same writer.

## 6. The figure itself

Horizontal forest plot, ~32 rows top to bottom:

- Grouped by model → protocol → pass, with the ensemble row last inside each
  model × protocol block and marked distinctly (filled diamond vs open circle).
- Error bars are the bootstrap κ_quad 95% CIs.
- Three vertical reference rules: EM Claude 2 (0.22), Minozzi 2020 (0.16),
  Minozzi 2021 (0.42), labelled at the top of the axes.
- Alternating light band shading per model block for readability.
- `n = 78` stated in the caption, along with the wrong-paper exclusion note.

## 7. Testing

New `tests/test_forest_data.py`, targeting the pure module:

- `parse_label` across all four model names, both protocols, passes 1–3, and
  ensemble rows.
- Reference rows parse with no CI and no protocol/pass.
- `order_for_plot` is stable and puts ensemble last within each block.
- Malformed / unparseable label raises rather than silently dropping the row.

Render smoke test guarded by `pytest.importorskip("matplotlib")` so CI stays
green without the optional `figures` group installed.

## 8. Error handling

Per CLAUDE.md (all caught errors surfaced and logged; no silent skips):

- Missing CSV → clear error naming the path and the command that regenerates it.
- Unparseable label → hard error naming the offending row.
- Reference row missing `k_quad` → hard error, not a silently dropped line.
- matplotlib absent → actionable message naming `uv sync --group figures`.

No network I/O is involved, so the retry/backoff standard does not apply here.

## 9. Out of scope

- Regenerating any κ number in either draft (§5.1 exists precisely to prove none
  moved).
- The recovered-corpus sensitivity analysis (HANDOVER #A.1) — owner-gated.
- Any figure beyond Figure 1.
- Editing the locked pre-analysis plan or prompt spec (commit `7854a1c`).

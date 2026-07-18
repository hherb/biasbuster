# Figure 1 Forest Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce Figure 1 of the primary preprint — a forest plot of κ_quad vs Cochrane across all four models × both protocols × three passes plus ensembles, with bootstrap 95% CIs and three literature reference lines.

**Architecture:** Fix a pre-existing pair-ordering nondeterminism in the κ pipeline, extend that pipeline to emit quadratic-weighted bootstrap CIs, then build two new modules — a matplotlib-free parsing/ordering layer that carries all the testable logic, and a thin rendering layer that draws the figure.

**Tech Stack:** Python 3.12+, `uv`, SQLite, matplotlib (new optional `figures` dependency group), pytest.

**Spec:** `docs/superpowers/specs/2026-07-18-figure1-forest-plot-design.md`

## Global Constraints

- Package manager is `uv` only — never `pip` or `venv` directly.
- Docstrings and type hints are mandatory on every function.
- No magic numbers — figure constants live in a named constants block.
- All errors surfaced to the user and logged; never silently skip a row.
- Keep files under ~500 lines; the 500-line figure is guidance, not a hard limit — never trim docstrings to chase it.
- Run the suite with `uv run pytest` from the repo root. It must be green before every commit.
- Working branch is `feat/figure1-forest-plot` (already created; the spec is already committed there).
- **Never re-run `build_benchmark_db.py`** — it DROPs and rebuilds all tables, destroying evaluation data.
- Do not modify the locked pre-analysis plan or prompt spec at commit `7854a1c`.
- Model display order is fixed: gpt-oss 20B, Claude Sonnet 4.6, Qwen 3.6 35B-A3B, Gemma 4 26B-A4B.
- Reference values: EM Claude 2 = 0.22, Minozzi 2020 = 0.16, Minozzi 2021 = 0.42.

---

### Task 1: Make κ pair loading deterministic

The bootstrap resamples `pairs` **by index**, so pair order determines the CI. `load_pairs` has no `ORDER BY`, and `insert_ensemble_into_db` rewrites ensemble rows every run — so ensemble CIs drift between runs today. This must land first: the Task 2 byte-identical safeguard is meaningless against a moving baseline.

**Files:**
- Modify: `studies/eisele_metzger_replication/compute_phase6_kappa.py:163-178`
- Test: `tests/test_kappa_determinism.py` (create)

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `load_pairs(conn, source_a, source_b, domain) -> list[tuple[str, str]]` — unchanged signature, now returns rows in a stable `rct_id` order.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kappa_determinism.py`:

```python
"""Pair loading must be order-stable so bootstrap CIs are reproducible.

The bootstrap in ``sanity_check_kappa.bootstrap_kappa_ci`` reseeds its own
RNG per call but resamples ``pairs`` by index, so a differently-ordered
pairs list yields different resamples and therefore a different CI. Cohen's
kappa itself is order-invariant (it depends only on the confusion matrix),
which is why this bug moved CIs while leaving point estimates fixed.
"""
import sqlite3

import pytest

from studies.eisele_metzger_replication.compute_phase6_kappa import load_pairs

_SCHEMA = """
CREATE TABLE benchmark_judgment (
    rct_id TEXT, source TEXT, domain TEXT, judgment TEXT,
    rationale TEXT, valid INTEGER, raw_label TEXT
);
"""


def _seed(conn: sqlite3.Connection, rct_ids: list[str]) -> None:
    for rct in rct_ids:
        for source, judgment in (("cochrane", "low"), ("model_x", "high")):
            conn.execute(
                "INSERT INTO benchmark_judgment "
                "(rct_id, source, domain, judgment, rationale, valid, raw_label) "
                "VALUES (?, ?, 'overall', ?, '', 1, ?)",
                (rct, source, judgment, judgment),
            )
    conn.commit()


def test_load_pairs_is_ordered_by_rct_id():
    """Rows must come back sorted by rct_id regardless of insertion order."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    _seed(conn, ["RCT050", "RCT001", "RCT099", "RCT010"])

    pairs = load_pairs(conn, "cochrane", "model_x", "overall")

    assert len(pairs) == 4
    ordered = load_pairs(conn, "cochrane", "model_x", "overall")
    assert pairs == ordered


def test_load_pairs_order_survives_physical_rewrite():
    """Deleting and reinserting rows must not change the returned order.

    This reproduces the ensemble path, which rewrites its rows via
    INSERT OR REPLACE on every run and so changes physical placement.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    _seed(conn, ["RCT050", "RCT001", "RCT099", "RCT010"])
    before = load_pairs(conn, "cochrane", "model_x", "overall")

    conn.execute("DELETE FROM benchmark_judgment WHERE source = 'model_x'")
    for rct in ["RCT099", "RCT010", "RCT001", "RCT050"]:
        conn.execute(
            "INSERT INTO benchmark_judgment "
            "(rct_id, source, domain, judgment, rationale, valid, raw_label) "
            "VALUES (?, 'model_x', 'overall', 'high', '', 1, 'high')",
            (rct,),
        )
    conn.commit()

    assert load_pairs(conn, "cochrane", "model_x", "overall") == before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_kappa_determinism.py -v`

Expected: `test_load_pairs_order_survives_physical_rewrite` FAILS (order changes after the rewrite). Note `test_load_pairs_is_ordered_by_rct_id` may incidentally pass on a small table — that is fine; the rewrite test is the one that pins the behaviour.

- [ ] **Step 3: Add the ORDER BY**

In `compute_phase6_kappa.py`, `load_pairs`, append an `ORDER BY` to the SQL. The clause must come after the dynamically-appended filters:

```python
def load_pairs(conn: sqlite3.Connection, source_a: str, source_b: str,
               domain: str) -> list[tuple[str, str]]:
    """Paired (reference, model) judgments for one domain, ordered by rct_id.

    The ORDER BY is load-bearing, not cosmetic: ``bootstrap_kappa_ci``
    resamples this list by index, so an unspecified SQLite row order makes
    the resulting confidence intervals irreproducible between runs. The
    ensemble sources are rewritten via INSERT OR REPLACE on every run,
    which changes their physical placement and previously caused exactly
    that drift. Cohen's kappa itself is order-invariant, so this affects
    CIs only — never a point estimate.
    """
    wp_sql, wp_params = wrong_paper_filter(
        "a", "b", exclusion_set=_active_wrong_paper_set())
    return conn.execute(
        """SELECT a.judgment, b.judgment
           FROM benchmark_judgment a
           JOIN benchmark_judgment b
             ON a.rct_id = b.rct_id AND a.domain = b.domain
           WHERE a.source = ? AND b.source = ? AND a.domain = ?
             AND a.judgment IS NOT NULL AND b.judgment IS NOT NULL
             AND a.valid = 1 AND b.valid = 1"""
        + _fallback_filter("a", "b")
        + wp_sql
        + " ORDER BY a.rct_id",
        (source_a, source_b, domain, *wp_params),
    ).fetchall()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_kappa_determinism.py -v`
Expected: 2 passed.

- [ ] **Step 5: Prove reproducibility end-to-end on the real DB**

```bash
uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py
cp studies/eisele_metzger_replication/phase6_forest_data.csv /tmp/forest_run1.csv
uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py
diff /tmp/forest_run1.csv studies/eisele_metzger_replication/phase6_forest_data.csv && echo "REPRODUCIBLE"
```

Expected: `REPRODUCIBLE`, no diff output. Runtime ~3.5 s per run.

- [ ] **Step 6: Confirm no point estimate moved**

```bash
git stash && cp studies/eisele_metzger_replication/phase6_results.csv /tmp/results_before.csv && git stash pop
uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py
uv run python - <<'PY'
import csv
KEYS = ("source", "domain")
POINTS = ("n", "raw_agr", "k_unw", "k_lin", "k_quad")
def load(p):
    with open(p) as fh:
        return {tuple(r[k] for k in KEYS): r for r in csv.DictReader(fh)}
before = load("/tmp/results_before.csv")
after = load("studies/eisele_metzger_replication/phase6_results.csv")
assert before.keys() == after.keys(), "row set changed"
moved = [(k, c, before[k][c], after[k][c])
         for k in before for c in POINTS if before[k][c] != after[k][c]]
print("POINT ESTIMATES UNCHANGED" if not moved else f"MOVED: {moved[:10]}")
PY
```

Expected: `POINT ESTIMATES UNCHANGED`. **If anything moved, stop and report — do not proceed.**

- [ ] **Step 7: Run full suite and commit**

```bash
uv run pytest
git add tests/test_kappa_determinism.py \
        studies/eisele_metzger_replication/compute_phase6_kappa.py \
        studies/eisele_metzger_replication/phase6_results.csv \
        studies/eisele_metzger_replication/phase6_results.md \
        studies/eisele_metzger_replication/phase6_forest_data.csv
git commit -m "fix(study): make kappa pair loading deterministic via ORDER BY rct_id

The bootstrap resamples pairs by index, so the unordered SELECT made
ensemble CIs irreproducible between runs. Point estimates were never
affected (kappa is order-invariant) and no published number changes."
```

---

### Task 2: Emit quadratic-weighted bootstrap CIs

**Files:**
- Modify: `studies/eisele_metzger_replication/compute_phase6_kappa.py` (`KappaRow` ~line 60, `build_kappa_row` ~line 73, the three `forest_rows.append` / `rows.append` blocks, and both fieldname lists ~line 517)
- Test: `tests/test_kappa_determinism.py` (extend)

**Interfaces:**
- Consumes: `load_pairs` from Task 1.
- Produces: `phase6_forest_data.csv` with fieldnames `label, k_lin, k_quad, ci_lin_lo, ci_lin_hi, ci_quad_lo, ci_quad_hi, n, kind`. `KappaRow` gains `ci_quad_low: float` and `ci_quad_high: float`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kappa_determinism.py`:

```python
def test_build_kappa_row_reports_quadratic_ci():
    """build_kappa_row must return a quadratic-weighted CI bracketing k_quad."""
    from studies.eisele_metzger_replication.compute_phase6_kappa import build_kappa_row

    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    for i, (gold, pred) in enumerate(
        [("low", "low"), ("high", "high"), ("some_concerns", "some_concerns"),
         ("low", "high"), ("high", "low"), ("low", "low"),
         ("high", "high"), ("some_concerns", "low")]
    ):
        rct = f"RCT{i:03d}"
        conn.execute(
            "INSERT INTO benchmark_judgment VALUES (?, 'cochrane', 'overall', ?, '', 1, ?)",
            (rct, gold, gold))
        conn.execute(
            "INSERT INTO benchmark_judgment VALUES (?, 'model_x', 'overall', ?, '', 1, ?)",
            (rct, pred, pred))
    conn.commit()

    row = build_kappa_row(conn, "model_x", "overall", n_resamples=100)

    assert row is not None
    assert row.ci_quad_low <= row.kappa_quad <= row.ci_quad_high
    assert (row.ci_quad_low, row.ci_quad_high) != (row.ci_lin_low, row.ci_lin_high)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_kappa_determinism.py::test_build_kappa_row_reports_quadratic_ci -v`
Expected: FAIL — `AttributeError: 'KappaRow' object has no attribute 'ci_quad_low'`.

- [ ] **Step 3: Extend `KappaRow` and `build_kappa_row`**

```python
@dataclass
class KappaRow:
    source: str
    domain: str
    n: int
    raw_agreement: float
    kappa_unw: float
    kappa_lin: float
    kappa_quad: float
    ci_lin_low: float
    ci_lin_high: float
    ci_quad_low: float
    ci_quad_high: float


def build_kappa_row(conn: sqlite3.Connection, source: str, domain: str,
                    reference: str = "cochrane",
                    n_resamples: int = 500) -> KappaRow | None:
    """One κ row (all three weightings) with bootstrap CIs at linear and
    quadratic weighting.

    Quadratic weighting is the manuscript's primary metric, so it needs its
    own interval — the linear CI does not bracket κ_quad. Each
    ``bootstrap_kappa_ci`` call reseeds its own RNG, so the two calls are
    independent and adding the quadratic one cannot perturb the linear one.
    """
    pairs = load_pairs(conn, reference, source, domain)
    if not pairs:
        return None
    lo, hi = bootstrap_kappa_ci(pairs, "linear", n_resamples=n_resamples)
    q_lo, q_hi = bootstrap_kappa_ci(pairs, "quadratic", n_resamples=n_resamples)
    return KappaRow(
        source=source,
        domain=domain,
        n=len(pairs),
        raw_agreement=raw_agreement(pairs),
        kappa_unw=cohen_kappa(pairs, "none"),
        kappa_lin=cohen_kappa(pairs, "linear"),
        kappa_quad=cohen_kappa(pairs, "quadratic"),
        ci_lin_low=lo,
        ci_lin_high=hi,
        ci_quad_low=q_lo,
        ci_quad_high=q_hi,
    )
```

- [ ] **Step 4: Thread the new fields through the writers**

In `write_results`, add `"ci_quad_lo": r.ci_quad_low, "ci_quad_hi": r.ci_quad_high` to **every** `rows.append({...})` and `forest_rows.append({...})` dict built from a `KappaRow` (the single-pass block ~line 412, the ensemble block ~line 480, and their `forest_rows` counterparts). For the run-to-run block (~line 452) and the reference-marker block (~line 506), which have no `KappaRow`, add `"ci_quad_lo": None, "ci_quad_hi": None` alongside the existing `ci_lin_*` Nones.

Then extend both fieldname lists:

```python
    fieldnames = ["source", "model", "protocol", "pass", "kind", "domain",
                  "n", "raw_agr", "k_unw", "k_lin", "k_quad",
                  "ci_lin_lo", "ci_lin_hi", "ci_quad_lo", "ci_quad_hi"]
    ...
    forest_fields = ["label", "k_lin", "k_quad", "ci_lin_lo", "ci_lin_hi",
                     "ci_quad_lo", "ci_quad_hi", "n", "kind"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_kappa_determinism.py -v`
Expected: 3 passed.

- [ ] **Step 6: Regenerate and prove the pre-existing columns are byte-identical**

```bash
cp studies/eisele_metzger_replication/phase6_results.csv /tmp/results_task1.csv
cp studies/eisele_metzger_replication/phase6_forest_data.csv /tmp/forest_task1.csv
uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py
uv run python - <<'PY'
import csv
def load(p, keys):
    with open(p) as fh:
        return {tuple(r[k] for k in keys): r for r in csv.DictReader(fh)}
for path_before, path_after, keys in [
    ("/tmp/results_task1.csv",
     "studies/eisele_metzger_replication/phase6_results.csv", ("source", "domain")),
    ("/tmp/forest_task1.csv",
     "studies/eisele_metzger_replication/phase6_forest_data.csv", ("label",)),
]:
    before, after = load(path_before, keys), load(path_after, keys)
    assert before.keys() == after.keys(), f"row set changed in {path_after}"
    cols = list(next(iter(before.values())).keys())
    moved = [(k, c) for k in before for c in cols if before[k][c] != after[k][c]]
    print(f"{path_after}: {'ALL PRE-EXISTING COLUMNS IDENTICAL' if not moved else f'MOVED: {moved[:10]}'}")
PY
```

Expected: `ALL PRE-EXISTING COLUMNS IDENTICAL` for both files. **If anything moved, stop and report.**

- [ ] **Step 7: Verify the sensitivity mode still composes**

```bash
uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py --exclude-fallback
head -1 studies/eisele_metzger_replication/phase6_forest_data.strict.csv
```

Expected: header includes `ci_quad_lo,ci_quad_hi`. Then restore the primary outputs:
`uv run python studies/eisele_metzger_replication/compute_phase6_kappa.py`

- [ ] **Step 8: Run full suite and commit**

```bash
uv run pytest
git add tests/test_kappa_determinism.py \
        studies/eisele_metzger_replication/compute_phase6_kappa.py \
        studies/eisele_metzger_replication/phase6_results.csv \
        studies/eisele_metzger_replication/phase6_results.md \
        studies/eisele_metzger_replication/phase6_forest_data.csv
git commit -m "feat(study): emit quadratic-weighted bootstrap CIs in phase-6 kappa

kappa_quad is the manuscript's primary metric but only linear CIs were
reported, so Figure 1 had no valid error bars. Purely additive: all
pre-existing columns verified byte-identical after regeneration."
```

---

### Task 3: Forest-plot data layer (matplotlib-free)

**Files:**
- Create: `studies/eisele_metzger_replication/figures/__init__.py`
- Create: `studies/eisele_metzger_replication/figures/forest_data.py`
- Test: `tests/test_forest_data.py`

**Interfaces:**
- Consumes: the CSV schema produced by Task 2.
- Produces:
  - `ForestPoint` dataclass with fields `label: str`, `model: str | None`, `protocol: str | None`, `pass_id: str | None`, `k_quad: float`, `ci_lo: float | None`, `ci_hi: float | None`, `kind: str`, `n: int | None`
  - `MODEL_DISPLAY_ORDER: tuple[str, ...]`
  - `parse_label(label: str) -> tuple[str | None, str | None, str | None]`
  - `load_forest_points(csv_path: Path) -> list[ForestPoint]`
  - `split_references(points) -> tuple[list[ForestPoint], list[ForestPoint]]`
  - `order_for_plot(points: list[ForestPoint]) -> list[ForestPoint]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_forest_data.py`:

```python
"""Parsing and ordering for the Figure 1 forest plot.

Deliberately free of matplotlib so it runs in CI without the optional
`figures` dependency group.
"""
from pathlib import Path

import pytest

from studies.eisele_metzger_replication.figures.forest_data import (
    ForestPoint,
    MODEL_DISPLAY_ORDER,
    load_forest_points,
    order_for_plot,
    parse_label,
    split_references,
)

_CSV = """label,k_lin,k_quad,ci_lin_lo,ci_lin_hi,ci_quad_lo,ci_quad_hi,n,kind
"gpt-oss 20B (fulltext, pass 1)",0.26,0.32,0.12,0.40,0.15,0.45,78,single_pass
"gpt-oss 20B (fulltext, ensemble)",0.18,0.22,0.06,0.31,0.08,0.35,78,ensemble
"gpt-oss 20B (abstract, pass 2)",0.004,0.041,-0.09,0.10,-0.10,0.12,78,single_pass
"Claude Sonnet 4.6 (fulltext, pass 2)",0.21,0.30,0.06,0.34,0.09,0.38,78,single_pass
"EM Claude 2 (published, single pass)",,0.22,,,,,,reference
"Minozzi 2021 — trained humans, with ID",,0.42,,,,,,reference_human
"""


@pytest.fixture()
def csv_path(tmp_path: Path) -> Path:
    p = tmp_path / "forest.csv"
    p.write_text(_CSV, encoding="utf-8")
    return p


@pytest.mark.parametrize("label,expected", [
    ("gpt-oss 20B (fulltext, pass 1)", ("gpt-oss 20B", "fulltext", "pass 1")),
    ("Gemma 4 26B-A4B (abstract, pass 3)", ("Gemma 4 26B-A4B", "abstract", "pass 3")),
    ("Qwen 3.6 35B-A3B (fulltext, ensemble)", ("Qwen 3.6 35B-A3B", "fulltext", "ensemble")),
    ("Claude Sonnet 4.6 (abstract, pass 2)", ("Claude Sonnet 4.6", "abstract", "pass 2")),
])
def test_parse_label_decomposes_model_protocol_pass(label, expected):
    assert parse_label(label) == expected


def test_parse_label_returns_none_triple_for_reference_rows():
    assert parse_label("EM Claude 2 (published, single pass)") == (None, None, None)


def test_parse_label_rejects_unparseable_label():
    with pytest.raises(ValueError, match="Unrecognised"):
        parse_label("Totally Unknown Model (fulltext, pass 1)")


def test_load_forest_points_reads_quadratic_cis(csv_path):
    points = load_forest_points(csv_path)
    single = next(p for p in points if p.pass_id == "pass 1")
    assert single.k_quad == pytest.approx(0.32)
    assert (single.ci_lo, single.ci_hi) == (pytest.approx(0.15), pytest.approx(0.45))


def test_reference_rows_carry_no_ci(csv_path):
    _, refs = split_references(load_forest_points(csv_path))
    em = next(r for r in refs if r.label.startswith("EM Claude 2"))
    assert em.k_quad == pytest.approx(0.22)
    assert em.ci_lo is None and em.ci_hi is None


def test_reference_row_without_k_quad_is_a_hard_error(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(
        "label,k_lin,k_quad,ci_lin_lo,ci_lin_hi,ci_quad_lo,ci_quad_hi,n,kind\n"
        '"EM Claude 2 (published, single pass)",,,,,,,,reference\n',
        encoding="utf-8")
    with pytest.raises(ValueError, match="missing k_quad"):
        load_forest_points(p)


def test_missing_csv_names_the_regeneration_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="compute_phase6_kappa"):
        load_forest_points(tmp_path / "absent.csv")


def test_order_puts_ensemble_last_within_each_block(csv_path):
    plotted, _ = split_references(load_forest_points(csv_path))
    ordered = order_for_plot(plotted)
    gpt_ft = [p for p in ordered
              if p.model == "gpt-oss 20B" and p.protocol == "fulltext"]
    assert gpt_ft[-1].kind == "ensemble"


def test_order_follows_model_display_order_then_protocol(csv_path):
    plotted, _ = split_references(load_forest_points(csv_path))
    ordered = order_for_plot(plotted)
    models = [p.model for p in ordered]
    assert models.index("gpt-oss 20B") < models.index("Claude Sonnet 4.6")
    assert MODEL_DISPLAY_ORDER[0] == "gpt-oss 20B"
    gpt = [p.protocol for p in ordered if p.model == "gpt-oss 20B"]
    assert gpt.index("abstract") < gpt.index("fulltext")


def test_order_is_stable(csv_path):
    plotted, _ = split_references(load_forest_points(csv_path))
    assert [p.label for p in order_for_plot(plotted)] == \
           [p.label for p in order_for_plot(plotted)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_forest_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'studies.eisele_metzger_replication.figures'`.

- [ ] **Step 3: Create the package marker**

Create `studies/eisele_metzger_replication/figures/__init__.py`:

```python
"""Manuscript figures for the Eisele-Metzger replication study."""
```

- [ ] **Step 4: Implement `forest_data.py`**

Create `studies/eisele_metzger_replication/figures/forest_data.py`:

```python
"""Parse and order `phase6_forest_data.csv` for the Figure 1 forest plot.

Deliberately free of any plotting dependency: all the decision-shaped logic
lives here so it can be unit-tested without matplotlib installed. The
rendering layer (`figure1_forest.py`) consumes what this module produces.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

# Fixed top-to-bottom model order: descending best-pass fulltext κ_quad, the
# order the numbers appear in the manuscript prose. Deliberately a constant
# rather than derived from the data, so the figure's row order cannot silently
# rearrange underneath the prose if a κ shifts.
MODEL_DISPLAY_ORDER: tuple[str, ...] = (
    "gpt-oss 20B",
    "Claude Sonnet 4.6",
    "Qwen 3.6 35B-A3B",
    "Gemma 4 26B-A4B",
)

# Abstract above fulltext within each model block.
PROTOCOL_DISPLAY_ORDER: tuple[str, ...] = ("abstract", "fulltext")

# Ensemble sorts after the numbered passes within a model × protocol block.
_ENSEMBLE_SORT_KEY = 99

REFERENCE_KINDS = frozenset({"reference", "reference_human"})

_LABEL_RE = re.compile(r"^(?P<model>.+?) \((?P<protocol>abstract|fulltext), "
                       r"(?P<pass_id>pass \d+|ensemble)\)$")

_REGENERATE_HINT = (
    "Regenerate it with: uv run python "
    "studies/eisele_metzger_replication/compute_phase6_kappa.py"
)


@dataclass(frozen=True)
class ForestPoint:
    """One plotted point: a model×protocol×pass estimate, or a reference marker."""

    label: str
    model: str | None
    protocol: str | None
    pass_id: str | None
    k_quad: float
    ci_lo: float | None
    ci_hi: float | None
    kind: str
    n: int | None


def parse_label(label: str) -> tuple[str | None, str | None, str | None]:
    """Decompose a forest CSV label into (model, protocol, pass_id).

    Reference rows carry no model/protocol parenthetical and yield an
    all-None triple. A label that looks like a model row but names a model
    outside `MODEL_DISPLAY_ORDER` is an error rather than a silent skip: it
    means the evaluation matrix changed and the figure needs updating.
    """
    match = _LABEL_RE.match(label)
    if match is None:
        return (None, None, None)
    model = match.group("model")
    if model not in MODEL_DISPLAY_ORDER:
        raise ValueError(
            f"Unrecognised model {model!r} in forest label {label!r}. "
            f"Known models: {', '.join(MODEL_DISPLAY_ORDER)}. "
            "Add it to MODEL_DISPLAY_ORDER if the evaluation matrix grew."
        )
    return (model, match.group("protocol"), match.group("pass_id"))


def _optional_float(raw: str | None) -> float | None:
    """Parse a CSV cell that is legitimately empty for some row kinds."""
    if raw is None or raw == "":
        return None
    return float(raw)


def load_forest_points(csv_path: Path) -> list[ForestPoint]:
    """Read the forest CSV into typed points, erroring loudly on bad rows."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Forest data not found at {csv_path}. {_REGENERATE_HINT}"
        )
    points: list[ForestPoint] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "ci_quad_lo" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} has no ci_quad_lo column, so κ_quad error bars "
                f"cannot be drawn. {_REGENERATE_HINT}"
            )
        for row in reader:
            label = row["label"]
            k_quad = _optional_float(row["k_quad"])
            if k_quad is None:
                raise ValueError(
                    f"Forest row {label!r} is missing k_quad; every plotted "
                    "point needs a quadratic-weighted κ."
                )
            model, protocol, pass_id = parse_label(label)
            n_raw = row.get("n")
            points.append(ForestPoint(
                label=label,
                model=model,
                protocol=protocol,
                pass_id=pass_id,
                k_quad=k_quad,
                ci_lo=_optional_float(row["ci_quad_lo"]),
                ci_hi=_optional_float(row["ci_quad_hi"]),
                kind=row["kind"],
                n=int(n_raw) if n_raw else None,
            ))
    return points


def split_references(points: list[ForestPoint]
                     ) -> tuple[list[ForestPoint], list[ForestPoint]]:
    """Partition into (plotted model rows, reference markers)."""
    plotted = [p for p in points if p.kind not in REFERENCE_KINDS]
    references = [p for p in points if p.kind in REFERENCE_KINDS]
    return plotted, references


def _sort_key(point: ForestPoint) -> tuple[int, int, int]:
    model_rank = MODEL_DISPLAY_ORDER.index(point.model) if point.model else 0
    protocol_rank = (PROTOCOL_DISPLAY_ORDER.index(point.protocol)
                     if point.protocol else 0)
    if point.pass_id == "ensemble":
        pass_rank = _ENSEMBLE_SORT_KEY
    elif point.pass_id:
        pass_rank = int(point.pass_id.removeprefix("pass "))
    else:
        pass_rank = 0
    return (model_rank, protocol_rank, pass_rank)


def order_for_plot(points: list[ForestPoint]) -> list[ForestPoint]:
    """Order rows model → protocol → pass, ensemble last within each block."""
    return sorted(points, key=_sort_key)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_forest_data.py -v`
Expected: all passed (13 test cases including the 4 parametrised label cases).

- [ ] **Step 6: Run full suite and commit**

```bash
uv run pytest
git add studies/eisele_metzger_replication/figures/__init__.py \
        studies/eisele_metzger_replication/figures/forest_data.py \
        tests/test_forest_data.py
git commit -m "feat(study): add matplotlib-free forest-plot data layer for Figure 1"
```

---

### Task 4: Render the figure

**Files:**
- Create: `studies/eisele_metzger_replication/figures/figure1_forest.py`
- Modify: `pyproject.toml` (add `figures` dependency group)
- Test: `tests/test_figure1_forest.py`

**Interfaces:**
- Consumes: `ForestPoint`, `MODEL_DISPLAY_ORDER`, `load_forest_points`, `order_for_plot`, `split_references` from Task 3.
- Produces: `render_figure(points, references, output_dir: Path) -> list[Path]` returning the written file paths; `main() -> None` CLI entry point.

- [ ] **Step 1: Add the optional dependency group**

```bash
uv add --group figures matplotlib
```

Verify `pyproject.toml` now contains a `figures` entry under `[dependency-groups]` alongside `dev` and `mlx`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_figure1_forest.py`:

```python
"""Smoke test for Figure 1 rendering.

Skipped entirely when the optional `figures` group is not installed, so CI
stays green on a lean install.
"""
from pathlib import Path

import pytest

pytest.importorskip("matplotlib", reason="requires the optional 'figures' group")

from studies.eisele_metzger_replication.figures.figure1_forest import render_figure
from studies.eisele_metzger_replication.figures.forest_data import (
    load_forest_points, order_for_plot, split_references,
)

_CSV = """label,k_lin,k_quad,ci_lin_lo,ci_lin_hi,ci_quad_lo,ci_quad_hi,n,kind
"gpt-oss 20B (abstract, pass 1)",0.03,0.01,-0.06,0.14,-0.07,0.16,78,single_pass
"gpt-oss 20B (abstract, ensemble)",0.02,-0.03,-0.05,0.12,-0.06,0.13,78,ensemble
"gpt-oss 20B (fulltext, pass 1)",0.26,0.32,0.12,0.40,0.15,0.45,78,single_pass
"Claude Sonnet 4.6 (fulltext, pass 2)",0.21,0.30,0.06,0.34,0.09,0.38,78,single_pass
"EM Claude 2 (published, single pass)",,0.22,,,,,,reference
"Minozzi 2021 — trained humans, with ID",,0.42,,,,,,reference_human
"""


def test_render_writes_pdf_and_png(tmp_path: Path):
    csv_path = tmp_path / "forest.csv"
    csv_path.write_text(_CSV, encoding="utf-8")
    plotted, references = split_references(load_forest_points(csv_path))

    written = render_figure(order_for_plot(plotted), references, tmp_path)

    suffixes = sorted(p.suffix for p in written)
    assert suffixes == [".pdf", ".png"]
    for path in written:
        assert path.exists() and path.stat().st_size > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_figure1_forest.py -v`
Expected: FAIL — `ModuleNotFoundError: ...figure1_forest`.

- [ ] **Step 4: Implement `figure1_forest.py`**

Create `studies/eisele_metzger_replication/figures/figure1_forest.py`:

```python
"""Render Figure 1 of the primary preprint: κ_quad vs Cochrane, forest plot.

Rendering only — all parsing and ordering lives in `forest_data.py`, which
carries no plotting dependency and holds the unit-tested logic.

Run:
    uv run python -m studies.eisele_metzger_replication.figures.figure1_forest
"""
from __future__ import annotations

import logging
from pathlib import Path

from .forest_data import (
    ForestPoint,
    MODEL_DISPLAY_ORDER,
    load_forest_points,
    order_for_plot,
    split_references,
)

logger = logging.getLogger(__name__)

STUDY_DIR = Path(__file__).resolve().parent.parent
FOREST_CSV = STUDY_DIR / "phase6_forest_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "figure1_forest"

# --- Figure constants (no magic numbers inline) ------------------------
FIG_WIDTH_IN = 7.5
ROW_HEIGHT_IN = 0.22
FIG_MIN_HEIGHT_IN = 4.0
DPI = 300
X_LIMITS = (-0.15, 0.50)
X_TICK_STEP = 0.1

SINGLE_PASS_MARKER = "o"
ENSEMBLE_MARKER = "D"
MARKER_SIZE = 22
POINT_COLOUR = "#1f2933"
ENSEMBLE_COLOUR = "#b04a1a"
ERRORBAR_LINEWIDTH = 1.0
ERRORBAR_CAPSIZE = 1.8

BAND_COLOUR = "#f2f4f6"
ZERO_LINE_COLOUR = "#9aa5b1"

# Reference rules, keyed by the CSV label prefix each matches.
REFERENCE_STYLES = {
    "EM Claude 2": {"colour": "#c0392b", "linestyle": "--",
                    "short": "Eisele-Metzger 2025\n(Claude 2)"},
    "Minozzi 2020": {"colour": "#2c7fb8", "linestyle": ":",
                     "short": "Minozzi 2020\n(humans, no ID)"},
    "Minozzi 2021": {"colour": "#2c7fb8", "linestyle": "-.",
                     "short": "Minozzi 2021\n(humans, with ID)"},
}
REFERENCE_LINEWIDTH = 1.1
REFERENCE_LABEL_FONTSIZE = 6.0

AXIS_LABEL_FONTSIZE = 9.0
TICK_FONTSIZE = 7.5
ROW_LABEL_FONTSIZE = 7.0
LEGEND_FONTSIZE = 7.0


def _row_label(point: ForestPoint) -> str:
    """Right-aligned row label, e.g. 'fulltext · pass 2'."""
    return f"{point.protocol} · {point.pass_id}"


def _reference_style(label: str) -> dict | None:
    for prefix, style in REFERENCE_STYLES.items():
        if label.startswith(prefix):
            return style
    return None


def render_figure(points: list[ForestPoint], references: list[ForestPoint],
                  output_dir: Path) -> list[Path]:
    """Draw the forest plot and write PDF + PNG. Returns the written paths.

    ``points`` must already be ordered by `order_for_plot`; this function does
    not reorder, so the caller controls row layout.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not points:
        raise ValueError("No plottable forest points — refusing to draw an "
                         "empty figure.")

    n_rows = len(points)
    height = max(FIG_MIN_HEIGHT_IN, n_rows * ROW_HEIGHT_IN + 1.6)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, height))

    # Rows run top to bottom, so invert the y positions.
    y_positions = list(range(n_rows - 1, -1, -1))

    # Alternating band shading per model block.
    for index, model in enumerate(MODEL_DISPLAY_ORDER):
        if index % 2:
            continue
        ys = [y for y, p in zip(y_positions, points) if p.model == model]
        if ys:
            ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, color=BAND_COLOUR,
                       zorder=0, linewidth=0)

    ax.axvline(0.0, color=ZERO_LINE_COLOUR, linewidth=0.8, zorder=1)

    for ref in references:
        style = _reference_style(ref.label)
        if style is None:
            logger.warning("Reference row %r has no configured style; "
                           "drawing it in the default colour.", ref.label)
            style = {"colour": ZERO_LINE_COLOUR, "linestyle": "--",
                     "short": ref.label}
        ax.axvline(ref.k_quad, color=style["colour"],
                   linestyle=style["linestyle"],
                   linewidth=REFERENCE_LINEWIDTH, zorder=2)
        ax.annotate(f"{style['short']}\nκ={ref.k_quad:.2f}",
                    xy=(ref.k_quad, n_rows - 0.3), xycoords=("data", "data"),
                    ha="center", va="bottom", fontsize=REFERENCE_LABEL_FONTSIZE,
                    color=style["colour"])

    for y, point in zip(y_positions, points):
        is_ensemble = point.kind == "ensemble"
        colour = ENSEMBLE_COLOUR if is_ensemble else POINT_COLOUR
        if point.ci_lo is not None and point.ci_hi is not None:
            ax.plot([point.ci_lo, point.ci_hi], [y, y], color=colour,
                    linewidth=ERRORBAR_LINEWIDTH, zorder=3,
                    solid_capstyle="butt")
            for bound in (point.ci_lo, point.ci_hi):
                ax.plot([bound, bound], [y - ERRORBAR_CAPSIZE / 10,
                                         y + ERRORBAR_CAPSIZE / 10],
                        color=colour, linewidth=ERRORBAR_LINEWIDTH, zorder=3)
        ax.scatter([point.k_quad], [y], s=MARKER_SIZE,
                   marker=ENSEMBLE_MARKER if is_ensemble else SINGLE_PASS_MARKER,
                   facecolor=colour if is_ensemble else "white",
                   edgecolor=colour, linewidths=0.9, zorder=4)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([_row_label(p) for p in points],
                       fontsize=ROW_LABEL_FONTSIZE)
    ax.set_ylim(-0.8, n_rows + 0.8)

    # Model names as a second, outer y-axis annotation.
    for model in MODEL_DISPLAY_ORDER:
        ys = [y for y, p in zip(y_positions, points) if p.model == model]
        if not ys:
            continue
        ax.annotate(model, xy=(0.0, sum(ys) / len(ys)),
                    xycoords=("axes fraction", "data"),
                    xytext=(-72, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=ROW_LABEL_FONTSIZE,
                    fontweight="bold", annotation_clip=False)

    ax.set_xlim(*X_LIMITS)
    ax.set_xticks([round(X_LIMITS[0] + i * X_TICK_STEP, 2)
                   for i in range(int((X_LIMITS[1] - X_LIMITS[0])
                                      / X_TICK_STEP) + 1)])
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel("Cohen's κ (quadratic weights) vs Cochrane RoB 2, overall "
                  "judgement", fontsize=AXIS_LABEL_FONTSIZE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    handles = [
        plt.Line2D([], [], marker=SINGLE_PASS_MARKER, linestyle="none",
                   markerfacecolor="white", markeredgecolor=POINT_COLOUR,
                   label="single pass"),
        plt.Line2D([], [], marker=ENSEMBLE_MARKER, linestyle="none",
                   markerfacecolor=ENSEMBLE_COLOUR,
                   markeredgecolor=ENSEMBLE_COLOUR,
                   label="ensemble of 3 (majority vote)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=LEGEND_FONTSIZE,
              frameon=False)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in (".pdf", ".png"):
        path = output_dir / f"{OUTPUT_STEM}{suffix}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    """CLI entry point: read the forest CSV, render, report what was written."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to render Figure 1. Install the optional "
            "group with: uv sync --group figures"
        ) from exc

    points = load_forest_points(FOREST_CSV)
    plotted, references = split_references(points)
    written = render_figure(order_for_plot(plotted), references, OUTPUT_DIR)
    for path in written:
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_figure1_forest.py -v`
Expected: 1 passed.

- [ ] **Step 6: Render the real figure and inspect it**

```bash
uv run python -m studies.eisele_metzger_replication.figures.figure1_forest
ls -la studies/eisele_metzger_replication/figures/figure1_forest.*
```

Expected: both files written, non-zero size. **Open the PNG and check by eye:** 32 rows, four model blocks in the fixed order, ensembles as filled diamonds at the foot of each block, three labelled reference rules, no clipped text, no overlapping labels. Adjust `FIG_WIDTH_IN` / `ROW_HEIGHT_IN` / offsets if anything collides.

- [ ] **Step 7: Run full suite and commit**

```bash
uv run pytest
git add pyproject.toml uv.lock \
        studies/eisele_metzger_replication/figures/figure1_forest.py \
        studies/eisele_metzger_replication/figures/figure1_forest.pdf \
        studies/eisele_metzger_replication/figures/figure1_forest.png \
        tests/test_figure1_forest.py
git commit -m "feat(study): render Figure 1 forest plot of kappa_quad vs Cochrane

Adds matplotlib as an optional 'figures' dependency group so runtime
installs and CI stay lean; the smoke test skips without it."
```

---

### Task 5: Wire Figure 1 into the primary draft and update session docs

**Files:**
- Modify: `docs/papers/drafts/20260501_medrxiv_harness_vs_naive_rob2_v1.md` (§3.2, starting line 121)
- Modify: `HANDOVER.md`
- Modify: `ROADMAP.md`

**Interfaces:**
- Consumes: the rendered `figure1_forest.pdf` / `.png` from Task 4.
- Produces: no code interface — documentation only.

- [ ] **Step 1: Read §3.2 to find the insertion point**

```bash
sed -n '121,168p' docs/papers/drafts/20260501_medrxiv_harness_vs_naive_rob2_v1.md
```

Identify the sentence that states the best-pass fulltext κ_quad spread across the four models. The in-text citation attaches there.

- [ ] **Step 2: Add the in-text reference**

Append ` (Figure 1)` to that sentence. **Additive only** — do not reword any existing prose, so the owner's pending read-through is not invalidated.

- [ ] **Step 3: Add the caption block at the end of §3.2**

Insert immediately before the `### 3.3` heading:

```markdown
![Figure 1](../../../studies/eisele_metzger_replication/figures/figure1_forest.png)

**Figure 1. Agreement with Cochrane RoB 2 overall judgements (κ_quad) for all
four models, both protocols, and all three passes.** Points are single passes
(open circles) and the per-domain ensemble-of-3 majority vote (filled diamonds);
horizontal bars are percentile bootstrap 95% confidence intervals (500
resamples) at quadratic weighting. Vertical rules mark Eisele-Metzger 2025's
published Claude 2 result (κ = 0.22) and two human-reliability reference points
from Minozzi et al. (2020, κ = 0.16 without item descriptions; 2021, κ = 0.42
with them). n = 78 RCTs — the pre-registered primary corpus, excluding the 13
wrong-paper acquisitions. Every model's fulltext intervals overlap Eisele-Metzger's
point estimate and all four sit below the with-descriptions human benchmark.
```

- [ ] **Step 4: Verify the numbers in the caption against the CSV**

```bash
grep -E "EM Claude 2|Minozzi" studies/eisele_metzger_replication/phase6_forest_data.csv
grep -c "single_pass\|ensemble" studies/eisele_metzger_replication/phase6_forest_data.csv
```

Confirm 0.22 / 0.16 / 0.42 and n = 78 match. **Confirm the caption's final sentence against the actual rendered intervals** — if any fulltext CI excludes 0.22, reword it rather than shipping a claim the figure contradicts.

- [ ] **Step 5: Update HANDOVER.md**

In "Open work #A.4", strike the forest-plot item as done. In "State of play → A", add a line recording that Figure 1 ships, that a pre-existing ensemble-CI nondeterminism was found and fixed, and that no published number moved. Update the "Last updated" header and the test count.

- [ ] **Step 6: Update ROADMAP.md**

Change the `⬜ Planned | Forest-plot figure` row to `✅ Done`, noting the κ_quad CI addition and the determinism fix. Add a row for the determinism fix under "Objective 3".

- [ ] **Step 7: Run full suite and commit**

```bash
uv run pytest
git add docs/papers/drafts/20260501_medrxiv_harness_vs_naive_rob2_v1.md \
        HANDOVER.md ROADMAP.md
git commit -m "docs: add Figure 1 to primary draft; update HANDOVER/ROADMAP"
```

- [ ] **Step 8: Push and open the PR**

```bash
git push -u origin feat/figure1-forest-plot
gh pr create --base main --title "feat(study): Figure 1 forest plot of κ_quad vs Cochrane" --body "$(cat <<'EOF'
## Summary

Builds Figure 1 for the primary preprint — a forest plot of κ_quad vs Cochrane
across four models × two protocols × three passes plus ensembles, with bootstrap
95% CIs and three literature reference lines. Closes HANDOVER Open work #A.4.

## Bug found and fixed along the way

Validating the "no published number moves" safeguard surfaced a **pre-existing
reproducibility bug**: `load_pairs` issued its SELECT with no `ORDER BY`, and
the bootstrap resamples by index, so ensemble-row CIs drifted between runs of
the unmodified script. Point estimates were never affected (Cohen's κ is
order-invariant), and no ensemble CI appears in either draft — so nothing
published was corrupted. Fixed with `ORDER BY a.rct_id` plus regression tests.

## Changes

- `fix(study)`: deterministic κ pair loading (+ 2 regression tests)
- `feat(study)`: quadratic-weighted bootstrap CIs in the phase-6 pipeline —
  κ_quad is the manuscript's primary metric but only linear CIs were reported,
  so the figure had no valid error bars
- `feat(study)`: `figures/forest_data.py` (matplotlib-free, unit-tested) and
  `figures/figure1_forest.py` (rendering); matplotlib added as an optional
  `figures` group so runtime installs and CI stay lean
- `docs`: Figure 1 + caption wired into draft §3.2; HANDOVER/ROADMAP updated

## Verification

- All pre-existing CSV columns verified byte-identical after regeneration
- κ script reproducible across consecutive runs (was not before)
- Full suite green

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Notes for the reviewer

- **Task 1 and Task 2 both regenerate `phase6_results.*` and `phase6_forest_data.csv`.** The committed diffs will show ensemble CI values changing (Task 1, the bug fix landing) and two new columns appearing (Task 2). No point estimate may change in either. Both tasks carry an explicit verification step that prints `POINT ESTIMATES UNCHANGED` / `ALL PRE-EXISTING COLUMNS IDENTICAL` — if either fails, stop.
- **The figure is committed** as PDF and PNG. It is a manuscript deliverable and the benchmark DB that regenerates it is gitignored, so an uncommitted figure would be unreproducible for anyone else.
- **Step 6 of Task 4 needs a human eye.** Automated tests can only confirm the files are non-empty; row collisions and clipped labels need looking at.

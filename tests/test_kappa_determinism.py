"""Pair loading must be order-stable so bootstrap CIs are reproducible.

The bootstrap in ``sanity_check_kappa.bootstrap_kappa_ci`` reseeds its own
RNG per call but resamples ``pairs`` by index, so a differently-ordered
pairs list yields different resamples and therefore a different CI. Cohen's
kappa itself is order-invariant (it depends only on the confusion matrix),
which is why this bug moved CIs while leaving point estimates fixed.
"""
import importlib.util
import sqlite3
import sys
from pathlib import Path

_STUDY_DIR = (
    Path(__file__).resolve().parents[1]
    / "studies" / "eisele_metzger_replication"
)


def _load(module_name: str):
    """Load a study module by file path (mirrors test_kappa_exclusions).

    ``studies/`` is not a package. Registers the module in ``sys.modules``
    before executing it: ``@dataclass`` (used by ``KappaRow``) looks the
    class's module up in ``sys.modules`` during class creation and fails if
    it is absent.
    """
    spec = importlib.util.spec_from_file_location(
        module_name, _STUDY_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_kappa = _load("compute_phase6_kappa")
load_pairs = _kappa.load_pairs

_sanity = _load("sanity_check_kappa")
load_paired = _sanity.load_paired

_SCHEMA = """
CREATE TABLE benchmark_judgment (
    rct_id TEXT, source TEXT, domain TEXT, judgment TEXT,
    rationale TEXT, valid INTEGER, raw_label TEXT
);
CREATE INDEX idx_judgment_source ON benchmark_judgment(source);
CREATE INDEX idx_judgment_domain ON benchmark_judgment(domain);
"""
# The two indexes above mirror the real benchmark_judgment table
# (dataset/eisele_metzger_benchmark.db). They are load-bearing for this test,
# not incidental: without them SQLite's query planner drives the join on the
# untouched 'cochrane' side and the rewrite below never surfaces the bug on
# this bare (unindexed) in-memory schema. With the indexes present, the
# planner searches the rewritten side by index and the physical-rewrite test
# reproduces the real drift.


_JUDGMENTS = ("low", "some_concerns", "high")


def _judgment_for(rct_ids: list[str], rct: str, offset: int) -> str:
    """A per-RCT judgment label, distinguishable across positions.

    ``load_pairs`` returns bare ``(judgment, judgment)`` tuples with no
    ``rct_id`` column, so a test seeded with the *same* judgment value for
    every RCT (as a naive fixture might) can never observe a reordering: a
    scrambled list of identical tuples is still list-equal to the original.
    Indexing into ``_JUDGMENTS`` by position (offset by column so the two
    sides of a pair differ too) makes each row's content depend on its
    identity, so `before == after` genuinely fails when rows are returned
    out of order.
    """
    i = rct_ids.index(rct)
    return _JUDGMENTS[(i + offset) % len(_JUDGMENTS)]


def _seed(conn: sqlite3.Connection, rct_ids: list[str]) -> None:
    for rct in rct_ids:
        for source, offset in (("cochrane", 0), ("model_x", 1)):
            judgment = _judgment_for(rct_ids, rct, offset)
            conn.execute(
                "INSERT INTO benchmark_judgment "
                "(rct_id, source, domain, judgment, rationale, valid, raw_label) "
                "VALUES (?, ?, 'overall', ?, '', 1, ?)",
                (rct, source, judgment, judgment),
            )
    conn.commit()


def test_load_pairs_is_ordered_by_rct_id() -> None:
    """Rows must come back sorted by rct_id regardless of insertion order."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    _seed(conn, ["RCT050", "RCT001", "RCT099", "RCT010"])

    pairs = load_pairs(conn, "cochrane", "model_x", "overall")

    assert len(pairs) == 4
    ordered = load_pairs(conn, "cochrane", "model_x", "overall")
    assert pairs == ordered


def test_load_pairs_order_survives_physical_rewrite() -> None:
    """Deleting and reinserting rows must not change the returned order.

    This reproduces the ensemble path, which rewrites its rows via
    INSERT OR REPLACE on every run and so changes physical placement.
    """
    rct_ids = ["RCT050", "RCT001", "RCT099", "RCT010"]
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    _seed(conn, rct_ids)
    before = load_pairs(conn, "cochrane", "model_x", "overall")

    conn.execute("DELETE FROM benchmark_judgment WHERE source = 'model_x'")
    for rct in ["RCT099", "RCT010", "RCT001", "RCT050"]:
        judgment = _judgment_for(rct_ids, rct, offset=1)
        conn.execute(
            "INSERT INTO benchmark_judgment "
            "(rct_id, source, domain, judgment, rationale, valid, raw_label) "
            "VALUES (?, 'model_x', 'overall', ?, '', 1, ?)",
            (rct, judgment, judgment),
        )
    conn.commit()

    assert load_pairs(conn, "cochrane", "model_x", "overall") == before


def test_load_paired_is_ordered_by_rct_id() -> None:
    """``sanity_check_kappa.load_paired`` rows must come back sorted by rct_id
    regardless of insertion order.

    Mirrors ``test_load_pairs_is_ordered_by_rct_id`` above for the sibling
    ``compute_phase6_kappa.load_pairs`` loader — this is the identical
    reproducibility bug (no ``ORDER BY``) in the Phase 4 sanity-check module.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    _seed(conn, ["RCT050", "RCT001", "RCT099", "RCT010"])

    pairs = load_paired(conn, "cochrane", "model_x", "overall")

    assert len(pairs) == 4
    ordered = load_paired(conn, "cochrane", "model_x", "overall")
    assert pairs == ordered


def test_load_paired_order_survives_physical_rewrite() -> None:
    """Deleting and reinserting rows must not change ``load_paired``'s order.

    Mirrors ``test_load_pairs_order_survives_physical_rewrite`` above: this
    reproduces the ensemble-style rewrite pattern (delete + reinsert changes
    physical row placement) that first surfaced the ordering bug in the
    sibling module. ``sanity_check_kappa.load_paired`` never rewrites rows
    itself (it is only ever run against static ``cochrane`` /
    ``em_claude2_run1`` source labels), but the missing ``ORDER BY`` is the
    same latent bug: any DB state where the two sources' physical row order
    diverges reproduces it.
    """
    rct_ids = ["RCT050", "RCT001", "RCT099", "RCT010"]
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    _seed(conn, rct_ids)
    before = load_paired(conn, "cochrane", "model_x", "overall")

    conn.execute("DELETE FROM benchmark_judgment WHERE source = 'model_x'")
    for rct in ["RCT099", "RCT010", "RCT001", "RCT050"]:
        judgment = _judgment_for(rct_ids, rct, offset=1)
        conn.execute(
            "INSERT INTO benchmark_judgment "
            "(rct_id, source, domain, judgment, rationale, valid, raw_label) "
            "VALUES (?, 'model_x', 'overall', ?, '', 1, ?)",
            (rct, judgment, judgment),
        )
    conn.commit()

    assert load_paired(conn, "cochrane", "model_x", "overall") == before


def test_build_kappa_row_reports_quadratic_ci() -> None:
    """build_kappa_row must return a quadratic-weighted CI bracketing k_quad."""
    build_kappa_row = _kappa.build_kappa_row

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

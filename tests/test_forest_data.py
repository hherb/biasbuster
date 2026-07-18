"""Parsing and ordering for the Figure 1 forest plot.

Deliberately free of matplotlib so it runs in CI without the optional
`figures` dependency group.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_FIGURES_DIR = (
    Path(__file__).resolve().parents[1]
    / "studies" / "eisele_metzger_replication" / "figures"
)


def _load(module_name: str) -> types.ModuleType:
    """Load a figures module by file path — `studies/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        module_name, _FIGURES_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_fd = _load("forest_data")
ForestPoint = _fd.ForestPoint
MODEL_DISPLAY_ORDER = _fd.MODEL_DISPLAY_ORDER
load_forest_points = _fd.load_forest_points
order_for_plot = _fd.order_for_plot
parse_label = _fd.parse_label
split_references = _fd.split_references

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
    """Write the fixture forest CSV to a temp file and return its path."""
    p = tmp_path / "forest.csv"
    p.write_text(_CSV, encoding="utf-8")
    return p


@pytest.mark.parametrize("label,expected", [
    ("gpt-oss 20B (fulltext, pass 1)", ("gpt-oss 20B", "fulltext", "pass 1")),
    ("Gemma 4 26B-A4B (abstract, pass 3)", ("Gemma 4 26B-A4B", "abstract", "pass 3")),
    ("Qwen 3.6 35B-A3B (fulltext, ensemble)", ("Qwen 3.6 35B-A3B", "fulltext", "ensemble")),
    ("Claude Sonnet 4.6 (abstract, pass 2)", ("Claude Sonnet 4.6", "abstract", "pass 2")),
])
def test_parse_label_decomposes_model_protocol_pass(
        label: str, expected: tuple[str, str, str]) -> None:
    """Model-row labels split cleanly into (model, protocol, pass_id)."""
    assert parse_label(label) == expected


def test_parse_label_returns_none_triple_for_reference_rows() -> None:
    """Reference labels carry no parenthetical and yield an all-None triple."""
    assert parse_label("EM Claude 2 (published, single pass)") == (None, None, None)


def test_parse_label_rejects_unparseable_label() -> None:
    """A model-shaped label naming an unknown model is a hard error."""
    with pytest.raises(ValueError, match="Unrecognised"):
        parse_label("Totally Unknown Model (fulltext, pass 1)")


def test_load_forest_points_reads_quadratic_cis(csv_path: Path) -> None:
    """load_forest_points reads k_quad and its quadratic-weighted CI."""
    points = load_forest_points(csv_path)
    single = next(p for p in points if p.pass_id == "pass 1")
    assert single.k_quad == pytest.approx(0.32)
    assert (single.ci_lo, single.ci_hi) == (pytest.approx(0.15), pytest.approx(0.45))


def test_reference_rows_carry_no_ci(csv_path: Path) -> None:
    """Reference rows have a k_quad but no CI, n, or model/protocol split."""
    _, refs = split_references(load_forest_points(csv_path))
    em = next(r for r in refs if r.label.startswith("EM Claude 2"))
    assert em.k_quad == pytest.approx(0.22)
    assert em.ci_lo is None and em.ci_hi is None


def test_reference_row_without_k_quad_is_a_hard_error(tmp_path: Path) -> None:
    """A reference row missing k_quad must raise rather than load as blank."""
    p = tmp_path / "bad.csv"
    p.write_text(
        "label,k_lin,k_quad,ci_lin_lo,ci_lin_hi,ci_quad_lo,ci_quad_hi,n,kind\n"
        '"EM Claude 2 (published, single pass)",,,,,,,,reference\n',
        encoding="utf-8")
    with pytest.raises(ValueError, match="missing k_quad"):
        load_forest_points(p)


def test_missing_csv_names_the_regeneration_command(tmp_path: Path) -> None:
    """A missing CSV must raise with the exact command to regenerate it."""
    with pytest.raises(FileNotFoundError, match="compute_phase6_kappa"):
        load_forest_points(tmp_path / "absent.csv")


def test_missing_ci_quad_lo_column_is_a_hard_error(tmp_path: Path) -> None:
    """A pre-Task-2 CSV lacking ci_quad_lo must fail loudly, not silently.

    Without this column every point would silently lose its κ_quad error
    bar, so the schema check must raise rather than fall back to None.
    """
    p = tmp_path / "stale_schema.csv"
    p.write_text(
        "label,k_lin,k_quad,ci_lin_lo,ci_lin_hi,n,kind\n"
        '"gpt-oss 20B (fulltext, pass 1)",0.26,0.32,0.12,0.40,78,single_pass\n',
        encoding="utf-8")
    with pytest.raises(ValueError, match="ci_quad_lo"):
        load_forest_points(p)


def test_order_puts_ensemble_last_within_each_block(csv_path: Path) -> None:
    """Within one model×protocol block, the ensemble row sorts last."""
    plotted, _ = split_references(load_forest_points(csv_path))
    ordered = order_for_plot(plotted)
    gpt_ft = [p for p in ordered
              if p.model == "gpt-oss 20B" and p.protocol == "fulltext"]
    assert gpt_ft[-1].kind == "ensemble"


def test_order_follows_model_display_order_then_protocol(csv_path: Path) -> None:
    """Rows order by MODEL_DISPLAY_ORDER, then abstract before fulltext."""
    plotted, _ = split_references(load_forest_points(csv_path))
    ordered = order_for_plot(plotted)
    models = [p.model for p in ordered]
    assert models.index("gpt-oss 20B") < models.index("Claude Sonnet 4.6")
    assert MODEL_DISPLAY_ORDER[0] == "gpt-oss 20B"
    gpt = [p.protocol for p in ordered if p.model == "gpt-oss 20B"]
    assert gpt.index("abstract") < gpt.index("fulltext")


def test_order_is_stable(csv_path: Path) -> None:
    """Repeated calls on the same input produce the same row order."""
    plotted, _ = split_references(load_forest_points(csv_path))
    assert [p.label for p in order_for_plot(plotted)] == \
           [p.label for p in order_for_plot(plotted)]

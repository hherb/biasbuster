"""Smoke test for Figure 1 rendering.

Skipped entirely when the optional `figures` group is not installed, so CI
stays green on a lean install.
"""
from pathlib import Path

import pytest

pytest.importorskip("matplotlib", reason="requires the optional 'figures' group")

import importlib.util
import sys
import types

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
load_forest_points = _fd.load_forest_points
order_for_plot = _fd.order_for_plot
split_references = _fd.split_references
render_figure = _load("figure1_forest").render_figure

_CSV = """label,k_lin,k_quad,ci_lin_lo,ci_lin_hi,ci_quad_lo,ci_quad_hi,n,kind
"gpt-oss 20B (abstract, pass 1)",0.03,0.01,-0.06,0.14,-0.07,0.16,78,single_pass
"gpt-oss 20B (abstract, ensemble)",0.02,-0.03,-0.05,0.12,-0.06,0.13,78,ensemble
"gpt-oss 20B (fulltext, pass 1)",0.26,0.32,0.12,0.40,0.15,0.45,78,single_pass
"Claude Sonnet 4.6 (fulltext, pass 2)",0.21,0.30,0.06,0.34,0.09,0.38,78,single_pass
"EM Claude 2 (published, single pass)",,0.22,,,,,,reference
"Minozzi 2021 — trained humans, with ID",,0.42,,,,,,reference_human
"""


def test_render_writes_pdf_and_png(tmp_path: Path) -> None:
    """render_figure writes non-empty PDF and PNG files for a small dataset."""
    csv_path = tmp_path / "forest.csv"
    csv_path.write_text(_CSV, encoding="utf-8")
    plotted, references = split_references(load_forest_points(csv_path))

    written = render_figure(order_for_plot(plotted), references, tmp_path)

    suffixes = sorted(p.suffix for p in written)
    assert suffixes == [".pdf", ".png"]
    for path in written:
        assert path.exists() and path.stat().st_size > 0

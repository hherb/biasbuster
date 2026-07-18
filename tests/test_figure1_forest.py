"""Smoke test for Figure 1 rendering.

Skipped entirely when the optional `figures` group is not installed, so CI
stays green on a lean install.
"""
from pathlib import Path

import pytest

pytest.importorskip("matplotlib", reason="requires the optional 'figures' group")

import matplotlib.pyplot as plt

from tests.conftest import FIGURES_DIR, load_study_module

_fd = load_study_module("forest_data", FIGURES_DIR)
load_forest_points = _fd.load_forest_points
order_for_plot = _fd.order_for_plot
split_references = _fd.split_references
_ff = load_study_module("figure1_forest", FIGURES_DIR)
render_figure = _ff.render_figure
_compute_x_ticks = _ff._compute_x_ticks
X_LIMITS = _ff.X_LIMITS
X_TICK_STEP = _ff.X_TICK_STEP
X_TICK_BOUNDS_TOLERANCE = _ff.X_TICK_BOUNDS_TOLERANCE

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


_CSV_OUT_OF_BOUNDS = """label,k_lin,k_quad,ci_lin_lo,ci_lin_hi,ci_quad_lo,ci_quad_hi,n,kind
"gpt-oss 20B (abstract, pass 1)",0.03,0.01,-0.06,0.14,-0.07,0.16,78,single_pass
"gpt-oss 20B (abstract, pass 3)",-0.05,-0.05,-0.30,0.20,-0.60,0.13,78,single_pass
"""


def test_render_raises_when_ci_exceeds_x_limits(tmp_path: Path) -> None:
    """render_figure refuses to silently clip a CI bound outside X_LIMITS.

    A point whose interval extends past the axes would otherwise render as
    a silently truncated error bar — exactly the defect this guard exists
    to prevent — so it must raise a ValueError naming the offending row
    instead of drawing it.
    """
    csv_path = tmp_path / "forest_out_of_bounds.csv"
    csv_path.write_text(_CSV_OUT_OF_BOUNDS, encoding="utf-8")
    plotted, references = split_references(load_forest_points(csv_path))

    with pytest.raises(ValueError, match="gpt-oss 20B \\(abstract, pass 3\\)"):
        render_figure(order_for_plot(plotted), references, tmp_path)


def test_compute_x_ticks_never_overshoots_upper_limit() -> None:
    """No generated tick may exceed the upper X_LIMITS bound.

    Reproduces the exact ratio that triggered the historical bug: with
    limits (-0.20, 0.55) and step 0.1, ``(0.55 - -0.20) / 0.1 == 7.5``, and
    Python's round-half-to-even rounds 7.5 up to 8, generating a 9th tick at
    0.6 — one step past the declared upper limit. The old code passed that
    tick straight to ``ax.set_xticks()``, which silently widened the axes'
    xlim to include it. The fixed helper must filter it out instead.
    """
    limits = (-0.20, 0.55)
    ticks = _compute_x_ticks(limits, X_TICK_STEP, X_TICK_BOUNDS_TOLERANCE)

    assert max(ticks) <= limits[1] + X_TICK_BOUNDS_TOLERANCE
    assert all(limits[0] - X_TICK_BOUNDS_TOLERANCE <= t
               <= limits[1] + X_TICK_BOUNDS_TOLERANCE for t in ticks)
    assert 0.6 not in ticks


def test_compute_x_ticks_shipping_constants_produce_expected_set() -> None:
    """The shipping constants' tick set is exactly nine ticks ending at 0.5.

    With the project's real constants — limits (-0.30, 0.55), step 0.1 —
    the ratio is 8.5, which round-half-to-even rounds down to 8, so the
    generated ticks stop at 0.5 and never reach the 0.55 upper bound. This
    pins the published figure's tick set: filtering must drop nothing and
    precision derivation must displace nothing.
    """
    ticks = _compute_x_ticks(X_LIMITS, X_TICK_STEP, X_TICK_BOUNDS_TOLERANCE)

    assert ticks == [round(X_LIMITS[0] + i * X_TICK_STEP, 2)
                      for i in range(9)]
    assert max(ticks) == pytest.approx(0.5)


def test_compute_x_ticks_keeps_tick_landing_exactly_on_bound() -> None:
    """A tick landing exactly on the upper limit is kept, not filtered out.

    Exercises the tolerance path with a genuinely bound-landing tick:
    limits (0.0, 0.5) at step 0.1 put the final tick exactly on the bound,
    where an exclusive or tolerance-free comparison would drop it.
    """
    ticks = _compute_x_ticks((0.0, 0.5), 0.1, X_TICK_BOUNDS_TOLERANCE)

    assert ticks == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def test_compute_x_ticks_precision_follows_step() -> None:
    """Tick rounding precision derives from the step, not a hardcoded count.

    At step 0.025 a fixed two-decimal rounding would displace ticks (0.025
    to 0.03, 0.075 to 0.08); deriving the precision from the step keeps
    every position exact.
    """
    ticks = _compute_x_ticks((0.0, 0.1), 0.025, X_TICK_BOUNDS_TOLERANCE)

    assert ticks == [0.0, 0.025, 0.05, 0.075, 0.1]


def test_render_figure_xlim_matches_x_limits_and_ticks_are_within(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After rendering, the live axes' xlim equals X_LIMITS with every tick
    inside it.

    ``ax.set_xticks()`` silently widens the axes' xlim to include any tick
    value outside the current range (verified empirically against real
    matplotlib) — that is why `render_figure` must call `set_xlim` after
    `set_xticks`. This intercepts `plt.close` to inspect the live axes
    before `render_figure` closes the figure, so the assertion exercises
    real matplotlib behaviour rather than just the pure tick helper.
    """
    captured: list = []
    real_close = plt.close

    def _capture_then_close(fig: object) -> None:
        captured.append(fig)
        real_close(fig)

    monkeypatch.setattr(plt, "close", _capture_then_close)

    csv_path = tmp_path / "forest.csv"
    csv_path.write_text(_CSV, encoding="utf-8")
    plotted, references = split_references(load_forest_points(csv_path))

    render_figure(order_for_plot(plotted), references, tmp_path)

    assert len(captured) == 1
    ax = captured[0].axes[0]
    assert ax.get_xlim() == pytest.approx(X_LIMITS)
    assert all(X_LIMITS[0] - X_TICK_BOUNDS_TOLERANCE <= tick
               <= X_LIMITS[1] + X_TICK_BOUNDS_TOLERANCE
               for tick in ax.get_xticks())

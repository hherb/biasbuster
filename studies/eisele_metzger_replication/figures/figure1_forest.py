"""Render Figure 1 of the primary preprint: κ_quad vs Cochrane, forest plot.

Rendering only — all parsing and ordering lives in `forest_data.py`, which
carries no plotting dependency and holds the unit-tested logic.

Run:
    uv run python studies/eisele_metzger_replication/figures/figure1_forest.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# `studies/` is not a Python package; siblings are imported flat off sys.path,
# matching the convention in compute_phase6_kappa.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forest_data import (  # noqa: E402
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
FIG_WIDTH_IN = 8.5
ROW_HEIGHT_IN = 0.24
FIG_MIN_HEIGHT_IN = 4.0
FIG_TOP_MARGIN_IN = 1.0
DPI = 300
X_LIMITS = (-0.20, 0.55)
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

# Left-margin reserved for the outer model-name annotations, as a fraction of
# the axes width (via subplots_adjust rather than tight_layout, which cannot
# see annotations placed outside the axes in "axes fraction" / "data" mixed
# coordinates).
LEFT_MARGIN_FRACTION = 0.22
RIGHT_MARGIN_FRACTION = 0.97
TOP_MARGIN_FRACTION = 0.90
BOTTOM_MARGIN_FRACTION = 0.08
MODEL_LABEL_OFFSET_POINTS = -85
# Reference lines stop just above the top data row (rather than running the
# full axes height) so they never cross the staggered reference labels
# drawn above them.
REFERENCE_LINE_TOP_PAD_ROWS = 0.5

# Reference rules, keyed by the CSV label prefix each matches.
REFERENCE_STYLES = {
    "EM Claude 2": {"colour": "#c0392b", "linestyle": "--",
                    "short": "Eisele-Metzger 2025 (Claude 2)"},
    "Minozzi 2020": {"colour": "#2c7fb8", "linestyle": ":",
                     "short": "Minozzi 2020 (humans, no ID)"},
    "Minozzi 2021": {"colour": "#238b45", "linestyle": "-.",
                     "short": "Minozzi 2021 (humans, with ID)"},
}
REFERENCE_LINEWIDTH = 1.1
REFERENCE_LABEL_FONTSIZE = 6.5
# Vertical stagger for reference annotations so three closely-spaced rules
# don't collide with each other above the plot area.
REFERENCE_LABEL_ROW_GAP_IN = 0.16

AXIS_LABEL_FONTSIZE = 9.0
TICK_FONTSIZE = 7.5
ROW_LABEL_FONTSIZE = 7.0
MODEL_LABEL_FONTSIZE = 7.5
LEGEND_FONTSIZE = 7.0
# The legend sits below the x-axis label rather than inside a plot corner:
# with 32 rows of varying CI width, every corner is occupied by some model's
# wide interval at one point or another, so an in-plot legend location is
# never reliably collision-free.
LEGEND_Y_AXES_FRACTION = -0.09
LEGEND_NCOL = 2


def _row_label(point: ForestPoint) -> str:
    """Right-aligned row label, e.g. 'fulltext · pass 2'."""
    return f"{point.protocol} · {point.pass_id}"


def _reference_style(label: str) -> dict | None:
    """Look up the display style for a reference row by CSV label prefix."""
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
    # Reserve headroom above the rows for the staggered reference-rule labels.
    top_headroom_rows = len(references) * (REFERENCE_LABEL_ROW_GAP_IN
                                            / ROW_HEIGHT_IN)
    height = max(FIG_MIN_HEIGHT_IN,
                 n_rows * ROW_HEIGHT_IN + FIG_TOP_MARGIN_IN
                 + top_headroom_rows * ROW_HEIGHT_IN)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, height))
    fig.subplots_adjust(left=LEFT_MARGIN_FRACTION, right=RIGHT_MARGIN_FRACTION,
                         top=TOP_MARGIN_FRACTION, bottom=BOTTOM_MARGIN_FRACTION)

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

    # Reference lines share one top endpoint, just above the topmost data
    # row, so no line ever runs through the label header zone above it.
    ref_line_top = (n_rows - 1) + REFERENCE_LINE_TOP_PAD_ROWS
    for ref_index, ref in enumerate(references):
        style = _reference_style(ref.label)
        if style is None:
            logger.warning("Reference row %r has no configured style; "
                           "drawing it in the default colour.", ref.label)
            style = {"colour": ZERO_LINE_COLOUR, "linestyle": "--",
                     "short": ref.label}
        ax.plot([ref.k_quad, ref.k_quad], [-0.8, ref_line_top],
                color=style["colour"], linestyle=style["linestyle"],
                linewidth=REFERENCE_LINEWIDTH, zorder=2)
        # Stagger each reference label at a different height above the plot
        # so labels for closely-spaced κ values don't overlap each other.
        label_y_rows = n_rows + 0.6 + ref_index * (REFERENCE_LABEL_ROW_GAP_IN
                                                     / ROW_HEIGHT_IN)
        ax.annotate(f"{style['short']}, κ={ref.k_quad:.2f}",
                    xy=(ref.k_quad, label_y_rows), xycoords=("data", "data"),
                    ha="center", va="bottom", fontsize=REFERENCE_LABEL_FONTSIZE,
                    color=style["colour"], annotation_clip=False)

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
    ax.set_ylim(-0.8, n_rows + 0.8 + top_headroom_rows)

    # Model names as a second, outer y-axis annotation, placed in the margin
    # reserved by `LEFT_MARGIN_FRACTION` via `fig.subplots_adjust` above.
    for model in MODEL_DISPLAY_ORDER:
        ys = [y for y, p in zip(y_positions, points) if p.model == model]
        if not ys:
            continue
        ax.annotate(model, xy=(0.0, sum(ys) / len(ys)),
                    xycoords=("axes fraction", "data"),
                    xytext=(MODEL_LABEL_OFFSET_POINTS, 0),
                    textcoords="offset points",
                    ha="right", va="center", fontsize=MODEL_LABEL_FONTSIZE,
                    fontweight="bold", annotation_clip=False)

    ax.set_xlim(*X_LIMITS)
    ax.set_xticks([round(X_LIMITS[0] + i * X_TICK_STEP, 2)
                   for i in range(int(round((X_LIMITS[1] - X_LIMITS[0])
                                      / X_TICK_STEP)) + 1)])
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
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, LEGEND_Y_AXES_FRACTION),
              ncol=LEGEND_NCOL, fontsize=LEGEND_FONTSIZE, frameon=False)

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

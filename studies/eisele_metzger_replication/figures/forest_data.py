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

# Every column load_forest_points reads. Validated against the CSV header up
# front so a stale or hand-edited file fails with the regeneration command
# rather than a bare KeyError partway through parsing.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "label", "k_quad", "ci_quad_lo", "ci_quad_hi", "n", "kind",
)

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

    @property
    def is_ensemble(self) -> bool:
        """True for ensemble-of-3 rows.

        The single predicate both sorting (`_sort_key`) and marker styling
        (`figure1_forest.render_figure`) key off, so a row can never sort as
        an ensemble yet render as a single pass. ``kind`` is authoritative;
        `load_forest_points` rejects rows whose label disagrees with it.
        """
        return self.kind == "ensemble"


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
        header = reader.fieldnames or []
        missing_columns = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required column(s) "
                f"{', '.join(missing_columns)}, so the forest plot cannot be "
                f"drawn faithfully. {_REGENERATE_HINT}"
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
            kind = row["kind"]
            if model is not None and (pass_id == "ensemble") != (kind == "ensemble"):
                raise ValueError(
                    f"Forest row {label!r} is internally inconsistent: "
                    f"kind={kind!r} but the label's pass field says "
                    f"{pass_id!r}. The ensemble flag drives both row order "
                    "and marker style, so the row would sort as one thing "
                    "and render as another."
                )
            if (model, protocol, pass_id) == (None, None, None) and kind not in REFERENCE_KINDS:
                raise ValueError(
                    f"Forest row {label!r} (kind={kind!r}) failed to parse as a "
                    "model row and is not a reference kind, so it would plot as "
                    "'None · None'. Check the label format against _LABEL_RE, or "
                    "add its kind to REFERENCE_KINDS if it is a legitimate "
                    "reference row."
                )
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
    """Sort key implementing model → protocol → pass, ensemble last."""
    model_rank = MODEL_DISPLAY_ORDER.index(point.model) if point.model else 0
    protocol_rank = (PROTOCOL_DISPLAY_ORDER.index(point.protocol)
                     if point.protocol else 0)
    if point.is_ensemble:
        pass_rank = _ENSEMBLE_SORT_KEY
    elif point.pass_id:
        pass_rank = int(point.pass_id.removeprefix("pass "))
    else:
        pass_rank = 0
    return (model_rank, protocol_rank, pass_rank)


def order_for_plot(points: list[ForestPoint]) -> list[ForestPoint]:
    """Order rows model → protocol → pass, ensemble last within each block."""
    return sorted(points, key=_sort_key)

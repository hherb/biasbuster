"""Manual-gate manifest generator for the OA-first RoB benchmark (Task 7).

Renders a bounded sample of ``benchmark_item`` rows as Markdown so the
owner can eyeball-verify each RoB 2 label against its source review before
the benchmark is scaled beyond the initial manual-gate batch — the
mandatory manual gate called out in the study design.

``render_manifest`` is a pure function (items in, Markdown string out); all
file/DB I/O lives in the ``__main__`` runner below.
"""
from __future__ import annotations

from studies.oa_rob_benchmark.store import BenchmarkStore

MANUAL_SAMPLE_SIZE = 20

_MISSING = "—"


def _cell(value: object) -> str:
    """Render a possibly-``None`` field as a Markdown-safe table cell.

    ROBoto2-sourced rows leave ``table_index``/``row_index``/
    ``similarity_score`` as ``None`` (structural-table provenance doesn't
    apply to signalling-question-derived rows), so every field must
    tolerate ``None`` without crashing.
    """
    if value is None or value == "":
        return _MISSING
    return str(value)


def _render_item(item: dict, index: int) -> str:
    """Render one benchmark item as a Markdown section.

    The heading embeds the raw ``trial_title``/``trial_pmid`` field values
    only (no static "Trial " literal of our own in the template) so a
    title like the test fixture's ``"Trial 0"`` appears exactly once per
    item — required for ``render_manifest``'s row-count test, which counts
    occurrences of the substring ``"Trial "`` in the rendered output.
    """
    trial_pmid = _cell(item.get("trial_pmid"))
    trial_title = _cell(item.get("trial_title"))
    lines = [
        f"## {index}. {trial_title} (PMID {trial_pmid})",
        "",
        f"- **Source review**: PMID {_cell(item.get('source_review_pmid'))} "
        f"/ PMCID {_cell(item.get('source_review_pmcid'))}",
        f"- **Table/row index**: table {_cell(item.get('table_index'))}, "
        f"row {_cell(item.get('row_index'))}",
        "- **RoB 2 tuple**: "
        f"overall={_cell(item.get('rob2_overall'))}, "
        f"D1={_cell(item.get('rob2_d1'))}, "
        f"D2={_cell(item.get('rob2_d2'))}, "
        f"D3={_cell(item.get('rob2_d3'))}, "
        f"D4={_cell(item.get('rob2_d4'))}, "
        f"D5={_cell(item.get('rob2_d5'))}",
        f"- **Resolution**: {_cell(item.get('resolution_method'))} "
        f"(similarity={_cell(item.get('similarity_score'))})",
        f"- **License**: {_cell(item.get('trial_license'))}",
        f"- **Label source**: {_cell(item.get('label_source'))}",
    ]
    return "\n".join(lines)


def render_manifest(items: list[dict], *, limit: int = MANUAL_SAMPLE_SIZE) -> str:
    """Render up to ``limit`` benchmark items as a Markdown manual-gate manifest.

    Pure: no file or database I/O. Each item dict is expected to use the
    store's column names, as returned by ``BenchmarkStore.all_items()``.
    Fields that may be ``None`` (e.g. ``table_index``, ``row_index``,
    ``similarity_score`` for ROBoto2-sourced rows) are rendered as an
    em-dash rather than raising or printing the literal string ``"None"``.
    """
    sample = items[:limit]
    header = (
        "# OA-First RoB Benchmark — Manual Gate Manifest\n\n"
        f"{len(sample)} of {len(items)} item(s) shown "
        f"(manual-gate sample size: {limit}).\n"
    )
    sections = [_render_item(item, i) for i, item in enumerate(sample, start=1)]
    return header + "\n" + "\n\n".join(sections) + "\n"


if __name__ == "__main__":
    store = BenchmarkStore("dataset/oa_rob_benchmark.db")
    manifest = render_manifest(store.all_items())
    with open("dataset/oa_rob_benchmark_manual_check.md", "w", encoding="utf-8") as fh:
        fh.write(manifest)
    print(f"Wrote dataset/oa_rob_benchmark_manual_check.md ({len(manifest)} chars)")

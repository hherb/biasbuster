"""Redistributable export for the OA-first RoB benchmark (spec §9).

Filters ``benchmark_item`` rows down to those whose license actually
permits redistribution, then whitelists a fixed set of shareable columns.
``fulltext_path`` is a local machine cache path (points into the
operator's own filesystem) and is never included — it is neither useful
nor safe to hand to a downstream consumer of the export.

Only real ``BenchmarkStore`` schema columns are ever emitted: the
whitelist tuples below are drawn directly from the columns defined in
``studies/oa_rob_benchmark/store.py``. ``fulltext_path`` is deliberately
absent from every tuple below.
"""
from __future__ import annotations

import json
from pathlib import Path

from studies.oa_rob_benchmark.store import BenchmarkStore

DEFAULT_DB_PATH = "dataset/oa_rob_benchmark.db"
DEFAULT_EXPORT_PATH = "dataset/oa_rob_benchmark_export.jsonl"

_IDENTIFIER_FIELDS = ("trial_pmid", "trial_pmcid", "trial_doi", "trial_title")
_LICENSE_FIELDS = ("trial_license", "license_redistributable")
_NC_ND_FIELDS = ("non_commercial", "no_derivatives")
_ROB2_TUPLE_FIELDS = ("rob2_overall", "rob2_d1", "rob2_d2", "rob2_d3",
                      "rob2_d4", "rob2_d5")
_PROVENANCE_FIELDS = ("label_source", "source_review_pmid",
                      "source_review_pmcid", "table_index", "row_index",
                      "resolution_method", "similarity_score",
                      "pubtype_check", "extraction_method",
                      "manual_verified", "per_outcome_variant",
                      "benchmark_version")

# Single source of truth for the redistributable whitelist: identifiers +
# license + NC/ND flags + the six-field RoB 2 tuple + provenance.
# `fulltext_path` (local cache path) is intentionally excluded.
EXPORT_FIELDS = (_IDENTIFIER_FIELDS + _LICENSE_FIELDS + _NC_ND_FIELDS
                 + _ROB2_TUPLE_FIELDS + _PROVENANCE_FIELDS)


def export_redistributable(items: list[dict]) -> list[dict]:
    """Return the redistributable subset of ``items``, whitelisted.

    Drops any item whose ``license_redistributable`` flag is falsy
    (defensive — items reaching here should already satisfy the litmus
    test, but the export must never depend on that being true). Each
    surviving item is reduced to ``EXPORT_FIELDS`` only, so no field
    outside the whitelist (e.g. a local ``fulltext_path`` cache path, or
    any bookkeeping field carried by an ingest task) can leak into the
    export.

    Pure function: takes data, returns data, no I/O.
    """
    return [
        {field: item.get(field) for field in EXPORT_FIELDS}
        for item in items
        if item.get("license_redistributable")
    ]


def main() -> None:
    """Load the benchmark DB, export the redistributable subset to JSONL."""
    store = BenchmarkStore(DEFAULT_DB_PATH)
    rows = export_redistributable(store.all_items())
    Path(DEFAULT_EXPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_EXPORT_PATH, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"exported {len(rows)} redistributable items to {DEFAULT_EXPORT_PATH}")


if __name__ == "__main__":
    main()

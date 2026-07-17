"""Litmus audit for the OA-first RoB benchmark (spec §9).

Re-checks every persisted ``benchmark_item`` row against the same
four-part inclusion litmus test (spec §4) enforced on insert by
``BenchmarkStore``. Both the store and this audit call the shared
``litmus_violations`` pure function in ``studies/oa_rob_benchmark/store.py``
so there is exactly one place the four rules are implemented — this
script catches drift (e.g. a row edited directly in the DB, or a future
schema change that bypasses ``upsert_item``) rather than re-implementing
the rules.

Usage::

    uv run python scripts/audit_oa_rob_benchmark.py
"""
from __future__ import annotations

import sys

from studies.oa_rob_benchmark.store import BenchmarkStore, litmus_violations

DEFAULT_DB_PATH = "dataset/oa_rob_benchmark.db"


def audit_benchmark(items: list[dict]) -> list[str]:
    """Re-run the litmus test over ``items``; return all violations found.

    Each violation string is prefixed with the offending item's
    ``trial_pmid`` so a flat list of strings is enough to locate the bad
    row. An empty return value means the whole cohort is clean.

    Pure function: takes data, returns data, no I/O.
    """
    violations: list[str] = []
    for item in items:
        pmid = item.get("trial_pmid")
        for v in litmus_violations(item):
            violations.append(f"{pmid}: {v}")
    return violations


def main() -> int:
    """Load the benchmark DB, audit every row, print results, set exit code."""
    store = BenchmarkStore(DEFAULT_DB_PATH)
    items = store.all_items()
    violations = audit_benchmark(items)
    if violations:
        for v in violations:
            print(v)
        return 1
    print(f"clean: {len(items)} items")
    return 0


if __name__ == "__main__":
    sys.exit(main())

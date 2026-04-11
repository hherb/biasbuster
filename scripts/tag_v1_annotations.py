#!/usr/bin/env python3
"""Rename existing v1 annotations so they don't collide with v3 re-annotation.

Before re-annotating with the new two-call pipeline, we rename existing
v1 annotations by appending ``_v1`` to their ``model_name``. This preserves
the old annotations for comparison while freeing up the original model_name
for fresh v3 annotations.

Safety:
- Only renames rows whose annotation JSON does NOT contain
  ``_annotation_mode: "two_call_v3"`` or ``"two_call_full_text_v3"``. Rows
  that are already v3 are left untouched.
- Refuses to rename if a ``<model>_v1`` row already exists for the same
  PMID (avoids overwriting a previous tagging run).
- Dry-run by default; pass ``--apply`` to actually perform the update.

Usage:
    uv run python scripts/tag_v1_annotations.py              # dry run
    uv run python scripts/tag_v1_annotations.py --apply      # commit
    uv run python scripts/tag_v1_annotations.py --apply --models deepseek
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from config import Config
from biasbuster.database import Database


V3_MODES = {"two_call_v3", "two_call_full_text_v3"}


def tag_v1(db: Database, models: list[str] | None, apply: bool) -> int:
    """Rename v1 annotations to <model>_v1.

    Args:
        db: Open database.
        models: Restrict to these model names, or None for all.
        apply: If False, only report what would change.

    Returns:
        Number of rows that would be (or were) renamed.
    """
    conn = db.conn

    # Build the list of candidate rows
    if models:
        placeholders = ",".join("?" * len(models))
        query = (
            f"SELECT pmid, model_name, annotation "
            f"FROM annotations WHERE model_name IN ({placeholders})"
        )
        rows = conn.execute(query, models).fetchall()
    else:
        rows = conn.execute(
            "SELECT pmid, model_name, annotation FROM annotations"
        ).fetchall()

    to_rename: list[tuple[str, str, str]] = []  # (pmid, old_name, new_name)
    already_v3 = Counter()
    already_tagged = Counter()
    collisions: list[tuple[str, str]] = []

    for r in rows:
        pmid = r["pmid"]
        name = r["model_name"]
        try:
            ann = json.loads(r["annotation"]) if r["annotation"] else {}
        except json.JSONDecodeError:
            ann = {}

        if name.endswith("_v1"):
            already_tagged[name] += 1
            continue

        mode = ann.get("_annotation_mode")
        if mode in V3_MODES:
            already_v3[name] += 1
            continue

        new_name = f"{name}_v1"
        # Check target slot is free
        exists = conn.execute(
            "SELECT 1 FROM annotations WHERE pmid = ? AND model_name = ?",
            (pmid, new_name),
        ).fetchone()
        if exists:
            collisions.append((pmid, new_name))
            continue

        to_rename.append((pmid, name, new_name))

    # Report
    by_transition = Counter((old, new) for _, old, new in to_rename)
    print(f"Scanned {len(rows)} annotation row(s).")
    if already_v3:
        print(f"  Already v3 (leaving alone):")
        for name, count in already_v3.most_common():
            print(f"    {name}: {count}")
    if already_tagged:
        print(f"  Already tagged with _v1 suffix (leaving alone):")
        for name, count in already_tagged.most_common():
            print(f"    {name}: {count}")
    if collisions:
        print(f"  Collisions (target slot already occupied — SKIPPED):")
        for pmid, new_name in collisions[:10]:
            print(f"    {pmid} → {new_name}")
        if len(collisions) > 10:
            print(f"    ... and {len(collisions) - 10} more")
    print(f"  Candidates to rename: {len(to_rename)}")
    for (old, new), count in by_transition.most_common():
        print(f"    {old} → {new}: {count}")

    if not to_rename:
        print("Nothing to do.")
        return 0

    if not apply:
        print("\nDRY RUN — pass --apply to perform the rename.")
        return len(to_rename)

    # Actually apply. Do it in a single transaction.
    # Note: human_reviews has FK to annotations via (pmid, model_name), so
    # we update those too. But get_annotated_pmids found 0 human_reviews in
    # this DB, so the update is defensive.
    print(f"\nApplying rename to {len(to_rename)} row(s)...")
    for pmid, old_name, new_name in to_rename:
        conn.execute(
            "UPDATE annotations SET model_name = ? WHERE pmid = ? AND model_name = ?",
            (new_name, pmid, old_name),
        )
        conn.execute(
            "UPDATE human_reviews SET model_name = ? WHERE pmid = ? AND model_name = ?",
            (new_name, pmid, old_name),
        )
    conn.commit()
    print(f"Renamed {len(to_rename)} annotation row(s) (and any matching human_reviews).")
    return len(to_rename)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tag existing v1 annotations with _v1 suffix"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply changes. Without this flag, runs as a dry run.",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of model names to tag (e.g. 'deepseek'). "
             "Default: all non-v3, non-tagged annotations.",
    )
    args = parser.parse_args()

    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    config = Config()
    db = Database(config.db_path)
    db.initialize()
    try:
        tag_v1(db, models=models, apply=args.apply)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

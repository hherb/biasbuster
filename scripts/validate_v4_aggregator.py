#!/usr/bin/env python3
"""Validate the v4 algorithmic aggregator against existing v3 extractions.

Hypothesis under test (Option B from the v4 architecture discussion):

    "If the bias-domain severity rules are moved out of the LLM prompt and
    into Python, the calibration test failures observed in v3 disappear —
    because every one of those failures was an arithmetic or boolean-logic
    bug, not a text-reasoning failure."

This script does NOT call any LLM. It pulls the existing v3 extraction
JSON blobs from the SQLite DB (the same blobs that v3 Stage 2 worked
from), feeds them to ``biasbuster.assessment.assess_extraction``, and
prints a side-by-side comparison against:

  1. Claude's full-text v3 assessment (anthropic_fulltext) — the
     ground truth used in the §3.13 calibration test
  2. Each local model's f2 v3 assessment (the cells where Round 10
     failed)
  3. The new algorithmic v4 assessment, computed from each model's
     own extraction (so we can isolate "algorithm fixes severity"
     from "extraction quality varies across models")

Test set: the 4 calibration papers from §6.2 + the Seed Health
motivating failure case = 5 papers.

Output: a printed table per paper showing
  - the v3 result (severity/probability) for each annotator
  - the v4 algorithmic result for each annotator
  - whether v4 matches Claude's headline severity
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

# Add project root to path so the script can import biasbuster + config
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from biasbuster.assessment import assess_extraction  # noqa: E402

from config import Config  # noqa: E402


CALIBRATION_PAPERS = [
    ("32382720", "low",           "rTMS depression (Cochrane 5/5 LOW)"),
    ("39777610", "low",           "tapinarof phase 3 (industry LOW)"),
    ("39905419", "some_concerns", "balneotherapy post-COVID (public)"),
    ("39691748", "high",          "lidocaine patch (Cochrane HIGH)"),
    ("41750436", "n/a",           "Seed Health synbiotic (motivating failure case)"),
]

ANNOTATOR_TAGS = [
    ("Claude GT",  "anthropic_fulltext"),
    ("120b f2",    "ollama_gpt-oss_120b_fulltext_twocall"),
    ("20b f2",     "ollama_gpt-oss_20b_fulltext_twocall"),
    ("gemma4 f2",  "ollama_gemma4_26b-a4b-it-q8_0_fulltext_twocall"),
]


def fetch_annotation(cur: sqlite3.Cursor, pmid: str, tag: str) -> dict | None:
    """Pull a single annotation JSON from the DB, or None if missing."""
    cur.execute(
        "SELECT annotation FROM annotations WHERE pmid=? AND model_name=?",
        (pmid, tag),
    )
    row = cur.fetchone()
    return json.loads(row[0]) if row else None


def fmt_severity(sev: str | None, prob: float | None) -> str:
    """Format severity/probability for table display, normalising case."""
    if sev is None:
        return "—"
    sev = sev.lower()
    if prob is None:
        return sev
    return f"{sev}/{prob:.2f}"


def main() -> int:
    db_path = Config().db_path
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Track headline-severity matches across papers
    v3_match_counts: dict[str, int] = {label: 0 for label, _ in ANNOTATOR_TAGS if label != "Claude GT"}
    v4_match_counts: dict[str, int] = {label: 0 for label, _ in ANNOTATOR_TAGS if label != "Claude GT"}
    n_papers_with_gt = 0

    for pmid, cochrane_rob, description in CALIBRATION_PAPERS:
        print()
        print("=" * 90)
        print(f"PMID {pmid} — {description}")
        print(f"Cochrane RoB: {cochrane_rob}")
        print("=" * 90)

        # Fetch all annotations
        rows: list[tuple[str, dict | None]] = []
        for label, tag in ANNOTATOR_TAGS:
            rows.append((label, fetch_annotation(cur, pmid, tag)))

        # Find Claude headline for the comparison
        claude_ann = next((a for label, a in rows if label == "Claude GT" and a is not None), None)
        claude_headline = (claude_ann or {}).get("overall_severity", "").lower() if claude_ann else None
        if claude_headline:
            n_papers_with_gt += 1

        # Print header
        print(f"\n{'annotator':12s}  {'v3 stored':22s}  {'v4 algorithmic':22s}  {'v3 match':9s}  {'v4 match':9s}")
        print("-" * 90)

        for label, ann in rows:
            if ann is None:
                print(f"{label:12s}  MISSING")
                continue

            # v3 result (as stored)
            v3_sev = (ann.get("overall_severity") or "").lower() or None
            v3_prob = ann.get("overall_bias_probability")
            v3_str = fmt_severity(v3_sev, v3_prob)

            # v4 algorithmic — recompute from this annotator's own extraction
            extraction = ann.get("extraction") or {}
            if not extraction:
                v4_str = "no extraction"
                v4_sev = None
                v4_prob = None
            else:
                v4_assessment = assess_extraction(extraction)
                v4_sev = v4_assessment["overall_severity"]
                v4_prob = v4_assessment["overall_bias_probability"]
                v4_str = fmt_severity(v4_sev, v4_prob)

            # Match indicators (vs Claude GT headline category only)
            if label == "Claude GT":
                v3_match = "(GT)"
                v4_match = "(GT)"
            elif claude_headline:
                v3_match = "✓" if v3_sev == claude_headline else "✗"
                v4_match = "✓" if v4_sev == claude_headline else "✗"
                if v3_match == "✓":
                    v3_match_counts[label] += 1
                if v4_match == "✓":
                    v4_match_counts[label] += 1
            else:
                v3_match = "?"
                v4_match = "?"

            print(f"{label:12s}  {v3_str:22s}  {v4_str:22s}  {v3_match:9s}  {v4_match:9s}")

        # Per-paper provenance for v4 (which rules fired) — show only when v4 disagrees with Claude
        for label, ann in rows:
            if ann is None or label == "Claude GT" or claude_headline is None:
                continue
            extraction = ann.get("extraction") or {}
            if not extraction:
                continue
            v4_assessment = assess_extraction(extraction)
            v4_sev = v4_assessment["overall_severity"]
            if v4_sev != claude_headline:
                print(f"\n  ⚠ v4 on {label} = {v4_sev} (Claude={claude_headline}) — domain rationales:")
                for dom, rat in v4_assessment["_provenance"]["domain_rationales"].items():
                    sev = v4_assessment["_provenance"]["domain_severities"][dom]
                    print(f"    {dom:25s} = {sev:9s} — {rat}")

    # Summary
    print()
    print("=" * 90)
    print("SUMMARY — headline severity match against Claude GT")
    print("=" * 90)
    print(f"\nn_papers with Claude GT: {n_papers_with_gt}")
    print(f"\n{'annotator':12s}  {'v3 match':10s}  {'v4 match':10s}  {'delta':10s}")
    print("-" * 60)
    for label in ["120b f2", "20b f2", "gemma4 f2"]:
        v3 = v3_match_counts[label]
        v4 = v4_match_counts[label]
        delta = v4 - v3
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        print(f"{label:12s}  {v3}/{n_papers_with_gt:<8}  {v4}/{n_papers_with_gt:<8}  {delta_str}")

    print()
    print("Interpretation:")
    print("  Each cell = how many of the 5 papers each annotator's v3/v4 result")
    print("  matched Claude's full-text headline severity exactly.")
    print()
    print("  v4 holds the EXTRACTION constant and only changes the assessment")
    print("  algorithm. So v4 - v3 = the contribution of moving the rule logic")
    print("  out of the LLM prompt and into Python. Positive delta means the")
    print("  model's extraction was correct but its in-prompt assessment was")
    print("  applying the rules wrong; negative delta means the model's in-prompt")
    print("  reasoning was actually compensating for an extraction defect.")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

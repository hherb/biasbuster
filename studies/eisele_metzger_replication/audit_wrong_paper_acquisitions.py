"""Detect wrong-paper acquisitions in the EM benchmark DB (issue #29, step 5).

Background
----------
RCT030 is a known *wrong-paper acquisition*: Phase 1 resolved the parent
Cochrane review instead of the underlying primary trial, so every model
judgement for it describes a different document than the Cochrane ground truth
it is scored against. ``exclusions.WRONG_PAPER_RCTS`` records the RCTs that must
therefore be dropped from the κ analysis. That set was populated *reactively*
(from the recovery incident), so issue #29 asks whether RCT030 is the **only**
such case. This module is the reproducible audit that answers that question.

Two independent signals
-----------------------
1. **Reference coverage** — the fraction of the acquired ``benchmark_rct.title``
   content words (the title of the PMID Phase 1 actually fetched) that also
   appear anywhere in ``em_rct_ref`` (EM 2025's own reference for the intended
   primary trial). A correctly-fetched paper's title is essentially a substring
   of its own reference, so coverage ≈ 1.0; a wrong-topic document shares few
   content words with the intended reference. Comparing against the *whole*
   reference (not a parsed-out title) is deliberate: it is robust to both
   authors-first and title-first reference formats, which a title extractor is
   not.
2. **Model-flagged mismatch** — our own assessor models frequently *notice* when
   the supplied text is the wrong document ("the source materials describe an
   entirely unrelated study on...", "is a systematic review rather than a single
   RCT", "the study protocol..."). Counting rationales that contain such phrases
   is a second, orthogonal detector.

Known limitations (documented, not bugs)
----------------------------------------
Neither signal reliably catches these, so they are called out in the report as
a standing manual-review reminder:

* **Same-platform, different-arm** swaps where the model proceeds without
  complaint — e.g. RCT093, where the RECOVERY **empagliflozin** report was
  fetched for the intended RECOVERY **aspirin** arm. The platform-trial
  boilerplate dominates the title tokens (high coverage) and the model assessed
  it as a valid RCT (no mismatch phrase).
* **Same trial, different report** — a sub-analysis or companion paper of the
  right trial (e.g. RCT009 TTM2 oxygen sub-study vs the temperature main paper;
  RCT095 STOIC mechanistic analysis vs the budesonide RCT). Trial-level domains
  (randomisation, deviations) are unaffected, but outcome-dependent domains
  (missing data, measurement, selective reporting) may diverge.

This tool flags candidates for a human read of the acquired title against
``em_rct_ref``; it is not the final authority. Final exclusion-set membership is
a research decision for the repo owner.

Usage::

    uv run python studies/eisele_metzger_replication/audit_wrong_paper_acquisitions.py
    uv run python studies/eisele_metzger_replication/audit_wrong_paper_acquisitions.py --verbose

Read-only: this module never mutates the database.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "studies/eisele_metzger_replication"))

from exclusions import WRONG_PAPER_RCTS  # noqa: E402

DEFAULT_DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"

# Content words that carry no discriminating signal between two RCT titles:
# study-design boilerplate and syntactic glue. Removing them stops two
# unrelated papers from scoring "similar" merely because both are RCTs.
TITLE_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "vs", "versus",
    "with", "to", "at", "by", "study", "trial", "randomized", "randomised",
    "controlled", "clinical", "double", "blind", "blinded", "placebo",
    "prospective", "pilot", "phase", "open", "label", "labelled", "labeled",
    "multicentre", "multicenter", "et", "al", "effect", "effects", "efficacy",
    "safety", "patients", "using", "among", "versus",
})

# Lowercased phrases our assessor models use when they detect that the supplied
# text is the wrong document. Kept deliberately specific: generic phrases like
# "no information about the randomization process" (a legitimate abstract-only
# RoB judgement) or "the protocol and trial registration are not available" (a
# legitimate Domain-5 judgement) must NOT match. The protocol phrases below all
# assert the protocol IS the assessed document ("...was pre-specified"), not
# merely that it is absent.
MISMATCH_PHRASES: tuple[str, ...] = (
    "do not contain any information about the",
    "does not contain any information about the",
    "entirely unrelated",
    "unrelated study on",
    "rather than a single rct",
    "rather than the trial",
    "the study protocol was pre-specified",
    "the study protocol was registered",
    "is a study protocol",
    "are entirely from",
    "entirely from a",
    "describe a different",
    "systematic review rather than",
    "only a systematic review",
    "source material is only a systematic review",
    "source material (abstract only) provides no information",
    "protocol was pre-specified and registered",
)

# Default suspicion thresholds. Tuned against the 2026-07-17 corpus so the known
# wrong-topic cases (RCT030/080/088/100/095) fall below LOW_COVERAGE and the
# model-flagged cases (RCT008/017/074) clear MIN_MISMATCH_HITS. See module
# docstring for what these thresholds deliberately do not catch.
LOW_COVERAGE_THRESHOLD = 0.35
MIN_MISMATCH_HITS = 2

# Coverage below this is always shown in the report (even if not auto-flagged),
# so the borderline same-trial-different-report band — which the heuristics
# deliberately do not flag — still lands in front of the human reviewer.
REVIEW_CEILING = 0.85


def title_tokens(text: str) -> set[str]:
    """Lowercased content words of a title, minus stopwords and short tokens."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in TITLE_STOPWORDS}


def reference_coverage(acquired_title: str, em_rct_ref: str) -> float:
    """Fraction of acquired-title content words present in the reference.

    ``|tokens(acquired) ∩ tokens(reference)| / |tokens(acquired)|`` in
    ``[0.0, 1.0]``. A correctly-fetched paper's title is (near) a substring of
    its own reference, so coverage ≈ 1.0; a wrong-topic document shares few
    content words. Comparing against the *whole* reference sidesteps the
    fragile authors-vs-title parsing that a title extractor needs.

    Returns ``0.0`` when the acquired title has no content words (never-fetched
    rows), so it reads as maximally dissimilar rather than dividing by zero.
    """
    ta = title_tokens(acquired_title)
    if not ta:
        return 0.0
    tr = title_tokens(em_rct_ref)
    return len(ta & tr) / len(ta)


def count_mismatch_rationales(rationales: Iterable[str]) -> int:
    """Number of rationales containing at least one wrong-document phrase."""
    hits = 0
    for rationale in rationales:
        if not rationale:
            continue
        low = rationale.lower()
        if any(phrase in low for phrase in MISMATCH_PHRASES):
            hits += 1
    return hits


def is_suspected(
    coverage: float,
    mismatch_hits: int,
    *,
    low_coverage: float = LOW_COVERAGE_THRESHOLD,
    min_mismatch: int = MIN_MISMATCH_HITS,
) -> bool:
    """True if either detector fires: low reference coverage OR model-flagged."""
    return coverage < low_coverage or mismatch_hits >= min_mismatch


@dataclass
class AuditRow:
    """One RCT's wrong-paper audit result. Ordering key is the suspicion."""

    rct_id: str
    pmid: str
    acquired_title: str
    em_rct_ref: str
    coverage: float
    mismatch_hits: int
    model_valid_rows: int
    already_excluded: bool

    @property
    def suspected(self) -> bool:
        return is_suspected(self.coverage, self.mismatch_hits)


def audit(conn: sqlite3.Connection) -> list[AuditRow]:
    """Score every fetched RCT for wrong-paper acquisition, most-suspect first.

    An RCT with no acquired title (never fetched) is skipped: it contributes no
    model judgements, so it cannot pollute κ and is not a wrong-paper case.
    """
    conn.row_factory = sqlite3.Row
    rcts = conn.execute(
        "SELECT rct_id, pmid, title, em_rct_ref FROM benchmark_rct ORDER BY rct_id"
    ).fetchall()

    results: list[AuditRow] = []
    for rct in rcts:
        acquired = (rct["title"] or "").strip()
        if not acquired:
            continue
        rationales = [
            row[0]
            for row in conn.execute(
                "SELECT rationale FROM benchmark_judgment "
                "WHERE rct_id = ? AND source LIKE '%fulltext%'",
                (rct["rct_id"],),
            ).fetchall()
        ]
        model_valid = conn.execute(
            "SELECT COUNT(*) FROM benchmark_judgment "
            "WHERE rct_id = ? AND valid = 1 AND source != 'cochrane' "
            "AND source NOT LIKE 'em_claude2_%'",
            (rct["rct_id"],),
        ).fetchone()[0]
        results.append(
            AuditRow(
                rct_id=rct["rct_id"],
                pmid=rct["pmid"] or "",
                acquired_title=acquired,
                em_rct_ref=rct["em_rct_ref"],
                coverage=reference_coverage(acquired, rct["em_rct_ref"]),
                mismatch_hits=count_mismatch_rationales(rationales),
                model_valid_rows=model_valid,
                already_excluded=rct["rct_id"] in WRONG_PAPER_RCTS,
            )
        )

    # Most suspect first: lowest coverage, then most model-flagged.
    results.sort(key=lambda r: (r.coverage, -r.mismatch_hits))
    return results


def _print_report(rows: list[AuditRow], *, verbose: bool) -> None:
    suspected = [r for r in rows if r.suspected]
    new_suspects = [r for r in suspected if not r.already_excluded]

    print(f"Audited {len(rows)} fetched RCTs. "
          f"{len(suspected)} suspected wrong-paper "
          f"({len(new_suspects)} not yet in WRONG_PAPER_RCTS).\n")
    print(f"{'susp':>4} {'excl':>4} {'covg':>5} {'flag':>4} {'rows':>5}  rct")
    print("-" * 70)
    for r in rows:
        # Show auto-flagged rows plus the borderline band (coverage below the
        # review ceiling), so same-trial-different-report cases still surface.
        if not (verbose or r.suspected or r.coverage < REVIEW_CEILING):
            continue
        print(f"{'YES' if r.suspected else '   ':>4} "
              f"{'yes' if r.already_excluded else '   ':>4} "
              f"{r.coverage:5.2f} {r.mismatch_hits:4d} {r.model_valid_rows:5d}  "
              f"{r.rct_id}")
        print(f"       ACQ: {r.acquired_title[:96]}")
        print(f"       REF: {r.em_rct_ref[:96]}")

    if new_suspects:
        ids = ", ".join(r.rct_id for r in new_suspects)
        print(f"\nSuspected wrong-paper RCTs NOT yet excluded: {ids}")
    else:
        print("\nNo new auto-flagged suspects beyond WRONG_PAPER_RCTS.")
    print("Every row above needs a human read of the acquired title vs "
          "em_rct_ref before any exclusion decision.")
    print("Heuristics do NOT auto-flag same-platform-different-arm swaps "
          "(e.g. RCT093 empagliflozin vs the intended aspirin RECOVERY arm) — "
          "check the borderline band by hand.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                        help="benchmark DB path")
    parser.add_argument("--verbose", action="store_true",
                        help="print every RCT, not just suspected ones")
    args = parser.parse_args(argv)

    if not args.db.exists():
        parser.error(f"database not found: {args.db}")

    conn = sqlite3.connect(args.db)
    try:
        rows = audit(conn)
    finally:
        conn.close()
    _print_report(rows, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Obtainability audit + surgical recovery for wrong-paper acquisitions (#29).

Context
-------
`audit_wrong_paper_acquisitions.py` found that Phase 1 fetched the *wrong
document* for a set of RCTs (RCT030 and several more). Because the `cochrane`
and `em_claude2_*` ground-truth rows are for the *correct* primary trial, our
four models were scored against a different paper than they read. Two responses
are possible per RCT:

* **recover** — re-resolve the correct primary trial, re-fetch its text, and
  re-run the four models on it (converts a wrong-paper exclusion back into a
  valid data point); or
* **exclude** — when the correct paper is not obtainable (not indexed / no OA
  full text), drop the RCT via `exclusions.WRONG_PAPER_RCTS`.

This module decides which is which, and performs the recovery surgery.

The resolution bug, fixed here
------------------------------
The original `acquire_fulltext.resolve_pmid_for` picked the first non-review
candidate, with no check that the candidate actually *is* the trial EM cited.
`resolve_correct_pmid()` below adds that missing gate: it ranks candidates by
`reference_coverage(candidate_title, em_rct_ref)` (the audit's metric) and
accepts one only if it clears `MIN_MATCH_COVERAGE`. That is exactly what would
have prevented RCT093 (fetched the RECOVERY *empagliflozin* report for the
intended *aspirin* arm) and the topic-adjacent mis-matches.

Two modes
---------
- ``report`` (default, read-only on the DB; makes network calls) — for each
  candidate RCT, re-resolve + validate the correct paper and check Europe PMC
  open-access full text, then write a recoverable/exclude report. Checkpointed:
  per-RCT results are flushed to a JSON sidecar so a killed run resumes.
- ``apply`` (mutating; **dry-run unless ``--apply``**) — for the named RCTs,
  re-fetch the correct document into the Phase-1 file layout, update the
  ``benchmark_rct`` metadata row, and delete the stale *model* judgement /
  evaluation rows (never the ``cochrane`` / ``em_claude2_*`` ground truth), so a
  subsequent `run_evaluation.py` re-populates them. A DB file backup is written
  before any mutation.

Re-assessment (the expensive model re-run) is deliberately **not** performed
here — it is slow, costs API credits, and is owner-gated. ``apply`` prints the
exact `run_evaluation.py` commands to run next; their (rct, source, domain)
existence check means only the deleted rows are recomputed.

Usage::

    uv run python studies/eisele_metzger_replication/recover_wrong_papers.py report
    uv run python studies/eisele_metzger_replication/recover_wrong_papers.py apply RCT088 RCT093            # dry-run
    uv run python studies/eisele_metzger_replication/recover_wrong_papers.py apply RCT088 RCT093 --apply    # mutate
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "studies/eisele_metzger_replication"))

from acquire_fulltext import (  # noqa: E402
    EMRow,
    FULLTEXT_DIR,
    USER_AGENT,
    europepmc_fulltext_xml,
    extract_identifiers,
    fetch_pubmed_record,
    pubmed_search,
    pubmed_summary,
    _title_keywords,
)
from audit_wrong_paper_acquisitions import reference_coverage  # noqa: E402
from exclusions import RECOVERY_NOTE_MARKER, WRONG_PAPER_RCTS  # noqa: E402

DEFAULT_DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"
STUDY_DIR = PROJECT_ROOT / "studies/eisele_metzger_replication"
REPORT_MD = STUDY_DIR / "recovery_obtainability.md"
CHECKPOINT_JSON = STUDY_DIR / "recovery_obtainability.checkpoint.json"

# Candidate wrong-paper RCTs from the 2026-07-17 audit + manual review, with the
# adjudication tier (see the issue #29 comment). This is the human-confirmed set
# the report runs over by default; `audit_wrong_paper_acquisitions.py` is the
# generator that surfaced most of them. Ordering is by RCT id for stable output.
RECOVERY_CANDIDATES: dict[str, str] = {
    "RCT008": "A: systematic review, not the primary RCT",
    "RCT009": "B: TTM2 oxygen sub-analysis, not the temperature main paper",
    "RCT017": "B: study protocol, not the results paper",
    "RCT019": "B: fluocinolone 3-year results, not the 12-month paper",
    "RCT030": "A: parent Cochrane review, not the trial (already excluded)",
    "RCT040": "B: insulin antibody 104-wk study vs the 52-wk efficacy RCT",
    "RCT064": "B: PRET-PD cognition report, not the 2-year main RCT",
    "RCT074": "B: study protocol, not the results paper",
    "RCT080": "A: Scandinavian mortality stats, not the kindergarten RCT",
    "RCT088": "A: concrete-engineering paper, not the calcifediol COVID RCT",
    "RCT093": "A: RECOVERY empagliflozin arm, not the intended aspirin arm",
    "RCT095": "B: STOIC mechanistic sub-study, not the budesonide RCT",
    "RCT100": "B: pooled four-trial analysis, not the single COVE trial",
}

# Human-verified correct PMIDs for RCTs the auto-resolver cannot select (supply
# via the ``RCTxxx=PMID`` apply syntax; the override is still coverage-validated).
# The resolver misses these for structural reasons: a compound surname query
# (RCT088 "Entrenas Castillo"), a same-platform-different-arm title (RCT093
# empagliflozin vs aspirin), and near-identical protocol/results titles (RCT017 —
# the resolver locks onto the protocol PMID, so the results paper is supplied here).
MANUAL_PMIDS: dict[str, str] = {
    "RCT017": "31968595",  # Ho et al ERAS results paper (Nutrients 2020), not the protocol
    "RCT088": "32871238",  # Entrenas Castillo calcifediol COVID-19 pilot RCT
    "RCT093": "34800427",  # aspirin RECOVERY arm (Lancet 2022)
}

# A resolved candidate must cover at least this fraction of its own title words
# in em_rct_ref to be accepted as the correct trial. Correct papers score
# ~0.8-1.0 (title is ~a substring of its reference); wrong-but-adjacent papers
# fall well below. Reported alongside the verdict so borderline calls are visible.
MIN_MATCH_COVERAGE = 0.6

# Model judgement rows read the wrong document and must be recomputed. Ground
# truth (Cochrane RoB 2 + EM's own Claude 2) assessed the correct trial and is
# preserved. evaluation_run holds only model rows, so it is cleared wholesale
# for the RCT.
_KEEP_JUDGMENT_SOURCES_SQL = "source != 'cochrane' AND source NOT LIKE 'em_claude2_%'"


@dataclass
class ResolvedCandidate:
    """The best-matching PubMed candidate for an RCT's intended trial."""

    pmid: str
    title: str
    coverage: float
    doi: str = ""
    pmcid: str = ""


@dataclass
class ObtainabilityResult:
    """Per-RCT obtainability verdict for the report."""

    rct_id: str
    tier: str
    wrong_pmid: str
    resolved_pmid: str
    resolved_title: str
    coverage: float
    oa_fulltext: bool
    verdict: str  # 'recover_fulltext' | 'recover_abstract' | 'exclude'
    note: str = ""


def classify_verdict(coverage: float, resolved_pmid: str,
                     oa_fulltext: bool,
                     *, min_coverage: float = MIN_MATCH_COVERAGE) -> str:
    """Verdict from a resolution result. Pure — no I/O.

    - no confident match (empty pmid or coverage below the gate) -> ``exclude``
    - confident match with OA full text -> ``recover_fulltext``
    - confident match, abstract only -> ``recover_abstract``
    """
    if not resolved_pmid or coverage < min_coverage:
        return "exclude"
    return "recover_fulltext" if oa_fulltext else "recover_abstract"


def em_row_from_db(row: sqlite3.Row) -> EMRow:
    """Build an EMRow (for the resolver) from a benchmark_rct row.

    Uses the DB's own ``em_rct_ref`` / ``authors_text`` / ``nct_nr`` so the
    report does not depend on the gitignored EM CSV being present.
    """
    em = EMRow(
        rct_id=row["rct_id"],
        cr_id=row["cr_id"] or "",
        rct_author=row["authors_text"] or "",
        rct_ref=row["em_rct_ref"] or "",
        rct_regnr=row["nct_nr"] or "",
        rct_condition=row["condition"] or "",
        rct_intervention=row["intervention"] or "",
    )
    extract_identifiers(em)
    return em


def gather_candidate_pmids(em: EMRow, client: httpx.Client) -> list[str]:
    """Collect candidate PMIDs for the intended trial from several queries.

    Strategy union (deduped, order-preserving): NCT[si], then a title-keyword +
    author + year citation search, then a looser author + year search. Unlike
    the original resolver we do NOT stop at the first hit — every candidate is
    scored against em_rct_ref afterwards, so the correct paper can win even when
    a topic-adjacent wrong paper also matches the query.
    """
    keywords = _title_keywords(em.rct_ref, em.first_author_surname)[:8]
    seen: set[str] = set()
    out: list[str] = []

    def add(pmids: list[str]) -> None:
        for p in pmids:
            if p not in seen:
                seen.add(p)
                out.append(p)

    if em.extracted_nct:
        add(pubmed_search(client, f"{em.extracted_nct}[si]"))
    if em.first_author_surname and em.publication_year and keywords:
        tight = " AND ".join(f"{kw}[tiab]" for kw in keywords[:3])
        add(pubmed_search(
            client,
            f"{em.first_author_surname}[au] AND {em.publication_year}[pdat] AND {tight}",
        ))
    if em.first_author_surname and em.publication_year:
        add(pubmed_search(
            client,
            f"{em.first_author_surname}[au] AND {em.publication_year}[pdat]",
        ))
    return out[:15]


def resolve_correct_pmid(em: EMRow, client: httpx.Client) -> ResolvedCandidate | None:
    """Re-resolve the intended trial, validated by title coverage vs em_rct_ref.

    Returns the best candidate (highest coverage) even when it is below the
    acceptance gate, so the caller can report the near-miss; returns ``None``
    only when no candidate PMID could be found at all. Network-bound.
    """
    candidates = gather_candidate_pmids(em, client)
    if not candidates:
        return None
    summaries = pubmed_summary(client, candidates)
    best: ResolvedCandidate | None = None
    for pmid in candidates:
        title = (summaries.get(pmid, {}) or {}).get("title", "") or ""
        cov = reference_coverage(title, em.rct_ref)
        if best is None or cov > best.coverage:
            best = ResolvedCandidate(pmid=pmid, title=title, coverage=cov)
    return best


def assess_obtainability(row: sqlite3.Row, client: httpx.Client) -> ObtainabilityResult:
    """Resolve + validate + OA-check one candidate RCT (network-bound)."""
    rct_id = row["rct_id"]
    em = em_row_from_db(row)
    tier = RECOVERY_CANDIDATES.get(rct_id, "?")

    best = resolve_correct_pmid(em, client)
    if best is None:
        return ObtainabilityResult(
            rct_id=rct_id, tier=tier, wrong_pmid=row["pmid"] or "",
            resolved_pmid="", resolved_title="", coverage=0.0,
            oa_fulltext=False, verdict="exclude",
            note="no PubMed candidate resolved",
        )

    oa = False
    doi = pmcid = ""
    accepted = best.coverage >= MIN_MATCH_COVERAGE
    if accepted:
        record = fetch_pubmed_record(client, best.pmid)
        doi = record.get("doi", "")
        pmcid = record.get("pmcid", "")
        if pmcid:
            oa = bool(europepmc_fulltext_xml(client, pmcid))

    verdict = classify_verdict(best.coverage, best.pmid, oa)
    note = "" if accepted else f"best coverage {best.coverage:.2f} < {MIN_MATCH_COVERAGE} gate"
    resolved_pmid = best.pmid if accepted else ""
    if best.pmid == (row["pmid"] or "") and accepted:
        # The resolver re-selected the same wrong PMID: not a genuine recovery,
        # so report no resolved paper and force exclude (needs a manual PMID).
        note = "resolver still returns the same (wrong) PMID — needs manual PMID"
        verdict = "exclude"
        resolved_pmid = ""
    return ObtainabilityResult(
        rct_id=rct_id, tier=tier, wrong_pmid=row["pmid"] or "",
        resolved_pmid=resolved_pmid,
        resolved_title=best.title, coverage=best.coverage,
        oa_fulltext=oa, verdict=verdict, note=note,
    )


# --- report mode -------------------------------------------------------

def _load_checkpoint() -> dict[str, dict]:
    if CHECKPOINT_JSON.exists():
        try:
            return json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_checkpoint(done: dict[str, dict]) -> None:
    CHECKPOINT_JSON.write_text(json.dumps(done, indent=2), encoding="utf-8")


def run_report(conn: sqlite3.Connection, rct_ids: list[str],
               client: httpx.Client) -> list[ObtainabilityResult]:
    """Assess every candidate (resumable via the JSON checkpoint)."""
    conn.row_factory = sqlite3.Row
    done = _load_checkpoint()
    results: list[ObtainabilityResult] = []
    for rct_id in rct_ids:
        if rct_id in done:
            results.append(ObtainabilityResult(**done[rct_id]))
            print(f"[skip] {rct_id} (checkpointed)", flush=True)
            continue
        row = conn.execute(
            "SELECT rct_id, cr_id, pmid, nct_nr, authors_text, condition, "
            "intervention, em_rct_ref FROM benchmark_rct WHERE rct_id = ?",
            (rct_id,),
        ).fetchone()
        if row is None:
            print(f"[warn] {rct_id} not in benchmark_rct; skipping", flush=True)
            continue
        res = assess_obtainability(row, client)
        results.append(res)
        done[rct_id] = asdict(res)
        _save_checkpoint(done)  # flush per-RCT so a kill resumes here
        print(f"[{rct_id}] {res.verdict}  cov={res.coverage:.2f}  "
              f"pmid={res.resolved_pmid or '-'}  {res.note}", flush=True)
    return results


def write_report(results: list[ObtainabilityResult]) -> None:
    by_verdict: dict[str, int] = {}
    for r in results:
        by_verdict[r.verdict] = by_verdict.get(r.verdict, 0) + 1
    # Excludes that a human-verified PMID can still recover (auto-resolver
    # cannot select them) are a lower bound on the automated verdict, not a
    # true dead end. Split them out so the counts are not misread.
    manual_recoverable = [r for r in results
                          if r.verdict == "exclude" and r.rct_id in MANUAL_PMIDS]
    n_auto = by_verdict.get('recover_fulltext', 0) + by_verdict.get('recover_abstract', 0)

    lines = [
        "# Wrong-paper recovery: obtainability report",
        "",
        "Generated by `studies/eisele_metzger_replication/recover_wrong_papers.py report`.",
        "Read-only on the DB; verdicts from re-resolving each intended trial and "
        "validating the candidate title against `em_rct_ref` "
        f"(coverage gate {MIN_MATCH_COVERAGE}).",
        "",
        "## Counts",
        "",
        f"- auto-recover (full text): {by_verdict.get('recover_fulltext', 0)}",
        f"- auto-recover (abstract only): {by_verdict.get('recover_abstract', 0)}",
        f"- **auto-recover total: {n_auto}**",
        f"- manual-recover (verified PMID, resolver cannot select): "
        f"{len(manual_recoverable)} ({', '.join(r.rct_id for r in manual_recoverable) or '—'})",
        f"- exclude (correct paper not obtainable): "
        f"{by_verdict.get('exclude', 0) - len(manual_recoverable)}",
        "",
        "## Per-RCT",
        "",
        "| RCT | tier | verdict | cov | wrong→correct PMID | OA FT | note |",
        "|---|---|---|--:|---|:-:|---|",
    ]
    for r in sorted(results, key=lambda x: x.rct_id):
        pmid_col = f"{r.wrong_pmid or '-'} → {r.resolved_pmid or '-'}"
        oa = "✓" if r.oa_fulltext else " "
        note = r.note
        if r.rct_id in MANUAL_PMIDS:
            verdict = "recover (manual)"
            pmid_col = f"{r.wrong_pmid or '-'} → {MANUAL_PMIDS[r.rct_id]}"
            note = "auto-resolver cannot select; apply with "
            note += f"`{r.rct_id}={MANUAL_PMIDS[r.rct_id]}`"
        else:
            verdict = r.verdict
        lines.append(
            f"| {r.rct_id} | {r.tier.split(':')[0]} | {verdict} | "
            f"{r.coverage:.2f} | {pmid_col} | {oa} | {note} |"
        )
    lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[write] {REPORT_MD}")


# --- apply mode --------------------------------------------------------

@dataclass
class RecoveryPlan:
    """What `apply` would do for one RCT (populated in dry-run too)."""

    rct_id: str
    old_pmid: str
    new_pmid: str
    new_title: str
    new_doi: str
    has_abstract: bool
    has_fulltext: bool
    fulltext_source: str
    model_judgment_rows: int
    eval_run_rows: int
    abstract_text: str = field(default="", repr=False)
    jats_xml: str = field(default="", repr=False)


def plan_recovery(conn: sqlite3.Connection, row: sqlite3.Row,
                  client: httpx.Client,
                  override_pmid: str | None = None) -> RecoveryPlan | None:
    """Resolve + fetch the correct document and count stale rows. No writes.

    ``override_pmid`` supplies a human-verified correct PMID directly, for the
    cases the auto-resolver cannot select (near-identical protocol/results
    titles, same-platform different-arm, or compound-surname query misses). The
    override is still coverage-validated against ``em_rct_ref`` — a supplied
    PMID whose title does not match the intended trial is rejected, so a typo
    cannot silently recover the wrong paper.
    """
    em = em_row_from_db(row)
    if override_pmid:
        record = fetch_pubmed_record(client, override_pmid)
        coverage = reference_coverage(record.get("title", ""), em.rct_ref)
        if not record.get("title") or coverage < MIN_MATCH_COVERAGE:
            print(f"[reject] {row['rct_id']}: override pmid {override_pmid} "
                  f"title coverage {coverage:.2f} < {MIN_MATCH_COVERAGE} gate")
            return None
        best = ResolvedCandidate(pmid=override_pmid,
                                 title=record.get("title", ""), coverage=coverage)
    else:
        resolved = resolve_correct_pmid(em, client)
        if resolved is None or resolved.coverage < MIN_MATCH_COVERAGE:
            return None
        best = resolved
        record = fetch_pubmed_record(client, best.pmid)
    abstract = record.get("abstract", "")
    pmcid = record.get("pmcid", "")
    jats = europepmc_fulltext_xml(client, pmcid) if pmcid else ""

    n_judg = conn.execute(
        f"SELECT COUNT(*) FROM benchmark_judgment "
        f"WHERE rct_id = ? AND {_KEEP_JUDGMENT_SOURCES_SQL}",
        (row["rct_id"],),
    ).fetchone()[0]
    n_eval = conn.execute(
        "SELECT COUNT(*) FROM evaluation_run WHERE rct_id = ?",
        (row["rct_id"],),
    ).fetchone()[0]
    return RecoveryPlan(
        rct_id=row["rct_id"], old_pmid=row["pmid"] or "",
        new_pmid=best.pmid, new_title=record.get("title", best.title),
        new_doi=record.get("doi", ""),
        has_abstract=bool(abstract), has_fulltext=bool(jats),
        fulltext_source="europepmc_xml" if jats else "",
        model_judgment_rows=n_judg, eval_run_rows=n_eval,
        abstract_text=abstract, jats_xml=jats,
    )


def apply_recovery(conn: sqlite3.Connection, plan: RecoveryPlan) -> None:
    """Perform the surgical recovery for one RCT. Mutates DB + FULLTEXT_DIR."""
    rct_dir = FULLTEXT_DIR / plan.rct_id
    rct_dir.mkdir(parents=True, exist_ok=True)
    if plan.abstract_text:
        (rct_dir / "abstract.txt").write_text(plan.abstract_text, encoding="utf-8")
    jats_path = rct_dir / "paper.jats.xml"
    if plan.jats_xml:
        jats_path.write_text(plan.jats_xml, encoding="utf-8")
    elif jats_path.exists():
        jats_path.unlink()  # remove the stale wrong-paper full text
    # The recovery path only ever re-fetches JATS full text, so a leftover
    # paper.pdf is always the wrong paper's full text. eval_input.py counts a
    # stale paper.pdf toward has_fulltext, so drop it to keep the on-disk state
    # consistent with the recovered (abstract-or-JATS) document.
    pdf_path = rct_dir / "paper.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
    meta_path = rct_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    meta.update({
        "rct_id": plan.rct_id, "pmid": plan.new_pmid, "doi": plan.new_doi,
        "title": plan.new_title, "has_abstract": plan.has_abstract,
        "has_fulltext": plan.has_fulltext,
        "fulltext_source": plan.fulltext_source or None,
        "recovered_from_wrong_pmid": plan.old_pmid, "complete": True,
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # RECOVERY_NOTE_MARKER is the substring compute_phase6_kappa keys on to tell
    # which wrong papers have been recovered (the --sensitivity precondition).
    note = (f"Recovered {plan.rct_id}: wrong pmid {plan.old_pmid} -> correct "
            f"pmid {plan.new_pmid} ({RECOVERY_NOTE_MARKER}).")
    conn.execute(
        "UPDATE benchmark_rct SET pmid=?, doi=?, title=?, has_abstract=?, "
        "has_fulltext=?, fulltext_source=?, "
        "notes = COALESCE(notes || ' ', '') || ? WHERE rct_id=?",
        (plan.new_pmid, plan.new_doi or None, plan.new_title,
         1 if plan.has_abstract else 0, 1 if plan.has_fulltext else 0,
         plan.fulltext_source or None, note, plan.rct_id),
    )
    conn.execute(
        f"DELETE FROM benchmark_judgment WHERE rct_id=? AND {_KEEP_JUDGMENT_SOURCES_SQL}",
        (plan.rct_id,),
    )
    conn.execute("DELETE FROM evaluation_run WHERE rct_id=?", (plan.rct_id,))
    conn.commit()


def _backup_db(db_path: Path) -> Path:
    backup = db_path.with_suffix(db_path.suffix + ".pre_recovery.bak")
    shutil.copy2(db_path, backup)
    return backup


def parse_apply_targets(tokens: list[str]) -> dict[str, str | None]:
    """Parse apply targets: ``RCTxxx`` (auto-resolve) or ``RCTxxx=PMID`` (manual).

    Returns ``{rct_id: override_pmid_or_None}``, order-preserving via dict.
    """
    targets: dict[str, str | None] = {}
    for tok in tokens:
        if "=" in tok:
            rct_id, pmid = tok.split("=", 1)
            targets[rct_id.strip()] = pmid.strip() or None
        else:
            targets[tok.strip()] = None
    return targets


def run_apply(conn: sqlite3.Connection, targets: dict[str, str | None],
              client: httpx.Client, db_path: Path, do_apply: bool) -> None:
    conn.row_factory = sqlite3.Row
    plans: list[RecoveryPlan] = []
    for rct_id, override_pmid in targets.items():
        row = conn.execute(
            "SELECT rct_id, cr_id, pmid, nct_nr, authors_text, condition, "
            "intervention, em_rct_ref FROM benchmark_rct WHERE rct_id = ?",
            (rct_id,),
        ).fetchone()
        if row is None:
            print(f"[warn] {rct_id} not in benchmark_rct; skipping")
            continue
        plan = plan_recovery(conn, row, client, override_pmid=override_pmid)
        if plan is None:
            print(f"[skip] {rct_id}: correct paper not confidently resolved — "
                  "exclude instead, do not recover")
            continue
        plans.append(plan)
        print(f"[plan] {rct_id}: {plan.old_pmid} -> {plan.new_pmid} "
              f"({'FT' if plan.has_fulltext else 'abstract'}); would delete "
              f"{plan.model_judgment_rows} judgement + {plan.eval_run_rows} eval rows")

    if not do_apply:
        print("\n[dry-run] no changes written. Re-run with --apply to mutate the DB.")
        return
    if not plans:
        print("\n[apply] nothing to do.")
        return

    backup = _backup_db(db_path)
    print(f"\n[backup] {backup}")
    for plan in plans:
        apply_recovery(conn, plan)
        print(f"[applied] {plan.rct_id}")

    _print_reassess_instructions([p.rct_id for p in plans])


def _print_reassess_instructions(rct_ids: list[str]) -> None:
    print("\n=== NEXT: re-assess the recovered RCTs (owner-gated, not run here) ===")
    print("The deleted model rows are regenerated by run_evaluation.py's "
          "(rct, source, domain) existence check — only the missing rows recompute:")
    for model in ("gpt_oss_20b", "gemma4_26b", "qwen3_6_35b"):
        for protocol in ("abstract", "fulltext"):
            print(f"  uv run python studies/eisele_metzger_replication/run_evaluation.py "
                  f"--model {model} --protocol {protocol}")
    print("  # Sonnet: uv run python studies/eisele_metzger_replication/run_evaluation_anthropic.py "
          "--protocol abstract   (and fulltext)")
    print(f"Recovered RCTs: {', '.join(rct_ids)}")
    print("If any recovered RCT is in the gpt-oss temperature-sweep subset, "
          "re-run temperature_sweep.py for it too.")
    print("Then regenerate kappa (compute_phase6_kappa.py, both modes) and "
          "re-derive the drafts.")


# --- main --------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("report", "apply"))
    parser.add_argument("rct_ids", nargs="*",
                        help="apply mode: RCT ids, or RCTxxx=PMID to supply a "
                             "verified correct PMID the auto-resolver cannot "
                             "select. report mode defaults to all candidates.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--apply", action="store_true",
                        help="apply mode only: actually mutate the DB (else dry-run)")
    args = parser.parse_args(argv)

    if not args.db.exists():
        parser.error(f"database not found: {args.db}")
    if args.mode == "apply" and not args.rct_ids:
        parser.error("apply mode requires explicit RCT ids")

    rct_ids = args.rct_ids or sorted(RECOVERY_CANDIDATES)
    conn = sqlite3.connect(args.db)
    try:
        with httpx.Client(headers={"User-Agent": USER_AGENT},
                          follow_redirects=True) as client:
            if args.mode == "report":
                results = run_report(conn, rct_ids, client)
                write_report(results)
            else:
                targets = parse_apply_targets(rct_ids)
                run_apply(conn, targets, client, args.db, do_apply=args.apply)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

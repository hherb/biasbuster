"""Post-hoc recovery of parse-failure rows via Cochrane's per-domain algorithm.

When an LLM emits valid JSON containing ``signalling_answers`` but omits
the ``judgement`` field, we can deterministically derive what the
judgement *should* be by applying the per-domain algorithm published in
the Cochrane RoB 2 documentation — the same algorithm the model was
instructed to apply in the system prompt
(``biasbuster/methodologies/cochrane_rob2/prompts.py``). The model
sometimes drops the ``judgement`` field while preserving the signalling
answers and rationale; this is a known schema-drift mode and was first
documented in CLAUDE.md (PMID 36101416 case in BiasBuster's annotation
pipeline).

The recovery is purely deterministic — no new model calls, no new
information beyond what the model already emitted. Recovered judgements
are tagged in the DB so they can be filtered or reported separately:

  benchmark_judgment.judgment   <- canonical low/some_concerns/high
  benchmark_judgment.valid      <- 1
  benchmark_judgment.raw_label  <- 'FALLBACK'   (sentinel for sensitivity analysis)
  benchmark_judgment.rationale  <- preserved from the model's JSON

  evaluation_run.parse_status   <- 'retry_succeeded' (overloaded — see note below)
  evaluation_run.parse_attempts <- existing + 1
  evaluation_run.error          <- 'algorithmic_fallback: derived judgement from signalling_answers'

Pre-registration consideration: the locked methodology (commit 7854a1c)
specifies the model-emitted judgement as the primary signal. This
recovery is a *sensitivity analysis* — running the published Cochrane
algorithm against the model's signalling outputs is not introducing new
model behaviour, but it is a methodological addition. Downstream
analyses can filter on ``raw_label != 'FALLBACK'`` to reproduce the
original primary metric.

Schema-status note: ``evaluation_run.parse_status`` has a CHECK
constraint that does not include a 'fallback' value. We overload
'retry_succeeded' (semantically: the original parse failed, a retry
strategy succeeded — here the retry is algorithmic rather than another
LLM call). The error column carries the human-readable note. Future
work could add a dedicated status via SQLite ALTER + table rebuild;
not worth it for the present sensitivity-analysis use case.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from biasbuster.methodologies.cochrane_rob2.algorithms import (  # noqa: E402
    derive_domain_judgement,
    synthesis_worst_wins,
)

DEFAULT_DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"


import re

# Targeted-extract regexes used when the full JSON is malformed (a
# common Sonnet drift mode is to emit ``{"text": "...", "Methods"}``
# inside ``evidence_quotes`` — missing the ``"section":`` key — which
# breaks the whole document under strict json.loads but leaves the
# load-bearing fields (judgement, signalling_answers, justification)
# textually intact and individually parseable).
_JUDGEMENT_RE = re.compile(
    r'"judgement"\s*:\s*"(low|some_concerns|some\s+concerns|high)"',
    re.IGNORECASE,
)
_OVERALL_JUDGEMENT_RE = re.compile(
    r'"overall_judgement"\s*:\s*"(low|some_concerns|some\s+concerns|high)"',
    re.IGNORECASE,
)
_JUSTIFICATION_RE = re.compile(
    r'"justification"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
_OVERALL_RATIONALE_RE = re.compile(
    r'"overall_rationale"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)


def _normalize_label(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip().lower().replace(" ", "_")
    if cleaned in ("low", "some_concerns", "high"):
        return cleaned
    return None


def _extract_signalling_block(text: str) -> dict[str, str] | None:
    """Find ``"signalling_answers": {...}`` and return the parsed sub-object.

    Uses a depth counter scoped to the inner braces so a malformed
    evidence_quotes array later in the document does not break this
    extraction.
    """
    marker = '"signalling_answers"'
    idx = text.find(marker)
    if idx < 0:
        return None
    open_brace = text.find("{", idx)
    if open_brace < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(open_brace, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    blob = text[open_brace:i + 1]
                    try:
                        return {str(k): str(v)
                                for k, v in json.loads(blob).items()}
                    except json.JSONDecodeError:
                        return None
    return None


def extract_json_object(text: str) -> dict | None:
    """Best-effort JSON object extraction (mirrors eval_ollama parser)."""
    if not text:
        return None
    # Strip code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Find the closing fence
        end = cleaned.rfind("```")
        if end > 0 and end != cleaned.find("```"):
            inner = cleaned[cleaned.find("\n") + 1:end].strip()
            cleaned = inner
    # Find first { and matching close brace
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    blob = cleaned[start:i + 1]
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError:
                        return None
    return None


def lenient_extract(text: str) -> dict:
    """Pull just the fields we need from a malformed-JSON response.

    Returns a dict with whatever fields could be located. Always returns
    a dict (possibly empty). This is the "Strategy B" parser used when
    full json.loads fails but the document's load-bearing fields
    (judgement, signalling_answers, justification) are individually
    intact — typically because the model malformed only the
    ``evidence_quotes`` array at the tail.
    """
    out: dict = {}
    if not text:
        return out
    if m := _JUDGEMENT_RE.search(text):
        out["judgement"] = _normalize_label(m.group(1))
    if m := _OVERALL_JUDGEMENT_RE.search(text):
        out["overall_judgement"] = _normalize_label(m.group(1))
    if m := _JUSTIFICATION_RE.search(text):
        # Decode escape sequences so the stored rationale matches what
        # a strict json.loads would produce.
        try:
            out["justification"] = json.loads('"' + m.group(1) + '"')
        except json.JSONDecodeError:
            out["justification"] = m.group(1)
    if m := _OVERALL_RATIONALE_RE.search(text):
        try:
            out["overall_rationale"] = json.loads('"' + m.group(1) + '"')
        except json.JSONDecodeError:
            out["overall_rationale"] = m.group(1)
    sa = _extract_signalling_block(text)
    if sa is not None:
        out["signalling_answers"] = sa
    return out


def attempt_recovery(raw_response: str, domain: str
                     ) -> tuple[str | None, str | None, dict | None]:
    """Try to recover (judgement, rationale, parsed_json) from a parse-failure row.

    Two strategies, in order:

    A. Strict JSON parse via ``extract_json_object``. If the document is
       fully valid, prefer the model's explicit ``judgement`` field; if
       the field is missing but ``signalling_answers`` is present,
       derive the judgement via Cochrane's per-domain algorithm.

    B. ``lenient_extract`` for documents where strict JSON parse fails
       (typically because of a malformed ``evidence_quotes`` array at
       the document tail). Pulls each field individually via regex /
       sub-object brace-matching, so the load-bearing fields can be
       recovered even if the surrounding document is broken.

    Returns (None, None, None) when neither strategy yields a usable
    judgement.
    """
    if not raw_response:
        return None, None, None

    if domain == "overall":
        # Synthesis call. Try strict first, then lenient.
        parsed = extract_json_object(raw_response)
        if isinstance(parsed, dict):
            j = parsed.get("overall_judgement")
            if isinstance(j, str) and j.lower().strip() in (
                    "low", "some_concerns", "high"):
                return j.lower().strip(), parsed.get("overall_rationale"), parsed
        # Lenient fallback
        lenient = lenient_extract(raw_response)
        if j := lenient.get("overall_judgement"):
            return j, lenient.get("overall_rationale"), lenient or None
        return None, None, parsed

    # Per-domain call.
    parsed = extract_json_object(raw_response)
    if isinstance(parsed, dict):
        # Strategy A1: explicit judgement field present
        explicit = parsed.get("judgement")
        if isinstance(explicit, str):
            normalized = explicit.lower().strip().replace(" ", "_")
            if normalized in ("low", "some_concerns", "high"):
                rationale = parsed.get("justification")
                return (normalized,
                        rationale if isinstance(rationale, str) else None,
                        parsed)
        # Strategy A2: explicit judgement missing, derive from signalling_answers
        answers = parsed.get("signalling_answers")
        if isinstance(answers, dict):
            answers = {str(k): str(v) for k, v in answers.items()}
            derived = derive_domain_judgement(domain, answers)
            if derived is not None:
                rationale = parsed.get("justification")
                return (derived,
                        rationale if isinstance(rationale, str) else None,
                        parsed)

    # Strategy B: lenient extraction (handles malformed evidence_quotes)
    lenient = lenient_extract(raw_response)
    # B1: explicit judgement
    if j := lenient.get("judgement"):
        return j, lenient.get("justification"), lenient or None
    # B2: derive from extracted signalling_answers
    if answers := lenient.get("signalling_answers"):
        derived = derive_domain_judgement(domain, answers)
        if derived is not None:
            return derived, lenient.get("justification"), lenient or None

    return None, None, parsed


# --- Per-RCT synthesis recovery -----------------------------------------

def recover_synthesis_overall(conn: sqlite3.Connection, rct_id: str,
                              source: str) -> tuple[str | None, list[str]]:
    """If all 5 domains for (rct_id, source) are now valid (post-domain
    recovery), apply Cochrane's worst-wins rule to derive the overall
    judgement. Returns (overall_judgement_or_None, list_of_domains_used).
    """
    cur = conn.cursor()
    rows = cur.execute(
        """SELECT domain, judgment FROM benchmark_judgment
           WHERE rct_id = ? AND source = ? AND valid = 1
             AND domain IN ('d1','d2','d3','d4','d5')""",
        (rct_id, source),
    ).fetchall()
    domain_judgments = {d: j for d, j in rows if j}
    if len(domain_judgments) < 5:
        return None, list(domain_judgments)
    overall = synthesis_worst_wins(domain_judgments.values())
    return overall, list(domain_judgments)


# --- Main recovery loop --------------------------------------------------

def run_recovery(conn: sqlite3.Connection, dry_run: bool,
                 source_filter: str | None = None) -> dict[str, int]:
    """Iterate parse_failure rows, attempt recovery, optionally write."""
    cur = conn.cursor()
    where = "parse_status = 'parse_failure'"
    params: list = []
    if source_filter:
        where += " AND source LIKE ?"
        params.append(source_filter)
    rows = cur.execute(
        f"""SELECT rct_id, source, domain, raw_response, parse_attempts
            FROM evaluation_run WHERE {where}
            ORDER BY rct_id, source, domain""",
        params,
    ).fetchall()

    counts = {
        "candidates": len(rows),
        "domain_recovered": 0,
        "domain_unrecoverable": 0,
        "synthesis_recovered_post_hoc": 0,
    }
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Stage 1: per-domain recovery
    for rct_id, source, domain, raw_response, parse_attempts in rows:
        if domain == "overall":
            # Synthesis recovery is handled in stage 2 after domain recovery
            continue
        judgement, rationale, _parsed = attempt_recovery(raw_response or "", domain)
        if judgement is None:
            counts["domain_unrecoverable"] += 1
            continue
        counts["domain_recovered"] += 1
        if not dry_run:
            cur.execute(
                """UPDATE benchmark_judgment
                   SET judgment = ?, rationale = ?, valid = 1, raw_label = 'FALLBACK'
                   WHERE rct_id = ? AND source = ? AND domain = ?""",
                (judgement, rationale, rct_id, source, domain),
            )
            cur.execute(
                """UPDATE evaluation_run
                   SET parse_status = 'retry_succeeded',
                       parse_attempts = ?,
                       error = 'algorithmic_fallback: derived judgement from signalling_answers',
                       completed_at = ?
                   WHERE rct_id = ? AND source = ? AND domain = ?""",
                ((parse_attempts or 1) + 1, now, rct_id, source, domain),
            )

    if not dry_run:
        conn.commit()

    # Stage 2: synthesis recovery for any (rct_id, source) where
    # all 5 domains are now valid but the overall row is missing or
    # currently has valid=0. Synthesis here is fully deterministic
    # (worst-wins rule applied to the recovered domain judgements).
    cur.execute(
        """SELECT DISTINCT rct_id, source
           FROM benchmark_judgment
           WHERE source LIKE 'sonnet_4_6_%' OR source LIKE 'gpt_oss_%'
              OR source LIKE 'gemma4_%' OR source LIKE 'qwen3_6_%'"""
    )
    candidate_pairs = cur.fetchall()
    for rct_id, source in candidate_pairs:
        overall_row = cur.execute(
            "SELECT judgment, valid FROM benchmark_judgment "
            "WHERE rct_id = ? AND source = ? AND domain = 'overall'",
            (rct_id, source),
        ).fetchone()
        if overall_row and overall_row[0] and overall_row[1] == 1:
            continue  # already valid; leave alone
        derived_overall, domains_used = recover_synthesis_overall(
            conn, rct_id, source)
        if derived_overall is None:
            continue
        counts["synthesis_recovered_post_hoc"] += 1
        if not dry_run:
            cur.execute(
                """INSERT OR REPLACE INTO benchmark_judgment
                   (rct_id, source, domain, judgment, rationale, valid, raw_label)
                   VALUES (?, ?, 'overall', ?, ?, 1, 'FALLBACK')""",
                (rct_id, source, derived_overall,
                 f"Worst-wins synthesis from algorithmic-fallback domain "
                 f"judgements ({','.join(sorted(domains_used))})."),
            )
            cur.execute(
                """INSERT OR REPLACE INTO evaluation_run
                   (rct_id, source, domain, model_id, protocol, pass_n,
                    started_at, completed_at, parse_status, parse_attempts, error)
                   SELECT rct_id, source, 'overall', model_id, protocol, pass_n,
                          ?, ?, 'retry_succeeded', 1,
                          'synthesis: derived via worst-wins from fallback-recovered domains'
                   FROM evaluation_run
                   WHERE rct_id = ? AND source = ? AND domain = 'd1' LIMIT 1""",
                (now, now, rct_id, source),
            )
    if not dry_run:
        conn.commit()

    return counts


# --- main ---------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--source-filter", default=None,
        help="Optional SQL LIKE pattern to scope recovery to one model "
             "(e.g. 'sonnet_4_6_%%' — include the SQL wildcards yourself).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be recovered without writing to the DB.",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"[error] DB not found at {args.db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(args.db_path)
    try:
        # Show pre-state
        cur = conn.cursor()
        n_pre = cur.execute(
            "SELECT COUNT(*) FROM evaluation_run WHERE parse_status = 'parse_failure'"
        ).fetchone()[0]
        print(f"[pre]  parse_failure rows in evaluation_run: {n_pre}")

        counts = run_recovery(conn, dry_run=args.dry_run,
                              source_filter=args.source_filter)
        print(f"[recovery] candidates: {counts['candidates']}")
        print(f"[recovery] domain rows recovered: {counts['domain_recovered']}")
        print(f"[recovery] domain rows unrecoverable: {counts['domain_unrecoverable']}")
        print(f"[recovery] synthesis rows derived post-hoc: "
              f"{counts['synthesis_recovered_post_hoc']}")

        if args.dry_run:
            print("\n[dry-run] no changes written; re-run without --dry-run to apply.")
            return 0

        n_post = cur.execute(
            "SELECT COUNT(*) FROM evaluation_run WHERE parse_status = 'parse_failure'"
        ).fetchone()[0]
        n_fallback = cur.execute(
            "SELECT COUNT(*) FROM benchmark_judgment WHERE raw_label = 'FALLBACK'"
        ).fetchone()[0]
        print(f"[post] parse_failure rows remaining: {n_post}")
        print(f"[post] benchmark_judgment rows tagged FALLBACK: {n_fallback}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

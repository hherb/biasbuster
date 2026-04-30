"""Phase 5 evaluation orchestrator: drives model runs over the EM benchmark.

Per the locked pre-analysis plan (§4 Models, §5 Procedure):
  - 4 models: Claude Sonnet 4.6, gpt-oss:20b, gemma4:26b-a4b-it-q8_0,
    qwen3.6:35b-a3b-q8_0
  - 2 protocols: abstract-only, full-text
  - 3 independent passes per (model × RCT × protocol)
  - 6 LLM calls per pass per RCT (5 domains + 1 synthesis)

This script supports the strategy chosen for execution: **local models
first, Sonnet last**. The CLI takes a model identifier and runs that one
model end-to-end, writing per-call rows to ``evaluation_run`` and the
parsed judgment to ``benchmark_judgment``. Resumability is via the
existence check before each call: if a row already exists in
``benchmark_judgment`` for (rct_id, source, domain), the call is skipped.

Source label convention:  ``{model_short}_{protocol}_pass{n}``
  e.g.  gpt_oss_20b_abstract_pass1
        qwen3_6_35b_fulltext_pass2
        sonnet_4_6_abstract_pass3

Source labels are stable identifiers; the corresponding model_id (the
exact Ollama tag or Anthropic model name) is recorded in
``evaluation_run.model_id`` for full reproducibility.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path("/Users/hherb/src/biasbuster")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "studies/eisele_metzger_replication"))

from biasbuster.methodologies.cochrane_rob2.prompts import build_system_prompt  # noqa: E402

from eval_input import (  # noqa: E402
    RctInput,
    build_domain_user_message,
    build_synthesis_user_message,
    load_rct_input,
)
from eval_ollama import DOMAIN_TO_STAGE, OllamaCallResult, OllamaRunner  # noqa: E402

DEFAULT_DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"

# Model registry: short label → (Ollama tag, runner kind, "human-friendly" name).
# The short label is the prefix for source strings in the DB.
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "gpt_oss_20b": {
        "model_id": "gpt-oss:20b",
        "runner": "ollama",
        "display": "gpt-oss 20B",
    },
    "gemma4_26b": {
        "model_id": "gemma4:26b-a4b-it-q8_0",
        "runner": "ollama",
        "display": "Gemma 4 26B-A4B-IT-Q8",
    },
    "qwen3_6_35b": {
        "model_id": "qwen3.6:35b-a3b-q8_0",
        "runner": "ollama",
        "display": "Qwen 3.6 35B-A3B-Q8",
    },
    "sonnet_4_6": {
        "model_id": "claude-sonnet-4-6",
        "runner": "anthropic",
        "display": "Claude Sonnet 4.6",
    },
}

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "overall")  # 6 calls per pass

# The pre-reg parse-failure halt threshold.
PARSE_FAILURE_HALT_FRACTION = 0.20


# --- Storage helpers ----------------------------------------------------

@dataclass
class CallContext:
    """All identifiers needed to write one call's results to the DB."""
    rct_id: str
    source_label: str
    domain: str
    model_id: str
    protocol: str
    pass_n: int


def is_call_complete(conn: sqlite3.Connection, rct_id: str,
                     source_label: str, domain: str) -> bool:
    """Return True if a benchmark_judgment row already exists with valid=1.

    A failed call (valid=0 or no row) is re-runnable. A successful call
    is skipped on resume.
    """
    cur = conn.cursor()
    row = cur.execute(
        """SELECT valid FROM benchmark_judgment
           WHERE rct_id = ? AND source = ? AND domain = ?""",
        (rct_id, source_label, domain),
    ).fetchone()
    return bool(row and row[0])


def record_result(conn: sqlite3.Connection, ctx: CallContext,
                  result: OllamaCallResult) -> None:
    """Atomically write to evaluation_run AND benchmark_judgment.

    On parse_failure or api_error we still write a benchmark_judgment row
    with valid=0 and judgment=NULL so the rest of the pipeline can join
    against the run table without missing keys; the call is re-runnable
    via re-execution of the orchestrator (the orchestrator deletes the
    invalid row before retrying).
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = conn.cursor()

    cur.execute(
        """INSERT OR REPLACE INTO evaluation_run
           (rct_id, source, domain, model_id, protocol, pass_n,
            started_at, completed_at, duration_seconds,
            input_tokens, output_tokens,
            cache_read_tokens, cache_write_tokens,
            raw_response, parse_status, parse_attempts, error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ctx.rct_id, ctx.source_label, ctx.domain,
            ctx.model_id, ctx.protocol, ctx.pass_n,
            now, now, result.duration_seconds,
            result.input_tokens, result.output_tokens,
            None, None,  # cache fields populated by Anthropic runner only
            result.raw_response, result.parse_status,
            result.parse_attempts, result.error,
        ),
    )

    valid = 1 if result.judgment is not None else 0
    cur.execute(
        """INSERT OR REPLACE INTO benchmark_judgment
           (rct_id, source, domain, judgment, rationale, valid, raw_label)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            ctx.rct_id, ctx.source_label, ctx.domain,
            result.judgment, result.rationale, valid,
            result.judgment,  # raw_label = canonical for our own outputs
        ),
    )
    conn.commit()


def delete_invalid_row(conn: sqlite3.Connection, rct_id: str,
                       source_label: str, domain: str) -> None:
    """Clear any prior failed attempt before re-running."""
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM benchmark_judgment WHERE rct_id = ? AND source = ? AND domain = ? AND valid = 0",
        (rct_id, source_label, domain),
    )
    cur.execute(
        "DELETE FROM evaluation_run WHERE rct_id = ? AND source = ? AND domain = ? AND parse_status != 'ok'",
        (rct_id, source_label, domain),
    )
    conn.commit()


# --- RCT iteration ------------------------------------------------------

def load_eligible_rcts(conn: sqlite3.Connection, protocol: str,
                       limit: int | None = None) -> list[dict]:
    """Return rcts that have the materials needed for this protocol.

    For abstract: any RCT with has_abstract=1 (91 of 100 in the current state).
    For fulltext: any RCT with has_fulltext=1 OR has_abstract=1 (we fall back
    to abstract under the FULLTEXT protocol when full text is missing).
    """
    cur = conn.cursor()
    if protocol == "abstract":
        rows = cur.execute(
            "SELECT rct_id, em_rct_ref, outcome_text FROM benchmark_rct "
            "WHERE has_abstract = 1 ORDER BY rct_id"
        ).fetchall()
    elif protocol == "fulltext":
        rows = cur.execute(
            "SELECT rct_id, em_rct_ref, outcome_text FROM benchmark_rct "
            "WHERE has_fulltext = 1 OR has_abstract = 1 ORDER BY rct_id"
        ).fetchall()
    else:
        raise ValueError(f"unknown protocol: {protocol!r}")
    if limit:
        rows = rows[:limit]
    return [
        {"rct_id": r[0], "em_rct_ref": r[1] or "", "outcome_text": r[2] or ""}
        for r in rows
    ]


# --- One pass over all RCTs --------------------------------------------

def run_one_pass(conn: sqlite3.Connection, runner: OllamaRunner,
                 model_short: str, protocol: str, pass_n: int,
                 rct_limit: int | None = None) -> dict[str, int]:
    """Run all 6 calls × all eligible RCTs for one (model, protocol, pass_n).

    Returns counters: ok, retry_succeeded, parse_failure, api_error, skipped.
    """
    source_label = f"{model_short}_{protocol}_pass{pass_n}"
    model_id = MODEL_REGISTRY[model_short]["model_id"]
    rcts = load_eligible_rcts(conn, protocol, limit=rct_limit)
    counts: dict[str, int] = {"ok": 0, "retry_succeeded": 0,
                              "parse_failure": 0, "api_error": 0, "skipped": 0}
    total_calls = len(rcts) * len(DOMAINS)
    print(f"\n[run] source={source_label} model={model_id} rcts={len(rcts)} "
          f"calls={total_calls}")
    t_start = time.monotonic()

    for i, rct in enumerate(rcts, 1):
        rct_id = rct["rct_id"]
        rct_input = load_rct_input(
            rct_id=rct_id, protocol=protocol,
            rct_label=rct["em_rct_ref"][:120],
            outcome_text=rct["outcome_text"],
        )
        if rct_input is None:
            print(f"  [{i:>3}/{len(rcts)}] {rct_id}: no input materials, skip")
            for d in DOMAINS:
                counts["skipped"] += 1
            continue

        # Per-domain calls (5) feed into the synthesis call (1).
        domain_judgments: dict[str, str] = {}
        for domain in DOMAINS:
            ctx = CallContext(
                rct_id=rct_id, source_label=source_label, domain=domain,
                model_id=model_id, protocol=protocol, pass_n=pass_n,
            )
            if is_call_complete(conn, rct_id, source_label, domain):
                # Already done — load the existing judgment so synthesis can use it.
                cur = conn.cursor()
                row = cur.execute(
                    "SELECT judgment FROM benchmark_judgment "
                    "WHERE rct_id=? AND source=? AND domain=?",
                    (rct_id, source_label, domain),
                ).fetchone()
                if row and row[0]:
                    domain_judgments[domain] = row[0]
                counts["skipped"] += 1
                continue

            delete_invalid_row(conn, rct_id, source_label, domain)
            stage = DOMAIN_TO_STAGE[domain]
            system_prompt = build_system_prompt(stage)

            if domain == "overall":
                user_msg = build_synthesis_user_message(rct_input, domain_judgments)
                result = runner.synthesis_call(system_prompt, user_msg)
            else:
                user_msg = build_domain_user_message(rct_input)
                result = runner.domain_call(domain, system_prompt, user_msg)

            record_result(conn, ctx, result)
            counts[result.parse_status] = counts.get(result.parse_status, 0) + 1
            if result.judgment is not None:
                domain_judgments[domain] = result.judgment

        if i % 5 == 0 or i == len(rcts):
            elapsed = time.monotonic() - t_start
            rate = (i * len(DOMAINS)) / elapsed if elapsed > 0 else 0
            n_failures = counts["parse_failure"] + counts["api_error"]
            n_attempted = sum(counts[k] for k in ("ok", "retry_succeeded", "parse_failure", "api_error"))
            failure_rate = n_failures / n_attempted if n_attempted else 0
            print(
                f"  [{i:>3}/{len(rcts)}] elapsed={elapsed:.0f}s "
                f"rate={rate:.1f} calls/s  "
                f"ok={counts['ok']} retry={counts['retry_succeeded']} "
                f"parse_fail={counts['parse_failure']} api_err={counts['api_error']} "
                f"skipped={counts['skipped']}  failure_rate={failure_rate:.1%}"
            )
            if n_attempted >= 30 and failure_rate > PARSE_FAILURE_HALT_FRACTION:
                print(f"  [HALT] parse-failure rate {failure_rate:.1%} > "
                      f"{PARSE_FAILURE_HALT_FRACTION:.0%} — pre-reg §8 halt rule.")
                return counts

    return counts


# --- main ---------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True,
        choices=list(MODEL_REGISTRY),
        help="Short model label (gpt_oss_20b, gemma4_26b, qwen3_6_35b, sonnet_4_6)",
    )
    parser.add_argument(
        "--protocol", required=True, choices=("abstract", "fulltext"),
    )
    parser.add_argument(
        "--passes", default="1,2,3",
        help="Comma-separated pass numbers to run, e.g. '1' or '1,2,3' (default: all three)",
    )
    parser.add_argument(
        "--rct-limit", type=int, default=None,
        help="Limit to first N RCTs (for smoke testing).",
    )
    parser.add_argument(
        "--ollama-host", default="http://localhost:11434",
        help="Ollama server URL.",
    )
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DB_PATH,
        help=(
            "SQLite DB to write judgments to. Default is the canonical "
            "dataset/eisele_metzger_benchmark.db. For multi-host parallel "
            "runs (e.g. Mac + Spark DGX), point each host at its own "
            "shard (e.g. dataset/eisele_metzger_benchmark.spark.db) and "
            "merge afterwards with merge_eval_dbs.py. Each shard must be "
            "initialised first with build_benchmark_db.py against the "
            "same EM CSV (output is byte-identical given frozen input)."
        ),
    )
    args = parser.parse_args()

    if MODEL_REGISTRY[args.model]["runner"] != "ollama":
        print(f"[error] this script supports Ollama runners only. "
              f"For {args.model} use run_evaluation_anthropic.py "
              "(Phase 5.7, after Sonnet runner is implemented).", file=sys.stderr)
        return 2

    pass_ns = [int(x) for x in args.passes.split(",") if x.strip()]
    for p in pass_ns:
        if p not in (1, 2, 3):
            print(f"[error] pass {p} not in 1..3", file=sys.stderr)
            return 2

    if not args.db_path.exists():
        print(f"[error] DB not found at {args.db_path}. Run "
              "build_benchmark_db.py first to initialise it with the EM "
              "RCTs and Cochrane judgments.", file=sys.stderr)
        return 2
    print(f"[db] writing to {args.db_path}")

    conn = sqlite3.connect(args.db_path)
    runner = OllamaRunner(
        model_id=MODEL_REGISTRY[args.model]["model_id"],
        host=args.ollama_host,
    )
    try:
        for pass_n in pass_ns:
            counts = run_one_pass(
                conn=conn, runner=runner,
                model_short=args.model, protocol=args.protocol,
                pass_n=pass_n, rct_limit=args.rct_limit,
            )
            print(f"\n[pass {pass_n}] final counts: {counts}")
    finally:
        runner.close()
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

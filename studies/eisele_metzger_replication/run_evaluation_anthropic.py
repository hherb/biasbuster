"""Phase 5.8 orchestrator — drive Sonnet 4.6 evaluation via the Anthropic
Batch API. Two-stage workflow with state persistence so the user can
submit, walk away, check back hours later, and ingest without keeping a
terminal open.

Subcommands::

    submit-domains     Build all 5-domain calls (RCTs × passes × protocols)
                       into one batch and submit. Saves batch_id to state.
    submit-synthesis   After domain results are ingested into the DB, build
                       the 600 synthesis calls (overall judgement) using
                       each pass's domain results as input. Submit batch.
    status             Poll all in-flight batches recorded in state.
    ingest-domains     Download completed domain batch, parse JSON
                       responses, write to benchmark_judgment + evaluation_run.
    ingest-synthesis   Same for the synthesis batch.
    wait               Optional: blocking poll on the most recent batch.
                       Useful for smoke tests; not recommended for the
                       full run (production run wall-time is hours).

Typical full-run sequence::

    submit-domains  →  status (until 'ended')  →  ingest-domains
                   →  submit-synthesis  →  status  →  ingest-synthesis

Smoke-test sequence (use --rct-limit 2 --protocols abstract --passes 1)::

    submit-domains --rct-limit 2 --protocols abstract --passes 1
    wait
    ingest-domains
    submit-synthesis  (same scope)
    wait
    ingest-synthesis

State file: ``.sonnet_batch_state.json`` in this directory. Keyed by
(model_short, scope_signature) so multiple smoke tests don't clobber a
production run's state.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "studies/eisele_metzger_replication"))

from biasbuster.methodologies.cochrane_rob2.prompts import build_system_prompt  # noqa: E402

from eval_anthropic import (  # noqa: E402
    AnthropicBatchRunner,
    BatchResult,
    BatchStatus,
    BatchSubmission,
)
from eval_input import (  # noqa: E402
    build_domain_user_message,
    build_synthesis_user_message,
    load_rct_input,
)
from eval_ollama import DOMAIN_TO_STAGE, parse_response  # noqa: E402

DEFAULT_DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"
STATE_PATH = (
    PROJECT_ROOT / "studies/eisele_metzger_replication/.sonnet_batch_state.json"
)
MODEL_SHORT = "sonnet_4_6"
MODEL_ID = "claude-sonnet-4-6"
DOMAINS_FOR_BATCH = ("d1", "d2", "d3", "d4", "d5")  # synthesis is stage 2
PASSES = (1, 2, 3)


# --- State management --------------------------------------------------

@dataclass
class StageState:
    """Per-stage tracking — populated when submitted, updated as ingested."""
    batch_id: str = ""
    submitted_at: str = ""
    n_requests: int = 0
    last_polled_status: str = ""
    last_polled_at: str = ""
    ingested_at: str = ""


@dataclass
class SonnetBatchState:
    """Top-level state file structure."""
    model_short: str = MODEL_SHORT
    model_id: str = MODEL_ID
    scope: dict = field(default_factory=dict)  # protocols, passes, rct_limit
    domains: StageState = field(default_factory=StageState)
    synthesis: StageState = field(default_factory=StageState)


def load_state() -> SonnetBatchState:
    if not STATE_PATH.exists():
        return SonnetBatchState()
    raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return SonnetBatchState(
        model_short=raw.get("model_short", MODEL_SHORT),
        model_id=raw.get("model_id", MODEL_ID),
        scope=raw.get("scope", {}),
        domains=StageState(**raw.get("domains", {})),
        synthesis=StageState(**raw.get("synthesis", {})),
    )


def save_state(state: SonnetBatchState) -> None:
    payload = {
        "model_short": state.model_short,
        "model_id": state.model_id,
        "scope": state.scope,
        "domains": asdict(state.domains),
        "synthesis": asdict(state.synthesis),
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --- DB helpers --------------------------------------------------------

def open_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"[error] DB not found at {db_path}. Build it with build_benchmark_db.py.",
              file=sys.stderr)
        raise SystemExit(2)
    return sqlite3.connect(db_path)


def load_eligible_rcts(conn: sqlite3.Connection, protocol: str,
                       limit: int | None = None) -> list[dict]:
    """Mirror of run_evaluation.load_eligible_rcts — kept duplicate-free
    via shared semantics; we re-implement here to avoid importing from
    the Ollama orchestrator (which pulls httpx etc.)."""
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


# --- Custom-id parsing -------------------------------------------------

def make_custom_id(source_label: str, rct_id: str, domain: str) -> str:
    """Build a Batch-API-legal custom_id.

    Anthropic constrains custom_id to ``^[a-zA-Z0-9_-]{1,64}$`` — only
    alphanumerics, underscore, and hyphen. Source labels already use
    underscores (``sonnet_4_6_fulltext_pass3``), so we use hyphen as
    the field separator. Max length under our schema:
        ``sonnet_4_6_fulltext_pass3-RCT100-overall`` = 40 chars (well under 64).
    """
    return f"{source_label}-{rct_id}-{domain}"


def parse_custom_id(custom_id: str) -> tuple[str, str, str]:
    """Inverse of make_custom_id. Returns (source_label, rct_id, domain).

    Splits from the right on ``-`` because the source label contains
    underscores but never hyphens, RCT id is ``RCT###`` (no hyphens),
    and domain is ``d1..d5`` or ``overall`` (no hyphens). So the last
    two hyphens are unambiguously the separators.
    """
    parts = custom_id.rsplit("-", 2)
    if len(parts) != 3:
        raise ValueError(f"unexpected custom_id shape: {custom_id!r}")
    return parts[0], parts[1], parts[2]


# --- Stage 1: domains --------------------------------------------------

def submit_domains_cmd(args: argparse.Namespace) -> int:
    """Build all 5-domain calls × selected scope and submit one big batch."""
    state = load_state()
    if state.domains.batch_id:
        print(
            f"[error] Domains batch already submitted: {state.domains.batch_id}\n"
            "Use `status` to check progress, or delete .sonnet_batch_state.json "
            "to start over.",
            file=sys.stderr,
        )
        return 2

    protocols = args.protocols.split(",")
    passes = [int(p) for p in args.passes.split(",")]
    state.scope = {
        "protocols": protocols, "passes": passes, "rct_limit": args.rct_limit,
    }

    conn = open_db(args.db_path)
    runner = AnthropicBatchRunner(model_id=args.model_id)
    requests = []
    n_skipped_no_input = 0

    for protocol in protocols:
        rcts = load_eligible_rcts(conn, protocol, limit=args.rct_limit)
        for rct in rcts:
            rct_input = load_rct_input(
                rct_id=rct["rct_id"], protocol=protocol,
                rct_label=rct["em_rct_ref"][:120],
                outcome_text=rct["outcome_text"],
            )
            if rct_input is None:
                n_skipped_no_input += 1
                continue
            user_msg = build_domain_user_message(rct_input)
            for domain in DOMAINS_FOR_BATCH:
                stage = DOMAIN_TO_STAGE[domain]
                system_prompt = build_system_prompt(stage)
                for pass_n in passes:
                    source_label = f"sonnet_4_6_{protocol}_pass{pass_n}"
                    custom_id = make_custom_id(source_label, rct["rct_id"], domain)
                    requests.append(runner.build_request(
                        custom_id=custom_id,
                        system_prompt=system_prompt,
                        user_message=user_msg,
                    ))
    conn.close()

    print(f"[build] {len(requests)} domain requests "
          f"(skipped {n_skipped_no_input} RCTs with no usable input)")
    if not requests:
        print("[error] no requests built; nothing to submit", file=sys.stderr)
        return 2
    if args.dry_run:
        print("[dry-run] not submitting. Sample custom_id:",
              requests[0]["custom_id"])
        return 0

    print(f"[submit] sending batch to Anthropic …")
    submission = runner.submit_batch(requests)
    state.domains.batch_id = submission.batch_id
    state.domains.submitted_at = submission.submitted_at or datetime.now(
        timezone.utc).isoformat()
    state.domains.n_requests = submission.n_requests
    save_state(state)
    print(f"[submit] OK  batch_id={submission.batch_id}  "
          f"n_requests={submission.n_requests}")
    print(f"[state] saved to {STATE_PATH}")
    return 0


def ingest_domains_cmd(args: argparse.Namespace) -> int:
    state = load_state()
    if not state.domains.batch_id:
        print("[error] no domains batch in state; submit-domains first.",
              file=sys.stderr)
        return 2
    if state.domains.ingested_at and not args.force:
        print(f"[error] domains already ingested at {state.domains.ingested_at}. "
              "Use --force to re-ingest.", file=sys.stderr)
        return 2

    runner = AnthropicBatchRunner(model_id=state.model_id)
    status = runner.poll_batch(state.domains.batch_id)
    if status.processing_status != "ended":
        print(f"[error] batch not finished yet (status={status.processing_status}). "
              "Use `status` to monitor.", file=sys.stderr)
        return 2

    conn = open_db(args.db_path)
    counts = ingest_results(conn, runner, state.domains.batch_id,
                            output_kind="domain")
    conn.close()
    state.domains.ingested_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    save_state(state)
    print(f"[ingest] domains complete: {counts}")
    return 0


# --- Stage 2: synthesis ------------------------------------------------

def submit_synthesis_cmd(args: argparse.Namespace) -> int:
    state = load_state()
    if not state.domains.ingested_at:
        print("[error] domains stage must be ingested first.", file=sys.stderr)
        return 2
    if state.synthesis.batch_id and not args.force:
        print(f"[error] synthesis batch already submitted: "
              f"{state.synthesis.batch_id}. Use --force to overwrite.",
              file=sys.stderr)
        return 2

    protocols = state.scope.get("protocols", [])
    passes = state.scope.get("passes", [])
    rct_limit = state.scope.get("rct_limit")

    conn = open_db(args.db_path)
    runner = AnthropicBatchRunner(model_id=state.model_id)
    requests = []
    n_skipped_no_domains = 0
    cur = conn.cursor()
    synthesis_system_prompt = build_system_prompt("synthesize")

    for protocol in protocols:
        rcts = load_eligible_rcts(conn, protocol, limit=rct_limit)
        for rct in rcts:
            rct_input = load_rct_input(
                rct_id=rct["rct_id"], protocol=protocol,
                rct_label=rct["em_rct_ref"][:120],
                outcome_text=rct["outcome_text"],
            )
            if rct_input is None:
                continue
            for pass_n in passes:
                source_label = f"sonnet_4_6_{protocol}_pass{pass_n}"
                domain_judgments = dict(cur.execute(
                    """SELECT domain, judgment FROM benchmark_judgment
                       WHERE rct_id = ? AND source = ? AND valid = 1
                         AND domain IN ('d1','d2','d3','d4','d5')""",
                    (rct["rct_id"], source_label),
                ).fetchall())
                if len(domain_judgments) < 5:
                    # Some domains parse-failed; skip synthesis for this RCT
                    # and record valid=0 placeholder so downstream can find it.
                    n_skipped_no_domains += 1
                    cur.execute(
                        """INSERT OR REPLACE INTO benchmark_judgment
                           (rct_id, source, domain, judgment, rationale, valid, raw_label)
                           VALUES (?, ?, 'overall', NULL, NULL, 0, NULL)""",
                        (rct["rct_id"], source_label),
                    )
                    continue
                user_msg = build_synthesis_user_message(rct_input, domain_judgments)
                custom_id = make_custom_id(source_label, rct["rct_id"], "overall")
                requests.append(runner.build_request(
                    custom_id=custom_id,
                    system_prompt=synthesis_system_prompt,
                    user_message=user_msg,
                ))
    conn.commit()
    conn.close()

    print(f"[build] {len(requests)} synthesis requests "
          f"(skipped {n_skipped_no_domains} for missing domain judgements)")
    if not requests:
        print("[error] no synthesis requests to submit", file=sys.stderr)
        return 2
    if args.dry_run:
        print("[dry-run] not submitting. Sample custom_id:",
              requests[0]["custom_id"])
        return 0

    print(f"[submit] sending synthesis batch to Anthropic …")
    submission = runner.submit_batch(requests)
    state.synthesis.batch_id = submission.batch_id
    state.synthesis.submitted_at = submission.submitted_at or datetime.now(
        timezone.utc).isoformat()
    state.synthesis.n_requests = submission.n_requests
    save_state(state)
    print(f"[submit] OK  batch_id={submission.batch_id}  "
          f"n_requests={submission.n_requests}")
    return 0


def ingest_synthesis_cmd(args: argparse.Namespace) -> int:
    state = load_state()
    if not state.synthesis.batch_id:
        print("[error] no synthesis batch in state.", file=sys.stderr)
        return 2

    runner = AnthropicBatchRunner(model_id=state.model_id)
    status = runner.poll_batch(state.synthesis.batch_id)
    if status.processing_status != "ended":
        print(f"[error] batch not finished yet (status={status.processing_status}).",
              file=sys.stderr)
        return 2

    conn = open_db(args.db_path)
    counts = ingest_results(conn, runner, state.synthesis.batch_id,
                            output_kind="synthesis")
    conn.close()
    state.synthesis.ingested_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    save_state(state)
    print(f"[ingest] synthesis complete: {counts}")
    return 0


# --- Shared ingest path ------------------------------------------------

def ingest_results(conn: sqlite3.Connection, runner: AnthropicBatchRunner,
                   batch_id: str, output_kind: str) -> dict[str, int]:
    """Stream batch results, parse JSON, write benchmark_judgment + evaluation_run.

    `output_kind` is "domain" (5-element JSON) or "synthesis" (overall-only).
    """
    cur = conn.cursor()
    counts = {"ok": 0, "parse_failure": 0, "api_error": 0, "expired": 0}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for r in runner.fetch_results(batch_id):
        source_label, rct_id, domain = parse_custom_id(r.custom_id)
        protocol = source_label.split("_")[3]  # sonnet_4_6_<protocol>_passN
        pass_n = int(source_label.rsplit("pass", 1)[1])

        if not r.succeeded:
            judgment = None
            rationale = None
            valid = 0
            parse_status = "api_error" if r.error_type else "expired"
            counts[parse_status] += 1
            error_str = (r.error_message or r.error_type or "")[:500]
        else:
            # Pass the domain code (e.g. "d1") so parse_response can apply
            # the algorithmic fallback if the model omits the explicit
            # judgement field. Synthesis calls (output_kind="synthesis")
            # have domain_code=None internally.
            judgment, rationale = parse_response(
                r.text_response, output_kind,
                domain_code=domain if output_kind == "domain" else None,
            )
            if judgment is None:
                valid = 0
                parse_status = "parse_failure"
                counts["parse_failure"] += 1
            else:
                valid = 1
                parse_status = "ok"
                counts["ok"] += 1
            error_str = None

        cur.execute(
            """INSERT OR REPLACE INTO evaluation_run
               (rct_id, source, domain, model_id, protocol, pass_n,
                started_at, completed_at, duration_seconds,
                input_tokens, output_tokens,
                cache_read_tokens, cache_write_tokens,
                raw_response, parse_status, parse_attempts, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rct_id, source_label, domain,
                runner.model_id, protocol, pass_n,
                now, now, None,
                r.input_tokens, r.output_tokens,
                r.cache_read_input_tokens, r.cache_creation_input_tokens,
                r.text_response[:10000] if r.text_response else None,
                parse_status, 1, error_str,
            ),
        )
        cur.execute(
            """INSERT OR REPLACE INTO benchmark_judgment
               (rct_id, source, domain, judgment, rationale, valid, raw_label)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (rct_id, source_label, domain, judgment, rationale, valid, judgment),
        )
    conn.commit()
    return counts


# --- Status / wait -----------------------------------------------------

def status_cmd(args: argparse.Namespace) -> int:
    state = load_state()
    runner = AnthropicBatchRunner(model_id=state.model_id)
    print(f"## Sonnet batch state — model={state.model_id}")
    print(f"   scope={state.scope}\n")
    for stage_name, stage in (("domains", state.domains),
                              ("synthesis", state.synthesis)):
        if not stage.batch_id:
            print(f"[{stage_name}] not submitted")
            continue
        live = runner.poll_batch(stage.batch_id)
        print(f"[{stage_name}] batch_id={stage.batch_id}")
        print(f"   submitted={stage.submitted_at}  n_requests={stage.n_requests}")
        print(f"   processing_status={live.processing_status}")
        print(f"   succeeded={live.succeeded}  errored={live.errored}  "
              f"processing={live.processing}  canceled={live.canceled}  "
              f"expired={live.expired}")
        if stage.ingested_at:
            print(f"   ingested_at={stage.ingested_at}")
        # Persist last poll for reference.
        stage.last_polled_status = live.processing_status
        stage.last_polled_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
    save_state(state)
    return 0


def wait_cmd(args: argparse.Namespace) -> int:
    state = load_state()
    target = state.synthesis if state.synthesis.batch_id else state.domains
    if not target.batch_id:
        print("[error] no batch in state to wait on.", file=sys.stderr)
        return 2
    runner = AnthropicBatchRunner(model_id=state.model_id)

    def on_status(s: BatchStatus) -> None:
        n_done = s.succeeded + s.errored + s.canceled + s.expired
        n_total = n_done + s.processing
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
              f"status={s.processing_status}  "
              f"done={n_done}/{n_total}  "
              f"succeeded={s.succeeded}  errored={s.errored}")

    print(f"[wait] polling batch {target.batch_id} every {args.poll_interval}s "
          f"(max {args.max_wait_h}h) …")
    try:
        runner.wait_for_completion(
            target.batch_id,
            poll_interval_s=args.poll_interval,
            max_wait_s=args.max_wait_h * 3600,
            on_status=on_status,
        )
        print("[wait] batch ended.")
        return 0
    except TimeoutError as exc:
        print(f"[wait] timeout: {exc}", file=sys.stderr)
        return 2


# --- main --------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--model-id", default=MODEL_ID,
                        help="Anthropic model identifier "
                             "(default: claude-sonnet-4-6)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sd = sub.add_parser("submit-domains")
    sd.add_argument("--protocols", default="abstract,fulltext",
                    help="comma-separated subset of {abstract,fulltext}")
    sd.add_argument("--passes", default="1,2,3",
                    help="comma-separated pass numbers in {1,2,3}")
    sd.add_argument("--rct-limit", type=int, default=None,
                    help="cap to first N RCTs (smoke test)")
    sd.add_argument("--dry-run", action="store_true",
                    help="build but do not submit; print sample custom_id")
    sd.set_defaults(func=submit_domains_cmd)

    ssyn = sub.add_parser("submit-synthesis")
    ssyn.add_argument("--dry-run", action="store_true")
    ssyn.add_argument("--force", action="store_true",
                      help="resubmit even if a synthesis batch is already in state")
    ssyn.set_defaults(func=submit_synthesis_cmd)

    sst = sub.add_parser("status")
    sst.set_defaults(func=status_cmd)

    sw = sub.add_parser("wait")
    sw.add_argument("--poll-interval", type=float, default=60.0)
    sw.add_argument("--max-wait-h", type=float, default=24.0)
    sw.set_defaults(func=wait_cmd)

    sid = sub.add_parser("ingest-domains")
    sid.add_argument("--force", action="store_true",
                     help="re-ingest even if ingested_at is set")
    sid.set_defaults(func=ingest_domains_cmd)

    sis = sub.add_parser("ingest-synthesis")
    sis.set_defaults(func=ingest_synthesis_cmd)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

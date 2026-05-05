"""Temperature sensitivity analysis (sensitivity to decoding parameters).

Sweep across decoding temperatures to characterise the contribution of
stochastic decoding to LLM run-to-run κ. This sits OUTSIDE the locked
pre-registered methodology — the primary analysis fixes temperature at
each model's defaults (Ollama 0.8, Anthropic 1.0). This script enables
controlled overrides for an explicit sensitivity analysis.

Results are written under a separate source-label namespace so they do
not contaminate the primary analysis::

    sonnet_4_6_fulltext_pass1   ← primary (model defaults)
    gpt_oss_20b_T0_fulltext_pass1   ← sweep, T=0
    gpt_oss_20b_T0p3_fulltext_pass1  ← sweep, T=0.3
    gpt_oss_20b_T1p2_fulltext_pass1  ← sweep, T=1.2

The decimal point in temperatures is encoded as ``p`` in source labels
(``T0p3`` for 0.3) so the labels stay alphanumeric+underscore-friendly.

Recommended experimental progression
------------------------------------

Stage 1 (extremes, 10 RCTs, 3 passes each):
    --temperature 0.0  --rct-limit 10  --passes 1,2,3
    --temperature 1.2  --rct-limit 10  --passes 1,2,3
    --temperature 0.8  --rct-limit 10  --passes 1,2,3   # calibration: same RCTs at default

Stage 2 (fill in the curve if Stage 1 shows a gradient):
    --temperature 0.3  --rct-limit 10  --passes 1,2,3
    --temperature 0.6  --rct-limit 10  --passes 1,2,3

Suggested cost (gpt-oss:20b on Spark DGX, ~14 sec/call):
  10 RCTs × 6 calls × 3 passes = 180 calls/temperature × 5 temperatures
  = 900 calls × 14 sec ≈ 3.5 hours total. Local GPU; no API spend.

Analysis
--------

After the sweep completes, run::

    uv run python studies/eisele_metzger_replication/temperature_analysis.py \\
        --model gpt_oss_20b --protocol fulltext

It computes run-to-run κ at each temperature and prints a comparison
table — making the temperature-sensitivity question empirical rather
than rhetorical.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "studies/eisele_metzger_replication"))

from biasbuster.methodologies.cochrane_rob2.prompts import build_system_prompt  # noqa: E402

from eval_input import (  # noqa: E402
    build_domain_user_message,
    build_synthesis_user_message,
    load_rct_input,
)
from eval_ollama import DOMAIN_TO_STAGE, OllamaRunner  # noqa: E402
from run_evaluation import (  # noqa: E402
    DOMAINS,
    CallContext,
    delete_invalid_row,
    is_call_complete,
    load_eligible_rcts,
    record_result,
)

DEFAULT_DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"

# Map from model short label to its Ollama tag. Mirrors run_evaluation.MODEL_REGISTRY
# but we keep this script self-contained so the user can run it standalone
# even if MODEL_REGISTRY changes upstream.
MODEL_TAGS: dict[str, str] = {
    "gpt_oss_20b": "gpt-oss:20b",
    "gemma4_26b": "gemma4:26b-a4b-it-q8_0",
    "qwen3_6_35b": "qwen3.6:35b-a3b-q8_0",
}


def temperature_label(temperature: float) -> str:
    """Encode a temperature as an alphanumeric source-label segment.

    >>> temperature_label(0.0)
    'T0'
    >>> temperature_label(0.3)
    'T0p3'
    >>> temperature_label(1.2)
    'T1p2'
    """
    if temperature == 0:
        return "T0"
    s = f"{temperature:g}"  # 0.3, 1.2, etc.
    return "T" + s.replace(".", "p")


def make_source_label(model_short: str, temperature: float, protocol: str,
                      pass_n: int) -> str:
    return f"{model_short}_{temperature_label(temperature)}_{protocol}_pass{pass_n}"


def run_sweep_pass(conn: sqlite3.Connection, runner: OllamaRunner,
                   model_short: str, temperature: float, protocol: str,
                   pass_n: int, rct_limit: int) -> dict[str, int]:
    """Run all 6 calls × N RCTs for one (temperature, protocol, pass)."""
    source_label = make_source_label(model_short, temperature, protocol, pass_n)
    model_id = runner.model_id
    rcts = load_eligible_rcts(conn, protocol, limit=rct_limit)
    counts: dict[str, int] = {"ok": 0, "retry_succeeded": 0,
                              "parse_failure": 0, "api_error": 0, "skipped": 0}
    total_calls = len(rcts) * len(DOMAINS)
    print(f"\n[run] source={source_label} model={model_id} "
          f"T={temperature} rcts={len(rcts)} calls={total_calls}")
    t_start = time.monotonic()

    for i, rct in enumerate(rcts, 1):
        rct_id = rct["rct_id"]
        rct_input = load_rct_input(
            rct_id=rct_id, protocol=protocol,
            rct_label=rct["em_rct_ref"][:120],
            outcome_text=rct["outcome_text"],
        )
        if rct_input is None:
            for _ in DOMAINS:
                counts["skipped"] += 1
            continue

        domain_judgments: dict[str, str] = {}
        for domain in DOMAINS:
            ctx = CallContext(
                rct_id=rct_id, source_label=source_label, domain=domain,
                model_id=model_id, protocol=protocol, pass_n=pass_n,
            )
            if is_call_complete(conn, rct_id, source_label, domain):
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

        if i % 2 == 0 or i == len(rcts):
            elapsed = time.monotonic() - t_start
            rate = (i * len(DOMAINS)) / elapsed if elapsed > 0 else 0
            print(
                f"  [{i:>3}/{len(rcts)}] elapsed={elapsed:.0f}s "
                f"rate={rate:.1f} calls/s  "
                f"ok={counts['ok']} retry={counts['retry_succeeded']} "
                f"parse_fail={counts['parse_failure']} api_err={counts['api_error']} "
                f"skipped={counts['skipped']}"
            )

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=list(MODEL_TAGS), default="gpt_oss_20b",
        help="Model short label (default: gpt_oss_20b — fastest local).",
    )
    parser.add_argument(
        "--temperature", type=float, required=True,
        help="Decoding temperature override (e.g. 0.0, 0.3, 0.6, 1.2). "
             "Outside the pre-reg locked default; results stored under "
             "a temperature-tagged source label.",
    )
    parser.add_argument(
        "--protocol", choices=("abstract", "fulltext"), default="fulltext",
        help="Protocol to evaluate (default: fulltext — the methodologically "
             "meaningful regime where signalling-question answers can vary).",
    )
    parser.add_argument(
        "--passes", default="1,2,3",
        help="Comma-separated pass numbers in {1,2,3} (default: 1,2,3).",
    )
    parser.add_argument(
        "--rct-limit", type=int, default=10,
        help="Cap to first N RCTs (default: 10 — sufficient for an extremes "
             "comparison; expand to 20–30 for tighter CIs once the gradient "
             "is established).",
    )
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DB_PATH,
        help="SQLite DB to write to. Default is the canonical "
             "dataset/eisele_metzger_benchmark.db. Spark DGX runs typically "
             "point at the local shard.",
    )
    parser.add_argument(
        "--ollama-host", default="http://localhost:11434",
        help="Ollama server URL.",
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help="Optional top_p override (default: rely on Modelfile default 0.9).",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Optional top_k override (default: rely on Modelfile default 40).",
    )
    args = parser.parse_args()

    pass_ns = [int(p) for p in args.passes.split(",") if p.strip()]
    for p in pass_ns:
        if p not in (1, 2, 3):
            print(f"[error] pass {p} not in 1..3", file=sys.stderr)
            return 2
    if not args.db_path.exists():
        print(f"[error] DB not found at {args.db_path}", file=sys.stderr)
        return 2

    sampling_params: dict[str, float | int] = {"temperature": args.temperature}
    if args.top_p is not None:
        sampling_params["top_p"] = args.top_p
    if args.top_k is not None:
        sampling_params["top_k"] = args.top_k

    print(f"[setup] db={args.db_path}")
    print(f"[setup] model={args.model} ({MODEL_TAGS[args.model]})")
    print(f"[setup] sampling_params={sampling_params}")
    print(f"[setup] protocol={args.protocol} passes={pass_ns} rct_limit={args.rct_limit}")
    print(f"[setup] source labels will start with: "
          f"{make_source_label(args.model, args.temperature, args.protocol, 1)!r}")

    conn = sqlite3.connect(args.db_path)
    runner = OllamaRunner(
        model_id=MODEL_TAGS[args.model],
        host=args.ollama_host,
        sampling_params=sampling_params,
    )
    try:
        for pass_n in pass_ns:
            counts = run_sweep_pass(
                conn=conn, runner=runner,
                model_short=args.model, temperature=args.temperature,
                protocol=args.protocol, pass_n=pass_n, rct_limit=args.rct_limit,
            )
            print(f"\n[pass {pass_n}] final counts: {counts}")
    finally:
        runner.close()
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Algorithm-conformance audit for the Eisele-Metzger replication benchmark.

Asks two distinct questions per (RCT, domain) cell:

  1. **Model self-conformance**: does the model's emitted ``judgement``
     equal what the Cochrane RoB 2 algorithm produces when applied to
     the model's own ``signalling_answers``? This is a sanity check —
     a self-conformance rate near 100% means the model is operating as
     a faithful algorithm executor, not a free-form judge.

  2. **Cochrane-vs-model-evidence**: given the model's signalling
     extraction (which the model also corroborates with verbatim
     quotes, recoverable from ``raw_response``), would applying the
     same Cochrane decision rule produce Cochrane's published rating?
     When the answer is "no, Cochrane is more lenient", that is
     prima-facie evidence that Cochrane's published rating is in
     algorithmic tension with the paper's content.

The audit is asymmetric by construction. The model's signalling
answers are the auditable artefact; Cochrane's reviewer notes were
not published. We use the model's structured extraction as a proxy
for "what the algorithm says the rating should be given the paper",
and where the multi-model consensus on signalling agrees, the proxy
is corroborated. This is the central methodological move of the
algorithm-conformance paper.

Inputs
------
Reads ``evaluation_run.raw_response`` (full JSON output) and joins
to ``benchmark_judgment`` for both the model's emitted judgement and
Cochrane's published rating. No writes.

Outputs
-------
- A summary table printed to stdout.
- An optional per-row TSV (``--out-tsv path``) for the conformance
  paper's supplementary audit appendix.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from biasbuster.methodologies.cochrane_rob2.algorithms import (  # noqa: E402
    derive_domain_judgement,
)

DEFAULT_DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"
DOMAINS = ("d1", "d2", "d3", "d4", "d5")
MODELS: dict[str, str] = {
    "gpt-oss:20b": "gpt_oss_20b_fulltext_pass",
    "gemma4:26b-A4B": "gemma4_26b_fulltext_pass",
    "qwen3.6:35b-A3B": "qwen3_6_35b_fulltext_pass",
    "Sonnet 4.6": "sonnet_4_6_fulltext_pass",
}
SEVERITY = {"low": 0, "some_concerns": 1, "high": 2}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?", re.MULTILINE)
_CLOSING_FENCE_RE = re.compile(r"\n?```\s*$", re.MULTILINE)


# --- JSON extraction (same balanced-brace scanner as eval_ollama.py) ---

def _extract_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` block, tolerating fenced output."""
    text = _FENCE_RE.sub("", text, count=1)
    text = _CLOSING_FENCE_RE.sub("", text, count=1)
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
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
                    return text[start:i + 1]
    return ""


def parse_signalling(raw_response: str) -> dict[str, str] | None:
    """Recover ``signalling_answers`` from a raw model response.

    Returns the dict if found and well-formed, else None. Values are
    normalised to upper-case so downstream rule application is robust
    to per-model casing drift.
    """
    if not raw_response:
        return None
    js = _extract_json_object(raw_response)
    if not js:
        return None
    try:
        data = json.loads(js)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    sa = data.get("signalling_answers")
    if not isinstance(sa, dict):
        return None
    return {str(k).strip(): str(v).strip().upper() for k, v in sa.items() if v is not None}


# --- Audit row + aggregation -----------------------------------------------

@dataclass(frozen=True)
class AuditRow:
    rct_id: str
    source: str
    domain: str
    signalling_answers: tuple[tuple[str, str], ...]  # frozen for dedup
    model_judgement: str | None        # what the model emitted
    algorithmic_judgement: str | None  # derived from model's signalling
    cochrane_judgement: str | None     # Cochrane's published rating
    has_evidence_quotes: bool

    @property
    def model_self_conformant(self) -> bool | None:
        if self.model_judgement is None or self.algorithmic_judgement is None:
            return None
        return self.model_judgement == self.algorithmic_judgement

    @property
    def cochrane_consistent_with_evidence(self) -> bool | None:
        if self.cochrane_judgement is None or self.algorithmic_judgement is None:
            return None
        return self.cochrane_judgement == self.algorithmic_judgement

    @property
    def direction(self) -> str:
        """Cochrane vs algorithmic judgement applied to model's signalling."""
        if self.cochrane_judgement is None or self.algorithmic_judgement is None:
            return "indeterminate"
        if self.cochrane_judgement == self.algorithmic_judgement:
            return "match"
        c = SEVERITY[self.cochrane_judgement]
        a = SEVERITY[self.algorithmic_judgement]
        return "cochrane_more_lenient" if c < a else "cochrane_more_strict"


def _has_evidence_quotes(raw_response: str) -> bool:
    """Best-effort: does the JSON include a non-empty evidence_quotes array?"""
    js = _extract_json_object(raw_response)
    if not js:
        return False
    try:
        data = json.loads(js)
    except json.JSONDecodeError:
        return False
    quotes = data.get("evidence_quotes") if isinstance(data, dict) else None
    return isinstance(quotes, list) and len(quotes) > 0


def load_cochrane_ratings(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """Map ``(rct_id, domain) -> cochrane_judgement`` for the per-domain rows."""
    rows = conn.execute(
        "SELECT rct_id, domain, judgment FROM benchmark_judgment "
        "WHERE source = 'cochrane' AND domain != 'overall'"
    ).fetchall()
    return {(rct, dom): j for rct, dom, j in rows}


def load_audit_rows(conn: sqlite3.Connection, source_pattern: str,
                    cochrane: dict[tuple[str, str], str]) -> list[AuditRow]:
    """Build AuditRow records for every per-domain cell of a model × pass."""
    rows = conn.execute(
        """
        SELECT er.rct_id, er.source, er.domain, er.raw_response, bj.judgment
        FROM evaluation_run er
        LEFT JOIN benchmark_judgment bj
          ON bj.rct_id = er.rct_id
         AND bj.source = er.source
         AND bj.domain = er.domain
        WHERE er.source LIKE ?
          AND er.domain IN ('d1','d2','d3','d4','d5')
          AND er.raw_response IS NOT NULL
        """,
        (source_pattern,),
    ).fetchall()
    audit: list[AuditRow] = []
    for rct_id, source, domain, raw, model_j in rows:
        sa = parse_signalling(raw or "")
        algo_j = derive_domain_judgement(domain, sa) if sa else None
        cochrane_j = cochrane.get((rct_id, domain))
        audit.append(AuditRow(
            rct_id=rct_id, source=source, domain=domain,
            signalling_answers=tuple(sorted(sa.items())) if sa else (),
            model_judgement=model_j,
            algorithmic_judgement=algo_j,
            cochrane_judgement=cochrane_j,
            has_evidence_quotes=_has_evidence_quotes(raw or ""),
        ))
    return audit


# --- Aggregation -----------------------------------------------------------

@dataclass
class ModelDomainStats:
    n_rows: int = 0
    n_signalling_parsed: int = 0
    n_self_conformant: int = 0           # model_judgement == algorithmic_judgement
    n_cochrane_match: int = 0            # cochrane_judgement == algorithmic_judgement
    n_cochrane_more_lenient: int = 0     # cochrane lower severity than algo
    n_cochrane_more_strict: int = 0      # cochrane higher severity than algo
    n_with_evidence_quotes: int = 0


def aggregate(audit: list[AuditRow]) -> dict[tuple[str, str], ModelDomainStats]:
    """Aggregate per (model_label, domain). model_label is the source's
    pass-stripped form (e.g. ``gpt_oss_20b_fulltext``)."""
    stats: dict[tuple[str, str], ModelDomainStats] = defaultdict(ModelDomainStats)
    for r in audit:
        # Strip the trailing ``_pass{N}`` so all 3 passes aggregate per model.
        model_label = re.sub(r"_pass\d+$", "", r.source)
        s = stats[(model_label, r.domain)]
        s.n_rows += 1
        if r.signalling_answers:
            s.n_signalling_parsed += 1
        if r.model_self_conformant is True:
            s.n_self_conformant += 1
        if r.has_evidence_quotes:
            s.n_with_evidence_quotes += 1
        d = r.direction
        if d == "match":
            s.n_cochrane_match += 1
        elif d == "cochrane_more_lenient":
            s.n_cochrane_more_lenient += 1
        elif d == "cochrane_more_strict":
            s.n_cochrane_more_strict += 1
    return stats


def _pct(num: int, den: int) -> str:
    return f"{(100 * num / den):>5.1f}%" if den else "    -"


def format_summary(stats: dict[tuple[str, str], ModelDomainStats]) -> str:
    """Pretty-print the per (model × domain) table."""
    lines: list[str] = []
    header = (
        f"{'model_protocol':<28} {'dom':>3} "
        f"{'n':>5} {'sig_ok':>7} "
        f"{'self_conf%':>10} {'coch_match%':>11} "
        f"{'coch_lenient%':>14} {'coch_stricter%':>15}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for (model_label, dom) in sorted(stats):
        s = stats[(model_label, dom)]
        lines.append(
            f"{model_label:<28} {dom:>3} "
            f"{s.n_rows:>5} {s.n_signalling_parsed:>7} "
            f"{_pct(s.n_self_conformant, s.n_signalling_parsed):>10} "
            f"{_pct(s.n_cochrane_match, s.n_signalling_parsed):>11} "
            f"{_pct(s.n_cochrane_more_lenient, s.n_signalling_parsed):>14} "
            f"{_pct(s.n_cochrane_more_strict, s.n_signalling_parsed):>15}"
        )
    return "\n".join(lines)


def overall_asymmetry(stats: dict[tuple[str, str], ModelDomainStats]
                      ) -> dict[str, int | float]:
    """Pooled across all (model × domain × pass). The headline numbers."""
    n = sum(s.n_signalling_parsed for s in stats.values())
    m = sum(s.n_cochrane_match for s in stats.values())
    le = sum(s.n_cochrane_more_lenient for s in stats.values())
    st = sum(s.n_cochrane_more_strict for s in stats.values())
    sc = sum(s.n_self_conformant for s in stats.values())
    return {
        "n_signalling_parsed": n,
        "n_cochrane_match": m,
        "n_cochrane_more_lenient": le,
        "n_cochrane_more_strict": st,
        "n_self_conformant": sc,
        "match_pct": (100 * m / n) if n else 0.0,
        "lenient_pct": (100 * le / n) if n else 0.0,
        "stricter_pct": (100 * st / n) if n else 0.0,
        "self_conformance_pct": (100 * sc / n) if n else 0.0,
        "asymmetry_ratio": (le / st) if st else float("inf"),
    }


def write_tsv(audit: list[AuditRow], path: Path) -> None:
    """Write one row per audit entry — supplementary appendix material."""
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "rct_id", "source", "domain",
            "signalling_answers_json",
            "model_judgement", "algorithmic_judgement", "cochrane_judgement",
            "model_self_conformant", "cochrane_match",
            "direction", "has_evidence_quotes",
        ])
        for r in audit:
            sa_json = json.dumps(dict(r.signalling_answers)) if r.signalling_answers else ""
            w.writerow([
                r.rct_id, r.source, r.domain, sa_json,
                r.model_judgement or "",
                r.algorithmic_judgement or "",
                r.cochrane_judgement or "",
                "" if r.model_self_conformant is None else int(r.model_self_conformant),
                "" if r.cochrane_consistent_with_evidence is None else int(r.cochrane_consistent_with_evidence),
                r.direction,
                int(r.has_evidence_quotes),
            ])


# --- CLI -------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH,
                        help="Benchmark SQLite DB.")
    parser.add_argument("--out-tsv", type=Path, default=None,
                        help="Optional path to write per-row audit TSV.")
    parser.add_argument(
        "--source-prefixes", default=",".join(MODELS.values()),
        help="Comma-separated source-prefix patterns (without the trailing "
             "pass index). Default audits all 4 models × fulltext × 3 passes.",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"[error] DB not found at {args.db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(f"file:{args.db_path}?mode=ro", uri=True)
    try:
        cochrane = load_cochrane_ratings(conn)
        all_audit: list[AuditRow] = []
        for prefix in args.source_prefixes.split(","):
            prefix = prefix.strip()
            if not prefix:
                continue
            audit = load_audit_rows(conn, f"{prefix}%", cochrane)
            all_audit.extend(audit)
            print(f"[load] {prefix}* → {len(audit)} per-domain rows")
        if not all_audit:
            print("[empty] no audit rows", file=sys.stderr)
            return 1

        stats = aggregate(all_audit)
        print()
        print(format_summary(stats))

        print()
        print("Pooled across all (model × domain × pass):")
        h = overall_asymmetry(stats)
        print(f"  signalling parsed:                {h['n_signalling_parsed']}")
        print(f"  model self-conformance:           {h['self_conformance_pct']:.1f}% "
              f"({h['n_self_conformant']}/{h['n_signalling_parsed']})")
        print(f"  Cochrane matches model-evidence:  {h['match_pct']:.1f}% "
              f"({h['n_cochrane_match']}/{h['n_signalling_parsed']})")
        print(f"  Cochrane MORE LENIENT than algo:  {h['lenient_pct']:.1f}% "
              f"({h['n_cochrane_more_lenient']}/{h['n_signalling_parsed']})")
        print(f"  Cochrane MORE STRICT than algo:   {h['stricter_pct']:.1f}% "
              f"({h['n_cochrane_more_strict']}/{h['n_signalling_parsed']})")
        print(f"  asymmetry ratio (lenient:strict): {h['asymmetry_ratio']:.1f}:1")

        if args.out_tsv:
            args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
            write_tsv(all_audit, args.out_tsv)
            print(f"\n[write] per-row audit TSV → {args.out_tsv} ({len(all_audit)} rows)")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

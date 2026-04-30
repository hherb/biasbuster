"""Phase 3: build the sanitized benchmark SQLite DB.

Combines two data sources into a single canonical schema:

1. The Eisele-Metzger 2025 supplementary CSVs (`Extracted_Data_Test_Data`):
   per-RCT identifiers and per-domain Cochrane + Claude 2 judgments
   (3 independent Claude 2 passes plus a re-run under a different
   condition).

2. Our Phase 1 acquisition results: PMID, DOI, title, has-abstract /
   has-fulltext / has-registration flags, and the source from which
   full text was obtained.

Output: ``dataset/eisele_metzger_benchmark.db`` (SQLite, gitignored).

The DB is the canonical input for all downstream phases:
- Phase 4 will compute κ between cochrane and claude2_run1 from this DB
  to confirm we reproduce Eisele-Metzger's published κ ≈ 0.22.
- Phase 5 will populate `benchmark_judgment` with our four models'
  outputs across protocols and passes.
- Phase 6 will read judgments back out for statistical analysis.

Design choices:
- Clean rebuild on each run (DROP + CREATE), not incremental upsert.
  The EM source data is frozen; rebuilds are cheap; idempotency by
  reconstruction is simpler than tracking a migration trail.
- All judgments stored in normalised form: ``low``, ``some_concerns``,
  ``high`` (per ``prompt_v1.md``). Empty/missing judgments stored as
  NULL with a ``valid=0`` flag for explicit exclusion downstream.
- One ``judgment`` row per (rct_id, source, domain) triple, with
  rationale text in the same row. PRIMARY KEY enforces uniqueness.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/Users/hherb/src/biasbuster")
EM_CSV = PROJECT_ROOT / "DATA/20240318_Data_for_analysis_full/Extracted_Data_Test_Data-Table 1.csv"
FULLTEXT_DIR = PROJECT_ROOT / "DATA/20240318_Data_for_analysis_full/fulltext"
DB_PATH = PROJECT_ROOT / "dataset/eisele_metzger_benchmark.db"
STUDY_DIR = PROJECT_ROOT / "studies/eisele_metzger_replication"
SUMMARY_MD = STUDY_DIR / "benchmark_summary.md"
SPOTCHECK_MD = STUDY_DIR / "benchmark_spotcheck.md"

EM_CSV_HEADER_ROW = 1

# RoB 2 domains as encoded in EM's CSV columns.
DOMAINS = ("d1", "d2", "d3", "d4", "d5", "overall")

# Sources of judgments encoded in the EM CSV. Each (column_prefix, source_label)
# pair becomes 6 rows per RCT (one per domain).
EM_JUDGMENT_SOURCES: list[tuple[str, str]] = [
    ("cr_", "cochrane"),
    ("claude1_", "em_claude2_run1"),
    ("claude2_", "em_claude2_run2"),
    ("claude3_", "em_claude2_run3"),
    ("c3_claude2_", "em_claude2_c3"),
]


# --- Schema --------------------------------------------------------------

SCHEMA_SQL = """
DROP TABLE IF EXISTS benchmark_judgment;
DROP TABLE IF EXISTS benchmark_rct;

CREATE TABLE benchmark_rct (
    rct_id TEXT PRIMARY KEY,
    cr_id TEXT NOT NULL,
    pmid TEXT,
    doi TEXT,
    nct_nr TEXT,
    title TEXT,
    authors_text TEXT,
    publication_year INTEGER,
    condition TEXT,
    intervention TEXT,
    outcome_text TEXT,
    has_abstract INTEGER NOT NULL DEFAULT 0,
    has_fulltext INTEGER NOT NULL DEFAULT 0,
    has_registration INTEGER NOT NULL DEFAULT 0,
    fulltext_source TEXT,
    em_rct_ref TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE benchmark_judgment (
    rct_id TEXT NOT NULL,
    source TEXT NOT NULL,
    domain TEXT NOT NULL,
    judgment TEXT,
    rationale TEXT,
    valid INTEGER NOT NULL DEFAULT 1,
    raw_label TEXT,
    PRIMARY KEY (rct_id, source, domain),
    FOREIGN KEY (rct_id) REFERENCES benchmark_rct(rct_id),
    CHECK (domain IN ('d1', 'd2', 'd3', 'd4', 'd5', 'overall')),
    CHECK (judgment IS NULL OR judgment IN ('low', 'some_concerns', 'high'))
);

CREATE INDEX idx_judgment_source ON benchmark_judgment(source);
CREATE INDEX idx_judgment_domain ON benchmark_judgment(domain);
CREATE INDEX idx_rct_has_fulltext ON benchmark_rct(has_fulltext);
"""


# --- Loaders -------------------------------------------------------------

@dataclass
class EMRow:
    """One full row of the Eisele-Metzger Extracted_Data CSV.

    Fields are dynamically populated from the CSV columns; column names
    are accessed by string key against ``raw`` which preserves the
    arbitrary 86-column structure.
    """
    rct_id: str
    cr_id: str
    raw: dict[str, str]  # full row keyed by header name


def load_em_rows() -> list[EMRow]:
    with open(EM_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[EM_CSV_HEADER_ROW]
    out: list[EMRow] = []
    for raw_row in rows[EM_CSV_HEADER_ROW + 1:]:
        if not any(c.strip() for c in raw_row):
            continue
        row_dict = dict(zip(header, raw_row, strict=False))
        out.append(EMRow(
            rct_id=row_dict.get("id", "").strip(),
            cr_id=row_dict.get("cr_id", "").strip(),
            raw=row_dict,
        ))
    return out


def load_acquisition_metadata() -> dict[str, dict[str, Any]]:
    """Load all per-RCT metadata.json files keyed by rct_id."""
    out: dict[str, dict[str, Any]] = {}
    if not FULLTEXT_DIR.exists():
        return out
    for sub in sorted(FULLTEXT_DIR.iterdir()):
        if not sub.is_dir():
            continue
        meta_path = sub / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            out[sub.name] = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


# --- Normalisation ------------------------------------------------------

# Map raw EM-CSV labels to our canonical scheme. Conservative on case;
# whitespace stripped before lookup.
_LABEL_MAP: dict[str, str] = {
    "low risk": "low",
    "low": "low",
    "low risk of bias": "low",
    "some concerns": "some_concerns",
    "some_concerns": "some_concerns",
    "some concern": "some_concerns",
    "high risk": "high",
    "high": "high",
    "high risk of bias": "high",
}


def normalize_judgment(raw: str) -> tuple[str | None, bool]:
    """Map a raw judgment string to canonical form.

    Returns (canonical, valid) where canonical is one of
    {'low', 'some_concerns', 'high'} or None when the raw value is empty
    or unrecognised. ``valid`` is True iff a recognised label was found.
    """
    if raw is None:
        return None, False
    cleaned = raw.strip().lower()
    if not cleaned:
        return None, False
    if cleaned in _LABEL_MAP:
        return _LABEL_MAP[cleaned], True
    # Try a fuzzier reduction — strip "of bias" and similar trailers
    cleaned2 = re.sub(r"\s+of\s+bias\b", "", cleaned).strip()
    if cleaned2 in _LABEL_MAP:
        return _LABEL_MAP[cleaned2], True
    return None, False


# --- Population ---------------------------------------------------------

def populate(conn: sqlite3.Connection,
             em_rows: list[EMRow],
             acq: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Build the DB from EM CSV + acquisition metadata. Return stats dict."""
    conn.executescript(SCHEMA_SQL)
    cur = conn.cursor()

    n_rct = 0
    n_judgments = 0
    n_invalid = 0
    n_unknown_label = 0
    unknown_label_examples: set[str] = set()

    for em in em_rows:
        meta = acq.get(em.rct_id, {})

        try:
            year = int(meta.get("extracted", {}).get("publication_year") or 0)
        except (TypeError, ValueError):
            year = 0

        cur.execute(
            """INSERT INTO benchmark_rct
            (rct_id, cr_id, pmid, doi, nct_nr, title, authors_text,
             publication_year, condition, intervention, outcome_text,
             has_abstract, has_fulltext, has_registration, fulltext_source,
             em_rct_ref, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                em.rct_id,
                em.cr_id,
                meta.get("pmid") or None,
                meta.get("doi") or None,
                em.raw.get("rct_regnr", "").strip() or None,
                meta.get("title") or None,
                em.raw.get("rct_author", "").strip(),
                year if year else None,
                em.raw.get("rct_condition", "").strip() or None,
                em.raw.get("rct_intervention", "").strip() or None,
                em.raw.get("rct_outcome", "").strip() or None,
                1 if meta.get("has_abstract") else 0,
                1 if meta.get("has_fulltext") else 0,
                1 if meta.get("has_registration") else 0,
                (meta.get("fulltext_source") or None),
                em.raw.get("rct_ref", "").strip(),
                None,
            ),
        )
        n_rct += 1

        # For each judgment source × domain, write one row.
        for col_prefix, source_label in EM_JUDGMENT_SOURCES:
            for domain in DOMAINS:
                column = f"{col_prefix}{domain}"
                rationale_column = f"{col_prefix}{domain}_text"
                raw_label = (em.raw.get(column) or "").strip()
                rationale = (em.raw.get(rationale_column) or "").strip() or None

                canonical, valid = normalize_judgment(raw_label)
                if not valid:
                    if raw_label and raw_label.lower() not in _LABEL_MAP:
                        unknown_label_examples.add(raw_label[:50])
                        n_unknown_label += 1
                    if not raw_label:
                        # Empty cell — store as NULL with valid=0
                        n_invalid += 1
                cur.execute(
                    """INSERT INTO benchmark_judgment
                    (rct_id, source, domain, judgment, rationale, valid, raw_label)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (em.rct_id, source_label, domain, canonical, rationale,
                     1 if valid else 0, raw_label or None),
                )
                n_judgments += 1

    conn.commit()
    return {
        "rcts": n_rct,
        "judgments_total": n_judgments,
        "judgments_invalid_empty": n_invalid,
        "judgments_unknown_label": n_unknown_label,
        "unknown_label_examples": sorted(unknown_label_examples),
    }


# --- Reporting ----------------------------------------------------------

def write_summary(conn: sqlite3.Connection, stats: dict[str, int]) -> None:
    cur = conn.cursor()

    n_rct = cur.execute("SELECT COUNT(*) FROM benchmark_rct").fetchone()[0]
    n_with_pmid = cur.execute(
        "SELECT COUNT(*) FROM benchmark_rct WHERE pmid IS NOT NULL"
    ).fetchone()[0]
    n_with_abstract = cur.execute(
        "SELECT COUNT(*) FROM benchmark_rct WHERE has_abstract = 1"
    ).fetchone()[0]
    n_with_fulltext = cur.execute(
        "SELECT COUNT(*) FROM benchmark_rct WHERE has_fulltext = 1"
    ).fetchone()[0]
    n_with_reg = cur.execute(
        "SELECT COUNT(*) FROM benchmark_rct WHERE has_registration = 1"
    ).fetchone()[0]

    sources = cur.execute(
        "SELECT source, COUNT(*) FROM benchmark_judgment GROUP BY source ORDER BY source"
    ).fetchall()

    label_dist = cur.execute(
        """SELECT source, judgment, COUNT(*)
           FROM benchmark_judgment
           WHERE valid = 1
           GROUP BY source, judgment
           ORDER BY source, judgment"""
    ).fetchall()

    invalid_per_source = cur.execute(
        """SELECT source, COUNT(*)
           FROM benchmark_judgment
           WHERE valid = 0
           GROUP BY source
           ORDER BY source"""
    ).fetchall()

    lines: list[str] = []
    lines.append("# Benchmark Database Build Summary")
    lines.append("")
    lines.append("**Generated:** by `studies/eisele_metzger_replication/build_benchmark_db.py`")
    lines.append(f"**SQLite path:** `{DB_PATH.relative_to(PROJECT_ROOT)}` (gitignored)")
    lines.append("**Companion:** `benchmark_spotcheck.md` (random 10-RCT manual verification sample)")
    lines.append("")
    lines.append("## RCT-level coverage")
    lines.append("")
    lines.append(f"- Total RCTs in benchmark: **{n_rct}**")
    lines.append(f"- With resolved PMID: {n_with_pmid} ({100*n_with_pmid/n_rct:.0f}%)")
    lines.append(f"- With abstract text: {n_with_abstract} ({100*n_with_abstract/n_rct:.0f}%)")
    lines.append(f"- With full text: {n_with_fulltext} ({100*n_with_fulltext/n_rct:.0f}%)")
    lines.append(f"- With trial registration: {n_with_reg} ({100*n_with_reg/n_rct:.0f}%)")
    lines.append("")
    lines.append("## Judgment-row counts by source")
    lines.append("")
    lines.append("| Source | Rows |")
    lines.append("|---|---:|")
    for src, n in sources:
        lines.append(f"| {src} | {n} |")
    lines.append("")
    lines.append(f"Expected: 6 domains × 100 RCTs = 600 rows per source. "
                 f"5 sources from EM × 600 = 3000 total.")
    lines.append(f"Actual: **{stats['judgments_total']}** rows inserted.")
    lines.append("")
    if stats["judgments_invalid_empty"] or stats["judgments_unknown_label"]:
        lines.append("## Validation issues")
        lines.append("")
        lines.append(f"- Empty/missing judgment cells: {stats['judgments_invalid_empty']}")
        lines.append(f"- Unrecognised labels: {stats['judgments_unknown_label']}")
        if stats["unknown_label_examples"]:
            lines.append("")
            lines.append("Example unknown labels (top 20, truncated to 50 chars):")
            lines.append("")
            for ex in stats["unknown_label_examples"][:20]:
                lines.append(f"- `{ex}`")
        if invalid_per_source:
            lines.append("")
            lines.append("| Source | Invalid rows |")
            lines.append("|---|---:|")
            for src, n in invalid_per_source:
                lines.append(f"| {src} | {n} |")
        lines.append("")
    else:
        lines.append("## Validation")
        lines.append("")
        lines.append("✅ No empty cells, no unrecognised labels — clean ingest.")
        lines.append("")
    lines.append("## Label distribution (valid judgments only)")
    lines.append("")
    lines.append("| Source | low | some_concerns | high |")
    lines.append("|---|---:|---:|---:|")
    by_source: dict[str, dict[str, int]] = {}
    for src, judgment, count in label_dist:
        by_source.setdefault(src, {})[judgment] = count
    for src in sorted(by_source):
        d = by_source[src]
        lines.append(
            f"| {src} | {d.get('low', 0)} | "
            f"{d.get('some_concerns', 0)} | {d.get('high', 0)} |"
        )
    lines.append("")
    lines.append("## Schema")
    lines.append("")
    lines.append("```sql")
    lines.append(SCHEMA_SQL.strip())
    lines.append("```")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def write_spotcheck(conn: sqlite3.Connection, em_rows: list[EMRow], n: int = 10) -> None:
    """Pick `n` evenly-spaced RCTs and emit a side-by-side comparison.

    Even spacing (rather than random) keeps the spot-check reproducible —
    re-running the script always picks the same RCTs to verify against the
    EM CSV. A reviewer can compare each block against the source CSV row
    in a few minutes.
    """
    cur = conn.cursor()
    em_by_id = {r.rct_id: r for r in em_rows}
    total = len(em_rows)
    indices = [int(i * total / n) for i in range(n)]
    sample_ids = [em_rows[i].rct_id for i in indices]

    lines: list[str] = []
    lines.append("# Benchmark DB Spot-Check")
    lines.append("")
    lines.append(f"Even-spaced sample of {n} RCTs across the 100-row dataset, "
                 "chosen for reproducibility (same RCTs each rebuild). "
                 "For each RCT we show the loaded `benchmark_rct` row and "
                 "the loaded `benchmark_judgment` rows alongside the "
                 "matching EM CSV cells.")
    lines.append("")
    lines.append(f"**Sample IDs:** {', '.join(sample_ids)}")
    lines.append("")

    for rct_id in sample_ids:
        em = em_by_id[rct_id]
        rct_row = cur.execute(
            "SELECT pmid, doi, nct_nr, title, condition, has_abstract, has_fulltext "
            "FROM benchmark_rct WHERE rct_id = ?",
            (rct_id,),
        ).fetchone()
        lines.append(f"## {rct_id}")
        lines.append("")
        lines.append(f"- **EM author/year:** {em.raw.get('rct_author', '')}")
        lines.append(f"- **EM cr_id:** {em.cr_id}")
        lines.append(f"- **EM citation (rct_ref):** {em.raw.get('rct_ref', '')[:240]}")
        lines.append("")
        if rct_row:
            pmid, doi, nct_nr, title, condition, has_abs, has_ft = rct_row
            lines.append("### Loaded benchmark_rct row")
            lines.append("")
            lines.append(f"- pmid: `{pmid or '-'}`  doi: `{doi or '-'}`  nct: `{nct_nr or '-'}`")
            lines.append(f"- condition: {condition or '-'}")
            lines.append(f"- has_abstract: **{'YES' if has_abs else 'no'}**, has_fulltext: **{'YES' if has_ft else 'no'}**")
            lines.append(f"- resolved title: {title or '-'}")
        lines.append("")
        lines.append("### Judgments — Cochrane vs EM CSV (raw → loaded)")
        lines.append("")
        lines.append("| domain | EM cr cell | EM cr_text first 60 | loaded judgment |")
        lines.append("|---|---|---|---|")
        for domain in DOMAINS:
            em_cell = em.raw.get(f"cr_{domain}", "").strip()
            em_text = em.raw.get(f"cr_{domain}_text", "").strip()[:60].replace("|", "\\|").replace("\n", " ")
            loaded = cur.execute(
                "SELECT judgment, raw_label FROM benchmark_judgment "
                "WHERE rct_id = ? AND source = 'cochrane' AND domain = ?",
                (rct_id, domain),
            ).fetchone()
            loaded_str = (loaded[0] if loaded and loaded[0] else "(NULL)")
            lines.append(f"| {domain} | {em_cell or '-'} | {em_text or '-'} | {loaded_str} |")
        lines.append("")
        lines.append("### Judgments — Claude run 1 vs EM CSV (raw → loaded)")
        lines.append("")
        lines.append("| domain | EM claude1 cell | loaded judgment |")
        lines.append("|---|---|---|")
        for domain in DOMAINS:
            em_cell = em.raw.get(f"claude1_{domain}", "").strip()
            loaded = cur.execute(
                "SELECT judgment FROM benchmark_judgment "
                "WHERE rct_id = ? AND source = 'em_claude2_run1' AND domain = ?",
                (rct_id, domain),
            ).fetchone()
            loaded_str = (loaded[0] if loaded and loaded[0] else "(NULL)")
            lines.append(f"| {domain} | {em_cell or '-'} | {loaded_str} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    SPOTCHECK_MD.write_text("\n".join(lines), encoding="utf-8")


# --- main ---------------------------------------------------------------

def main() -> None:
    print(f"[load] EM CSV: {EM_CSV}")
    em_rows = load_em_rows()
    print(f"[load] {len(em_rows)} EM rows")

    print(f"[load] acquisition metadata: {FULLTEXT_DIR}")
    acq = load_acquisition_metadata()
    print(f"[load] {len(acq)} per-RCT metadata.json files")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[build] {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        stats = populate(conn, em_rows, acq)
        print(f"[stats] {stats['rcts']} RCTs, {stats['judgments_total']} judgment rows")
        if stats["judgments_invalid_empty"] or stats["judgments_unknown_label"]:
            print(
                f"[stats] empty cells: {stats['judgments_invalid_empty']}, "
                f"unknown labels: {stats['judgments_unknown_label']}"
            )
            for ex in stats["unknown_label_examples"][:5]:
                print(f"        unknown label example: {ex!r}")
        write_summary(conn, stats)
        write_spotcheck(conn, em_rows)
    finally:
        conn.close()
    print(f"[write] {SUMMARY_MD}")
    print(f"[write] {SPOTCHECK_MD}")


if __name__ == "__main__":
    main()

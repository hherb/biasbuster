"""Diagnose where gemma4 and Sonnet disagree on V5A assessments.

For each of the 16 validation-set papers, compare both models' output
and classify each domain-level disagreement into one of:

  EXTRACT_GAP
      The Stage 1 extraction produced different inputs to the
      mechanical rules (different `_provenance.domain_severities`),
      so the two models started from different drafts. Fix: improve
      Stage 1 extraction.

  MECHANICAL_AGREE_LLM_DISAGREE
      Both models got the same mechanical draft severity, but their
      Stage 3 override decisions differed (one kept, one downgraded,
      or they downgraded to different targets). Fix: tighten per-domain
      override prompts.

  CALIBRATION_DRIFT
      Same mechanical draft AND same decision direction (both kept or
      both downgraded), but landed on different severity levels within
      adjacent bands (e.g. one said "moderate" the other "high"). Fix:
      explicit severity-boundary definitions in prompts.

  AGREE
      No disagreement at this domain.

Usage:
    uv run python diagnose_v5a_disagreements.py
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from typing import Optional

from biasbuster.database import Database


DOMAINS = [
    "statistical_reporting",
    "spin",
    "outcome_reporting",
    "conflict_of_interest",
    "methodology",
]

_SEVERITY_RANK = {"none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}


def _rank(sev: Optional[str]) -> int:
    if not sev:
        return 0
    return _SEVERITY_RANK.get(str(sev).lower(), 0)


def _decision_for(ann: dict, domain: str) -> dict:
    """Extract the Stage 3 decision for a domain, or synthesize one.

    V5A only records a decision for domains that were elevated (>=
    moderate) AND overridable. For non-reviewed domains, we synthesise
    a "keep" decision so the diagnostic can compare uniformly.
    """
    decisions = ann.get("_decomposed_decisions") or []
    for d in decisions:
        if d.get("domain") == domain:
            return {
                "decision": d.get("decision", "keep"),
                "mechanical": d.get("mechanical_severity", ""),
                "final": d.get("target_severity", ""),
                "reason": d.get("reason", ""),
                "reviewed": True,
            }

    # Not reviewed — read the final severity from the annotation block
    # and infer the mechanical severity from provenance
    prov = ann.get("_provenance") or ann.get("_mechanical_provenance") or {}
    mech_sev_map = prov.get("domain_severities") or {}
    mech_sev = mech_sev_map.get(domain, "")
    final_sev = str(ann.get(domain, {}).get("severity", "")).lower()
    return {
        "decision": "not_reviewed",
        "mechanical": mech_sev,
        "final": final_sev,
        "reason": "",
        "reviewed": False,
    }


def classify_disagreement(sonnet: dict, gemma: dict, domain: str) -> dict:
    """Return a classification dict for one domain's disagreement."""
    s = _decision_for(sonnet, domain)
    g = _decision_for(gemma, domain)

    s_final = s["final"]
    g_final = g["final"]
    s_mech = s["mechanical"]
    g_mech = g["mechanical"]

    # Agreement
    if s_final == g_final:
        return {
            "class": "AGREE",
            "sonnet_mech": s_mech,
            "gemma_mech": g_mech,
            "sonnet_final": s_final,
            "gemma_final": g_final,
            "sonnet_decision": s["decision"],
            "gemma_decision": g["decision"],
            "sonnet_reason": s["reason"],
            "gemma_reason": g["reason"],
        }

    # Extract gap — mechanical draft differs
    if s_mech != g_mech:
        return {
            "class": "EXTRACT_GAP",
            "sonnet_mech": s_mech,
            "gemma_mech": g_mech,
            "sonnet_final": s_final,
            "gemma_final": g_final,
            "sonnet_decision": s["decision"],
            "gemma_decision": g["decision"],
            "sonnet_reason": s["reason"],
            "gemma_reason": g["reason"],
        }

    # Same mechanical, different final — is it decision-direction drift or calibration?
    same_direction = (s["decision"] == g["decision"])
    if same_direction:
        cls = "CALIBRATION_DRIFT"
    else:
        cls = "MECHANICAL_AGREE_LLM_DISAGREE"

    return {
        "class": cls,
        "sonnet_mech": s_mech,
        "gemma_mech": g_mech,
        "sonnet_final": s_final,
        "gemma_final": g_final,
        "sonnet_decision": s["decision"],
        "gemma_decision": g["decision"],
        "sonnet_reason": s["reason"],
        "gemma_reason": g["reason"],
    }


def main() -> int:
    db = Database()
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    sonnet_tag = "anthropic_fulltext_decomposed"
    gemma_tag = "ollama:gemma4:26b-a4b-it-q8_0_fulltext_decomposed"

    def load(tag: str) -> dict[str, dict]:
        rows = conn.execute(
            "SELECT pmid, annotation FROM annotations WHERE model_name = ? ORDER BY pmid",
            (tag,),
        ).fetchall()
        return {r["pmid"]: json.loads(r["annotation"]) for r in rows}

    sonnet_anns = load(sonnet_tag)
    gemma_anns = load(gemma_tag)
    common = sorted(set(sonnet_anns) & set(gemma_anns))
    print(f"Comparing {len(common)} papers\n")

    # Tally per domain
    domain_class_counts: dict[str, Counter] = {d: Counter() for d in DOMAINS}
    paper_disagreements: list[dict] = []

    for pmid in common:
        s_ann = sonnet_anns[pmid]
        g_ann = gemma_anns[pmid]
        paper_row: dict = {"pmid": pmid, "disagreements": []}
        for dom in DOMAINS:
            d = classify_disagreement(s_ann, g_ann, dom)
            domain_class_counts[dom][d["class"]] += 1
            if d["class"] != "AGREE":
                paper_row["disagreements"].append({"domain": dom, **d})
        if paper_row["disagreements"]:
            paper_disagreements.append(paper_row)

    # ---- Summary tables ----
    print("## Disagreement class × domain (counts per 16-paper set)")
    print()
    header = f'{"Domain":<25} {"AGREE":>7} {"EXTRACT_GAP":>13} {"MECH=LLM≠":>11} {"CAL_DRIFT":>11}'
    print(header)
    print("─" * len(header))
    for dom in DOMAINS:
        counts = domain_class_counts[dom]
        print(
            f'{dom:<25} '
            f'{counts["AGREE"]:>7} '
            f'{counts["EXTRACT_GAP"]:>13} '
            f'{counts["MECHANICAL_AGREE_LLM_DISAGREE"]:>11} '
            f'{counts["CALIBRATION_DRIFT"]:>11}'
        )
    print()

    # ---- Per-paper detail ----
    print("## Per-paper disagreement detail")
    print()
    for paper in paper_disagreements:
        print(f"### PMID {paper['pmid']}  ({len(paper['disagreements'])} disagreements)")
        for dis in paper["disagreements"]:
            dom = dis["domain"]
            print(f"  {dom}: [{dis['class']}]")
            print(
                f"    Sonnet: mech={dis['sonnet_mech']} → final={dis['sonnet_final']} "
                f"(decision={dis['sonnet_decision']})"
            )
            print(
                f"    gemma4: mech={dis['gemma_mech']} → final={dis['gemma_final']} "
                f"(decision={dis['gemma_decision']})"
            )
            if dis["class"] == "MECHANICAL_AGREE_LLM_DISAGREE":
                if dis["sonnet_reason"]:
                    print(f"    Sonnet reason: {dis['sonnet_reason'][:200]}")
                if dis["gemma_reason"]:
                    print(f"    gemma4 reason: {dis['gemma_reason'][:200]}")
        print()

    # ---- Aggregate totals ----
    total_counts: Counter = Counter()
    for counts in domain_class_counts.values():
        for cls, n in counts.items():
            total_counts[cls] += n
    print("## Aggregate totals (across all domains × papers)")
    print()
    total = sum(total_counts.values())
    for cls in ("AGREE", "EXTRACT_GAP", "MECHANICAL_AGREE_LLM_DISAGREE", "CALIBRATION_DRIFT"):
        n = total_counts[cls]
        pct = 100.0 * n / total if total else 0.0
        print(f"  {cls:<36} {n:>4}  ({pct:5.1f}%)")
    print()
    disagreements = total - total_counts["AGREE"]
    if disagreements:
        print(f"Of {disagreements} total disagreements:")
        for cls in ("EXTRACT_GAP", "MECHANICAL_AGREE_LLM_DISAGREE", "CALIBRATION_DRIFT"):
            n = total_counts[cls]
            pct = 100.0 * n / disagreements
            print(f"  {cls:<36} {n:>4}  ({pct:5.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

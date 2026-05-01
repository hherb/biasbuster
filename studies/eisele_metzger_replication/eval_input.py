"""Load per-RCT input materials for the model evaluation.

Per the locked prompt specification (`prompt_v1.md` §4), the user message
varies by protocol:

- ABSTRACT: ``[ABSTRACT — main paper]\\n{abstract_text}``
- FULLTEXT: ``[MAIN PAPER — full text]\\n{main}\\n\\n[PROTOCOL]\\n{prot_or_NA}\\n\\n[TRIAL REGISTRATION]\\n{reg_or_NA}``

For the FULLTEXT protocol where we have only an abstract (because Phase 1
couldn't fetch full text), we fall back to the abstract and label it as
such. This is a graceful degradation mode rather than an error: the
benchmark is designed to be evaluated on the materials we actually have
for each RCT, with the asymmetry transparently reported in §6.5 of the
pre-reg as a sensitivity analysis subgroup.

JATS XML parsing is intentionally minimal — we strip tags and keep text.
PDF text extraction is deferred (currently only 6 of 100 RCTs have PDF
without JATS XML; revisit if needed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULLTEXT_DIR = PROJECT_ROOT / "DATA/20240318_Data_for_analysis_full/fulltext"

NOT_AVAILABLE = "NOT AVAILABLE"


@dataclass
class RctInput:
    """Loaded materials for one RCT under one protocol."""
    rct_id: str
    protocol: str  # 'abstract' or 'fulltext'
    rct_label: str
    outcome_text: str
    materials_block: str
    has_abstract: bool
    has_fulltext: bool
    has_registration: bool
    fulltext_source: str  # 'jats_xml', 'pdf', 'abstract_fallback', or ''


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _strip_jats(xml: str) -> str:
    """Convert a JATS XML body to plain text.

    Strategy: remove all XML tags, collapse runs of whitespace,
    keep paragraph breaks where they occur naturally. This is a
    minimum-fuss approach — we don't try to honour structural
    elements like <table>, <fig>, etc., because the LLM's job is
    to read text, not parse structure.
    """
    if not xml:
        return ""
    # Drop everything outside <body>...</body> if present (cuts journal
    # boilerplate, references, citation graphs, etc.).
    body_match = re.search(r"<body[^>]*>(.*?)</body>", xml, re.DOTALL | re.IGNORECASE)
    text_source = body_match.group(1) if body_match else xml
    # Convert paragraph breaks to newlines before stripping tags.
    text_source = re.sub(r"</p>", "\n\n", text_source, flags=re.IGNORECASE)
    text_source = re.sub(r"</sec>", "\n\n", text_source, flags=re.IGNORECASE)
    text_source = re.sub(r"<title[^>]*>", "## ", text_source, flags=re.IGNORECASE)
    text_source = re.sub(r"</title>", "\n", text_source, flags=re.IGNORECASE)
    # Strip remaining tags.
    text_source = re.sub(r"<[^>]+>", "", text_source)
    # Collapse whitespace runs but preserve blank lines.
    text_source = re.sub(r"[ \t]+", " ", text_source)
    text_source = re.sub(r"\n{3,}", "\n\n", text_source)
    return text_source.strip()


def _registration_summary(reg_json_path: Path) -> str:
    """Extract a compact human-readable view of a ClinicalTrials.gov record.

    The full record is several kilobytes of nested JSON; we surface the
    fields relevant to RoB 2 (design, allocation, masking, primary outcomes,
    eligibility) and skip administrative metadata.
    """
    if not reg_json_path.exists():
        return ""
    import json
    try:
        data = json.loads(reg_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    proto = data.get("protocolSection", {})
    out: list[str] = []
    iden = proto.get("identificationModule", {})
    if title := iden.get("officialTitle") or iden.get("briefTitle"):
        out.append(f"Title: {title}")
    if nct := iden.get("nctId"):
        out.append(f"NCT: {nct}")
    design = proto.get("designModule", {})
    if design:
        info = design.get("designInfo", {})
        bits = []
        if alloc := info.get("allocation"):
            bits.append(f"allocation={alloc}")
        if model := info.get("interventionModel"):
            bits.append(f"model={model}")
        if mask := info.get("maskingInfo", {}).get("masking"):
            bits.append(f"masking={mask}")
        if who := info.get("maskingInfo", {}).get("whoMasked"):
            bits.append(f"whoMasked={','.join(who)}")
        if bits:
            out.append("Design: " + "; ".join(bits))
    outcomes = proto.get("outcomesModule", {})
    if primary := outcomes.get("primaryOutcomes"):
        out.append("Primary outcomes:")
        for p in primary[:5]:
            measure = p.get("measure", "")
            timeframe = p.get("timeFrame", "")
            out.append(f"  - {measure} (timeframe: {timeframe})")
    eligibility = proto.get("eligibilityModule", {})
    if crit := eligibility.get("eligibilityCriteria"):
        out.append("Eligibility criteria:\n" + crit[:1500])
    return "\n".join(out).strip()


def load_rct_input(rct_id: str, protocol: str,
                   rct_label: str, outcome_text: str) -> RctInput | None:
    """Build the materials block for one RCT under one protocol.

    Returns None if the RCT has no usable materials at all (no abstract
    AND no full text). The orchestrator should skip such RCTs and record
    them as "no input available" without ever issuing model calls.
    """
    rct_dir = FULLTEXT_DIR / rct_id
    abstract_path = rct_dir / "abstract.txt"
    jats_path = rct_dir / "paper.jats.xml"
    pdf_path = rct_dir / "paper.pdf"
    reg_path = rct_dir / "registration.json"

    abstract_text = _read_file(abstract_path).strip() if abstract_path.exists() else ""
    has_abstract = bool(abstract_text)
    has_jats = jats_path.exists()
    has_pdf = pdf_path.exists()
    has_fulltext = has_jats or has_pdf
    has_registration = reg_path.exists()

    if protocol == "abstract":
        if not has_abstract:
            return None
        materials_block = f"[ABSTRACT — main paper]\n{abstract_text}"
        fulltext_source = ""
    elif protocol == "fulltext":
        # Prefer JATS, fall back to PDF (not yet implemented), then to abstract.
        main_paper_text = ""
        fulltext_source = ""
        if has_jats:
            main_paper_text = _strip_jats(_read_file(jats_path))
            fulltext_source = "jats_xml"
        # PDF parsing path intentionally omitted in v1; if the JATS XML
        # is missing we fall back to abstract for the FULLTEXT protocol
        # rather than introduce a heavy PDF-text dependency just yet.
        if not main_paper_text and has_abstract:
            main_paper_text = abstract_text
            fulltext_source = "abstract_fallback"
        if not main_paper_text:
            return None

        prot_text = NOT_AVAILABLE
        # Protocol PDF/text is a separate field in EM's CSV but rarely
        # populated; we skip the lookup in v1 and tell the model NOT AVAILABLE.

        reg_text = _registration_summary(reg_path) if has_registration else NOT_AVAILABLE
        if not reg_text:
            reg_text = NOT_AVAILABLE

        materials_block = (
            f"[MAIN PAPER — full text]\n{main_paper_text}\n\n"
            f"[PROTOCOL]\n{prot_text}\n\n"
            f"[TRIAL REGISTRATION]\n{reg_text}"
        )
    else:
        raise ValueError(f"unknown protocol: {protocol!r}")

    return RctInput(
        rct_id=rct_id,
        protocol=protocol,
        rct_label=rct_label,
        outcome_text=outcome_text,
        materials_block=materials_block,
        has_abstract=has_abstract,
        has_fulltext=has_fulltext,
        has_registration=has_registration,
        fulltext_source=fulltext_source,
    )


# --- User message construction (per prompt_v1.md §4.1) -----------------

USER_MESSAGE_TEMPLATE = """You are assessing one outcome of a single RCT for risk of bias.

RCT identification:
  Study label: {rct_label}
  Outcome under assessment: {outcome_text}

Available source materials follow. If a signalling question cannot be
answered from the available text, return "NI" (No Information) for that
question. Do not infer details that are not present in the text.

--- BEGIN SOURCE MATERIALS ---
{materials_block}
--- END SOURCE MATERIALS ---

Apply the RoB 2 procedure described in your system prompt and return
the JSON object specified."""


SYNTHESIS_USER_MESSAGE_TEMPLATE = """You are completing the RoB 2 overall judgement for a single RCT outcome.

RCT identification:
  Study label: {rct_label}
  Outcome under assessment: {outcome_text}

Per-domain judgements (your earlier outputs):
  Domain 1 (randomization process): {d1}
  Domain 2 (deviations from intended interventions): {d2}
  Domain 3 (missing outcome data): {d3}
  Domain 4 (measurement of the outcome): {d4}
  Domain 5 (selection of the reported result): {d5}

Apply Cochrane's worst-wins rule and return the JSON object specified."""


def build_domain_user_message(rct_input: RctInput) -> str:
    return USER_MESSAGE_TEMPLATE.format(
        rct_label=rct_input.rct_label,
        outcome_text=rct_input.outcome_text or "(no outcome specified)",
        materials_block=rct_input.materials_block,
    )


def build_synthesis_user_message(rct_input: RctInput,
                                 domain_judgments: dict[str, str]) -> str:
    return SYNTHESIS_USER_MESSAGE_TEMPLATE.format(
        rct_label=rct_input.rct_label,
        outcome_text=rct_input.outcome_text or "(no outcome specified)",
        d1=domain_judgments.get("d1", "(missing)"),
        d2=domain_judgments.get("d2", "(missing)"),
        d3=domain_judgments.get("d3", "(missing)"),
        d4=domain_judgments.get("d4", "(missing)"),
        d5=domain_judgments.get("d5", "(missing)"),
    )

"""
Tool wrappers for the verification agent.

Thin async wrappers around existing collectors and enrichers. Each function
manages its own async context manager lifecycle and returns a standardised
``ToolResult``.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from agent.agent_config import AgentConfig
from agent.tool_router import ToolCall

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standardised result from a verification tool."""

    tool_name: str
    success: bool
    summary: str = ""
    detail: str = ""
    raw_data: dict = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Individual tool wrappers
# ---------------------------------------------------------------------------


async def check_clinicaltrials(
    nct_id: str,
    abstract: str,
    title: str,
    pmid: str,
    config: AgentConfig,
) -> ToolResult:
    """Check ClinicalTrials.gov for trial registration and outcome switching.

    If no NCT ID is provided, attempts to extract one from the abstract
    or search by title keywords.
    """
    from collectors.clinicaltrials_gov import ClinicalTrialsGovCollector

    try:
        async with ClinicalTrialsGovCollector() as ctgov:
            # Try to find the NCT ID if not provided
            if not nct_id and abstract:
                nct_id = ctgov.extract_nct_from_abstract(abstract) or ""
            if not nct_id and title:
                results = await ctgov.search_by_title_keywords(title)
                if results:
                    nct_id = results[0].nct_id
                    logger.info("Found trial via title search: %s", nct_id)

            if not nct_id:
                return ToolResult(
                    tool_name="clinicaltrials_gov",
                    success=False,
                    summary="No NCT ID found in abstract or by title search.",
                    detail="Could not identify a ClinicalTrials.gov registration "
                           "for this study. The abstract does not contain an NCT "
                           "identifier and title-based search returned no matches.",
                )

            # Fetch registration
            reg = await ctgov.fetch_study(nct_id)
            if reg is None:
                return ToolResult(
                    tool_name="clinicaltrials_gov",
                    success=False,
                    summary=f"NCT ID {nct_id} not found on ClinicalTrials.gov.",
                )

            # Detect outcome switching
            switching = await ctgov.detect_outcome_switching(
                nct_id=nct_id,
                published_abstract=abstract,
                published_title=title,
                pmid=pmid,
            )

            # Format results
            primary_outcomes = [o.measure for o in reg.primary_outcomes]
            secondary_outcomes = [o.measure for o in reg.secondary_outcomes[:5]]

            detail_parts = [
                f"**Trial:** {reg.title}",
                f"**NCT ID:** {nct_id}",
                f"**Status:** {reg.status}",
                f"**Phase:** {reg.phase}",
                f"**Sponsor:** {reg.lead_sponsor} ({reg.lead_sponsor_type})",
                f"**Funding source:** {reg.funding_source}",
                f"**Registered primary outcomes:** {'; '.join(primary_outcomes) or 'None listed'}",
                f"**Registered secondary outcomes:** {'; '.join(secondary_outcomes) or 'None listed'}",
            ]

            if switching.primary_outcome_switched:
                detail_parts.append("\n**OUTCOME SWITCHING DETECTED**")
                detail_parts.append(f"Evidence: {'; '.join(switching.evidence)}")
            elif switching.outcomes_omitted:
                detail_parts.append("\n**OUTCOME OMISSION DETECTED**")
                detail_parts.append(f"Evidence: {'; '.join(switching.evidence)}")
            else:
                detail_parts.append("\nNo outcome switching detected.")

            summary = (
                f"Trial {nct_id} — Sponsor: {reg.lead_sponsor} ({reg.lead_sponsor_type})"
            )
            if switching.primary_outcome_switched:
                summary += " — OUTCOME SWITCHING DETECTED"

            return ToolResult(
                tool_name="clinicaltrials_gov",
                success=True,
                summary=summary,
                detail="\n".join(detail_parts),
                raw_data={
                    "nct_id": nct_id,
                    "sponsor": reg.lead_sponsor,
                    "sponsor_type": reg.lead_sponsor_type,
                    "funding_source": reg.funding_source,
                    "primary_outcomes": primary_outcomes,
                    "outcome_switched": switching.primary_outcome_switched,
                    "outcomes_omitted": switching.outcomes_omitted,
                },
            )

    except Exception as exc:
        logger.exception("ClinicalTrials.gov check failed")
        return ToolResult(
            tool_name="clinicaltrials_gov",
            success=False,
            error=str(exc),
            summary=f"ClinicalTrials.gov check failed: {exc}",
        )


async def check_open_payments(
    authors: list[dict[str, str]],
    config: AgentConfig,
) -> ToolResult:
    """Search CMS Open Payments for author payment records."""
    from enrichers.author_coi import AuthorCOIVerifier

    if not authors:
        return ToolResult(
            tool_name="open_payments",
            success=False,
            summary="No author names available for Open Payments search.",
        )

    try:
        async with AuthorCOIVerifier(
            mailto=config.crossref_mailto,
            ncbi_api_key=config.ncbi_api_key,
        ) as verifier:
            all_records: list[dict] = []
            for author in authors[:3]:  # Limit to first 3 authors
                last = author.get("last", "")
                first = author.get("first", "")
                if not last:
                    continue
                records = await verifier.search_open_payments(last, first)
                for rec in records:
                    all_records.append({
                        "physician": f"{first} {last}",
                        "company": rec.company,
                        "amount_usd": rec.amount_usd,
                        "payment_type": rec.payment_type,
                        "year": rec.year,
                    })

            if not all_records:
                return ToolResult(
                    tool_name="open_payments",
                    success=True,
                    summary="No payment records found in CMS Open Payments.",
                    detail="Searched for: " + ", ".join(
                        f"{a.get('first', '')} {a.get('last', '')}"
                        for a in authors[:3]
                    ) + ". Note: CMS Open Payments covers US physicians only.",
                )

            total = sum(r["amount_usd"] for r in all_records)
            companies = sorted(set(r["company"] for r in all_records))

            detail_lines = [
                f"**Total payments found:** ${total:,.2f}",
                f"**Companies:** {', '.join(companies)}",
                "",
            ]
            for rec in all_records[:10]:
                detail_lines.append(
                    f"- {rec['physician']}: ${rec['amount_usd']:,.2f} from "
                    f"{rec['company']} ({rec['payment_type']}, {rec['year']})"
                )

            return ToolResult(
                tool_name="open_payments",
                success=True,
                summary=f"Found {len(all_records)} payment records totalling ${total:,.2f}",
                detail="\n".join(detail_lines),
                raw_data={"records": all_records[:20], "total_usd": total},
            )

    except Exception as exc:
        logger.exception("Open Payments check failed")
        return ToolResult(
            tool_name="open_payments",
            success=False,
            error=str(exc),
            summary=f"Open Payments check failed: {exc}",
        )


async def check_orcid(
    authors: list[dict[str, str]],
    config: AgentConfig,
) -> ToolResult:
    """Search ORCID for author affiliation histories."""
    from enrichers.author_coi import AuthorCOIVerifier

    if not authors:
        return ToolResult(
            tool_name="orcid",
            success=False,
            summary="No author names available for ORCID search.",
        )

    try:
        async with AuthorCOIVerifier(
            mailto=config.crossref_mailto,
            ncbi_api_key=config.ncbi_api_key,
        ) as verifier:
            all_affiliations: list[dict] = []
            for author in authors[:3]:
                name = f"{author.get('first', '')} {author.get('last', '')}".strip()
                if not name:
                    continue
                results = await verifier.search_orcid(name)
                for result in results:
                    all_affiliations.append({
                        "query_name": name,
                        **result,
                    })

            if not all_affiliations:
                return ToolResult(
                    tool_name="orcid",
                    success=True,
                    summary="No ORCID profiles found for queried authors.",
                    detail="Searched for: " + ", ".join(
                        f"{a.get('first', '')} {a.get('last', '')}"
                        for a in authors[:3]
                    ),
                )

            detail_lines = []
            for aff in all_affiliations[:10]:
                detail_lines.append(
                    f"- **{aff.get('query_name', '')}**: "
                    f"{aff.get('organization', 'Unknown org')} "
                    f"({aff.get('role', 'role unknown')}, "
                    f"{aff.get('start_year', '?')}-{aff.get('end_year', 'present')})"
                )

            # Check for industry affiliations
            industry_flags = [
                aff for aff in all_affiliations
                if any(
                    ind in aff.get("organization", "").lower()
                    for ind in ["pharma", "biotech", "therapeutics", "pfizer",
                                "novartis", "roche", "merck", "amgen", "abbvie"]
                )
            ]

            summary = f"Found {len(all_affiliations)} affiliations"
            if industry_flags:
                summary += f" ({len(industry_flags)} industry-linked)"

            return ToolResult(
                tool_name="orcid",
                success=True,
                summary=summary,
                detail="\n".join(detail_lines),
                raw_data={
                    "affiliations": all_affiliations[:20],
                    "industry_flags": len(industry_flags),
                },
            )

    except Exception as exc:
        logger.exception("ORCID check failed")
        return ToolResult(
            tool_name="orcid",
            success=False,
            error=str(exc),
            summary=f"ORCID check failed: {exc}",
        )


async def check_europmc_funding(
    pmid: str,
    config: AgentConfig,
) -> ToolResult:
    """Get funder information from Europe PMC."""
    from enrichers.author_coi import AuthorCOIVerifier

    if not pmid:
        return ToolResult(
            tool_name="europmc",
            success=False,
            summary="No PMID provided for Europe PMC lookup.",
        )

    try:
        async with AuthorCOIVerifier(
            mailto=config.crossref_mailto,
            ncbi_api_key=config.ncbi_api_key,
        ) as verifier:
            grants = await verifier.get_europmc_funding(pmid)

            if not grants:
                return ToolResult(
                    tool_name="europmc",
                    success=True,
                    summary="No funder metadata found in Europe PMC.",
                    detail=f"No grant/funding records found for PMID {pmid} "
                           "in Europe PMC. This may mean the full text is not "
                           "indexed or funding is not machine-extractable.",
                )

            detail_lines = []
            for grant in grants:
                agency = grant.get("agency", "Unknown agency")
                grant_id = grant.get("id", "")
                detail_lines.append(f"- **{agency}** (Grant: {grant_id})" if grant_id else f"- **{agency}**")

            agencies = sorted(set(g.get("agency", "") for g in grants))

            return ToolResult(
                tool_name="europmc",
                success=True,
                summary=f"Found {len(grants)} funding source(s): {', '.join(agencies[:3])}",
                detail="\n".join(detail_lines),
                raw_data={"grants": grants},
            )

    except Exception as exc:
        logger.exception("Europe PMC check failed")
        return ToolResult(
            tool_name="europmc",
            success=False,
            error=str(exc),
            summary=f"Europe PMC check failed: {exc}",
        )


async def check_retraction_status(
    pmid: str,
    doi: str,
    config: AgentConfig,
) -> ToolResult:
    """Check for retraction notices via PubMed metadata."""
    from collectors.retraction_watch import RetractionWatchCollector

    if not pmid:
        return ToolResult(
            tool_name="retraction_watch",
            success=False,
            summary="No PMID provided for retraction check.",
        )

    try:
        async with RetractionWatchCollector(
            mailto=config.crossref_mailto or "biasbuster@example.com",
            ncbi_api_key=config.ncbi_api_key,
        ) as collector:
            article = await collector.fetch_pubmed_abstract(pmid)

            if article is None:
                return ToolResult(
                    tool_name="retraction_watch",
                    success=False,
                    summary=f"Could not fetch PubMed record for PMID {pmid}.",
                )

            title = article.get("title", "")
            is_retracted = any(
                kw in title.lower()
                for kw in ["retract", "withdraw", "expression of concern"]
            )

            if is_retracted:
                return ToolResult(
                    tool_name="retraction_watch",
                    success=True,
                    summary=f"RETRACTION/CONCERN DETECTED for PMID {pmid}",
                    detail=f"**Title:** {title}\n\nThis paper appears to have "
                           "a retraction notice, withdrawal, or expression of concern.",
                    raw_data={"retracted": True, "title": title},
                )

            return ToolResult(
                tool_name="retraction_watch",
                success=True,
                summary=f"No retraction notices found for PMID {pmid}.",
                detail=f"PubMed record exists. Title: {title}\n"
                       "No retraction, withdrawal, or expression of concern detected.",
                raw_data={"retracted": False, "title": title},
            )

    except Exception as exc:
        logger.exception("Retraction check failed")
        return ToolResult(
            tool_name="retraction_watch",
            success=False,
            error=str(exc),
            summary=f"Retraction check failed: {exc}",
        )


async def run_effect_size_audit(
    pmid: str,
    title: str,
    abstract: str,
) -> ToolResult:
    """Run the heuristic effect-size reporting audit (local, no network)."""
    from enrichers.effect_size_auditor import audit_abstract

    try:
        audit = audit_abstract(pmid=pmid, title=title, abstract=abstract)

        detail_lines = [
            f"**Reporting pattern:** {audit.pattern.value}",
            f"**Reporting bias score:** {audit.reporting_bias_score:.2f}",
        ]
        if audit.relative_measures_found:
            detail_lines.append(f"**Relative measures:** {', '.join(audit.relative_measures_found)}")
        if audit.absolute_measures_found:
            detail_lines.append(f"**Absolute measures:** {', '.join(audit.absolute_measures_found)}")
        if audit.flags:
            detail_lines.append(f"**Flags:** {'; '.join(audit.flags)}")

        return ToolResult(
            tool_name="effect_size_audit",
            success=True,
            summary=f"Pattern: {audit.pattern.value} (score: {audit.reporting_bias_score:.2f})",
            detail="\n".join(detail_lines),
            raw_data={
                "pattern": audit.pattern.value,
                "score": audit.reporting_bias_score,
                "relative_measures": audit.relative_measures_found,
                "absolute_measures": audit.absolute_measures_found,
                "flags": audit.flags,
            },
        )

    except Exception as exc:
        logger.exception("Effect size audit failed")
        return ToolResult(
            tool_name="effect_size_audit",
            success=False,
            error=str(exc),
            summary=f"Effect size audit failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TOOL_DISPLAY_NAMES = {
    "clinicaltrials_gov": "ClinicalTrials.gov",
    "open_payments": "CMS Open Payments",
    "orcid": "ORCID",
    "europmc": "Europe PMC",
    "retraction_watch": "Retraction Watch",
    "effect_size_audit": "Effect Size Audit",
    "unsupported": "Unsupported",
}


def get_tool_display_name(tool_name: str) -> str:
    """Return a human-readable display name for a tool."""
    return _TOOL_DISPLAY_NAMES.get(tool_name, tool_name)


async def execute_tool_call(call: ToolCall, config: AgentConfig) -> ToolResult:
    """Dispatch a ToolCall to the appropriate wrapper function.

    Args:
        call: The resolved tool call from the router.
        config: Agent configuration.

    Returns:
        A ToolResult with the verification findings.
    """
    if call.tool_name == "unsupported":
        return ToolResult(
            tool_name="unsupported",
            success=False,
            summary="Automatic verification not available for this step.",
            detail=f"**Original recommendation:** {call.original_step}\n\n"
                   "This verification step requires manual action.",
        )

    if call.tool_name == "clinicaltrials_gov":
        return await check_clinicaltrials(
            nct_id=call.params.get("nct_id", ""),
            abstract=call.params.get("abstract", ""),
            title=call.params.get("title", ""),
            pmid=call.params.get("pmid", ""),
            config=config,
        )

    if call.tool_name == "open_payments":
        return await check_open_payments(
            authors=call.params.get("authors", []),
            config=config,
        )

    if call.tool_name == "orcid":
        return await check_orcid(
            authors=call.params.get("authors", []),
            config=config,
        )

    if call.tool_name == "europmc":
        return await check_europmc_funding(
            pmid=call.params.get("pmid", ""),
            config=config,
        )

    if call.tool_name == "retraction_watch":
        return await check_retraction_status(
            pmid=call.params.get("pmid", ""),
            doi=call.params.get("doi", ""),
            config=config,
        )

    if call.tool_name == "effect_size_audit":
        return await run_effect_size_audit(
            pmid=call.params.get("pmid", ""),
            title=call.params.get("title", ""),
            abstract=call.params.get("abstract", ""),
        )

    return ToolResult(
        tool_name=call.tool_name,
        success=False,
        error=f"Unknown tool: {call.tool_name}",
        summary=f"Unknown tool: {call.tool_name}",
    )

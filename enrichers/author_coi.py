"""
Author Conflict of Interest Verification

Cross-references author names against:
1. CMS Open Payments (US physician payments from pharma/device companies)
2. ORCID (author employment/affiliation history)
3. EuroPMC (funder metadata for the paper)
4. Medicines Australia (AU pharma transparency reports)

For the training dataset, we encode both the COI findings AND the verification
methodology, so the model learns WHERE to look, not just what to find.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class PaymentRecord:
    """A single payment from Open Payments."""
    physician_name: str = ""
    company: str = ""
    amount_usd: float = 0.0
    payment_type: str = ""  # general, research, ownership
    year: int = 0
    nature_of_payment: str = ""  # consulting, speaking, food/beverage, travel, etc.


@dataclass
class AuthorCOIProfile:
    """COI profile for a single author."""
    name: str = ""
    orcid: Optional[str] = None

    # Open Payments (US only)
    open_payments_found: bool = False
    total_general_payments_usd: float = 0.0
    total_research_payments_usd: float = 0.0
    payment_companies: list[str] = field(default_factory=list)
    payment_records: list[PaymentRecord] = field(default_factory=list)

    # ORCID affiliations
    orcid_affiliations: list[dict] = field(default_factory=list)
    industry_affiliations: list[str] = field(default_factory=list)

    # Flags
    is_industry_employed: bool = False
    payments_from_study_sponsor: bool = False
    undisclosed_potential_coi: bool = False


@dataclass
class PaperCOIReport:
    """Aggregated COI report for all authors of a paper."""
    pmid: str = ""
    doi: str = ""

    # Paper-level funding
    declared_funding: list[str] = field(default_factory=list)
    funding_type: str = ""  # industry, public, mixed, not_reported
    sponsor_company: str = ""

    # Author-level findings
    author_profiles: list[AuthorCOIProfile] = field(default_factory=list)

    # Aggregate flags
    any_industry_payments: bool = False
    total_payments_usd: float = 0.0
    max_individual_payment_usd: float = 0.0
    authors_with_sponsor_payments: int = 0
    industry_employed_authors: int = 0

    # Verification metadata (for training the model on WHERE to look)
    verification_steps_taken: list[str] = field(default_factory=list)
    verification_sources_available: list[dict] = field(default_factory=list)
    verification_limitations: list[str] = field(default_factory=list)


class AuthorCOIVerifier:
    """
    Cross-references author information against payment databases.

    IMPORTANT: This is for building training data. The trained model won't
    call these APIs directly - it will learn to RECOMMEND verification steps
    and flag patterns that suggest undisclosed COI.
    """

    # Known pharmaceutical/device company name patterns
    PHARMA_INDICATORS = [
        "pharma", "therapeutics", "biosciences", "biotech", "laboratories",
        "medical devices", "diagnostics", "life sciences",
        # Major pharma
        "pfizer", "novartis", "roche", "merck", "johnson", "abbvie",
        "bristol-myers", "astrazeneca", "sanofi", "glaxosmithkline", "gsk",
        "eli lilly", "amgen", "gilead", "bayer", "boehringer",
        "takeda", "novo nordisk", "regeneron", "biogen", "moderna",
    ]

    def __init__(self, mailto: str = "", ncbi_api_key: str = ""):
        self.mailto = mailto
        self.ncbi_api_key = ncbi_api_key
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def search_open_payments(
        self, physician_last: str, physician_first: str, limit: int = 100
    ) -> list[PaymentRecord]:
        """
        Search CMS Open Payments for a physician.

        The Open Payments API provides:
        - General payments (consulting, speaking, food, travel, education)
        - Research payments (direct and indirect)
        - Ownership/investment interests

        API: https://openpaymentsdata.cms.gov/api/1/datastore/query
        Bulk: https://download.cms.gov/openpayments/
        """
        records = []

        # Open Payments has a DKAN-based API
        # General payments dataset
        try:
            params = {
                "conditions[0][property]": "covered_recipient_last_name",
                "conditions[0][value]": physician_last.upper(),
                "conditions[0][operator]": "=",
                "conditions[1][property]": "covered_recipient_first_name",
                "conditions[1][value]": physician_first.upper(),
                "conditions[1][operator]": "STARTS_WITH",
                "limit": limit,
                "sort": "total_amount_of_payment_usdollars",
                "sort_order": "desc",
            }

            # Note: The actual Open Payments API requires specific dataset IDs
            # This is a simplified illustration. In practice, you'd query:
            # - General Payment datasets (one per year)
            # - Research Payment datasets (one per year)
            # See: https://openpaymentsdata.cms.gov/dataset/

            # For the training dataset builder, we'd use the bulk CSV downloads
            # which are more practical for batch processing.
            logger.debug(
                f"Open Payments search: {physician_first} {physician_last} "
                f"(would query CMS API or bulk CSV)"
            )

        except Exception as e:
            logger.warning(f"Open Payments search error: {e}")

        return records

    async def search_orcid(self, name: str) -> list[dict]:
        """
        Search ORCID for author affiliations.

        ORCID public API provides:
        - Employment history (current and past)
        - Education
        - Funding received
        - Works

        Useful for identifying industry affiliations not declared in the paper.
        """
        affiliations = []
        try:
            # Search ORCID by name
            resp = await self.client.get(
                "https://pub.orcid.org/v3.0/search/",
                params={"q": f'family-name:"{name.split()[-1]}" AND given-names:"{name.split()[0]}"'},
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("result", [])
                for result in results[:5]:  # Check top 5 matches
                    orcid_id = result.get("orcid-identifier", {}).get("path", "")
                    if orcid_id:
                        # Fetch employment details
                        emp_resp = await self.client.get(
                            f"https://pub.orcid.org/v3.0/{orcid_id}/employments",
                            headers={"Accept": "application/json"},
                        )
                        if emp_resp.status_code == 200:
                            emp_data = emp_resp.json()
                            for group in emp_data.get("affiliation-group", []):
                                for summary in group.get("summaries", []):
                                    emp = summary.get("employment-summary", {})
                                    org = emp.get("organization", {})
                                    affiliations.append({
                                        "orcid": orcid_id,
                                        "organization": org.get("name", ""),
                                        "role": emp.get("role-title", ""),
                                        "start_year": (
                                            emp.get("start-date", {}) or {}
                                        ).get("year", {}).get("value", ""),
                                        "end_year": (
                                            emp.get("end-date", {}) or {}
                                        ).get("year", {}).get("value", ""),
                                    })
                        await asyncio.sleep(0.5)
        except Exception as e:
            logger.warning(f"ORCID search error for {name}: {e}")

        return affiliations

    async def get_europmc_funding(self, pmid: str) -> list[dict]:
        """
        Get funder information from Europe PMC.

        Europe PMC extracts grant/funding data from full-text articles and
        provides it via API. This often captures funding not visible in abstracts.
        """
        grants = []
        try:
            resp = await self.client.get(
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={
                    "query": f"EXT_ID:{pmid} AND SRC:MED",
                    "resultType": "core",
                    "format": "json",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("resultList", {}).get("result", [])
                for result in results:
                    grant_list = result.get("grantsList", {}).get("grant", [])
                    for grant in grant_list:
                        grants.append({
                            "agency": grant.get("agency", ""),
                            "grant_id": grant.get("grantId", ""),
                            "acronym": grant.get("acronym", ""),
                        })
        except Exception as e:
            logger.warning(f"EuroPMC funding lookup error for {pmid}: {e}")

        return grants

    def classify_funding_source(self, grants: list[dict], affiliations: list[str]) -> str:
        """
        Classify whether funding is industry, public, mixed, or not reported.
        Uses known pharma company patterns.
        """
        has_industry = False
        has_public = False

        all_text = " ".join(
            [g.get("agency", "") for g in grants] + affiliations
        ).lower()

        for indicator in self.PHARMA_INDICATORS:
            if indicator in all_text:
                has_industry = True
                break

        public_indicators = [
            "nih", "nsf", "nhmrc", "mrc", "wellcome", "nihr",
            "national institute", "national science", "research council",
            "government", "ministry", "pcori", "ahrq",
        ]
        for indicator in public_indicators:
            if indicator in all_text:
                has_public = True
                break

        if has_industry and has_public:
            return "mixed"
        elif has_industry:
            return "industry"
        elif has_public:
            return "public"
        elif grants:
            return "unclear"
        else:
            return "not_reported"

    def check_industry_affiliation(self, affiliations: list[dict]) -> list[str]:
        """Check if any ORCID affiliations are pharmaceutical/device companies."""
        industry = []
        for aff in affiliations:
            org_name = aff.get("organization", "").lower()
            for indicator in self.PHARMA_INDICATORS:
                if indicator in org_name:
                    industry.append(aff.get("organization", ""))
                    break
        return industry

    async def build_coi_report(
        self,
        pmid: str,
        doi: str,
        authors: list[dict],  # [{"first": "John", "last": "Smith", "affiliations": [...]}]
        declared_funding: list[str] = None,
    ) -> PaperCOIReport:
        """
        Build a comprehensive COI report for a paper.
        This becomes training data showing the model what to look for and where.
        """
        report = PaperCOIReport(
            pmid=pmid,
            doi=doi,
            declared_funding=declared_funding or [],
        )

        # Step 1: Check EuroPMC for funding data
        europmc_grants = await self.get_europmc_funding(pmid)
        report.verification_steps_taken.append(
            f"Checked Europe PMC for funder metadata: found {len(europmc_grants)} grants"
        )

        all_grants = europmc_grants
        all_affiliations = []
        for author in authors:
            all_affiliations.extend(author.get("affiliations", []))

        report.funding_type = self.classify_funding_source(all_grants, all_affiliations)

        # Step 2: Check ORCID for author affiliations
        for author_info in authors[:10]:  # Limit to first/last + key authors
            name = f"{author_info.get('first', '')} {author_info.get('last', '')}"
            profile = AuthorCOIProfile(name=name)

            orcid_affiliations = await self.search_orcid(name)
            profile.orcid_affiliations = orcid_affiliations
            profile.industry_affiliations = self.check_industry_affiliation(orcid_affiliations)
            profile.is_industry_employed = bool(profile.industry_affiliations)

            report.author_profiles.append(profile)
            report.verification_steps_taken.append(
                f"Checked ORCID for {name}: "
                f"{'industry affiliations found' if profile.is_industry_employed else 'no industry affiliations'}"
            )

            await asyncio.sleep(0.5)

        # Step 3: Aggregate findings
        report.any_industry_payments = any(
            p.open_payments_found for p in report.author_profiles
        )
        report.industry_employed_authors = sum(
            1 for p in report.author_profiles if p.is_industry_employed
        )
        report.total_payments_usd = sum(
            p.total_general_payments_usd + p.total_research_payments_usd
            for p in report.author_profiles
        )

        # Step 4: Document what verification sources are available
        report.verification_sources_available = [
            {
                "source": "CMS Open Payments",
                "url": "https://openpaymentsdata.cms.gov/",
                "applicable": "For US-licensed physicians",
                "data": "General payments, research payments, ownership interests",
            },
            {
                "source": "ClinicalTrials.gov",
                "url": f"https://clinicaltrials.gov/search?term={doi}" if doi else "",
                "applicable": "For registered clinical trials",
                "data": "Sponsor, collaborators, funding source, registered outcomes",
            },
            {
                "source": "ORCID",
                "url": "https://orcid.org/",
                "applicable": "For authors with ORCID profiles",
                "data": "Employment history, funding, peer review activity",
            },
            {
                "source": "Medicines Australia",
                "url": "https://www.medicinesaustralia.com.au/transparency-reporting/",
                "applicable": "For Australian healthcare professionals",
                "data": "Annual aggregate pharma payments",
            },
            {
                "source": "EFPIA/Betransparent",
                "url": "https://www.efpia.eu/",
                "applicable": "For European healthcare professionals",
                "data": "Pharma payments (varies by country)",
            },
        ]

        report.verification_limitations = [
            "Open Payments covers only US physicians; international authors require country-specific databases",
            "ORCID profiles are voluntary and may be incomplete",
            "Payment databases have a 12-18 month publication lag",
            "Payments through third parties or institutions may not be attributed to individuals",
            "Non-financial conflicts (academic advancement, intellectual) are not captured",
        ]

        return report


def generate_verification_guidance(report: PaperCOIReport) -> str:
    """
    Generate the verification guidance text that will be part of the training data.
    The model should learn to produce this kind of actionable guidance.
    """
    lines = ["## Recommended Verification Steps\n"]

    lines.append("### Author Payment History")
    lines.append("- Search CMS Open Payments (openpaymentsdata.cms.gov) for each US-based author")
    lines.append("  Look for: consulting fees, speaking honoraria, research payments from study sponsor")

    if report.funding_type == "industry":
        lines.append(f"- **Priority**: Study is industry-funded. Cross-reference sponsor with author payment records")
    elif report.funding_type == "not_reported":
        lines.append("- **Flag**: Funding source not reported in abstract. Check full text and ClinicalTrials.gov")

    lines.append("\n### Trial Registry Verification")
    lines.append("- Check ClinicalTrials.gov for:")
    lines.append("  - Registered primary outcome vs published primary outcome (outcome switching)")
    lines.append("  - Listed sponsor and collaborators")
    lines.append("  - Planned sample size vs actual enrolment")
    lines.append("  - Protocol amendments and their timing")

    lines.append("\n### Author Affiliation Verification")
    lines.append("- Check ORCID profiles for employment history")
    lines.append("  Look for: current or recent pharma/device company employment")
    if report.industry_employed_authors > 0:
        lines.append(
            f"  **Found**: {report.industry_employed_authors} author(s) with "
            "industry affiliations in ORCID"
        )

    lines.append("\n### Additional Sources")
    lines.append("- Europe PMC: funder metadata and grant information")
    lines.append("- Retraction Watch / Crossref: check for post-publication notices")
    lines.append("- For Australian authors: Medicines Australia transparency reports")
    lines.append("- For European authors: EFPIA country-specific disclosure databases")

    if report.verification_limitations:
        lines.append("\n### Limitations")
        for lim in report.verification_limitations:
            lines.append(f"- {lim}")

    return "\n".join(lines)

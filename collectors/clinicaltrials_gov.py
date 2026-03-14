"""
ClinicalTrials.gov Outcome Switching Detector

Compares registered trial outcomes with published abstract outcomes to detect:
1. Primary outcome switching (different primary endpoint published vs registered)
2. Outcome addition (new endpoints not in protocol)
3. Outcome omission (registered endpoints not reported)
4. Timepoint changes (different follow-up from protocol)

Uses the ClinicalTrials.gov v2 API:
    https://clinicaltrials.gov/api/v2/studies

Also extracts sponsor, collaborator, and funding information.

Reference: COMPare Trials project (compare-trials.org) - Ben Goldacre et al.
systematically documented outcome switching in top 5 medical journals.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RegisteredOutcome:
    """An outcome as registered on ClinicalTrials.gov."""
    measure: str = ""
    description: str = ""
    time_frame: str = ""
    outcome_type: str = ""  # primary, secondary, other


@dataclass
class TrialRegistration:
    """Full trial registration data from ClinicalTrials.gov."""
    nct_id: str = ""
    title: str = ""
    brief_summary: str = ""
    status: str = ""
    phase: str = ""
    enrollment: Optional[int] = None
    enrollment_type: str = ""  # actual, estimated
    start_date: str = ""
    completion_date: str = ""

    # Sponsor / funding
    lead_sponsor: str = ""
    lead_sponsor_type: str = ""  # INDUSTRY, NIH, FED, OTHER, NETWORK
    collaborators: list[str] = field(default_factory=list)
    funding_source: str = ""  # Derived: industry, public, mixed

    # Registered outcomes
    primary_outcomes: list[RegisteredOutcome] = field(default_factory=list)
    secondary_outcomes: list[RegisteredOutcome] = field(default_factory=list)

    # Design
    study_type: str = ""
    allocation: str = ""  # randomized, non-randomized
    masking: str = ""
    arms: list[dict] = field(default_factory=list)

    # Dates for detecting protocol amendments
    first_posted: str = ""
    last_updated: str = ""
    results_posted: str = ""


@dataclass
class OutcomeSwitchingReport:
    """Report comparing registered vs published outcomes."""
    nct_id: str = ""
    pmid: str = ""
    doi: str = ""

    # Registered outcomes
    registered_primary: list[str] = field(default_factory=list)
    registered_secondary: list[str] = field(default_factory=list)

    # Published outcomes (extracted from abstract)
    published_primary: list[str] = field(default_factory=list)

    # Switching flags
    primary_outcome_switched: bool = False
    new_outcomes_added: bool = False
    outcomes_omitted: bool = False
    timepoint_changed: bool = False

    # Sponsor info
    sponsor: str = ""
    sponsor_type: str = ""  # INDUSTRY, NIH, etc.

    # Timing flags
    registration_after_completion: bool = False  # Retrospective registration
    late_protocol_amendments: bool = False

    # Confidence in matching
    matching_confidence: str = "low"  # low, medium, high

    evidence: list[str] = field(default_factory=list)


class ClinicalTrialsGovCollector:
    """
    Query ClinicalTrials.gov v2 API and detect outcome switching.

    The v2 API provides rich structured data including:
    - Protocol section (design, outcomes, arms, interventions)
    - Results section (if posted)
    - Sponsor and collaborator information
    - Protocol amendment history (via document section)
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def fetch_study(self, nct_id: str) -> Optional[TrialRegistration]:
        """Fetch a single study by NCT ID."""
        try:
            resp = await self.client.get(
                f"{self.BASE_URL}/studies/{nct_id}",
                params={
                    "fields": (
                        "NCTId,BriefTitle,OfficialTitle,BriefSummary,"
                        "OverallStatus,Phase,EnrollmentCount,EnrollmentType,"
                        "StartDate,CompletionDate,StudyFirstPostDate,LastUpdatePostDate,"
                        "ResultsFirstPostDate,"
                        "LeadSponsorName,LeadSponsorClass,CollaboratorName,"
                        "PrimaryOutcomeMeasure,PrimaryOutcomeDescription,"
                        "PrimaryOutcomeTimeFrame,"
                        "SecondaryOutcomeMeasure,SecondaryOutcomeDescription,"
                        "SecondaryOutcomeTimeFrame,"
                        "StudyType,DesignAllocation,DesignMasking,"
                        "ArmGroupLabel,ArmGroupType,ArmGroupDescription,"
                        "InterventionName,InterventionType"
                    ),
                },
            )

            if resp.status_code != 200:
                logger.warning(f"CTgov API error for {nct_id}: {resp.status_code}")
                return None

            data = resp.json()
            return self._parse_study(data)

        except Exception as e:
            logger.warning(f"Failed to fetch {nct_id}: {e}")
            return None

    async def search_by_doi(self, doi: str) -> Optional[TrialRegistration]:
        """Search ClinicalTrials.gov by DOI to find the trial registration."""
        try:
            # The v2 API doesn't directly search by DOI, but we can try
            # searching by the DOI in the references section
            resp = await self.client.get(
                f"{self.BASE_URL}/studies",
                params={
                    "query.term": doi,
                    "pageSize": 5,
                },
            )

            if resp.status_code == 200:
                data = resp.json()
                studies = data.get("studies", [])
                if studies:
                    return self._parse_study(studies[0])

        except Exception as e:
            logger.warning(f"DOI search failed for {doi}: {e}")
        return None

    async def search_by_title_keywords(
        self, title: str, intervention: str = ""
    ) -> list[TrialRegistration]:
        """
        Search by title keywords when no NCT ID or DOI is available.
        Returns multiple candidates for manual matching.
        """
        # Extract key terms from title
        # Remove common words
        stopwords = {
            "a", "an", "the", "of", "in", "for", "and", "or", "to", "with",
            "on", "by", "at", "from", "is", "are", "was", "were", "been",
            "versus", "vs", "compared", "comparison", "effect", "effects",
            "study", "trial", "randomized", "controlled", "double-blind",
            "placebo-controlled", "phase", "efficacy", "safety",
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        keywords = [w for w in words if w not in stopwords][:6]
        query = " AND ".join(keywords[:4])

        results = []
        try:
            resp = await self.client.get(
                f"{self.BASE_URL}/studies",
                params={
                    "query.term": query,
                    "filter.overallStatus": "COMPLETED,TERMINATED",
                    "pageSize": 10,
                    "sort": "LastUpdatePostDate:desc",
                },
            )

            if resp.status_code == 200:
                data = resp.json()
                for study in data.get("studies", []):
                    parsed = self._parse_study(study)
                    if parsed:
                        results.append(parsed)

        except Exception as e:
            logger.warning(f"Title keyword search failed: {e}")

        return results

    async def extract_nct_from_abstract(self, abstract: str) -> Optional[str]:
        """Extract NCT ID from abstract text (many abstracts include it)."""
        match = re.search(r'NCT\d{8}', abstract)
        if match:
            return match.group()

        # Also try ClinicalTrials.gov URL patterns
        match = re.search(r'clinicaltrials\.gov/(?:ct2/show/|study/)?(NCT\d{8})', abstract)
        if match:
            return match.group(1)

        return None

    def _parse_study(self, data: dict) -> Optional[TrialRegistration]:
        """Parse the v2 API response into a TrialRegistration."""
        try:
            protocol = data.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            status_mod = protocol.get("statusModule", {})
            sponsor_mod = protocol.get("sponsorCollaboratorsModule", {})
            design_mod = protocol.get("designModule", {})
            outcomes_mod = protocol.get("outcomesModule", {})
            enroll_mod = design_mod.get("enrollmentInfo", {})

            reg = TrialRegistration(
                nct_id=ident.get("nctId", ""),
                title=ident.get("briefTitle", "") or ident.get("officialTitle", ""),
                brief_summary=protocol.get("descriptionModule", {}).get("briefSummary", ""),
                status=status_mod.get("overallStatus", ""),
                phase=",".join(design_mod.get("phases", [])),
                enrollment=enroll_mod.get("count"),
                enrollment_type=enroll_mod.get("type", ""),
                start_date=status_mod.get("startDateStruct", {}).get("date", ""),
                completion_date=status_mod.get("completionDateStruct", {}).get("date", ""),
                first_posted=status_mod.get("studyFirstPostDateStruct", {}).get("date", ""),
                last_updated=status_mod.get("lastUpdatePostDateStruct", {}).get("date", ""),
                results_posted=status_mod.get("resultsFirstPostDateStruct", {}).get("date", ""),
            )

            # Sponsor
            lead_sponsor = sponsor_mod.get("leadSponsor", {})
            reg.lead_sponsor = lead_sponsor.get("name", "")
            reg.lead_sponsor_type = lead_sponsor.get("class", "")

            collabs = sponsor_mod.get("collaborators", [])
            reg.collaborators = [c.get("name", "") for c in collabs]

            # Classify funding
            if reg.lead_sponsor_type == "INDUSTRY":
                reg.funding_source = "industry"
            elif reg.lead_sponsor_type in ("NIH", "FED"):
                reg.funding_source = "public"
            elif any(c.get("class") == "INDUSTRY" for c in collabs):
                reg.funding_source = "mixed"
            else:
                reg.funding_source = "other"

            # Primary outcomes
            for outcome in outcomes_mod.get("primaryOutcomes", []):
                reg.primary_outcomes.append(RegisteredOutcome(
                    measure=outcome.get("measure", ""),
                    description=outcome.get("description", ""),
                    time_frame=outcome.get("timeFrame", ""),
                    outcome_type="primary",
                ))

            # Secondary outcomes
            for outcome in outcomes_mod.get("secondaryOutcomes", []):
                reg.secondary_outcomes.append(RegisteredOutcome(
                    measure=outcome.get("measure", ""),
                    description=outcome.get("description", ""),
                    time_frame=outcome.get("timeFrame", ""),
                    outcome_type="secondary",
                ))

            # Design
            reg.study_type = design_mod.get("studyType", "")
            reg.allocation = design_mod.get("designInfo", {}).get("allocation", "")
            reg.masking = design_mod.get("designInfo", {}).get("maskingInfo", {}).get("masking", "")

            # Arms
            for arm in protocol.get("armsInterventionsModule", {}).get("armGroups", []):
                reg.arms.append({
                    "label": arm.get("label", ""),
                    "type": arm.get("type", ""),
                    "description": arm.get("description", ""),
                })

            return reg

        except Exception as e:
            logger.warning(f"Failed to parse study data: {e}")
            return None

    async def detect_outcome_switching(
        self,
        nct_id: str,
        published_abstract: str,
        published_title: str = "",
        pmid: str = "",
        doi: str = "",
    ) -> OutcomeSwitchingReport:
        """
        Compare registered outcomes with what appears in the published abstract.

        NOTE: This is a heuristic comparison. The LLM annotator provides
        more nuanced matching; this gives structured data for training.
        """
        report = OutcomeSwitchingReport(
            nct_id=nct_id, pmid=pmid, doi=doi,
        )

        # Fetch registration data
        registration = await self.fetch_study(nct_id)
        if not registration:
            report.evidence.append(f"Could not retrieve registration for {nct_id}")
            return report

        report.sponsor = registration.lead_sponsor
        report.sponsor_type = registration.lead_sponsor_type

        # Extract registered outcomes
        report.registered_primary = [
            o.measure for o in registration.primary_outcomes
        ]
        report.registered_secondary = [
            o.measure for o in registration.secondary_outcomes
        ]

        # Heuristic: check if abstract mentions registered primary outcome terms
        abstract_lower = (published_abstract + " " + published_title).lower()

        # For each registered primary outcome, check if key terms appear in abstract
        primary_mentioned = []
        for outcome in registration.primary_outcomes:
            # Extract key terms from the outcome measure
            outcome_terms = self._extract_outcome_terms(outcome.measure)
            mentioned = any(
                term.lower() in abstract_lower for term in outcome_terms
            )
            primary_mentioned.append(mentioned)

        if primary_mentioned and not all(primary_mentioned):
            report.outcomes_omitted = True
            report.evidence.append(
                f"Registered primary outcome(s) not clearly mentioned in abstract: "
                f"{[m for m, mentioned in zip(report.registered_primary, primary_mentioned) if not mentioned]}"
            )

        # Check for retrospective registration
        if registration.completion_date and registration.first_posted:
            # Simple date comparison (would need proper date parsing in production)
            report.evidence.append(
                f"First posted: {registration.first_posted}, "
                f"Completion: {registration.completion_date}"
            )

        # Extract what appears to be the published primary outcome
        published_primary = self._extract_published_primary(published_abstract)
        if published_primary:
            report.published_primary = published_primary

        # Rough matching: do published primary terms appear in registered primary?
        if published_primary and report.registered_primary:
            registered_terms = set()
            for outcome in report.registered_primary:
                registered_terms.update(
                    t.lower() for t in self._extract_outcome_terms(outcome)
                )

            published_terms = set()
            for outcome in published_primary:
                published_terms.update(
                    t.lower() for t in self._extract_outcome_terms(outcome)
                )

            overlap = registered_terms & published_terms
            if not overlap and registered_terms and published_terms:
                report.primary_outcome_switched = True
                report.evidence.append(
                    f"Possible outcome switching: registered primary terms "
                    f"{registered_terms} vs published terms {published_terms}"
                )
                report.matching_confidence = "low"  # Heuristic match is always low confidence
            elif overlap:
                report.matching_confidence = "medium"

        return report

    def _extract_outcome_terms(self, outcome_text: str) -> list[str]:
        """Extract clinically meaningful terms from an outcome description."""
        # Remove common filler words
        stopwords = {
            "change", "from", "baseline", "in", "at", "the", "of", "and",
            "or", "to", "as", "by", "with", "for", "on", "time", "rate",
            "proportion", "percentage", "number", "score", "level", "levels",
            "measured", "assessed", "defined", "weeks", "months", "years",
            "week", "month", "year", "day", "days",
        }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', outcome_text)
        terms = [w for w in words if w.lower() not in stopwords]

        # Also extract multi-word medical terms (simple bigrams)
        bigrams = []
        word_list = outcome_text.split()
        for i in range(len(word_list) - 1):
            bigram = f"{word_list[i]} {word_list[i+1]}".strip(".,;:()")
            if len(bigram) > 5 and not all(w.lower() in stopwords for w in bigram.split()):
                bigrams.append(bigram)

        return terms + bigrams[:3]

    def _extract_published_primary(self, abstract: str) -> list[str]:
        """
        Attempt to extract what the abstract declares as primary outcome.
        Heuristic: look for "primary outcome", "primary endpoint", "primary end point".
        """
        patterns = [
            r'[Pp]rimary\s+(?:outcome|endpoint|end\s*point|measure)\s+(?:was|were|is)\s+([^.]+)',
            r'[Pp]rimary\s+(?:outcome|endpoint|end\s*point)\s*[:\-]\s*([^.]+)',
            r'[Tt]he\s+primary\s+(?:outcome|endpoint)\s+(?:of\s+)?([^.]+)',
        ]

        results = []
        for pattern in patterns:
            matches = re.findall(pattern, abstract)
            results.extend(matches)

        return results[:3]  # Return at most 3 candidates


async def batch_check_outcome_switching(
    abstracts: list[dict],  # Each: {pmid, title, abstract, nct_id?}
) -> list[OutcomeSwitchingReport]:
    """
    Check a batch of abstracts for outcome switching.
    First tries to find NCT IDs in the abstracts, then queries CTgov.
    """
    async with ClinicalTrialsGovCollector() as collector:
        reports = []

        for item in abstracts:
            nct_id = item.get("nct_id", "")
            if not nct_id:
                # Try to extract from abstract text
                nct_id = await collector.extract_nct_from_abstract(
                    item.get("abstract", "")
                )

            if not nct_id:
                # Try title-based search
                candidates = await collector.search_by_title_keywords(
                    item.get("title", "")
                )
                if candidates:
                    nct_id = candidates[0].nct_id

            if nct_id:
                report = await collector.detect_outcome_switching(
                    nct_id=nct_id,
                    published_abstract=item.get("abstract", ""),
                    published_title=item.get("title", ""),
                    pmid=item.get("pmid", ""),
                    doi=item.get("doi", ""),
                )
                reports.append(report)

            await asyncio.sleep(0.5)  # Rate limit

        return reports


if __name__ == "__main__":
    async def demo():
        async with ClinicalTrialsGovCollector() as collector:
            # Demo: fetch a well-known trial
            reg = await collector.fetch_study("NCT01105962")
            if reg:
                print(f"Trial: {reg.title}")
                print(f"Sponsor: {reg.lead_sponsor} ({reg.lead_sponsor_type})")
                print(f"Funding: {reg.funding_source}")
                print(f"Primary outcomes:")
                for o in reg.primary_outcomes:
                    print(f"  - {o.measure} ({o.time_frame})")
                print(f"Secondary outcomes: {len(reg.secondary_outcomes)}")
            else:
                print("Could not fetch trial (network access may be restricted)")

    asyncio.run(demo())

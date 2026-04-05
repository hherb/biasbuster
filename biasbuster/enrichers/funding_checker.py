"""
Funding Source Classifier

Classifies funding sources from multiple signals:
1. Abstract text (explicit funding statements)
2. PubMed grant metadata
3. Europe PMC funder data
4. ClinicalTrials.gov sponsor info
5. Author affiliations

Also generates verification guidance for the training data,
teaching the model WHERE to look for funding information.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ---- Known entity databases ----

# Major pharmaceutical companies and common name variants
PHARMA_COMPANIES = {
    # Company name -> canonical name
    "pfizer": "Pfizer",
    "novartis": "Novartis",
    "roche": "Roche",
    "hoffmann-la roche": "Roche",
    "genentech": "Roche/Genentech",
    "merck": "Merck",
    "msd": "Merck",
    "johnson & johnson": "Johnson & Johnson",
    "janssen": "Johnson & Johnson/Janssen",
    "abbvie": "AbbVie",
    "abbott": "Abbott",
    "bristol-myers squibb": "Bristol-Myers Squibb",
    "bms": "Bristol-Myers Squibb",
    "astrazeneca": "AstraZeneca",
    "sanofi": "Sanofi",
    "glaxosmithkline": "GlaxoSmithKline",
    "gsk": "GlaxoSmithKline",
    "eli lilly": "Eli Lilly",
    "lilly": "Eli Lilly",
    "amgen": "Amgen",
    "gilead": "Gilead Sciences",
    "bayer": "Bayer",
    "boehringer ingelheim": "Boehringer Ingelheim",
    "takeda": "Takeda",
    "novo nordisk": "Novo Nordisk",
    "regeneron": "Regeneron",
    "biogen": "Biogen",
    "moderna": "Moderna",
    "medtronic": "Medtronic",
    "boston scientific": "Boston Scientific",
    "stryker": "Stryker",
    "edwards lifesciences": "Edwards Lifesciences",
    "daiichi sankyo": "Daiichi Sankyo",
    "astellas": "Astellas",
    "otsuka": "Otsuka",
    "eisai": "Eisai",
    "servier": "Servier",
    "ipsen": "Ipsen",
    "almirall": "Almirall",
    "teva": "Teva",
    "mylan": "Mylan/Viatris",
    "viatris": "Mylan/Viatris",
}

# Major public funding bodies
PUBLIC_FUNDERS = {
    "nih": "National Institutes of Health (US)",
    "national institutes of health": "National Institutes of Health (US)",
    "national heart, lung, and blood institute": "NIH/NHLBI",
    "nhlbi": "NIH/NHLBI",
    "nci": "NIH/NCI",
    "national cancer institute": "NIH/NCI",
    "nimh": "NIH/NIMH",
    "niddk": "NIH/NIDDK",
    "nsf": "National Science Foundation (US)",
    "nhmrc": "NHMRC (Australia)",
    "national health and medical research council": "NHMRC (Australia)",
    "mrc": "Medical Research Council (UK)",
    "medical research council": "Medical Research Council (UK)",
    "nihr": "NIHR (UK)",
    "national institute for health research": "NIHR (UK)",
    "wellcome trust": "Wellcome Trust",
    "wellcome": "Wellcome Trust",
    "british heart foundation": "British Heart Foundation",
    "cancer research uk": "Cancer Research UK",
    "deutsche forschungsgemeinschaft": "DFG (Germany)",
    "dfg": "DFG (Germany)",
    "inserm": "INSERM (France)",
    "pcori": "PCORI (US)",
    "ahrq": "AHRQ (US)",
    "va": "Veterans Affairs (US)",
    "department of veterans affairs": "Veterans Affairs (US)",
    "canadian institutes of health research": "CIHR (Canada)",
    "cihr": "CIHR (Canada)",
}

# Indicators of industry involvement (beyond direct naming)
INDUSTRY_INDICATORS = [
    r'\b(?:industry|pharmaceutical|pharma|drug\s+company|device\s+company)\s+'
    r'(?:sponsored|funded|supported|grant)',
    r'\b(?:sponsored|funded|supported)\s+by\s+(?:a\s+)?(?:grant\s+from\s+)?'
    r'(?:\w+\s+){0,3}(?:pharma|inc\.|ltd\.|corp\.|gmbh|plc|therapeutics|biosciences)',
    r'\b(?:medical\s+writing|editorial\s+)?(?:assistance|support)\s+(?:was\s+)?'
    r'(?:provided|funded)\s+by\b',
    r'\b(?:employees?\s+of|employed\s+by)\s+(?:\w+\s+){0,3}'
    r'(?:inc\.|ltd\.|corp\.|gmbh|therapeutics|pharma)',
]


@dataclass
class FundingAnalysis:
    """Complete funding analysis for a paper."""
    pmid: str = ""

    # Classification
    funding_type: str = "not_reported"  # industry, public, mixed, not_reported, unclear
    confidence: str = "low"             # low, medium, high

    # Identified funders
    industry_funders: list[str] = field(default_factory=list)
    public_funders: list[str] = field(default_factory=list)
    unknown_funders: list[str] = field(default_factory=list)

    # Evidence sources
    abstract_mentions: list[str] = field(default_factory=list)
    pubmed_grants: list[dict] = field(default_factory=list)
    author_industry_affiliations: list[str] = field(default_factory=list)

    # Flags
    undisclosed_industry_suspected: bool = False
    medical_writing_assistance: bool = False
    industry_employees_as_authors: bool = False

    # Verification guidance
    verification_steps: list[str] = field(default_factory=list)


def classify_funder(name: str) -> tuple[str, str]:
    """
    Classify a funder name as industry, public, or unknown.
    Returns (type, canonical_name).
    """
    name_lower = name.lower().strip()

    # Check pharma companies
    for pattern, canonical in PHARMA_COMPANIES.items():
        if pattern in name_lower:
            return ("industry", canonical)

    # Check public funders
    for pattern, canonical in PUBLIC_FUNDERS.items():
        if pattern in name_lower:
            return ("public", canonical)

    # Heuristic: check for corporate suffixes
    corporate_suffixes = [
        r'\b(?:inc|ltd|corp|gmbh|plc|ag|sa|sas|bv|pty)\b',
        r'\b(?:therapeutics|biosciences|biotech|pharma|pharmaceuticals|laboratories)\b',
        r'\b(?:medical\s+devices?|diagnostics|life\s+sciences)\b',
    ]
    for pattern in corporate_suffixes:
        if re.search(pattern, name_lower):
            return ("industry", name)

    # Heuristic: check for government/academic indicators
    academic_indicators = [
        r'\b(?:university|université|universität|college|institute|hospital|'
        r'foundation|trust|council|ministry|department\s+of|government|'
        r'national|federal|state|academy)\b',
    ]
    for pattern in academic_indicators:
        if re.search(pattern, name_lower):
            return ("public", name)

    return ("unknown", name)


def analyse_abstract_funding(abstract: str) -> dict:
    """
    Extract funding signals from abstract text.
    """
    findings = {
        "industry_mentions": [],
        "public_mentions": [],
        "disclosure_present": False,
        "medical_writing": False,
        "industry_employees": False,
    }

    text_lower = abstract.lower()

    # Check for explicit funding/disclosure sections
    funding_section = ""
    patterns = [
        r'(?:FUNDING|FINANCIAL\s+SUPPORT|GRANT\s+SUPPORT|DISCLOSURES?|'
        r'CONFLICT\s+OF\s+INTEREST|ACKNOWLEDGMENT)\s*[:\.]?\s*([^.]+(?:\.[^.]+){0,3})',
    ]
    for pattern in patterns:
        match = re.search(pattern, abstract, re.IGNORECASE)
        if match:
            funding_section = match.group(1)
            findings["disclosure_present"] = True

    search_text = funding_section if funding_section else abstract

    # Check for pharma companies
    for company_pattern, canonical in PHARMA_COMPANIES.items():
        if company_pattern in search_text.lower():
            findings["industry_mentions"].append(canonical)

    # Check for public funders
    for funder_pattern, canonical in PUBLIC_FUNDERS.items():
        if funder_pattern in search_text.lower():
            findings["public_mentions"].append(canonical)

    # Check for industry indicators
    for pattern in INDUSTRY_INDICATORS:
        if re.search(pattern, text_lower):
            if "medical writing" in text_lower or "editorial assistance" in text_lower:
                findings["medical_writing"] = True
            if "employee" in text_lower:
                findings["industry_employees"] = True

    return findings


def analyse_funding(
    pmid: str,
    abstract: str,
    pubmed_grants: Optional[list[dict]] = None,
    author_affiliations: Optional[list[str]] = None,
    ctgov_sponsor: str = "",
    ctgov_sponsor_type: str = "",
) -> FundingAnalysis:
    """
    Comprehensive funding analysis combining multiple data sources.
    """
    analysis = FundingAnalysis(pmid=pmid)

    # 1. Abstract text analysis
    abstract_findings = analyse_abstract_funding(abstract)
    analysis.abstract_mentions = (
        abstract_findings["industry_mentions"] + abstract_findings["public_mentions"]
    )
    analysis.medical_writing_assistance = abstract_findings["medical_writing"]
    analysis.industry_employees_as_authors = abstract_findings["industry_employees"]

    industry_found = set(abstract_findings["industry_mentions"])
    public_found = set(abstract_findings["public_mentions"])

    # 2. PubMed grant metadata
    if pubmed_grants:
        for grant in pubmed_grants:
            agency = grant.get("agency", "")
            ftype, canonical = classify_funder(agency)
            analysis.pubmed_grants.append({"agency": agency, "type": ftype, "canonical": canonical})
            if ftype == "industry":
                industry_found.add(canonical)
            elif ftype == "public":
                public_found.add(canonical)

    # 3. Author affiliations
    if author_affiliations:
        for affil in author_affiliations:
            for company_pattern, canonical in PHARMA_COMPANIES.items():
                if company_pattern in affil.lower():
                    analysis.author_industry_affiliations.append(canonical)
                    industry_found.add(canonical)

    # 4. ClinicalTrials.gov sponsor
    if ctgov_sponsor:
        ftype, canonical = classify_funder(ctgov_sponsor)
        if ftype == "industry":
            industry_found.add(canonical)
        elif ftype == "public":
            public_found.add(canonical)

    if ctgov_sponsor_type == "INDUSTRY":
        industry_found.add(ctgov_sponsor)

    # Compile results
    analysis.industry_funders = sorted(industry_found)
    analysis.public_funders = sorted(public_found)

    # Classify
    if industry_found and public_found:
        analysis.funding_type = "mixed"
    elif industry_found:
        analysis.funding_type = "industry"
    elif public_found:
        analysis.funding_type = "public"
    elif abstract_findings["disclosure_present"]:
        analysis.funding_type = "unclear"
    else:
        analysis.funding_type = "not_reported"

    # Confidence
    sources_checked = sum([
        bool(abstract),
        bool(pubmed_grants),
        bool(author_affiliations),
        bool(ctgov_sponsor),
    ])
    if sources_checked >= 3:
        analysis.confidence = "high"
    elif sources_checked >= 2:
        analysis.confidence = "medium"
    else:
        analysis.confidence = "low"

    # Flags
    if (analysis.author_industry_affiliations
            and analysis.funding_type in ("not_reported", "public")):
        analysis.undisclosed_industry_suspected = True

    # Verification guidance
    analysis.verification_steps = _generate_verification_steps(analysis)

    return analysis


def _generate_verification_steps(analysis: FundingAnalysis) -> list[str]:
    """Generate actionable verification steps based on findings."""
    steps = []

    if analysis.funding_type == "not_reported":
        steps.append(
            "Funding not reported in abstract. Check: (1) full-text funding statement, "
            "(2) ClinicalTrials.gov sponsor field, (3) Europe PMC grant metadata"
        )

    if analysis.funding_type in ("industry", "mixed"):
        steps.append(
            f"Industry funding identified ({', '.join(analysis.industry_funders)}). "
            "Cross-reference author payments via CMS Open Payments "
            "(openpaymentsdata.cms.gov) for US authors"
        )

    if analysis.undisclosed_industry_suspected:
        steps.append(
            "ALERT: Author(s) have industry affiliations but funding is reported as "
            f"'{analysis.funding_type}'. Check for undisclosed COI via ORCID "
            "employment records and Open Payments"
        )

    if analysis.medical_writing_assistance:
        steps.append(
            "Medical writing assistance acknowledged - check if this constitutes "
            "ghost authorship and whether the writer's employer is the study sponsor"
        )

    if analysis.industry_employees_as_authors:
        steps.append(
            "Industry employee(s) listed as authors. Verify their role and whether "
            "they had access to/control over data analysis"
        )

    # Always suggest these for completeness
    steps.append(
        "For Australian authors: check Medicines Australia transparency reports "
        "(medicinesaustralia.com.au/transparency-reporting/)"
    )
    steps.append(
        "For European authors: check country-specific EFPIA disclosure databases"
    )

    return steps


if __name__ == "__main__":
    # Test with synthetic examples
    result = analyse_funding(
        pmid="TEST001",
        abstract=(
            "BACKGROUND: We evaluated wonderdrug in hypertension. "
            "METHODS: Multicentre RCT sponsored by Pfizer Inc. "
            "RESULTS: Significant blood pressure reduction. "
            "FUNDING: This study was funded by Pfizer Inc. Medical writing "
            "assistance was provided by PAREXEL, funded by Pfizer."
        ),
        pubmed_grants=[{"agency": "Pfizer Inc", "id": "WS12345"}],
        author_affiliations=[
            "Pfizer Inc, New York, NY",
            "University of Melbourne, Australia",
        ],
        ctgov_sponsor="Pfizer",
        ctgov_sponsor_type="INDUSTRY",
    )

    print(f"Funding type: {result.funding_type} (confidence: {result.confidence})")
    print(f"Industry funders: {result.industry_funders}")
    print(f"Medical writing: {result.medical_writing_assistance}")
    print(f"Industry employees: {result.industry_employees_as_authors}")
    print(f"Undisclosed COI suspected: {result.undisclosed_industry_suspected}")
    print(f"\nVerification steps:")
    for step in result.verification_steps:
        print(f"  - {step}")

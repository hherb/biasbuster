"""
Configuration for the Bias Dataset Builder.

Copy this to config.py and fill in your API keys.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    """Central configuration for the Bias Dataset Builder.

    Contains all API endpoints, keys, rate limits, batch sizes, thresholds,
    and output paths. Copy config.example.py to config.py and edit.
    """

    # Crossref / Retraction Watch
    crossref_mailto: str = "your.email@example.com"  # Required for polite pool
    crossref_api_base: str = "https://api.crossref.org/v1"
    crossref_labs_base: str = "https://api.labs.crossref.org"
    retraction_watch_csv_url: str = (
        "https://api.labs.crossref.org/data/retractionwatch"
    )

    # PubMed / NCBI
    ncbi_api_key: str = ""  # Optional but increases rate limit
    pubmed_base: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    pubmed_fetch_batch: int = 200

    # ClinicalTrials.gov v2 API
    ctgov_base: str = "https://clinicaltrials.gov/api/v2"

    # CMS Open Payments
    open_payments_base: str = "https://openpaymentsdata.cms.gov/api/1"
    # Bulk download: https://download.cms.gov/openpayments/

    # Europe PMC (for funder metadata and full-text mining)
    europmc_base: str = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    # Anthropic API (for pre-labelling)
    anthropic_api_key: str = ""  # Set via env var ANTHROPIC_API_KEY preferred
    annotation_model: str = "claude-sonnet-4-6"
    annotation_max_tokens: int = 4000

    # DeepSeek API (alternative annotator for comparison)
    deepseek_api_key: str = ""  # Set via env var DEEPSEEK_API_KEY preferred
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-reasoner"
    deepseek_max_tokens: int = 4000

    # ORCID (for author affiliation verification)
    orcid_base: str = "https://pub.orcid.org/v3.0"

    # Retry / resilience
    max_retries: int = 3
    retry_base_delay: float = 1.0       # Seconds before first retry
    retry_max_delay: float = 60.0       # Cap on exponential backoff

    # Timeouts (seconds)
    http_timeout: float = 30.0
    http_timeout_long: float = 120.0    # For bulk downloads

    # Rate limiting (seconds between requests)
    crossref_rate_delay: float = 0.5
    ncbi_rate_delay: float = 0.35       # 10/s with API key, 3/s without
    ncbi_rate_delay_slow: float = 0.5   # For non-batch operations
    ctgov_rate_delay: float = 0.5
    orcid_rate_delay: float = 0.5
    annotation_delay: float = 1.0       # Delay between Claude API calls

    # Batch / page sizes
    crossref_rows_per_page: int = 100
    doi_batch_size: int = 100
    ctgov_page_size_doi: int = 5
    ctgov_page_size_title: int = 10
    annotation_concurrency: int = 3
    retraction_flush_every: int = 50    # Incremental save batch size
    outcome_switching_check_limit: int = 100

    # Output
    output_dir: str = "dataset"
    export_dir: str = "dataset/export"
    db_path: str = "dataset/biasbuster.db"

    # Collection targets
    retraction_watch_max: int = 2000      # Max retracted papers to collect
    cochrane_rob_max: int = 1000           # Max Cochrane high-RoB studies
    cochrane_max_reviews: int = 200        # Max Cochrane/systematic reviews to search
    spin_screening_max: int = 5000         # Max abstracts to screen for spin heuristics
    clean_examples_max: int = 500          # Negative examples (low-bias)
    pubmed_rct_start_date: str = "2020/01/01"
    cochrane_min_year: int = 2015

    # Annotation limits per source (for cost control)
    annotation_max_per_source: dict = field(default_factory=lambda: {
        "high_suspicion": 500,
        "retracted_papers": 200,
        "cochrane_rob": 300,
        "low_suspicion": 300,
    })

    # Enrichment thresholds
    high_suspicion_threshold: float = 0.3
    low_suspicion_threshold: float = 0.1

    # Export
    train_split: float = 0.8
    val_split: float = 0.1
    export_seed: int = 42

    # Display / truncation
    evidence_max_length: int = 200

    # Domains to focus on (MeSH terms for PubMed queries)
    focus_domains: list[str] = field(default_factory=lambda: [
        "cardiovascular diseases",
        "neoplasms",
        "diabetes mellitus",
        "mental disorders",
        "musculoskeletal diseases",
        "respiratory tract diseases",
        "anti-infective agents",
    ])

    # Verification sources to include in training examples
    verification_sources: dict = field(default_factory=lambda: {
        "author_payments": {
            "name": "CMS Open Payments",
            "url": "https://openpaymentsdata.cms.gov/",
            "scope": "US physicians - payments from pharma/device manufacturers",
            "api": "https://openpaymentsdata.cms.gov/api/1/datastore/query",
            "notes": "Bulk CSV also available. Covers general payments, research, ownership.",
        },
        "trial_registry": {
            "name": "ClinicalTrials.gov",
            "url": "https://clinicaltrials.gov/",
            "scope": "Registered trial protocols - primary outcomes, sponsors, funding",
            "api": "https://clinicaltrials.gov/api/v2/studies",
            "notes": "Compare registered vs published primary outcomes for switching.",
        },
        "international_registry": {
            "name": "WHO ICTRP",
            "url": "https://trialsearch.who.int/",
            "scope": "International trial registries aggregated",
            "notes": "Useful when ClinicalTrials.gov registration is absent.",
        },
        "retraction_status": {
            "name": "Crossref / Retraction Watch",
            "url": "https://www.crossref.org/documentation/retrieve-metadata/retraction-watch/",
            "scope": "Retraction, correction, expression of concern status",
            "api": "https://api.crossref.org/v1/works?filter=update-type:retraction",
        },
        "author_affiliations": {
            "name": "ORCID",
            "url": "https://orcid.org/",
            "scope": "Author employment history, funding, works",
            "api": "https://pub.orcid.org/v3.0/{orcid}/employments",
        },
        "funder_metadata": {
            "name": "Europe PMC / OpenAlex",
            "url": "https://europepmc.org/",
            "scope": "Funder information, grant IDs, full-text availability",
            "api": "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        },
        "cochrane_rob": {
            "name": "Cochrane Risk of Bias assessments",
            "url": "https://www.cochranelibrary.com/",
            "scope": "Expert risk-of-bias judgments for studies included in Cochrane reviews",
            "notes": "Scraped from Cochrane review supplementary data.",
        },
        "compare_trials": {
            "name": "COMPare Trials Tracker",
            "url": "https://www.compare-trials.org/",
            "scope": "Systematic outcome switching documentation in top 5 journals",
            "notes": "Ben Goldacre's project; covers NEJM, Lancet, BMJ, JAMA, Ann Int Med.",
        },
        "efpia_disclosure": {
            "name": "EFPIA Disclosure (EU equivalent)",
            "url": "https://www.efpia.eu/relationships-codes/disclosure-templates/",
            "scope": "European pharmaceutical industry payments to HCPs",
            "notes": "Varies by country; less centralised than US Open Payments.",
        },
        "aph_disclosure_au": {
            "name": "Medicines Australia Transparency Reports",
            "url": "https://www.medicinesaustralia.com.au/transparency-reporting/",
            "scope": "Australian pharma payments to healthcare professionals",
            "notes": "Annual reports; less granular than US Open Payments.",
        },
    })

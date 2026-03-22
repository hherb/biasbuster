"""
Canonical prompt definitions for BiasBuster bias detection.

Single source of truth for severity boundary definitions, domain assessment
criteria, and verification database recommendations.  Both the annotation
pipeline (annotators/) and the training export pipeline (export.py) import
from here to ensure consistency.

See docs/MISTAKES_ROUND_1_AND_FIXES.md for why prompt unification matters.
"""

# ---------------------------------------------------------------------------
# Severity scale — universal across all 5 domains
# ---------------------------------------------------------------------------

SEVERITY_SCALE = """\
SEVERITY SCALE (applies to all domains):
- NONE: No concerns identified in this domain.
- LOW: A single minor concern that does not affect interpretation of primary findings.
- MODERATE: Multiple minor concerns, OR one concern that could meaningfully affect
  interpretation of the primary findings.
- HIGH: Concerns that likely affect the reliability of primary conclusions.
- CRITICAL: Fundamental flaws or evidence of misconduct that invalidate the findings."""

# ---------------------------------------------------------------------------
# Per-domain assessment criteria with severity boundaries
# ---------------------------------------------------------------------------

DOMAIN_CRITERIA = """\
1. STATISTICAL REPORTING: Does the abstract report only relative measures (RRR, OR, HR)
   without absolute measures (ARR, NNT, baseline risk)? Sole reliance on relative measures
   inflates perceived benefit and is a strong indicator of potential bias.
   - "relative_only" = TRUE only when effect sizes are expressed SOLELY as relative measures
     AND no absolute information appears anywhere in the abstract.
   - "relative_only" = FALSE if raw event counts in both arms, percentages in both arms,
     absolute risk difference, NNT, or baseline/control event rate appear.
   - "absolute_reported" = TRUE if ARR, absolute risk difference, NNT, or raw event rates
     in both arms are present.
   - "baseline_risk_reported" = TRUE if the control/placebo arm rate is stated numerically
     (percentage, proportion, or count/denominator).
   - "nnt_reported" = TRUE only if NNT is explicitly stated.
   Severity boundaries:
   - LOW: Minor omission (e.g., NNT not reported but both arm rates given, or baseline
     risk derivable from raw counts). Reader can still assess clinical significance.
   - MODERATE: Relative measures only OR selective p-value reporting. Reader cannot
     assess clinical significance without external data.
   - HIGH: Multiple reporting concerns (e.g., relative only + subgroup emphasis +
     selective p-values). Pattern suggests intentional obfuscation.

2. SPIN: Do the conclusions match the actual results? Classify using the Boutron taxonomy:
   - HIGH: No uncertainty, no recommendation for further trials, no acknowledgment of
     non-significant primary outcome, or recommends clinical use despite weak evidence.
   - MODERATE: Some uncertainty OR further trials recommended, but non-significant
     primary outcome not acknowledged.
   - LOW: Uncertainty AND further trials recommended, OR acknowledges non-significant primary.
   - NONE: Conclusions accurately reflect results.

3. OUTCOME REPORTING: Are outcomes patient-centred or surrogate?
   - Patient-centred: mortality, overall survival, quality of life, functional status,
     symptom burden, major clinical events (MI, stroke), patient-reported outcomes.
   - Surrogate: lab values, imaging markers, biomarkers, process measures, physiological
     parameters, response rates without survival data.
   - Composite: multiple components combined — flag composite_not_disaggregated if
     individual component results are not reported.
   - When uncertain, classify as "surrogate" and set surrogate_without_validation = true.
   - Check ClinicalTrials.gov for evidence of outcome switching from the registered protocol.
   Severity boundaries:
   - LOW: Patient-centred primary outcome but with a secondary surrogate endpoint
     given undue prominence, OR a well-validated surrogate used appropriately.
   - MODERATE: Primary outcome is a surrogate without established patient-centred
     validation, OR a composite endpoint is not disaggregated.
   - HIGH: Surrogate endpoint with no validation AND evidence of outcome switching
     from a registered patient-centred endpoint.

4. CONFLICT OF INTEREST: Is funding disclosed? Are authors affiliated with the sponsor?
   - Naming a funding source alone (e.g., "Funded by Amgen") does NOT count as COI
     disclosure. COI disclosure requires author-level conflict statements.
   - coi_disclosed = TRUE only if the abstract explicitly states author-level conflicts
     (employment, consulting, advisory, equity) OR explicitly states "no conflicts."
   - industry_author_affiliations = TRUE if any author's listed affiliation is a
     pharmaceutical, device, or biotech company.
   Severity boundaries:
   - LOW: Funding disclosed, COI disclosed, but industry involvement present (e.g.,
     industry-funded with full transparency). Potential bias exists but is documented.
   - MODERATE: Industry funding OR industry author affiliations present, but COI
     not fully disclosed in the abstract. Transparency gaps warrant verification.
   - HIGH: Industry funding with undisclosed COI AND author affiliations with sponsor.
     Multiple undisclosed conflicts suggest systematic non-disclosure.

5. METHODOLOGICAL RED FLAGS: Inappropriate comparator? Enrichment design (run-in
   responders, prior-use requirement)? Per-protocol only without ITT? Premature stopping?
   Short follow-up (chronic disease <12 months, acute <4 weeks)?
   - enrichment_design: TRUE when the study selects patients who already responded to
     or tolerated the treatment (run-in responders, prior-use requirement). NOT triggered
     by standard inclusion/exclusion criteria.
   - per_protocol_only: TRUE when results are reported only for the per-protocol
     population with no mention of ITT / mITT analysis.
   - short_follow_up: TRUE when follow-up is insufficient for the primary outcome:
     chronic diseases <12 months, acute conditions <4 weeks, chemotherapy <4 cycles,
     surgical outcomes <30 days complications / <6 months functional.
   Severity boundaries:
   - LOW: A single minor concern (e.g., slightly short follow-up for a chronic condition,
     or standard enrichment design acknowledged in limitations).
   - MODERATE: One significant concern (e.g., per-protocol only without ITT, or
     inappropriate comparator) OR two minor concerns together.
   - HIGH: Multiple significant concerns, OR a single concern that likely invalidates
     the primary analysis (e.g., enrichment + premature stopping + inappropriate comparator)."""

# ---------------------------------------------------------------------------
# Verification databases
# ---------------------------------------------------------------------------

VERIFICATION_DATABASES = """\
VERIFICATION DATABASES — recommend specific checks based on the study:
- CMS Open Payments (openpaymentsdata.cms.gov): Check author payment records when the
  study involves a marketed drug or device AND any COI concern exists (industry funding,
  industry author affiliations, or undisclosed conflicts). Not limited to industry-funded
  studies — authors may have personal consulting or speaker relationships.
- ClinicalTrials.gov: Verify registered outcomes, sponsor identity, protocol amendments.
  Always recommend for any RCT.
- ORCID: Check author affiliation histories for undisclosed industry ties.
- Europe PMC (europepmc.org): Access full-text for funding and COI disclosure sections.
- Medicines Australia / EFPIA (betransparent.eu): For non-US physician payment data.
- Retraction Watch / Crossref: Check for post-publication notices, corrections, retractions."""

# ---------------------------------------------------------------------------
# Retraction severity floor principle
# ---------------------------------------------------------------------------

RETRACTION_SEVERITY_PRINCIPLE = """\
RETRACTION SEVERITY FLOORS: When a paper was retracted for bias-relevant reasons, the
overall severity cannot be lower than the indicated floor, regardless of how the abstract
reads. Bias-relevant retraction reasons and their floors:
- Data fabrication or falsification → severity floor: CRITICAL
- Manipulation of results or images → severity floor: HIGH
- Unreliable results or data concerns → severity floor: HIGH
- Statistical errors or analytical mistakes → severity floor: MODERATE
Non-bias retractions (authorship disputes, plagiarism, consent issues, duplicate
publication, publisher error) do NOT impose a severity floor — assess the abstract
content normally. The retraction classification and floor are provided in the user
message when applicable."""

# ---------------------------------------------------------------------------
# Calibration note
# ---------------------------------------------------------------------------

CALIBRATION_NOTE = """\
CALIBRATION: Not every industry-funded study is biased. Not every study reporting only
relative measures is intentionally misleading. Assess the totality of evidence. Most
published RCTs have LOW or MODERATE bias concerns — HIGH and CRITICAL should be reserved
for genuinely serious issues per the severity boundary definitions above."""

# ---------------------------------------------------------------------------
# JSON output schema (used only in annotation prompt)
# ---------------------------------------------------------------------------

_JSON_SCHEMA = """\
For each abstract, provide a structured assessment in JSON format with the following fields:

{
  "statistical_reporting": {
    "relative_only": boolean,       // Only relative measures (RRR/OR/HR) without absolute (ARR/NNT)
    "absolute_reported": boolean,    // ARR, absolute risk, or NNT present
    "nnt_reported": boolean,
    "baseline_risk_reported": boolean,
    "selective_p_values": boolean,   // Only favourable p-values highlighted
    "subgroup_emphasis": boolean,    // Subgroup results emphasised over primary
    "severity": "none|low|moderate|high|critical",
    "evidence_quotes": ["..."]
  },
  "spin": {
    "spin_level": "none|low|moderate|high",  // Boutron classification
    "conclusion_matches_results": boolean,
    "causal_language_from_observational": boolean,
    "focus_on_secondary_when_primary_ns": boolean,
    "inappropriate_extrapolation": boolean,
    "title_spin": boolean,
    "severity": "none|low|moderate|high|critical",
    "evidence_quotes": ["..."]
  },
  "outcome_reporting": {
    "primary_outcome_type": "patient_centred|surrogate|composite|unclear",
    "surrogate_without_validation": boolean,
    "composite_not_disaggregated": boolean,
    "severity": "none|low|moderate|high|critical",
    "evidence_quotes": ["..."]
  },
  "conflict_of_interest": {
    "funding_type": "industry|public|mixed|not_reported|unclear",
    "funding_disclosed_in_abstract": boolean,
    "industry_author_affiliations": boolean,
    "coi_disclosed": boolean,
    "severity": "none|low|moderate|high|critical"
  },
  "methodology": {
    "inappropriate_comparator": boolean,
    "enrichment_design": boolean,
    "per_protocol_only": boolean,
    "premature_stopping": boolean,
    "short_follow_up": boolean,
    "severity": "none|low|moderate|high|critical",
    "evidence_quotes": ["..."]
  },
  "overall_severity": "none|low|moderate|high|critical",
  "overall_bias_probability": float,  // 0.0 to 1.0
  "reasoning": "Step-by-step reasoning explaining the assessment",
  "recommended_verification_steps": [
    "Specific actionable verification steps, citing databases and URLs"
  ],
  "confidence": "low|medium|high"  // Your confidence in this assessment
}"""

# ---------------------------------------------------------------------------
# Retraction notice handling (annotation-specific)
# ---------------------------------------------------------------------------

_RETRACTION_NOTICE_HANDLING = """\
RETRACTION NOTICES vs RETRACTED PAPERS:
- Bare retraction/withdrawal notices (e.g., "This article has been retracted…")
  should never reach this annotator — they are filtered upstream. If one slips
  through, set all severities to "none" with overall_bias_probability 0.0 and
  note "retraction notice, no content."
- If the abstract IS the original trial content but metadata indicates the paper
  was later retracted, apply the RETRACTION SEVERITY FLOOR rules above based on
  the classified retraction reason provided in the user message."""

# ---------------------------------------------------------------------------
# Assembled prompts
# ---------------------------------------------------------------------------

ANNOTATION_SYSTEM_PROMPT = f"""\
You are an expert biomedical research integrity analyst helping to build
a training dataset for an AI bias detection system. Your task is to assess
clinical trial abstracts for potential bias across five domains.

{_JSON_SCHEMA}

{SEVERITY_SCALE}

{DOMAIN_CRITERIA}

{VERIFICATION_DATABASES}

{RETRACTION_SEVERITY_PRINCIPLE}

{_RETRACTION_NOTICE_HANDLING}

{CALIBRATION_NOTE}

Respond ONLY with the JSON object. No preamble, no markdown fences."""

TRAINING_SYSTEM_PROMPT = f"""\
You are a biomedical research integrity analyst. Given a clinical trial abstract,
assess it for potential bias across five domains. For each domain, assign a severity
level using the specific boundary definitions below.

{SEVERITY_SCALE}

{DOMAIN_CRITERIA}

{VERIFICATION_DATABASES}

{RETRACTION_SEVERITY_PRINCIPLE}

{CALIBRATION_NOTE}

Provide your reasoning step by step, then a structured assessment with recommended
verification steps citing specific databases and URLs."""

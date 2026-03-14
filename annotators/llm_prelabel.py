"""
LLM Pre-Labelling Annotator

Uses Claude API to generate structured bias assessments for abstracts.
These are PRE-LABELS that must be human-validated before use in training.

Strategy:
1. Send abstract + metadata to Claude with detailed bias assessment prompt
2. Parse structured response into BiasAssessment objects
3. Flag low-confidence assessments for priority human review
4. Export as JSONL for review in a simple web UI or spreadsheet
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

# The system prompt encodes the full bias taxonomy and verification knowledge
ANNOTATION_SYSTEM_PROMPT = """You are an expert biomedical research integrity analyst helping to build
a training dataset for an AI bias detection system. Your task is to assess clinical trial
abstracts for potential bias across multiple dimensions.

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
}

KEY PRINCIPLES:
1. RELATIVE vs ABSOLUTE: The sole use of relative measures (RRR, OR, HR) without
   absolute measures (ARR, NNT, baseline risk) is a STRONG indicator of potential bias.
   A "50% reduction" means very different things at 2% vs 20% baseline risk.

2. VERIFICATION SOURCES: Always suggest specific verification steps:
   - CMS Open Payments (openpaymentsdata.cms.gov) for US physician payments
   - ClinicalTrials.gov for registered outcomes, sponsors, protocol amendments
   - ORCID for author affiliation histories
   - Europe PMC for funder metadata
   - Medicines Australia for AU physician payments
   - EFPIA databases for European physician payments
   - Retraction Watch / Crossref for post-publication notices

3. SPIN CLASSIFICATION (Boutron):
   - HIGH: No uncertainty in conclusions, no recommendation for further trials,
     no acknowledgment of non-significant primary outcomes, or recommends clinical use
   - MODERATE: Some uncertainty OR recommendation for further trials,
     but no acknowledgment of non-significant primary
   - LOW: Uncertainty AND further trials recommended, OR acknowledges NS primary
   - NONE: Conclusions accurately reflect results

4. Be calibrated. Not every industry-funded study is biased. Not every study
   reporting only relative measures is intentionally misleading. Assess the totality.

Respond ONLY with the JSON object. No preamble, no markdown fences."""


class LLMAnnotator:
    """Pre-label abstracts using Claude API via the official Anthropic SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4000,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.client: Optional[anthropic.AsyncAnthropic] = None

    async def __aenter__(self):
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.close()

    async def annotate_abstract(
        self,
        pmid: str,
        title: str,
        abstract: str,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Send a single abstract to Claude for bias assessment.
        Returns parsed JSON assessment or None on failure.
        """
        if not self.api_key:
            logger.error("No Anthropic API key configured")
            return None

        # Build user message with all available context
        user_parts = [
            f"PMID: {pmid}",
            f"Title: {title}",
            f"\nAbstract:\n{abstract}",
        ]

        if metadata:
            if metadata.get("authors"):
                author_str = "; ".join(
                    f"{a.get('last', '')}, {a.get('first', '')} "
                    f"({', '.join(a.get('affiliations', [])[:2])})"
                    for a in metadata["authors"][:5]
                )
                user_parts.append(f"\nAuthors: {author_str}")

            if metadata.get("grants"):
                grant_str = "; ".join(
                    f"{g.get('agency', '')} ({g.get('id', '')})"
                    for g in metadata["grants"]
                )
                user_parts.append(f"Funding: {grant_str}")

            if metadata.get("journal"):
                user_parts.append(f"Journal: {metadata['journal']}")

            if metadata.get("mesh_terms"):
                user_parts.append(f"MeSH: {', '.join(metadata['mesh_terms'][:10])}")

            if metadata.get("retraction_reasons"):
                user_parts.append(
                    f"NOTE: This paper has been RETRACTED. "
                    f"Reasons: {', '.join(metadata['retraction_reasons'])}"
                )

            if metadata.get("effect_size_audit"):
                audit = metadata["effect_size_audit"]
                user_parts.append(
                    f"\nHeuristic pre-screen: {audit.get('pattern', 'unknown')} "
                    f"(score: {audit.get('reporting_bias_score', 0):.2f})"
                )
                if audit.get("flags"):
                    user_parts.append(f"Flags: {'; '.join(audit['flags'])}")

        user_message = "\n".join(user_parts)

        text = ""
        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=ANNOTATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            text = "".join(
                block.text for block in message.content
                if block.type == "text"
            )

            # Strip markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            assessment = json.loads(text)
            assessment["pmid"] = pmid
            assessment["title"] = title
            assessment["_annotation_model"] = self.model
            return assessment

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for PMID {pmid}: {e}")
            logger.debug(f"Raw response: {text[:500]}")
            return None
        except anthropic.APIError as e:
            logger.error(f"API error for PMID {pmid}: {e}")
            return None
        except Exception as e:
            logger.error(f"Annotation failed for PMID {pmid}: {e}")
            return None

    async def annotate_batch(
        self,
        items: list[dict],  # Each: {pmid, title, abstract, metadata?}
        concurrency: int = 3,
        delay: float = 1.0,
    ) -> list[dict]:
        """
        Annotate a batch of abstracts with rate limiting.
        Returns list of successful annotations.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def process_one(item):
            async with semaphore:
                result = await self.annotate_abstract(
                    pmid=item["pmid"],
                    title=item["title"],
                    abstract=item["abstract"],
                    metadata=item.get("metadata"),
                )
                await asyncio.sleep(delay)
                return result

        batch_results = await asyncio.gather(
            *(process_one(item) for item in items)
        )
        results = [r for r in batch_results if r is not None]

        logger.info(
            f"Annotated {len(results)}/{len(items)} abstracts successfully"
        )
        return results

    def save_annotations(self, annotations: list[dict], output_path: Path):
        """Save annotations as JSONL for human review."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ann in annotations:
                f.write(json.dumps(ann) + "\n")
        logger.info(f"Saved {len(annotations)} annotations to {output_path}")

    def generate_review_csv(self, annotations: list[dict], output_path: Path):
        """
        Generate a CSV for easier human review in a spreadsheet.
        Includes key fields and a validation column.
        """
        import csv

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pmid", "title", "overall_severity", "overall_bias_probability",
                "statistical_severity", "relative_only", "spin_level",
                "funding_type", "confidence",
                "reasoning_summary",
                "HUMAN_VALIDATED", "HUMAN_OVERRIDE_SEVERITY", "HUMAN_NOTES",
            ])

            for ann in annotations:
                stat = ann.get("statistical_reporting", {})
                spin = ann.get("spin", {})
                coi = ann.get("conflict_of_interest", {})

                writer.writerow([
                    ann.get("pmid", ""),
                    ann.get("title", "")[:100],
                    ann.get("overall_severity", ""),
                    ann.get("overall_bias_probability", ""),
                    stat.get("severity", ""),
                    stat.get("relative_only", ""),
                    spin.get("spin_level", ""),
                    coi.get("funding_type", ""),
                    ann.get("confidence", ""),
                    ann.get("reasoning", "")[:200],
                    "",  # HUMAN_VALIDATED
                    "",  # HUMAN_OVERRIDE_SEVERITY
                    "",  # HUMAN_NOTES
                ])

        logger.info(f"Generated review CSV at {output_path}")

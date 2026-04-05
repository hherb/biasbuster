"""BiasBuster CLI entry point.

Analyses biomedical publications for risk of bias using LLM-based
assessment with optional programmatic verification.

Usage:
    biasbuster <identifier> [options]

Examples:
    biasbuster 12345678                                    # PMID, default model
    biasbuster 10.1234/example.2024 --format markdown      # DOI, markdown output
    biasbuster 12345678 --format markdown -o report.md     # save report to file
    biasbuster ./paper.xml --model anthropic:claude-sonnet-4-6  # local JATS file
    biasbuster 12345678 --verify --save                    # with verification + DB save
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from typing import Any

from biasbuster.cli.settings import CLIConfig, load_config
from biasbuster.cli.content import AcquiredContent, acquire_content, classify_identifier
from biasbuster.cli.analysis import analyse
from biasbuster.cli.formatting import build_metadata, format_json, format_markdown
from biasbuster.cli.verification import VerificationResult, run_verification

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="biasbuster",
        description="Analyse biomedical publications for risk of bias.",
        epilog="Config file: ~/.biasbuster/config.toml",
    )

    parser.add_argument(
        "identifier",
        help=(
            "Publication identifier: PMID (e.g. 12345678), "
            "DOI (e.g. 10.1234/example.2024), "
            "or local file path (.pdf, .xml, .jats)"
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "LLM model as provider:model_name "
            "(e.g. anthropic:claude-sonnet-4-6, ollama:qwen3.5-9b-biasbuster). "
            "Default: from config file."
        ),
    )

    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        dest="output_format",
        help="Output format (default: json)",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Enable verification pipeline: cross-check assessment against "
            "ClinicalTrials.gov, CMS Open Payments, ORCID, Europe PMC, "
            "Retraction Watch, and effect size audit."
        ),
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist results to the BiasBuster SQLite database.",
    )

    parser.add_argument(
        "--db",
        default=None,
        help="Database path (default: from config or dataset/biasbuster.db)",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Config file path (default: ~/.biasbuster/config.toml)",
    )

    parser.add_argument(
        "--email",
        default=None,
        help="Contact email for API polite pools (PubMed, Unpaywall).",
    )

    parser.add_argument(
        "--outfile",
        "-o",
        default=None,
        metavar="PATH",
        help="Write the report to this file instead of stdout.",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Bypass the download cache and re-fetch content from APIs.",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging to stderr.",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages, output only the result.",
    )

    return parser


def _setup_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging based on verbosity flags."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def _progress(msg: str, quiet: bool) -> None:
    """Print a progress message to stderr unless quiet mode."""
    if not quiet:
        print(msg, file=sys.stderr)


_NCT_RE = re.compile(r"NCT\d{7,8}", re.IGNORECASE)


def _extract_nct_from_content(content: AcquiredContent) -> str:
    """Extract an NCT ID from whatever text is available, cheapest first."""
    for text in (content.abstract, content.plain_fulltext):
        if text:
            m = _NCT_RE.search(text)
            if m:
                return m.group(0)
    if content.jats_xml:
        m = _NCT_RE.search(content.jats_xml.decode("utf-8", errors="ignore"))
        if m:
            return m.group(0)
    return ""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code (0 = success, 1 = error)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.verbose, args.quiet)

    # Load config with CLI overrides
    config = load_config(
        config_path=args.config,
        cli_model=args.model,
        cli_email=args.email,
        cli_db_path=args.db,
    )

    _progress(f"Using model: {config.model}", args.quiet)

    # Classify identifier
    try:
        id_type, id_value = classify_identifier(args.identifier)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _progress(f"Identifier: {id_type} = {id_value}", args.quiet)

    # Acquire content
    _progress("Fetching content...", args.quiet)
    try:
        content = acquire_content(
            args.identifier, config, force_download=args.force_download,
        )
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"Error acquiring content: {exc}", file=sys.stderr)
        return 1

    _progress(
        f"Content type: {content.content_type} | "
        f"Title: {content.title[:80] + '...' if len(content.title) > 80 else content.title}",
        args.quiet,
    )

    # Run bias analysis
    _progress("Running bias analysis...", args.quiet)
    try:
        assessment = analyse(content, config)
    except (ValueError, RuntimeError) as exc:
        print(f"Error during analysis: {exc}", file=sys.stderr)
        return 1

    overall = assessment.get("overall_severity", "unknown").upper()
    prob = assessment.get("overall_bias_probability", 0.0)
    _progress(f"Assessment: {overall} (probability: {prob:.0%})", args.quiet)

    # Optional verification
    verification_dict: dict[str, Any] | None = None
    if args.verify:
        _progress("Running verification pipeline...", args.quiet)
        try:
            vresult = run_verification(
                initial_assessment=assessment,
                content_meta={
                    "pmid": content.pmid,
                    "doi": content.doi,
                    "title": content.title,
                    "abstract": content.abstract,
                    "authors": content.authors,
                    "nct_id": _extract_nct_from_content(content),
                },
                config=config,
            )
            verification_dict = {
                "verification_steps": vresult.verification_steps,
                "tool_results": vresult.tool_results,
                "refined_assessment": vresult.refined_assessment,
                "error": vresult.error,
            }
            if vresult.refined_assessment:
                refined_sev = vresult.refined_assessment.get("overall_severity", "unknown").upper()
                _progress(f"Refined assessment: {refined_sev}", args.quiet)
        except Exception as exc:
            logger.warning("Verification failed: %s", exc)
            _progress(f"Verification failed: {exc}", args.quiet)

    # Build metadata
    metadata = build_metadata(
        identifier=args.identifier,
        identifier_type=id_type,
        title=content.title,
        model=config.model,
        content_type=content.content_type,
        verified=args.verify,
        pmid=content.pmid,
        doi=content.doi,
    )

    # Format output
    if args.output_format == "markdown":
        output = format_markdown(assessment, metadata, verification_dict)
    else:
        output = format_json(assessment, metadata, verification_dict)

    if args.outfile:
        try:
            with open(args.outfile, "w", encoding="utf-8") as fh:
                fh.write(output)
            _progress(f"Report written to {args.outfile}", args.quiet)
        except OSError as exc:
            print(f"Error writing to {args.outfile}: {exc}", file=sys.stderr)
            return 1
    else:
        print(output)

    # Optional DB persistence
    if args.save:
        _save_to_db(content, assessment, config, args.quiet)

    return 0


def _save_to_db(
    content: AcquiredContent,
    assessment: dict[str, Any],
    config: CLIConfig,
    quiet: bool,
) -> None:
    """Persist assessment to the BiasBuster SQLite database."""
    if not content.pmid:
        _progress("Warning: Cannot save without a PMID", quiet)
        return

    try:
        from biasbuster.database import Database

        db = Database(config.db_path)
        db.initialize()

        # Ensure paper exists in DB
        paper_dict = {
            "pmid": content.pmid,
            "title": content.title,
            "abstract": content.abstract,
            "doi": content.doi,
            "journal": content.journal,
            "source": "cli_import",
        }
        db.insert_paper(paper_dict)

        # Store annotation with CLI model name
        model_label = f"cli_{config.model.replace(':', '_')}"
        db.insert_annotation(content.pmid, model_label, assessment)
        _progress(f"Saved to {config.db_path} (model: {model_label})", quiet)
    except Exception as exc:
        logger.exception("Failed to save to database: %s", exc)
        _progress(f"Warning: Failed to save to database: {exc}", quiet)


if __name__ == "__main__":
    sys.exit(main())

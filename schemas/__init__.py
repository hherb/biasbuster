"""Shared utilities and schema definitions."""

import re


def extract_abstract_sections(abstract: str) -> dict[str, str]:
    """
    Split a structured PubMed abstract into labelled sections.

    Many PubMed abstracts have labelled sections (BACKGROUND, METHODS, RESULTS, CONCLUSIONS).
    Returns a dict mapping uppercase section names to their content.
    If the abstract is unstructured, returns {"FULL": abstract}.
    """
    pattern = (
        r'(?:^|\n)\s*(BACKGROUND|INTRODUCTION|OBJECTIVE|PURPOSE|AIM|METHODS|'
        r'RESULTS|FINDINGS|CONCLUSION|CONCLUSIONS|DISCUSSION|INTERPRETATION|'
        r'SIGNIFICANCE|CONTEXT|SETTING|DESIGN|MAIN\s+OUTCOMES?|PARTICIPANTS?|'
        r'INTERVENTIONS?|MEASUREMENTS?)\s*[:\.]?\s*'
    )
    parts = re.split(pattern, abstract, flags=re.IGNORECASE)

    sections = {}
    if len(parts) > 1:
        for i in range(1, len(parts) - 1, 2):
            header = parts[i].strip().upper()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections[header] = content
    else:
        sections["FULL"] = abstract

    return sections

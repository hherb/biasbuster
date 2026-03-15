"""Shared utilities for the bias dataset builder."""

import json
from pathlib import Path


def load_pmids_from_jsonl(path: Path) -> set[str]:
    """Load all PMIDs from a JSONL file."""
    pmids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pmid = record.get("pmid", "")
                if pmid:
                    pmids.add(str(pmid))
            except json.JSONDecodeError:
                continue
    return pmids


def load_records_from_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_records_by_pmid(path: Path) -> dict[str, dict]:
    """Load all records from a JSONL file, indexed by PMID."""
    by_pmid = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pmid = record.get("pmid", "")
                if pmid:
                    by_pmid[str(pmid)] = record
            except json.JSONDecodeError:
                continue
    return by_pmid

"""PMC Open-Access-subset membership and license classification.

Given a PubMed/PMC identifier, determine whether a trial is in the PMC
Open Access Subset and under which license, so the OA-first benchmark can
admit only redistributable full text (spec §4.1) and flag NC/ND per item
(spec §2 license-strictness decision).

Pure classification (`classify_license`) is separated from network I/O
(`fetch_oa_status`) so the licensing logic is unit-testable offline.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

from biasbuster.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

#: Europe PMC returns license strings like "cc by", "cc by-nc", "cc0".
_CC_TOKEN = re.compile(r"cc[\s\-]?by", re.IGNORECASE)
_CC0_TOKEN = re.compile(r"cc0|public domain", re.IGNORECASE)


@dataclass(frozen=True)
class LicenseInfo:
    """Normalised license facts for one document."""
    raw: str
    spdx: str
    redistributable: bool
    non_commercial: bool
    no_derivatives: bool


def classify_license(raw_license: str) -> LicenseInfo:
    """Classify a raw license string into redistribution facts.

    Every Creative Commons license permits verbatim redistribution, so
    ``redistributable`` is True for any CC/CC0 license; ``non_commercial``
    and ``no_derivatives`` flag the NC and ND downstream constraints. A
    blank or all-rights-reserved string is not redistributable.
    """
    raw = (raw_license or "").strip()
    low = raw.lower()
    if _CC0_TOKEN.search(low):
        return LicenseInfo(raw, "CC0-1.0", True, False, False)
    if _CC_TOKEN.search(low) and "no-cc" not in low:
        nc = "nc" in low.replace("no-cc", "")
        nd = "nd" in low
        sa = "sa" in low
        parts = ["CC-BY"]
        if nc:
            parts.append("NC")
        if nd:
            parts.append("ND")
        elif sa:
            parts.append("SA")
        spdx = "-".join(parts) + "-4.0"
        return LicenseInfo(raw, spdx, True, nc, nd)
    return LicenseInfo(raw, "", False, False, False)


@dataclass
class OAStatus:
    """OA-subset membership + license for one PMID."""
    pmid: str
    pmcid: str
    in_oa_subset: bool
    license: LicenseInfo


async def fetch_oa_status(
    client: httpx.AsyncClient, pmid: str, *, base: str
) -> OAStatus:
    """Query Europe PMC for OA-subset membership and license of ``pmid``.

    A trial is treated as in the OA subset only when Europe PMC reports it
    open-access, present in EPMC full text, and under a redistributable
    (CC/CC0) license. Any query or parse failure yields a non-OA status
    (fail-closed) after retries — logged, never raised to the caller.

    ``retry_with_backoff`` (not ``fetch_with_retry``) wraps the request
    here because it retries an arbitrary zero-argument coroutine factory
    rather than requiring ``client.request`` specifically, letting this
    function call ``client.get`` directly (matching both a real
    ``httpx.AsyncClient`` and simpler injected test doubles) while still
    reusing the shared exponential-backoff retry logic.
    """
    url = f"{base}/search"
    params = {
        "query": f"EXT_ID:{pmid} AND SRC:MED",
        "resultType": "core",
        "format": "json",
    }

    async def _request() -> httpx.Response:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp

    try:
        resp = await retry_with_backoff(
            _request, operation_name=f"Europe PMC OA status fetch for PMID {pmid}"
        )
        results = resp.json().get("resultList", {}).get("result", [])
    except Exception as exc:  # network/parse — fail closed, logged
        logger.warning("OA status fetch failed for PMID %s: %s", pmid, exc)
        return OAStatus(pmid, "", False, classify_license(""))

    if not results:
        return OAStatus(pmid, "", False, classify_license(""))
    r = results[0]
    pmcid = str(r.get("pmcid", "") or "")
    lic = classify_license(str(r.get("license", "") or ""))
    in_oa = (
        str(r.get("isOpenAccess", "")).upper() == "Y"
        and str(r.get("inEPMC", "")).upper() == "Y"
        and bool(pmcid)
        and lic.redistributable
    )
    return OAStatus(pmid, pmcid, in_oa, lic)

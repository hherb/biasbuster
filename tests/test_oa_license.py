"""Tests for PMC Open-Access-subset membership and license classification."""
import pytest

from biasbuster.collectors.oa_license import classify_license, fetch_oa_status


def test_cc_by_is_redistributable_unrestricted():
    info = classify_license("CC BY")
    assert info.spdx == "CC-BY-4.0"
    assert info.redistributable is True
    assert info.non_commercial is False
    assert info.no_derivatives is False


def test_cc_by_nc_nd_flags_both():
    info = classify_license("CC BY-NC-ND")
    assert info.redistributable is True   # all CC permit verbatim redistribution
    assert info.non_commercial is True
    assert info.no_derivatives is True


def test_cc0_public_domain():
    info = classify_license("CC0")
    assert info.spdx == "CC0-1.0"
    assert info.redistributable is True


def test_unknown_or_all_rights_reserved_not_redistributable():
    assert classify_license("").redistributable is False
    assert classify_license("NO-CC BY").redistributable is False
    assert classify_license("copyright, all rights reserved").redistributable is False


class _FakeResponse:
    def __init__(self, payload): self._payload = payload; self.status_code = 200
    def json(self): return self._payload
    def raise_for_status(self): pass


class _FakeClient:
    def __init__(self, payload): self._payload = payload; self.calls = []
    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs)); return _FakeResponse(self._payload)


@pytest.mark.asyncio
async def test_fetch_oa_status_parses_epmc_result():
    payload = {"resultList": {"result": [
        {"pmid": "12345", "pmcid": "PMC999", "isOpenAccess": "Y",
         "inEPMC": "Y", "license": "cc by"}]}}
    client = _FakeClient(payload)
    status = await fetch_oa_status(client, "12345", base="https://epmc.test")
    assert status.pmcid == "PMC999"
    assert status.in_oa_subset is True
    assert status.license.redistributable is True
    assert status.license.non_commercial is False


@pytest.mark.asyncio
async def test_fetch_oa_status_missing_pmc_is_not_oa():
    payload = {"resultList": {"result": [
        {"pmid": "222", "isOpenAccess": "N"}]}}
    status = await fetch_oa_status(_FakeClient(payload), "222", base="https://epmc.test")
    assert status.in_oa_subset is False
    assert status.license.redistributable is False

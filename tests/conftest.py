"""Shared fixtures for biasbuster tests."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# Repo-root modules (`config`, `annotate_single_paper`, `main`, `seed_database`,
# `export`) live at the repository root — they are NOT part of the installed
# `biasbuster` package — and are imported bare, e.g. `from config import Config`.
# They resolve only when the repo root is on `sys.path`. In a settled venv that
# holds three ways: the editable install's `.pth`, `pyproject.toml`'s
# `[tool.pytest.ini_options] pythonpath = ["."]`, and this pin. The pin exists so
# the guarantee does not rely on the editable install being fully materialised
# (it can lag a fresh `uv sync` — the likely cause of the one-off
# `ModuleNotFoundError: No module named 'config'` at annotate_single_paper.py
# that motivated it) or on pytest's import mode. It is idempotent — a no-op when
# the root is already present — and makes the otherwise-implicit "bare root
# imports work in tests" contract explicit where maintainers will look.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Directories of the active study's plain-file modules. `studies/` is not a
# package: its modules import each other bare (`from forest_data import ...`)
# off sys.path, so tests load them by file path instead.
STUDY_DIR = Path(_REPO_ROOT) / "studies" / "eisele_metzger_replication"
FIGURES_DIR = STUDY_DIR / "figures"


def load_study_module(module_name: str,
                      directory: Path = STUDY_DIR) -> types.ModuleType:
    """Load a study module by file path, shared by test modules.

    Registers the module in ``sys.modules`` under its bare name BEFORE
    executing it, for two reasons: ``@dataclass`` (used by e.g.
    ``compute_phase6_kappa.KappaRow``) looks the class's module up in
    ``sys.modules`` during class creation and fails if absent, and study
    modules import their siblings bare (``from forest_data import ...``),
    which must resolve to the already-loaded copy rather than a duplicate.

    Returns the already-registered module when one exists, so every test
    file shares a single copy per module name instead of re-executing the
    file and shadowing the previous registration.
    """
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        module_name, directory / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def biased_abstract():
    """Abstract with relative-only reporting (high bias signal)."""
    return {
        "pmid": "TEST001",
        "title": "Wonderdrug reduces major adverse cardiac events by 47% in statin-intolerant patients",
        "abstract": (
            "BACKGROUND: Statin-intolerant patients remain at elevated cardiovascular risk. "
            "We evaluated wonderdrug in this population. "
            "METHODS: Randomized, double-blind, placebo-controlled trial of 8,000 patients. "
            "Primary endpoint was time to first major adverse cardiac event (MACE). "
            "RESULTS: Wonderdrug significantly reduced MACE (HR 0.53, 95% CI 0.40-0.70, "
            "p<0.001), representing a 47% relative risk reduction. Subgroup analyses showed "
            "consistent benefit across age groups. "
            "CONCLUSIONS: Wonderdrug provides substantial cardiovascular protection in "
            "statin-intolerant patients and should be considered as first-line alternative therapy."
        ),
    }


@pytest.fixture
def balanced_abstract():
    """Abstract with both relative and absolute measures (good reporting)."""
    return {
        "pmid": "TEST002",
        "title": "Effect of gooddrug on cardiovascular outcomes: a randomized trial",
        "abstract": (
            "BACKGROUND: We assessed whether gooddrug reduces cardiovascular events. "
            "METHODS: Multicentre RCT, n=5000, median follow-up 3.2 years. "
            "RESULTS: The primary endpoint occurred in 8.2% of the gooddrug group vs "
            "10.8% of the placebo group (HR 0.75, 95% CI 0.63-0.89, p=0.001). "
            "The absolute risk reduction was 2.6% (95% CI 1.1-4.1%), corresponding to "
            "a number needed to treat of 39 over 3 years. "
            "CONCLUSIONS: Gooddrug modestly reduced cardiovascular events with an NNT of 39. "
            "The absolute benefit should be weighed against the cost and side effect profile."
        ),
    }


@pytest.fixture
def no_effect_size_abstract():
    """Abstract with no quantitative effect sizes."""
    return {
        "pmid": "TEST003",
        "title": "Patient satisfaction with a new care model",
        "abstract": (
            "BACKGROUND: We explored patient satisfaction with a novel care delivery model. "
            "METHODS: Qualitative interviews with 30 participants. "
            "RESULTS: Patients reported high satisfaction and improved communication. "
            "CONCLUSIONS: The new care model was well received by patients."
        ),
    }


@pytest.fixture
def spun_abstract():
    """Abstract exhibiting spin: NS primary, but benefit claimed in conclusion."""
    return {
        "pmid": "SPIN001",
        "title": "Wonderdrug improves outcomes in heart failure patients",
        "abstract": (
            "BACKGROUND: Heart failure remains a major cause of morbidity.\n"
            "METHODS: Randomized, double-blind trial of wonderdrug vs placebo in 500 patients. "
            "Primary endpoint: all-cause mortality at 12 months.\n"
            "RESULTS: The primary endpoint did not reach statistical significance "
            "(HR 0.82, 95% CI 0.65-1.04, p=0.10). However, there was a trend toward "
            "reduced mortality. In the pre-specified subgroup of patients with EF<30%, "
            "wonderdrug significantly reduced mortality (HR 0.60, p=0.02). "
            "Quality of life scores improved significantly from baseline in the treatment group.\n"
            "CONCLUSIONS: Wonderdrug shows promising efficacy in heart failure patients, "
            "particularly those with severely reduced ejection fraction. These results "
            "suggest wonderdrug should be considered as an addition to standard therapy."
        ),
    }


@pytest.fixture
def clean_abstract():
    """Well-reported abstract with no spin."""
    return {
        "pmid": "CLEAN001",
        "title": "Effect of gooddrug on mortality in heart failure: a randomized trial",
        "abstract": (
            "BACKGROUND: We assessed whether gooddrug reduces mortality in heart failure.\n"
            "METHODS: Multicentre RCT, n=3000, median follow-up 2.5 years.\n"
            "RESULTS: All-cause mortality occurred in 12.1% of the gooddrug group and "
            "14.8% of the placebo group (HR 0.80, 95% CI 0.68-0.94, p=0.006). "
            "The absolute risk reduction was 2.7% (NNT 37).\n"
            "CONCLUSIONS: Gooddrug reduced all-cause mortality in heart failure patients. "
            "The modest absolute benefit should be weighed against cost and adverse effects. "
            "Longer-term follow-up data are needed."
        ),
    }


@pytest.fixture
def industry_funded_abstract():
    """Abstract with clear industry funding signals."""
    return {
        "pmid": "FUND001",
        "abstract": (
            "BACKGROUND: We evaluated wonderdrug in hypertension. "
            "METHODS: Multicentre RCT sponsored by Pfizer Inc. "
            "RESULTS: Significant blood pressure reduction. "
            "FUNDING: This study was funded by Pfizer Inc. Medical writing "
            "assistance was provided by PAREXEL, funded by Pfizer."
        ),
    }


@pytest.fixture
def sample_annotation():
    """Minimal annotation dict for export tests."""
    return {
        "pmid": "ANN001",
        "title": "Test Drug Trial",
        "abstract": "Test abstract text here.",
        "statistical_reporting": {
            "relative_only": True,
            "severity": "high",
            "evidence_quotes": ["HR 0.50, p<0.001"],
        },
        "spin": {
            "spin_level": "moderate",
            "severity": "moderate",
            "conclusion_matches_results": False,
            "focus_on_secondary_when_primary_ns": True,
        },
        "conflict_of_interest": {
            "funding_type": "industry",
            "severity": "high",
            "industry_author_affiliations": True,
            "coi_disclosed": False,
        },
        "overall_severity": "high",
        "overall_bias_probability": 0.85,
        "reasoning": "The abstract only reports relative measures.",
        "recommended_verification_steps": [
            "Check ClinicalTrials.gov for registered primary outcome",
            "Search CMS Open Payments for author-sponsor payments",
        ],
    }


@pytest.fixture
def pubmed_single_xml():
    """Minimal valid PubMed XML for a single article."""
    return """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal>
          <Title>Test Journal</Title>
          <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Effect of TestDrug on mortality</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">We tested a drug.</AbstractText>
          <AbstractText Label="RESULTS">Mortality was reduced.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
            <AffiliationInfo><Affiliation>University of Testing</Affiliation></AffiliationInfo>
          </Author>
        </AuthorList>
        <ELocationID EIdType="doi">10.1234/test.2024</ELocationID>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Mortality</DescriptorName></MeshHeading>
      </MeshHeadingList>
      <GrantList>
        <Grant>
          <GrantID>R01-HL12345</GrantID>
          <Agency>NHLBI</Agency>
        </Grant>
      </GrantList>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""

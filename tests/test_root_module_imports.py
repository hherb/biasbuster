"""Regression guard: bare repo-root modules import under collection-time churn.

`config`, `annotate_single_paper`, `main`, and `seed_database` live at the
repository root (not inside the installed `biasbuster` package) and are imported
bare, e.g. `from config import Config`. They resolve only when the repo root is
on `sys.path`. In a settled venv that holds three ways (the editable install's
`.pth`, `pyproject.toml`'s `pythonpath = ["."]`, and `conftest.py`'s explicit
pin); a one-off `ModuleNotFoundError: No module named 'config'` was seen at
`annotate_single_paper.py`, most plausibly during the window right after a fresh
`uv sync` before the editable install had fully materialised.

The suite makes that contract easy to break in subtler ways: several study
modules call `sys.path.insert(0, <study dir>)` at import time and are loaded
during collection (by `test_kappa_exclusions` / `test_wrong_paper_audit` /
`test_recover_wrong_papers`), and some tests import the bare root modules only at
call time (e.g. `test_applicability_guard` imports `annotate_single_paper` inside
its methods). This module reproduces that adversarial collection-time state — a
representative heavy import plus the study-module `sys.path` churn — and asserts
every bare root module still imports from the repo root. It fails if the
repo-root-on-`sys.path` contract regresses, if a shadowing `config.py` appears on
a study `sys.path` entry, or if a root module gains an import-time error.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

# A representative heavy collection-time import (this is the one whose top-level
# placement perturbed the earlier run before it was moved into a test body).
import biasbuster.collectors.retraction_watch  # noqa: F401,E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STUDY_DIR = _REPO_ROOT / "studies" / "eisele_metzger_replication"

#: Bare top-level modules at the repo root that must remain importable in tests.
_ROOT_MODULES = ("config", "annotate_single_paper", "main", "seed_database")


def _load_study_module(name: str) -> None:
    """Load a study module by path, mirroring the wrong-paper/kappa tests.

    These modules call `sys.path.insert(0, <study dir>)` at import time, so
    loading them here reproduces the collection-time `sys.path` churn this
    regression guards against.
    """
    spec = importlib.util.spec_from_file_location(name, _STUDY_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


# Reproduce the adversarial sys.path state at collection time.
for _study in ("exclusions", "compute_phase6_kappa"):
    _load_study_module(_study)


@pytest.mark.parametrize("module_name", _ROOT_MODULES)
def test_repo_root_module_imports_cleanly(module_name: str) -> None:
    """Each bare repo-root module imports fresh despite study `sys.path` churn."""
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    assert module is not None
    assert Path(module.__file__).resolve().parent == _REPO_ROOT

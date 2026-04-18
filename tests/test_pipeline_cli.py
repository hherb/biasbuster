"""CLI-integration smoke tests for the --methodology flag.

These tests exercise the argparse layer of ``biasbuster.pipeline`` and
``annotate_single_paper`` as a subprocess so we hit the exact code path
users hit, without mocking argparse away. Scope is limited to argument
parsing / validation — actual annotation runs need API keys and network
and are out of scope.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module", autouse=True)
def _ensure_local_config() -> None:
    """Ensure config.py exists for subprocess-invoked CLI scripts.

    Both ``biasbuster.pipeline`` and ``annotate_single_paper`` import
    ``from config import Config`` at module load time. ``config.py`` is
    gitignored (per project convention — contains API keys) so a fresh
    worktree has only ``config.example.py``. For CLI argparse tests we
    only need the class to exist; a copy of the example is sufficient.
    The copy is made once per test-module and left in place (harmless —
    it's gitignored).
    """
    cfg = REPO_ROOT / "config.py"
    example = REPO_ROOT / "config.example.py"
    if not cfg.exists() and example.exists():
        shutil.copy(example, cfg)


def _run_python(args: list[str]) -> subprocess.CompletedProcess:
    """Run a Python script using the worktree's uv-managed interpreter.

    Using ``sys.executable`` would pick whichever Python the test runner
    launched under, which may be a system Python without the biasbuster
    package installed. ``uv run`` resolves to the worktree's ``.venv``.
    """
    return subprocess.run(
        ["uv", "run", "python", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


class TestPipelineCliMethodologyFlag:
    def test_help_mentions_methodology_flag(self) -> None:
        result = _run_python(["-m", "biasbuster.pipeline", "--help"])
        assert result.returncode == 0, result.stderr
        assert "--methodology" in result.stdout
        assert "biasbuster" in result.stdout.lower()

    def test_unknown_methodology_rejected(self) -> None:
        """stage_annotate refuses a methodology slug that isn't registered.

        Uses --stage annotate so the validation path actually runs.
        Expected: non-zero exit + UnknownMethodologyError message on stderr.
        """
        result = _run_python([
            "-m", "biasbuster.pipeline",
            "--stage", "annotate",
            "--methodology", "fake_does_not_exist",
        ])
        assert result.returncode != 0
        combined = result.stderr + result.stdout
        assert "unknown methodology" in combined.lower() \
            or "fake_does_not_exist" in combined

    def test_single_call_rejected_with_non_biasbuster_methodology(self) -> None:
        """Orchestration flag + non-biasbuster methodology must be rejected at argparse.

        At this point cochrane_rob2 isn't registered, but the argparse
        validator runs BEFORE the registry lookup — it only checks the
        flag-combination rule (``--single-call`` is biasbuster-only).
        So any non-biasbuster slug, even a fake one, should trip the
        validator and exit with a usage error rather than reach
        stage_annotate.
        """
        result = _run_python([
            "-m", "biasbuster.pipeline",
            "--stage", "annotate",
            "--methodology", "cochrane_rob2",
            "--single-call",
        ])
        assert result.returncode != 0
        assert "--single-call is only valid with --methodology=biasbuster" \
            in (result.stderr + result.stdout)


class TestAnnotateSinglePaperMethodologyFlag:
    def test_help_mentions_methodology_flag(self) -> None:
        result = _run_python(["annotate_single_paper.py", "--help"])
        assert result.returncode == 0, result.stderr
        assert "--methodology" in result.stdout

    @pytest.mark.parametrize("flag", [
        "--single-call", "--full-text", "--agentic", "--decomposed",
    ])
    def test_orchestration_flag_rejected_with_non_biasbuster(
        self, flag: str
    ) -> None:
        """Every orchestration flag must be rejected with non-biasbuster methodology."""
        result = _run_python([
            "annotate_single_paper.py",
            "--pmid", "99999999",
            "--methodology", "cochrane_rob2",
            flag,
        ])
        assert result.returncode != 0
        combined = result.stderr + result.stdout
        assert flag in combined
        assert "--methodology=biasbuster" in combined

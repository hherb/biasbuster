"""Shared factory for ``status='stub'`` methodology modules.

A stub methodology is registered in the :data:`REGISTRY` so users see
it in ``list_methodologies()`` and the plan stays discoverable, but
every behavioural callable raises :class:`NotImplementedError` with a
clear message so an accidental invocation fails loud rather than
silently producing garbage.

Each stub submodule (``robins_i``, ``robins_e``, ``probast``,
``probast_plus_ai``, ``rob_me``, ``syrcle``, ``robis``, ``amstar_2``)
is a one-liner that calls :func:`make_stub_methodology` and registers
the result. Adding the ninth stub later costs one file and one line
in the package ``__init__``.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import Methodology, OrchestrationShape
from .registry import REGISTRY, register


def _stub_callable(method: str, methodology_name: str):
    """Build a NotImplementedError-raising callable for a stub methodology.

    The method name and methodology slug are baked into the error so
    the traceback names exactly what's missing, not just a generic
    "not implemented". The returned callable accepts any args/kwargs
    because the :class:`Methodology` dataclass callable fields are
    typed as untyped callables (intentional — see Step 4 discussion).
    """
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(
            f"methodology {methodology_name!r} is registered as a stub; "
            f"'{method}' is not implemented yet. See the plan for the "
            "intended spec."
        )

    _raise.__name__ = f"_stub_{methodology_name}_{method}"
    _raise.__qualname__ = _raise.__name__
    return _raise


def make_stub_methodology(
    *,
    name: str,
    display_name: str,
    version: str,
    applicable_study_designs: frozenset[str],
    requires_full_text: bool,
    orchestration: OrchestrationShape,
    severity_rollup_levels: tuple[str, ...],
    notes: Optional[dict[str, str]] = None,
) -> Methodology:
    """Construct a :class:`Methodology` whose behavioural callables all raise.

    All declarative fields must be provided so the stub still appears
    in ``list_methodologies()`` with meaningful metadata (study-design
    applicability, full-text requirement, display name). A stub's
    ``check_applicability`` also raises — an applicability check against
    a stub methodology is itself a bug in the caller, since any
    decision-making code should have filtered stubs out by looking at
    ``methodology.status`` first.
    """
    return Methodology(
        name=name,
        display_name=display_name,
        version=version,
        applicable_study_designs=applicable_study_designs,
        requires_full_text=requires_full_text,
        orchestration=orchestration,
        severity_rollup_levels=severity_rollup_levels,
        status="stub",
        build_system_prompt=_stub_callable("build_system_prompt", name),
        build_user_message=_stub_callable("build_user_message", name),
        parse_output=_stub_callable("parse_output", name),
        aggregate=_stub_callable("aggregate", name),
        check_applicability=_stub_callable("check_applicability", name),
        evaluation_mapping_to_ground_truth=_stub_callable(
            "evaluation_mapping_to_ground_truth", name,
        ),
        build_training_example=None,
        notes=notes or {},
    )


def register_stub_once(methodology: Methodology) -> Methodology:
    """Idempotent stub registration helper.

    Mirrors the ``_register_once`` pattern used by the active methodology
    submodules so repeat imports don't raise ``DuplicateMethodologyError``.
    """
    if REGISTRY.get(methodology.name) is methodology:
        return methodology
    return register(methodology, replace=True)

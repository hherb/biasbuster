"""Methodology registry package.

Importing this package registers every methodology submodule on the
module-level :data:`REGISTRY`. Callers then look up methodologies by slug:

    >>> from biasbuster.methodologies import get_methodology
    >>> m = get_methodology("biasbuster")
    >>> m.display_name
    'Biasbuster default'

Importing this package registers the built-in methodology submodules.
Additional methodologies (``cochrane_rob2``, ``quadas_2``, stubs...) are
added in subsequent build steps; they each self-register on import.
"""

from __future__ import annotations

from .base import (
    ANY_STUDY_DESIGN,
    STUDY_DESIGN_SLUGS,
    ApplicabilityError,
    DuplicateMethodologyError,
    FullTextRequiredError,
    Methodology,
    MethodologyError,
    MethodologyStatus,
    OrchestrationShape,
    UnknownMethodologyError,
    UnsupportedOrchestrationError,
    check_or_raise,
)
from .registry import (
    REGISTRY,
    clear_registry_for_testing,
    get_methodology,
    list_active_methodologies,
    list_methodologies,
    register,
)
from . import study_design
# Trigger self-registration of built-in methodologies. Importing a
# methodology submodule inserts its Methodology into REGISTRY. Further
# methodologies (cochrane_rob2, quadas_2, stubs) will be added to this
# block in later build steps — each new submodule registers once on import.
from . import biasbuster as _biasbuster  # noqa: F401  (registration side effect)


def _register_builtin_methodologies() -> None:
    """Re-install every built-in methodology into :data:`REGISTRY`.

    The module-level ``from . import biasbuster`` above registers the
    default pathway as an import side effect. That side effect only
    fires once per process — after a ``clear_registry_for_testing()``
    call the modules remain cached in ``sys.modules`` and the registry
    stays empty. Test fixtures that clear the registry should call this
    function at teardown so subsequent test modules don't inherit an
    empty registry (which would otherwise make ``get_methodology(\"biasbuster\")``
    fail based purely on alphabetical test-file ordering).

    Add new built-in methodologies to the body of this function as they
    are introduced (cochrane_rob2, quadas_2, ...).
    """
    _biasbuster._register_once()


__all__ = [
    "ANY_STUDY_DESIGN",
    "STUDY_DESIGN_SLUGS",
    "ApplicabilityError",
    "DuplicateMethodologyError",
    "FullTextRequiredError",
    "Methodology",
    "MethodologyError",
    "MethodologyStatus",
    "OrchestrationShape",
    "REGISTRY",
    "UnknownMethodologyError",
    "UnsupportedOrchestrationError",
    "_register_builtin_methodologies",
    "check_or_raise",
    "clear_registry_for_testing",
    "get_methodology",
    "list_active_methodologies",
    "list_methodologies",
    "register",
    "study_design",
]

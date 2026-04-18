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
    "check_or_raise",
    "clear_registry_for_testing",
    "get_methodology",
    "list_active_methodologies",
    "list_methodologies",
    "register",
    "study_design",
]

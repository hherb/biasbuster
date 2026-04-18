"""Schema types for the biasbuster methodology (re-exports).

Biasbuster's assessment schema is authoritative in
:mod:`biasbuster.schemas.bias_taxonomy`. This module re-exports
``BiasAssessment`` so every methodology subpackage has the same symmetric
structure (``<methodology>.schema.Assessment``), and so future refactors
of the biasbuster schema only need to touch the canonical location.
"""

from __future__ import annotations

from biasbuster.schemas.bias_taxonomy import BiasAssessment

# Alias under a methodology-agnostic name so callers can write
# ``from biasbuster.methodologies.biasbuster.schema import Assessment``
# symmetrically with, e.g., cochrane_rob2.schema.
Assessment = BiasAssessment

__all__ = ["Assessment", "BiasAssessment"]

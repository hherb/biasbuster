"""Mutable registry of methodology instances.

Methodology modules call :func:`register` at import time. The
:mod:`biasbuster.methodologies.__init__` module triggers those imports so
the registry is populated before any caller looks up a methodology.
"""

from __future__ import annotations

from .base import (
    DuplicateMethodologyError,
    Methodology,
    UnknownMethodologyError,
)

# Public only to the methodologies package + tests. External callers should
# use ``get_methodology()`` / ``list_methodologies()`` rather than touching
# the dict directly.
REGISTRY: dict[str, Methodology] = {}


def register(methodology: Methodology, *, replace: bool = False) -> Methodology:
    """Add a methodology to the registry.

    Args:
        methodology: The methodology to register.
        replace: If True, overwrite an existing entry with the same name.
            Default False — duplicate registrations raise. This prevents
            silent mis-registration when two modules reuse a slug.

    Returns:
        The same methodology (convenient for decorator-style use).

    Raises:
        DuplicateMethodologyError: If ``methodology.name`` is already
            registered and ``replace=False``.
    """
    existing = REGISTRY.get(methodology.name)
    if existing is not None and not replace:
        raise DuplicateMethodologyError(
            f"methodology {methodology.name!r} is already registered "
            f"(existing: {existing.display_name!r}, new: "
            f"{methodology.display_name!r}). Use replace=True if this "
            "is intentional."
        )
    REGISTRY[methodology.name] = methodology
    return methodology


def get_methodology(name: str) -> Methodology:
    """Look up a registered methodology by its slug.

    Raises:
        UnknownMethodologyError: If no methodology with that name is registered.
    """
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise UnknownMethodologyError(
            f"unknown methodology {name!r}. Registered methodologies: "
            f"{sorted(REGISTRY)}"
        ) from exc


def list_methodologies() -> list[str]:
    """Return all registered methodology slugs (active + stub), sorted."""
    return sorted(REGISTRY)


def list_active_methodologies() -> list[str]:
    """Return only the ``status='active'`` methodology slugs, sorted."""
    return sorted(n for n, m in REGISTRY.items() if m.status == "active")


def clear_registry_for_testing() -> None:
    """Reset the registry. For use in test fixtures only — never call from app code."""
    REGISTRY.clear()

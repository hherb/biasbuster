"""Annotation form component for crowd annotation.

Wraps the existing build_review_form() and collect_form_data() from
utils/review_form.py. For the blind phase, passes an empty dict to get
a blank form. For the revision phase, passes the user's blind annotation.
"""

from nicegui import ui

from utils.review_form import build_review_form, collect_form_data


def build_blind_form(container: ui.element) -> dict:
    """Build an empty annotation form for the blind phase.

    Args:
        container: NiceGUI element to build the form inside.

    Returns:
        Widget refs dict — pass to collect_form_data() to harvest JSON.
    """
    return build_review_form({}, container)


def build_revision_form(
    blind_annotation: dict, container: ui.element
) -> dict:
    """Build a form pre-populated with the user's blind annotation for revision.

    Args:
        blind_annotation: The user's blind phase annotation dict.
        container: NiceGUI element to build the form inside.

    Returns:
        Widget refs dict — pass to collect_form_data() to harvest JSON.
    """
    return build_review_form(blind_annotation, container)


def harvest_form(refs: dict) -> dict:
    """Harvest form widget values into an annotation dict.

    Delegates to the shared collect_form_data() to produce the same
    JSON structure as the LLM annotation schema.
    """
    return collect_form_data(refs)

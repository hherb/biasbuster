"""Study-design gate for the Cochrane RoB 2 methodology.

RoB 2 applies to parallel-group randomized trials. Cluster-RCTs and
crossover RCTs have their own Cochrane variants with additional domains
(recruitment bias, carry-over effects) — those are out of scope for v1
and should be registered as separate methodologies when/if added.

The check is two-tier:

1. The declarative ``applicable_study_designs`` on the Methodology
   dataclass filters by the ``study_design.detect`` slug.
2. This ``check_applicability`` function gives the methodology a chance
   to refuse based on finer-grained paper state (e.g. a paper flagged
   ``rct_parallel`` by heuristics but missing essential trial-registry
   metadata).

For MVP (Step 7 scaffold) the check is minimal — refuse non-RCT designs
and note the detected slug in the reason. Future versions can inspect
Cochrane review linkage, protocol registration, etc.
"""

from __future__ import annotations

from biasbuster.methodologies import study_design


def check_applicability(
    paper: dict, enrichment: dict, full_text_available: bool,
) -> tuple[bool, str]:
    """Return ``(True, "")`` if RoB 2 applies; ``(False, reason)`` otherwise.

    Does not itself enforce the full-text requirement — that's the
    methodology-level ``requires_full_text=True`` flag, checked by the
    shared ``check_or_raise`` guard before this function is called.
    """
    del enrichment, full_text_available  # not used in the MVP check
    detected = study_design.detect(paper)
    if detected == "rct_parallel":
        return True, ""
    # Cochrane RoB 2 variants for cluster/crossover RCTs are separate
    # methodologies (not yet implemented). Refuse with a specific
    # message so the user knows which pathway would fit instead.
    if detected == "rct_cluster":
        return False, (
            "paper appears to be a cluster-randomized trial; standard "
            "RoB 2 does not apply. The cluster-RCT RoB 2 variant adds a "
            "recruitment-bias domain and is not yet registered in "
            "biasbuster."
        )
    if detected == "rct_crossover":
        return False, (
            "paper appears to be a crossover trial; standard RoB 2 does "
            "not apply. The crossover RoB 2 variant handles period/"
            "carry-over effects and is not yet registered in biasbuster."
        )
    return False, (
        f"Cochrane RoB 2 applies only to parallel-group randomized "
        f"trials; detected study design was {detected!r}. Use "
        "--methodology=biasbuster for design-agnostic assessment."
    )

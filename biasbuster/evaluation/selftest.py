#!/usr/bin/env python3
"""
Self-Test for the Evaluation Pipeline

Validates the full harness -> scorer -> metrics -> comparison chain
using synthetic test data and simulated model outputs.
No model endpoints required — this tests the evaluation machinery itself.

Run:
    python -m evaluation.selftest
"""

import json
import sys
import tempfile
from pathlib import Path

# ---- Synthetic test examples with ground truth ----

SYNTHETIC_TEST_SET = [
    {
        "pmid": "SYN001",
        "title": "Wonderdrug reduces major cardiac events by 47% in high-risk patients",
        "abstract": (
            "BACKGROUND: Statin-intolerant patients need alternatives. "
            "METHODS: Randomized, double-blind, placebo-controlled trial, N=4000. "
            "Primary endpoint: first MACE at 24 months. "
            "RESULTS: Wonderdrug significantly reduced MACE (HR 0.53, 95% CI 0.40-0.70, "
            "p<0.001), a 47% relative risk reduction. LDL-cholesterol fell by 55%. "
            "CONCLUSIONS: Wonderdrug provides substantial cardiovascular protection and "
            "should be considered as first-line therapy in statin-intolerant patients."
        ),
        "source": "high_suspicion",
        "statistical_reporting": {"severity": "high", "relative_only": True, "absolute_reported": False, "nnt_reported": False, "baseline_risk_reported": False},
        "spin": {"severity": "moderate", "spin_level": "moderate", "conclusion_matches_results": False, "focus_on_secondary_when_primary_ns": False},
        "outcome_reporting": {"severity": "low", "primary_outcome_type": "patient_centred", "surrogate_without_validation": False},
        "conflict_of_interest": {"severity": "high", "funding_type": "not_reported", "coi_disclosed": False, "industry_author_affiliations": False},
        "methodology": {"severity": "none", "inappropriate_comparator": False, "per_protocol_only": False},
        "overall_severity": "high",
        "overall_bias_probability": 0.80,
    },
    {
        "pmid": "SYN002",
        "title": "Effect of gooddrug on mortality in heart failure: the GOOD-HF trial",
        "abstract": (
            "BACKGROUND: We assessed gooddrug in systolic heart failure. "
            "METHODS: Multicentre RCT, N=6000, median follow-up 3.2 years. "
            "Funded by the National Heart Foundation. "
            "RESULTS: All-cause mortality occurred in 12.1% (gooddrug) vs 14.8% (placebo); "
            "HR 0.80, 95% CI 0.68-0.94, p=0.006. Absolute risk reduction 2.7% "
            "(95% CI 1.1-4.1%), NNT 37 over 3 years. "
            "CONCLUSIONS: Gooddrug modestly reduced mortality (NNT 37). The absolute benefit "
            "should be weighed against cost and adverse effects. Further long-term data needed."
        ),
        "source": "low_suspicion",
        "statistical_reporting": {"severity": "none", "relative_only": False, "absolute_reported": True, "nnt_reported": True, "baseline_risk_reported": True},
        "spin": {"severity": "none", "spin_level": "none", "conclusion_matches_results": True, "focus_on_secondary_when_primary_ns": False},
        "outcome_reporting": {"severity": "none", "primary_outcome_type": "patient_centred", "surrogate_without_validation": False},
        "conflict_of_interest": {"severity": "none", "funding_type": "public", "coi_disclosed": True, "industry_author_affiliations": False},
        "methodology": {"severity": "none", "inappropriate_comparator": False, "per_protocol_only": False},
        "overall_severity": "none",
        "overall_bias_probability": 0.10,
    },
    {
        "pmid": "SYN003",
        "title": "Newpill improves glycaemic control in type 2 diabetes",
        "abstract": (
            "BACKGROUND: We evaluated newpill for T2DM. "
            "METHODS: Phase III RCT, N=800, 52 weeks. Funded by NewPharma Inc. "
            "Primary endpoint: HbA1c change from baseline. "
            "RESULTS: Primary endpoint was not statistically significant (between-group "
            "difference -0.15%, p=0.12). However, in the pre-specified subgroup with "
            "HbA1c >9%, significant improvement was observed (-0.8%, p=0.01). "
            "Fasting glucose improved significantly from baseline in the treatment group. "
            "CONCLUSIONS: Newpill demonstrates promising efficacy, particularly in patients "
            "with poorly controlled diabetes, and represents an important therapeutic advance."
        ),
        "source": "high_suspicion",
        "statistical_reporting": {"severity": "moderate", "relative_only": False, "absolute_reported": False, "nnt_reported": False, "baseline_risk_reported": False},
        "spin": {"severity": "high", "spin_level": "high", "conclusion_matches_results": False, "focus_on_secondary_when_primary_ns": True},
        "outcome_reporting": {"severity": "moderate", "primary_outcome_type": "surrogate", "surrogate_without_validation": True},
        "conflict_of_interest": {"severity": "moderate", "funding_type": "industry", "coi_disclosed": True, "industry_author_affiliations": False},
        "methodology": {"severity": "none", "inappropriate_comparator": False, "per_protocol_only": False},
        "overall_severity": "high",
        "overall_bias_probability": 0.75,
    },
    {
        "pmid": "SYN004",
        "title": "Aspirin versus clopidogrel for secondary stroke prevention: a pragmatic trial",
        "abstract": (
            "BACKGROUND: Optimal antiplatelet therapy after stroke is uncertain. "
            "METHODS: Pragmatic open-label RCT, N=3500, funded by NIHR. "
            "Primary: recurrent stroke or TIA at 12 months. "
            "RESULTS: Recurrent events in 8.1% (aspirin) vs 7.2% (clopidogrel); "
            "HR 0.88, 95% CI 0.72-1.08, p=0.22. ARD 0.9%, NNT 111. "
            "CONCLUSIONS: No significant difference between aspirin and clopidogrel. "
            "Aspirin remains a reasonable first-line option given its lower cost. "
            "Larger trials may be warranted."
        ),
        "source": "low_suspicion",
        "statistical_reporting": {"severity": "none", "relative_only": False, "absolute_reported": True, "nnt_reported": True, "baseline_risk_reported": True},
        "spin": {"severity": "none", "spin_level": "none", "conclusion_matches_results": True, "focus_on_secondary_when_primary_ns": False},
        "outcome_reporting": {"severity": "none", "primary_outcome_type": "patient_centred", "surrogate_without_validation": False},
        "conflict_of_interest": {"severity": "none", "funding_type": "public", "coi_disclosed": True, "industry_author_affiliations": False},
        "methodology": {"severity": "low", "inappropriate_comparator": False, "per_protocol_only": False},
        "overall_severity": "low",
        "overall_bias_probability": 0.15,
    },
]


def _simulate_model_output_good(example: dict) -> str:
    """Simulate a model that mostly gets it right (for testing scoring)."""
    gt = example
    sr = gt.get("statistical_reporting", {})
    sp = gt.get("spin", {})
    coi = gt.get("conflict_of_interest", {})

    return json.dumps({
        "statistical_reporting": {
            "severity": sr.get("severity", "none"),
            "relative_only": sr.get("relative_only", False),
            "absolute_reported": sr.get("absolute_reported", False),
            "nnt_reported": sr.get("nnt_reported", False),
            "baseline_risk_reported": sr.get("baseline_risk_reported", False),
            "selective_p_values": False,
            "subgroup_emphasis": sp.get("focus_on_secondary_when_primary_ns", False),
        },
        "spin": {
            "severity": sp.get("severity", "none"),
            "spin_level": sp.get("spin_level", "none"),
            "conclusion_matches_results": sp.get("conclusion_matches_results", True),
            "focus_on_secondary_when_primary_ns": sp.get("focus_on_secondary_when_primary_ns", False),
            "causal_language_from_observational": False,
            "inappropriate_extrapolation": False,
            "title_spin": False,
        },
        "outcome_reporting": {
            "severity": gt.get("outcome_reporting", {}).get("severity", "none"),
            "primary_outcome_type": gt.get("outcome_reporting", {}).get("primary_outcome_type", "unclear"),
            "surrogate_without_validation": gt.get("outcome_reporting", {}).get("surrogate_without_validation", False),
            "composite_not_disaggregated": False,
        },
        "conflict_of_interest": {
            "severity": coi.get("severity", "none"),
            "funding_type": coi.get("funding_type", "not_reported"),
            "funding_disclosed_in_abstract": coi.get("coi_disclosed", False),
            "industry_author_affiliations": coi.get("industry_author_affiliations", False),
            "coi_disclosed": coi.get("coi_disclosed", False),
        },
        "methodology": {
            "severity": gt.get("methodology", {}).get("severity", "none"),
            "inappropriate_comparator": False,
            "per_protocol_only": False,
            "premature_stopping": False,
            "enrichment_design": False,
            "short_follow_up": False,
        },
        "overall_severity": gt.get("overall_severity", "none"),
        "overall_bias_probability": gt.get("overall_bias_probability", 0.0),
        "recommended_verification_steps": [
            "Check ClinicalTrials.gov for registered primary outcome",
            "Search CMS Open Payments for author payment history",
            "Verify author affiliations via ORCID",
        ],
        "confidence": "high",
    })


def _simulate_model_output_mediocre(example: dict) -> str:
    """Simulate a model that gets some things wrong (for testing differentiation)."""
    gt = example
    sr = gt.get("statistical_reporting", {})

    # This model misses relative-only reporting and underrates spin
    return json.dumps({
        "statistical_reporting": {
            "severity": "low" if sr.get("severity") in ("moderate", "high") else sr.get("severity", "none"),
            "relative_only": False,  # Always misses this
            "absolute_reported": True,
            "nnt_reported": False,
            "baseline_risk_reported": False,
            "selective_p_values": False,
            "subgroup_emphasis": False,
        },
        "spin": {
            "severity": "low" if gt.get("spin", {}).get("severity") == "high" else gt.get("spin", {}).get("severity", "none"),
            "spin_level": "low",
            "conclusion_matches_results": True,  # Misses spin
            "focus_on_secondary_when_primary_ns": False,
            "causal_language_from_observational": False,
            "inappropriate_extrapolation": False,
            "title_spin": False,
        },
        "outcome_reporting": {
            "severity": gt.get("outcome_reporting", {}).get("severity", "none"),
            "primary_outcome_type": gt.get("outcome_reporting", {}).get("primary_outcome_type", "unclear"),
            "surrogate_without_validation": gt.get("outcome_reporting", {}).get("surrogate_without_validation", False),
            "composite_not_disaggregated": False,
        },
        "conflict_of_interest": {
            "severity": gt.get("conflict_of_interest", {}).get("severity", "none"),
            "funding_type": gt.get("conflict_of_interest", {}).get("funding_type", "not_reported"),
            "funding_disclosed_in_abstract": True,
            "industry_author_affiliations": False,
            "coi_disclosed": True,
        },
        "methodology": {
            "severity": "none",
            "inappropriate_comparator": False,
            "per_protocol_only": False,
            "premature_stopping": False,
            "enrichment_design": False,
            "short_follow_up": False,
        },
        "overall_severity": "low" if gt.get("overall_severity") in ("moderate", "high") else gt.get("overall_severity", "none"),
        "overall_bias_probability": max(0, gt.get("overall_bias_probability", 0) - 0.3),
        "recommended_verification_steps": [
            "Check ClinicalTrials.gov for trial details",
        ],
        "confidence": "medium",
    })


def run_selftest():
    """Run the full pipeline self-test."""
    from .scorer import parse_model_output, attach_ground_truth
    from .metrics import evaluate_model
    from .comparison import generate_comparison, save_report

    print("=" * 70)
    print("EVALUATION PIPELINE SELF-TEST")
    print("=" * 70)

    # 1. Create synthetic test set
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        test_path = tmpdir / "test.jsonl"
        with open(test_path, "w") as f:
            for ex in SYNTHETIC_TEST_SET:
                f.write(json.dumps(ex) + "\n")
        print(f"\n[OK] Created {len(SYNTHETIC_TEST_SET)} synthetic test examples")

        # 2. Simulate model outputs
        model_a_id = "simulated-good-model"
        model_b_id = "simulated-mediocre-model"

        outputs_a = []
        outputs_b = []
        for ex in SYNTHETIC_TEST_SET:
            outputs_a.append({
                "pmid": ex["pmid"],
                "model_id": model_a_id,
                "raw_output": _simulate_model_output_good(ex),
                "latency_seconds": 2.5,
                "output_tokens": 500,
                "error": None,
            })
            outputs_b.append({
                "pmid": ex["pmid"],
                "model_id": model_b_id,
                "raw_output": _simulate_model_output_mediocre(ex),
                "latency_seconds": 3.8,
                "output_tokens": 350,
                "error": None,
            })
        print(f"[OK] Simulated outputs for 2 models")

        # 3. Parse and score
        all_assessments = {}
        all_evaluations = {}

        for model_id, outputs in [(model_a_id, outputs_a), (model_b_id, outputs_b)]:
            assessments = []
            for output, gt_example in zip(outputs, SYNTHETIC_TEST_SET):
                parsed = parse_model_output(
                    raw_output=output["raw_output"],
                    pmid=output["pmid"],
                    model_id=model_id,
                )
                parsed = attach_ground_truth(parsed, gt_example)
                assessments.append(parsed)

            evaluation = evaluate_model(assessments, model_id=model_id)
            all_assessments[model_id] = assessments
            all_evaluations[model_id] = evaluation

            print(f"\n[{model_id}]")
            print(f"  Overall F1:           {evaluation.overall_binary.f1:.3f}")
            print(f"  Overall κ:            {evaluation.overall_ordinal.weighted_kappa():.3f}")
            print(f"  Calibration Error:    {evaluation.calibration_error:.3f}")
            print(f"  Verification Score:   {evaluation.mean_verification_score:.3f}")
            print(f"  Parse failures:       {evaluation.n_parse_failures}")
            for dim in ["statistical_reporting", "spin", "outcome_reporting",
                         "conflict_of_interest", "methodology"]:
                dim_eval = getattr(evaluation, dim)
                print(f"  {dim:30s} F1={dim_eval.binary.f1:.3f}  κ={dim_eval.ordinal.weighted_kappa():.3f}")

        print(f"\n[OK] Scoring complete for both models")

        # 4. Generate comparison report
        report = generate_comparison(
            evaluations=all_evaluations,
            assessments=all_assessments,
            raw_outputs={model_a_id: outputs_a, model_b_id: outputs_b},
            mode="zero-shot",
        )

        output_dir = tmpdir / "results"
        save_report(report, output_dir, evaluations=all_evaluations)
        print(f"\n[OK] Comparison report generated")

        # 5. Print the summary
        print("\n" + "=" * 70)
        print("GENERATED REPORT:")
        print("=" * 70)
        print(report.summary)

        # 6. Validate expectations
        print("\n" + "=" * 70)
        print("VALIDATION CHECKS")
        print("=" * 70)

        eval_good = all_evaluations[model_a_id]
        eval_med = all_evaluations[model_b_id]

        checks = [
            ("Good model F1 > mediocre F1",
             eval_good.overall_binary.f1 >= eval_med.overall_binary.f1),
            ("Good model κ > mediocre κ",
             eval_good.overall_ordinal.weighted_kappa() >= eval_med.overall_ordinal.weighted_kappa()),
            ("Good model better verification score",
             eval_good.mean_verification_score >= eval_med.mean_verification_score),
            ("No parse failures",
             eval_good.n_parse_failures == 0 and eval_med.n_parse_failures == 0),
            ("Report has 2 models",
             len(report.models) == 2),
            ("Report has pairwise tests",
             len(report.pairwise_tests) > 0),
            ("Report has dimension winners",
             len(report.dimension_winners) > 0),
            ("Statistical reporting flag comparison exists",
             "statistical_reporting.relative_only" in report.flag_comparison),
        ]

        all_passed = True
        for name, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")
            if not passed:
                all_passed = False

        print("\n" + "=" * 70)
        if all_passed:
            print("ALL CHECKS PASSED - Evaluation pipeline is working correctly")
        else:
            print("SOME CHECKS FAILED - Review output above")
        print("=" * 70)

        return all_passed


if __name__ == "__main__":
    success = run_selftest()
    sys.exit(0 if success else 1)

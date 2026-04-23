# Annotation JSON Spec (v1)

**Status:** active
**Applies to:** `cochrane_rob2`, `quadas_2`
**Schema files:** [`schemas/rob2_annotation.schema.json`](../schemas/rob2_annotation.schema.json), [`schemas/quadas2_annotation.schema.json`](../schemas/quadas2_annotation.schema.json)
**Enforced at:** [`biasbuster/database.py`](../biasbuster/database.py) — `Database.insert_annotation()` validates every annotation against the matching schema before writing to SQLite.

---

## 1. Why this document exists

Every AI-applied RoB 2 / QUADAS-2 annotation we store is intended to be independently auditable. An auditor must be able to, given only the JSON record and the original paper:

- Reconstruct the model's per-domain reasoning (signalling answers + justification + verbatim evidence quotes),
- Check each judgement against the Cochrane / Whiting decision rules printed in the prompt, and
- Pinpoint disagreements with a human expert rating at the *per-domain* level (not just the overall).

The schemas codify the exact shape the biasbuster pipeline emits, so downstream tooling (the faithfulness harness, future calibration studies, supplementary materials for publications) can rely on stable field names and value vocabularies.

**Philosophy:** additional top-level keys prefixed with `_` (e.g. `_annotation_mode`, `_annotation_model`) are allowed and non-semantic — they're pipeline metadata. The *semantic* fields (below) are fixed.

---

## 2. Shared top-level fields

Both methodologies emit these identically. None are required by the JSON Schema — they are convenience mirrors for fields the DB row already carries (or trivially derivable from the methodology's worst-wins rollup). Production assessors set them all; tests and supplementary-material exporters MAY omit them.

| Field | Type | Required? | Meaning |
|---|---|---|---|
| `pmid` | string | optional | PubMed ID. Redundant with the `annotations.pmid` PK column. |
| `title` | string | optional | Paper title at assessment time. |
| `abstract_text` | string | optional | Abstract text used by the assessor (empty if full text was sufficient). |
| `source` | string | optional | Always `"decomposed_full_text"` for the current orchestration. |
| `methodology_version` | string | optional | Tool-version tag (e.g. `"rob2-2019"`, `"quadas2-2011"`). Redundant with the `annotations.methodology_version` column. |
| `overall_severity` | enum | optional | Mirror of the methodology's worst-wins rollup (`worst_across_outcomes` for RoB 2, `worst_bias` for QUADAS-2). |
| `notes` | string\|null | optional | Free-form notes, typically empty. |

Internal metadata fields (optional, prefixed `_`):

| Field | Meaning |
|---|---|
| `_annotation_mode` | Orchestration mode tag (e.g. `"decomposed_rob2"`, `"decomposed_quadas2"`). |
| `_annotation_model` | Backend model identifier (e.g. `"anthropic/claude-sonnet-4-6"`). |
| `_methodology` | Methodology slug (redundant with the DB column but stored for portability). |
| `_methodology_version` | Redundant with top-level `methodology_version`. |

---

## 3. RoB 2 (`cochrane_rob2`) shape

### 3.1 Top-level (RoB 2-specific)

| Field | Type | Required? | Meaning |
|---|---|---|---|
| `outcomes` | array (≥1 item) | **required** | Per-outcome assessments. Most trials report ≥1 outcome; the decomposed assessor currently emits exactly one with `outcome_label="primary outcome"`. |
| `worst_across_outcomes` | enum | **required** | Worst per-outcome overall judgement. For single-outcome records this equals `outcomes[0].overall_judgement`. Values: `low`, `some_concerns`, `high`. |
| `overall_severity` | enum | optional | Mirror of `worst_across_outcomes` (column compatibility with the `annotations.overall_severity` DB column). |

### 3.2 Per-outcome (`outcomes[i]`)

| Field | Type | Meaning |
|---|---|---|
| `outcome_label` | string | Human-readable outcome name. |
| `result_label` | string\|null | Specific estimand / analysis arm if applicable. |
| `overall_judgement` | enum | Worst-wins across the five domains. `low`, `some_concerns`, `high`. |
| `overall_rationale` | string | 2-3 sentence synthesis naming the domain(s) driving the overall. |
| `domains` | object | Keyed by the 5 canonical domain slugs (below). |

### 3.3 Per-domain (`domains[slug]`)

Slugs: `randomization`, `deviations_from_interventions`, `missing_outcome_data`, `outcome_measurement`, `selection_of_reported_result`.

| Field | Type | Meaning |
|---|---|---|
| `domain` | string | Echo of the slug. |
| `signalling_answers` | object | `{question_id: answer}` with `answer ∈ {Y, PY, PN, N, NI}`. Question IDs follow Cochrane's numbering (e.g. `"1.1"`, `"2.6"`). |
| `judgement` | enum | `low`, `some_concerns`, `high`. |
| `justification` | string | 1-3 sentence rationale. |
| `evidence_quotes` | array | Zero or more `{text, section?}` quotes from the paper. `section` is null-able. |

**Cochrane decision rules** (printed in each domain prompt, authoritative source: Sterne et al. 2019):

- D1 randomization: `low` iff Q1.1 Y/PY AND Q1.2 Y/PY AND Q1.3 N/PN; `high` iff Q1.1 N/PN OR Q1.2 N/PN OR Q1.3 Y/PY; `some_concerns` otherwise.
- D2 deviations: `low` iff 2.1/2.2 both N/PN AND 2.6 Y/PY; `high` iff 2.5 N/PN OR 2.7 Y/PY; `some_concerns` otherwise.
- D3 missing outcome: `low` iff 3.1 Y/PY OR (3.2 Y/PY AND 3.3/3.4 both N/PN); `high` iff 3.3 Y/PY AND 3.4 Y/PY; `some_concerns` otherwise.
- D4 outcome measurement: `low` iff 4.1 N/PN AND 4.2 N/PN AND 4.3 Y/PY; `high` iff 4.4 Y/PY OR 4.5 Y/PY; `some_concerns` otherwise.
- D5 reporting: `low` iff 5.1 Y/PY AND 5.2/5.3 both N/PN; `high` iff 5.2 Y/PY OR 5.3 Y/PY; `some_concerns` otherwise.

### 3.4 Example (excerpt — PMID 36101416, Marzouki 2022, D2 only)

```json
{
  "pmid": "36101416",
  "methodology_version": "rob2-2019",
  "outcomes": [
    {
      "outcome_label": "primary outcome",
      "overall_judgement": "some_concerns",
      "domains": {
        "deviations_from_interventions": {
          "domain": "deviations_from_interventions",
          "signalling_answers": {
            "2.1": "Y",
            "2.2": "Y",
            "2.3": "PN",
            "2.4": "NI",
            "2.5": "NI",
            "2.6": "PN",
            "2.7": "PN"
          },
          "judgement": "some_concerns",
          "justification": "Participants and those delivering the intervention were aware of group assignment (open-label exercise trial), but there is no indication of meaningful deviations from the intended plyometric protocols arising from the trial context. However, 17 participants were excluded from analysis due to low attendance, meaning the analysis was not a true intention-to-treat analysis; the potential impact is likely modest given the relatively small proportion excluded and the large effect sizes observed.",
          "evidence_quotes": [
            {
              "section": "Methods/Results",
              "text": "Due to low PT attendance, data for 17 children (11 girls and 6 boys) were excluded from further analysis."
            }
          ]
        }
      }
    }
  ],
  "worst_across_outcomes": "some_concerns",
  "overall_severity": "some_concerns"
}
```

---

## 4. QUADAS-2 (`quadas_2`) shape

QUADAS-2 is primary-study diagnostic-accuracy only (not review-level) and carries two rating dimensions per domain (bias + applicability) except for Flow and Timing (bias only, per Whiting 2011).

### 4.1 Top-level (QUADAS-2-specific)

| Field | Type | Required? | Meaning |
|---|---|---|---|
| `domains` | object (≥1 entry) | **required** | Keyed by the 4 canonical domain slugs (below). No outcome decomposition — QUADAS-2 is per-study. |
| `worst_bias` | enum | **required** | Worst per-domain `bias_rating` across the 4 domains. Values: `low`, `unclear`, `high`. |
| `worst_applicability` | enum | **required** | Worst per-domain `applicability` across the 3 domains that carry it. Same value vocabulary. |
| `overall_applicability` | enum | optional | Mirror of `worst_applicability`. |
| `overall_severity` | enum | optional | Mirror of `worst_bias`. |

### 4.2 Per-domain (`domains[slug]`)

Slugs: `patient_selection`, `index_test`, `reference_standard`, `flow_and_timing`.

| Field | Type | Meaning |
|---|---|---|
| `domain` | string | Echo of the slug. |
| `signalling_answers` | object | `{question_id: answer}` with `answer ∈ {yes, no, unclear}`. Question IDs follow Whiting's numbering. |
| `bias_rating` | enum | `low`, `unclear`, `high`. |
| `applicability` | enum\|null | `low`, `unclear`, `high`; `null` when `domain="flow_and_timing"`. |
| `justification` | string | 1-3 sentence rationale. |
| `evidence_quotes` | array | Zero or more `{text, section?}` quotes. |

---

## 5. Worked audit example — how to disagree with an expert in 60 seconds

Below is the **actual** JSON excerpt for PMID 34059568 (Radwan 2021, "Effect of adding plyometric training to physical education sessions") as assessed against Cochrane RoB 2 by our pipeline, alongside the Deng 2024 review's expert rating for the same paper + same domain. This is the audit workflow we propose for publication-grade use.

### 5.1 What the model wrote for D2 (deviations from interventions)

```json
{
  "domain": "deviations_from_interventions",
  "signalling_answers": {
    "2.1": "Y", "2.2": "Y",
    "2.3": "PN", "2.4": "NI", "2.5": "NI",
    "2.6": "PN", "2.7": "PN"
  },
  "judgement": "some_concerns",
  "justification": "Participants and those delivering interventions were clearly aware of group assignment (no blinding), but no notable deviations from the intended intervention were reported. However, the analysis population is not explicitly stated as ITT, and the statistical approach (repeated-measures ANOVA with paired t-tests) does not clearly conform to a pre-specified ITT framework, introducing some concerns."
}
```

### 5.2 What the expert recorded (Deng 2024, Figure 2)

```
D2 deviations_from_interventions: low
```

### 5.3 The 60-second audit

Step through the Cochrane RoB 2 D2 decision rule (§3.3):

> **`low`** iff 2.1/2.2 both **N/PN** AND 2.6 **Y/PY**.

Reading the signalling answers:

- 2.1 = **Y** (not N/PN) ✗
- 2.2 = **Y** (not N/PN) ✗
- 2.6 = **PN** (not Y/PY) ✗

**Every one of the three preconditions for a `low` D2 rating fails.** The expert's `low` cannot be reconciled with their own tool's decision rule given the paper's content.

Reading the justification confirms the facts underlying the signalling answers:
- "Participants and those delivering interventions were clearly aware of group assignment (no blinding)" → 2.1=Y, 2.2=Y: correct.
- "the analysis population is not explicitly stated as ITT" → 2.6=PN: correct.

Reading the paper itself (confirmed independently in our run) confirms neither blinding nor ITT were reported.

**Conclusion of this audit:** The model's `some_concerns` is the algorithmic answer. The expert's `low` is algorithmically impossible given the facts. This is the pattern the spec is designed to make checkable — give an auditor 60 seconds per disagreement and a stable JSON shape, and they can decide for themselves.

### 5.4 What to look at, in order

1. **The signalling_answers block** (lowest-cost fact-check). Match each question ID to the decision rule in §3.3 / §4.
2. **The judgement** (did the model apply the rule correctly from its own answers?).
3. **The justification** (do the answers match the rationale the model wrote?).
4. **The evidence_quotes** (do the rationale claims trace to verbatim text in the paper?).
5. Only then, **the paper itself** (do the quotes accurately reflect the source?).

If any step fails, the disagreement is the model's. If all four hold, the disagreement is the expert's.

---

## 6. DB-level validation

`Database.insert_annotation()` calls `jsonschema.validate()` against the matching methodology schema before every INSERT. Invalid annotations raise `jsonschema.ValidationError` and never land in the `annotations` table.

Schema selection is driven by the `methodology` column:

| methodology | schema file |
|---|---|
| `cochrane_rob2` | `schemas/rob2_annotation.schema.json` |
| `quadas_2` | `schemas/quadas2_annotation.schema.json` |
| anything else | no validation (current behaviour; legacy `biasbuster` and other methodologies are not yet covered) |

Validation errors are logged at ERROR level and re-raised; there is no silent-skip path. This is deliberate — a malformed annotation in the DB is worse than a failed insert that the caller can retry.

---

## 7. Change log

| Version | Date | Change |
|---|---|---|
| v1 | 2026-04-23 | Initial spec. Covers `cochrane_rob2` and `quadas_2`. |

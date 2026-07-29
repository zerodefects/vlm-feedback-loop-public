# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Canonical Exact Match Evaluator.

Pure-logic tests: no database, no async, no mocking.  The evaluator is
a standalone reusable module consumed by proposal validation, evaluation
scoring, Batch Labeling, and TAO re-scoring.
"""

from __future__ import annotations

from vlm_feedback_loop.services.exact_match_evaluator import (
    FieldMatchResult,
    compute_aggregate_metrics,
    match_fields,
    normalize_field_value,
    normalize_ground_truth,
    validate_proposal,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_field(name: str, ftype: str, role: str = "core", **kwargs) -> dict:
    """Build a minimal SchemaCore field definition."""
    d: dict = {
        "field_id": f"fid_{name}",
        "field_name": name,
        "type": ftype,
        "role": role,
        "display_order": kwargs.pop("display_order", 0),
    }
    d.update(kwargs)
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Section A: Per-type normalization
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalizeEnum:
    """AC: Enum normalization — trim/lowercase, match against allowed_values."""

    def test_exact_match_returns_canonical_case(self):
        f = _make_field("damage", "enum", allowed_values=["Crack", "Dent"])
        r = normalize_field_value("Crack", f)
        assert r.valid is True
        assert r.normalized_value == "Crack"

    def test_case_insensitive_match(self):
        f = _make_field("damage", "enum", allowed_values=["Crack", "Dent"])
        r = normalize_field_value("crack", f)
        assert r.valid is True
        assert r.normalized_value == "Crack"  # canonical case returned

    def test_whitespace_trimmed(self):
        f = _make_field("damage", "enum", allowed_values=["Crack", "Dent"])
        r = normalize_field_value("  Crack  ", f)
        assert r.valid is True
        assert r.normalized_value == "Crack"

    def test_invalid_value_returns_error(self):
        f = _make_field("damage", "enum", allowed_values=["Crack", "Dent"])
        r = normalize_field_value("Scratch", f)
        assert r.valid is False
        assert r.error is not None
        assert "Scratch" in r.error

    def test_non_string_input_returns_error(self):
        f = _make_field("damage", "enum", allowed_values=["Crack", "Dent"])
        r = normalize_field_value(42, f)
        assert r.valid is False
        assert "int" in r.error

    def test_none_input_returns_error(self):
        f = _make_field("damage", "enum", allowed_values=["Crack", "Dent"])
        r = normalize_field_value(None, f)
        assert r.valid is False
        assert "missing" in r.error.lower() or "null" in r.error.lower()


class TestNormalizeEnumSet:
    """AC: Enum set normalization — normalize, dedupe, sort, compare as sets."""

    def test_valid_set_normalized_and_sorted(self):
        f = _make_field("types", "enum_set", allowed_values=["Crack", "Dent", "Leak"])
        r = normalize_field_value(["Dent", "Crack"], f)
        assert r.valid is True
        assert r.normalized_value == ["Crack", "Dent"]  # sorted

    def test_duplicates_removed(self):
        f = _make_field("types", "enum_set", allowed_values=["Crack", "Dent"])
        r = normalize_field_value(["Crack", "crack", "CRACK"], f)
        assert r.valid is True
        assert r.normalized_value == ["Crack"]  # deduped to one canonical

    def test_case_insensitive_elements(self):
        f = _make_field("types", "enum_set", allowed_values=["Crack", "Dent"])
        r = normalize_field_value(["DENT", "crack"], f)
        assert r.valid is True
        assert set(r.normalized_value) == {"Crack", "Dent"}

    def test_non_list_returns_error(self):
        f = _make_field("types", "enum_set", allowed_values=["Crack"])
        r = normalize_field_value("Crack", f)
        assert r.valid is False
        assert "array" in r.error.lower() or "list" in r.error.lower()

    def test_invalid_element_returns_error(self):
        f = _make_field("types", "enum_set", allowed_values=["Crack", "Dent"])
        r = normalize_field_value(["Crack", "Scratch"], f)
        assert r.valid is False
        assert "Scratch" in r.error

    def test_empty_list_valid(self):
        f = _make_field("types", "enum_set", allowed_values=["Crack", "Dent"])
        r = normalize_field_value([], f)
        assert r.valid is True
        assert r.normalized_value == []

    def test_none_returns_error(self):
        f = _make_field("types", "enum_set", allowed_values=["Crack"])
        r = normalize_field_value(None, f)
        assert r.valid is False


class TestNormalizeBoolean:
    """AC: Boolean normalization — ONLY JSON true/false. String and numeric proxies are INVALID."""

    def test_true_accepted(self):
        f = _make_field("damaged", "boolean")
        r = normalize_field_value(True, f)
        assert r.valid is True
        assert r.normalized_value is True

    def test_false_accepted(self):
        f = _make_field("damaged", "boolean")
        r = normalize_field_value(False, f)
        assert r.valid is True
        assert r.normalized_value is False

    def test_string_true_rejected(self):
        """CRITICAL: String 'true' is NOT a valid boolean."""
        f = _make_field("damaged", "boolean")
        r = normalize_field_value("true", f)
        assert r.valid is False
        assert "string" in r.error.lower()

    def test_string_false_rejected(self):
        f = _make_field("damaged", "boolean")
        r = normalize_field_value("false", f)
        assert r.valid is False

    def test_yes_no_rejected(self):
        f = _make_field("damaged", "boolean")
        for val in ("yes", "no", "Yes", "No"):
            r = normalize_field_value(val, f)
            assert r.valid is False, f"'{val}' should be rejected"

    def test_int_1_rejected(self):
        """CRITICAL: Numeric proxy 1 is NOT a valid boolean."""
        f = _make_field("damaged", "boolean")
        r = normalize_field_value(1, f)
        assert r.valid is False
        assert "numeric" in r.error.lower()

    def test_int_0_rejected(self):
        f = _make_field("damaged", "boolean")
        r = normalize_field_value(0, f)
        assert r.valid is False

    def test_none_rejected(self):
        f = _make_field("damaged", "boolean")
        r = normalize_field_value(None, f)
        assert r.valid is False


class TestNormalizeInteger:
    """AC: Integer normalization — type check, float conversion, range validation."""

    def test_valid_in_range(self):
        f = _make_field("severity", "integer", minimum=0, maximum=4)
        r = normalize_field_value(2, f)
        assert r.valid is True
        assert r.normalized_value == 2

    def test_below_minimum_invalid(self):
        f = _make_field("severity", "integer", minimum=0, maximum=4)
        r = normalize_field_value(-1, f)
        assert r.valid is False
        assert "below" in r.error.lower() or "minimum" in r.error.lower()

    def test_above_maximum_invalid(self):
        f = _make_field("severity", "integer", minimum=0, maximum=4)
        r = normalize_field_value(5, f)
        assert r.valid is False
        assert "above" in r.error.lower() or "maximum" in r.error.lower()

    def test_float_whole_number_accepted(self):
        f = _make_field("severity", "integer", minimum=0, maximum=4)
        r = normalize_field_value(3.0, f)
        assert r.valid is True
        assert r.normalized_value == 3
        assert isinstance(r.normalized_value, int)

    def test_float_fractional_rejected(self):
        f = _make_field("severity", "integer")
        r = normalize_field_value(3.5, f)
        assert r.valid is False

    def test_string_rejected(self):
        f = _make_field("severity", "integer")
        r = normalize_field_value("3", f)
        assert r.valid is False

    def test_no_bounds_any_int_valid(self):
        f = _make_field("count", "integer")
        r = normalize_field_value(999999, f)
        assert r.valid is True
        assert r.normalized_value == 999999

    def test_boolean_rejected_as_integer(self):
        """Python bool is subclass of int — must be rejected."""
        f = _make_field("severity", "integer")
        r = normalize_field_value(True, f)
        assert r.valid is False


class TestNormalizeString:
    """AC: String normalization — trim, validate length constraints."""

    def test_trimmed(self):
        f = _make_field("notes", "string", role="aux")
        r = normalize_field_value("  hello world  ", f)
        assert r.valid is True
        assert r.normalized_value == "hello world"

    def test_below_min_length_invalid(self):
        f = _make_field("notes", "string", role="aux", min_length=5)
        r = normalize_field_value("hi", f)
        assert r.valid is False

    def test_above_max_length_invalid(self):
        f = _make_field("notes", "string", role="aux", max_length=10)
        r = normalize_field_value("a" * 20, f)
        assert r.valid is False

    def test_non_string_rejected(self):
        f = _make_field("notes", "string", role="aux")
        r = normalize_field_value(42, f)
        assert r.valid is False

    def test_none_rejected(self):
        f = _make_field("notes", "string", role="core")
        r = normalize_field_value(None, f)
        assert r.valid is False


# ══════════════════════════════════════════════════════════════════════════════
# Section B: Proposal validation
# ══════════════════════════════════════════════════════════════════════════════

FIXTURE_FIELDS = [
    _make_field("rationale_note", "string", role="aux", display_order=-1),
    _make_field(
        "severity",
        "enum",
        role="core",
        allowed_values=["low", "medium", "high"],
        display_order=1,
    ),
    _make_field("damaged", "boolean", role="core", display_order=2),
]


class TestValidateProposal:
    """AC: Full proposal validation separating Core vs Aux errors."""

    def test_valid_all_fields(self):
        content = (
            '{"rationale_note": "dent visible", "severity": "high", "damaged": true}'
        )
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is True
        assert report.core_errors == []
        assert report.normalized_json is not None
        assert report.normalized_json["severity"] == "high"
        assert report.normalized_json["damaged"] is True

    def test_invalid_json_parse_error(self):
        report = validate_proposal("{invalid json", FIXTURE_FIELDS)
        assert report.schema_valid_core is False
        assert report.parse_error is not None
        assert len(report.core_errors) > 0

    def test_none_content_parse_error(self):
        report = validate_proposal(None, FIXTURE_FIELDS)
        assert report.schema_valid_core is False
        assert report.parse_error is not None

    def test_missing_core_field_is_core_error(self):
        content = '{"rationale_note": "ok", "damaged": true}'  # missing severity
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is False
        assert any("severity" in e for e in report.core_errors)

    def test_missing_aux_field_is_aux_warning(self):
        content = (
            '{"severity": "high", "damaged": true}'  # missing rationale_note (aux)
        )
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is True  # Core is still valid
        assert any("rationale_note" in e for e in report.aux_errors)

    def test_extra_field_is_warning(self):
        content = (
            '{"rationale_note": "ok", "severity": "high", "damaged": true, "extra": 1}'
        )
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is True
        assert any("extra" in e.lower() for e in report.aux_errors)

    def test_core_valid_when_all_pass(self):
        content = '{"rationale_note": "ok", "severity": "HIGH", "damaged": false}'
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is True

    def test_core_invalid_when_any_core_fails(self):
        content = '{"rationale_note": "ok", "severity": "invalid_val", "damaged": true}'
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is False

    def test_truncation_attributed_when_length_and_invalid(self):
        content = '{"rationale_note": "ok", "severity": "hi'  # truncated JSON
        report = validate_proposal(content, FIXTURE_FIELDS, finish_reason="length")
        assert report.schema_valid_core is False
        assert report.truncation_attributed_schema_invalid is True

    def test_truncation_not_attributed_when_stop_and_invalid(self):
        content = '{"severity": "invalid_val", "damaged": true}'
        report = validate_proposal(content, FIXTURE_FIELDS, finish_reason="stop")
        assert report.truncation_attributed_schema_invalid is False

    def test_mixed_core_aux_errors(self):
        # Core field invalid (boolean as string) + aux field missing
        content = '{"severity": "low", "damaged": "true"}'
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is False  # damaged is a string proxy
        assert len(report.core_errors) > 0
        assert len(report.aux_errors) > 0  # rationale_note missing

    def test_normalized_json_built_from_valid_fields(self):
        content = '{"rationale_note": "  test  ", "severity": "LOW", "damaged": true}'
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.normalized_json is not None
        assert report.normalized_json["severity"] == "low"  # canonical lowercase
        assert report.normalized_json["rationale_note"] == "test"  # trimmed

    def test_non_object_json_is_parse_error(self):
        report = validate_proposal("[1, 2, 3]", FIXTURE_FIELDS)
        assert report.schema_valid_core is False
        assert report.parse_error is not None
        assert "object" in report.parse_error.lower()

    def test_empty_string_content(self):
        report = validate_proposal("", FIXTURE_FIELDS)
        assert report.schema_valid_core is False
        assert report.parse_error is not None

    # ── Markdown code-fence handling ─────────────────────────────────────────
    #
    # Prompt-only mode exposes a common LLM failure: the model wraps its
    # JSON output in a ```json … ``` fenced block. The parser strips a
    # well-formed fence before json.loads so valid-JSON-inside-fence
    # succeeds; malformed fences pass through and surface as parse errors.

    def test_fenced_json_with_language_tag(self):
        content = (
            "```json\n"
            '{"rationale_note": "dent visible", "severity": "high", "damaged": true}\n'
            "```"
        )
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is True
        assert report.parse_error is None
        assert report.normalized_json["severity"] == "high"

    def test_fenced_json_without_language_tag(self):
        content = (
            '```\n{"rationale_note": "ok", "severity": "low", "damaged": false}\n```'
        )
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is True
        assert report.parse_error is None

    def test_fenced_json_with_trailing_whitespace(self):
        content = (
            "```json\n"
            '{"rationale_note": "ok", "severity": "medium", "damaged": true}\n'
            "```   \n\n"
        )
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is True

    def test_fenced_missing_close_is_parse_error(self):
        # Opening fence but no closing ``` — pass through unchanged so
        # the malformed response surfaces as a parse error.
        content = (
            '```json\n{"rationale_note": "ok", "severity": "medium", "damaged": true}'
        )
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is False
        assert report.parse_error is not None

    def test_plain_json_unaffected_by_fence_stripper(self):
        # Regression: normal JSON must not be perturbed by the fence logic.
        content = '{"rationale_note": "ok", "severity": "high", "damaged": true}'
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is True
        assert report.normalized_json["damaged"] is True

    def test_fenced_output_with_wrong_field_names_is_schema_error(self):
        # Real Teacher failure mode: the model wraps output in a fence
        # AND invents field names (`core_label` instead of the schema's
        # `severity`). Fence stripping gets the parser past the syntax
        # barrier so the validator reports a proper schema error for the
        # missing Core field instead of an opaque "JSON parse failure".
        content = (
            "```json\n"
            '{"rationale_note": "ok", "core_label": "high", "damaged": true}\n'
            "```"
        )
        report = validate_proposal(content, FIXTURE_FIELDS)
        assert report.schema_valid_core is False
        assert report.parse_error is None  # parse succeeded
        assert any("severity" in e for e in report.core_errors)


# ══════════════════════════════════════════════════════════════════════════════
# Section C: Field matching
# ══════════════════════════════════════════════════════════════════════════════


class TestMatchFields:
    """AC: Per-field comparison between predicted and ground-truth values."""

    def test_enum_match(self):
        fields = [_make_field("severity", "enum", allowed_values=["low", "high"])]
        results = match_fields({"severity": "high"}, {"severity": "high"}, fields)
        assert len(results) == 1
        assert results[0].matched is True

    def test_enum_mismatch(self):
        fields = [_make_field("severity", "enum", allowed_values=["low", "high"])]
        results = match_fields({"severity": "low"}, {"severity": "high"}, fields)
        assert results[0].matched is False

    def test_enum_set_order_independent(self):
        fields = [_make_field("types", "enum_set", allowed_values=["a", "b", "c"])]
        results = match_fields(
            {"types": ["b", "a"]},
            {"types": ["a", "b"]},
            fields,
        )
        assert results[0].matched is True

    def test_enum_set_missing_prediction_is_miss_against_empty_truth(self):
        """A schema-invalid response (scored as predicted={}) must be a miss
        on every core field — including an enum_set whose correct answer is
        the empty set, common in defect/anomaly schemas."""
        fields = [_make_field("types", "enum_set", allowed_values=["a", "b"])]
        results = match_fields({}, {"types": []}, fields)
        assert results[0].matched is False

    def test_enum_set_empty_prediction_matches_empty_truth(self):
        """A schema-valid explicit empty set is a legitimate exact match
        against an empty ground-truth set."""
        fields = [_make_field("types", "enum_set", allowed_values=["a", "b"])]
        results = match_fields({"types": []}, {"types": []}, fields)
        assert results[0].matched is True

    def test_boolean_match(self):
        fields = [_make_field("damaged", "boolean")]
        results = match_fields({"damaged": True}, {"damaged": True}, fields)
        assert results[0].matched is True

    def test_integer_match(self):
        fields = [_make_field("severity", "integer")]
        results = match_fields({"severity": 3}, {"severity": 3}, fields)
        assert results[0].matched is True

    def test_string_match(self):
        fields = [_make_field("notes", "string")]
        results = match_fields({"notes": "hello"}, {"notes": "hello"}, fields)
        assert results[0].matched is True

    def test_aux_fields_excluded(self):
        fields = [
            _make_field(
                "severity", "enum", role="core", allowed_values=["low", "high"]
            ),
            _make_field("rationale_note", "string", role="aux"),
        ]
        results = match_fields(
            {"severity": "high", "rationale_note": "a"},
            {"severity": "high", "rationale_note": "b"},
            fields,
        )
        assert len(results) == 1  # only core field
        assert results[0].field_name == "severity"


# ══════════════════════════════════════════════════════════════════════════════
# Section D: Aggregate metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestAggregateMetrics:
    """AC: Aggregate accuracy metrics across a set of examples."""

    def _core_fields(self):
        return [
            _make_field(
                "severity",
                "enum",
                role="core",
                allowed_values=["low", "medium", "high"],
            ),
            _make_field("damaged", "boolean", role="core"),
        ]

    def test_all_pass(self):
        results = [
            [
                FieldMatchResult("severity", True, "high", "high"),
                FieldMatchResult("damaged", True, True, True),
            ],
            [
                FieldMatchResult("severity", True, "low", "low"),
                FieldMatchResult("damaged", True, False, False),
            ],
        ]
        metrics = compute_aggregate_metrics(results, self._core_fields())
        assert metrics.overall_exact_match_rate == 1.0
        assert metrics.example_count == 2

    def test_some_fail(self):
        results = [
            [
                FieldMatchResult("severity", True, "high", "high"),
                FieldMatchResult("damaged", True, True, True),
            ],
            [
                FieldMatchResult("severity", False, "low", "high"),
                FieldMatchResult("damaged", True, False, False),
            ],
        ]
        metrics = compute_aggregate_metrics(results, self._core_fields())
        assert metrics.overall_exact_match_rate == 0.5

    def test_per_field_rate(self):
        results = [
            [
                FieldMatchResult("severity", True, "high", "high"),
                FieldMatchResult("damaged", False, True, False),
            ],
            [
                FieldMatchResult("severity", True, "low", "low"),
                FieldMatchResult("damaged", True, True, True),
            ],
        ]
        metrics = compute_aggregate_metrics(results, self._core_fields())
        assert metrics.per_core_field_match_rate["severity"] == 1.0
        assert metrics.per_core_field_match_rate["damaged"] == 0.5

    def test_per_value_f1_enum(self):
        fields = [
            _make_field("severity", "enum", role="core", allowed_values=["low", "high"])
        ]
        results = [
            [FieldMatchResult("severity", True, "high", "high")],
            [FieldMatchResult("severity", False, "high", "low")],
            [FieldMatchResult("severity", True, "low", "low")],
        ]
        metrics = compute_aggregate_metrics(results, fields)
        # 'high': predicted 2 times, truth 1 time. TP=1, FP=1, FN=0
        # precision=1/2=0.5, recall=1/1=1.0, F1=2*0.5*1/(0.5+1)=0.667
        high_m = metrics.per_value_metrics["severity"]["high"]
        assert abs(high_m.precision - 0.5) < 0.01
        assert abs(high_m.recall - 1.0) < 0.01

    def test_per_value_f1_boolean(self):
        fields = [_make_field("damaged", "boolean", role="core")]
        results = [
            [FieldMatchResult("damaged", True, True, True)],
            [FieldMatchResult("damaged", True, False, False)],
        ]
        metrics = compute_aggregate_metrics(results, fields)
        assert "damaged" in metrics.per_value_metrics
        assert "true" in metrics.per_value_metrics["damaged"]
        assert "false" in metrics.per_value_metrics["damaged"]
        # Both correct: precision=1, recall=1, F1=1
        assert metrics.per_value_metrics["damaged"]["true"].f1 == 1.0
        assert metrics.per_value_metrics["damaged"]["false"].f1 == 1.0

    def test_macro_f1_exposes_minority_collapse(self):
        # Always-predict-majority on an 80/20 split: Exact Match looks healthy
        # (0.8) but the minority class has 0 recall. Macro-F1 must drop well
        # below Exact Match — this is the metric the ICL diagnostic relies on.
        fields = [
            _make_field(
                "material", "enum", role="core", allowed_values=["majority", "minority"]
            )
        ]
        results = [[FieldMatchResult("material", True, "majority", "majority")]] * 8 + [
            [FieldMatchResult("material", False, "majority", "minority")]
        ] * 2
        metrics = compute_aggregate_metrics(results, fields)
        assert metrics.overall_exact_match_rate == 0.8
        # majority: P=8/10, R=8/8 -> F1=0.8889 ; minority: P=R=F1=0.0
        assert abs(metrics.per_value_metrics["material"]["minority"].f1) < 1e-9
        assert abs(metrics.per_field_macro_f1["material"] - 0.4444) < 0.001
        assert abs(metrics.overall_macro_f1 - 0.4444) < 0.001
        # The whole point: macro-F1 is far below Exact Match under collapse.
        assert metrics.overall_macro_f1 < metrics.overall_exact_match_rate

    def test_macro_f1_averages_across_fields(self):
        # Field A perfect (macro-F1 1.0), field B collapsed (macro-F1 ~0.4444);
        # overall macro-F1 is their unweighted mean.
        fields = [
            _make_field("a", "enum", role="core", allowed_values=["x", "y"]),
            _make_field(
                "b", "enum", role="core", allowed_values=["majority", "minority"]
            ),
        ]
        results = (
            [
                [
                    FieldMatchResult("a", True, "x", "x"),
                    FieldMatchResult("b", True, "majority", "majority"),
                ]
            ]
            * 4
            + [
                [
                    FieldMatchResult("a", True, "y", "y"),
                    FieldMatchResult("b", True, "majority", "majority"),
                ]
            ]
            * 4
            + [
                [
                    FieldMatchResult("a", True, "x", "x"),
                    FieldMatchResult("b", False, "majority", "minority"),
                ]
            ]
            * 2
        )
        metrics = compute_aggregate_metrics(results, fields)
        assert abs(metrics.per_field_macro_f1["a"] - 1.0) < 1e-9
        assert abs(metrics.per_field_macro_f1["b"] - 0.4444) < 0.001
        assert abs(metrics.overall_macro_f1 - (1.0 + 0.4444) / 2) < 0.001

    def test_macro_f1_empty_is_zero(self):
        metrics = compute_aggregate_metrics([], self._core_fields())
        assert metrics.overall_macro_f1 == 0.0
        assert metrics.per_field_macro_f1 == {}

    def test_per_value_f1_enum_set(self):
        fields = [
            _make_field("types", "enum_set", role="core", allowed_values=["a", "b"])
        ]
        results = [
            [FieldMatchResult("types", True, ["a", "b"], ["a", "b"])],
            [FieldMatchResult("types", False, ["a"], ["a", "b"])],  # missed 'b'
        ]
        metrics = compute_aggregate_metrics(results, fields)
        assert "types" in metrics.per_value_metrics
        # 'a': TP=2, FP=0, FN=0 -> perfect
        a_m = metrics.per_value_metrics["types"]["a"]
        assert a_m.f1 == 1.0
        # 'b': predicted 1 time, truth 2 times. TP=1, FP=0, FN=1
        b_m = metrics.per_value_metrics["types"]["b"]
        assert abs(b_m.recall - 0.5) < 0.01

    def test_empty_results(self):
        metrics = compute_aggregate_metrics([], [])
        assert metrics.overall_exact_match_rate == 0.0
        assert metrics.example_count == 0


# ══════════════════════════════════════════════════════════════════════════════
# Section E: Evaluator independence from proposal endpoint
# ══════════════════════════════════════════════════════════════════════════════


class TestEvaluatorIndependence:
    """AC: Evaluator invocable independently of the proposal endpoint."""

    def test_per_example_matching_standalone(self):
        """Given a SchemaCore, predicted label, and ground-truth label,
        the module returns per-field match booleans."""
        fields = [
            _make_field(
                "severity", "enum", role="core", allowed_values=["low", "high"]
            ),
            _make_field("count", "integer", role="core", minimum=0, maximum=10),
        ]
        results = match_fields(
            {"severity": "high", "count": 3},
            {"severity": "high", "count": 5},
            fields,
        )
        assert len(results) == 2
        assert results[0].matched is True  # severity matches
        assert results[1].matched is False  # count differs

    def test_aggregate_metrics_standalone(self):
        """Given N pairs, the module returns aggregate metrics."""
        fields = [
            _make_field("severity", "enum", role="core", allowed_values=["low", "high"])
        ]
        all_results = [
            [FieldMatchResult("severity", True, "high", "high")],
            [FieldMatchResult("severity", False, "low", "high")],
        ]
        metrics = compute_aggregate_metrics(all_results, fields)
        assert metrics.overall_exact_match_rate == 0.5
        assert metrics.per_core_field_match_rate["severity"] == 0.5
        assert "severity" in metrics.per_value_metrics


class TestGroundTruthNormalization:
    def test_normalize_ground_truth_canonicalizes_valid_values_only(self):
        """Legacy Verified rows saved before the save-time normalization
        boundary may hold un-normalized values; the read-side helper
        canonicalizes what it can and never makes a stored value worse."""
        fields = [
            {
                "field_name": "severity",
                "type": "enum",
                "role": "core",
                "allowed_values": ["low", "high"],
            },
            {"field_name": "damaged", "type": "boolean", "role": "core"},
        ]
        gt = {"severity": "  HIGH ", "damaged": 1, "extra": "kept"}
        out = normalize_ground_truth(gt, fields)
        assert out["severity"] == "high"
        # Invalid legacy value (numeric boolean proxy) stays as stored.
        assert out["damaged"] == 1
        assert out["extra"] == "kept"
        # Input is not mutated.
        assert gt["severity"] == "  HIGH "

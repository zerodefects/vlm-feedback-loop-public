# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SchemaCore type system (services/schema_core.py).

Pure logic tests — no database, no HTTP.  Covers the type system,
generation_order derivation, JSON Schema derivation, and all validation
issue codes.
"""

from __future__ import annotations

import re
from unittest.mock import patch

from vlm_feedback_loop.services.schema_core import (
    DUPLICATE_FIELD_NAME,
    ENUM_DUPLICATE_VALUE,
    ENUM_EMPTY_VALUE,
    ENUM_TOO_FEW_VALUES,
    FIELD_NAME_TOO_LONG,
    INVALID_FIELD_NAME,
    MIN_EXCEEDS_MAX,
    MINLENGTH_EXCEEDS_MAXLENGTH,
    MISSING_FIELD_NAME,
    MISSING_TYPE,
    NO_CORE_FIELDS,
    RATIONALE_NOTE_WRONG_ROLE,
    RATIONALE_NOTE_WRONG_TYPE,
    SCHEMA_COMPILE_FAILURE,
    strip_guided_decoding_unsupported_keys,
    validate_and_derive,
)

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_core_enum(
    name: str = "category",
    values: list[str] | None = None,
    display_order: int = 1,
) -> dict:
    return {
        "field_name": name,
        "type": "enum",
        "role": "core",
        "allowed_values": values or ["a", "b"],
        "display_order": display_order,
    }


def _make_core_boolean(name: str = "flag", display_order: int = 2) -> dict:
    return {
        "field_name": name,
        "type": "boolean",
        "role": "core",
        "display_order": display_order,
    }


def _make_aux_string(name: str = "notes", display_order: int = 0) -> dict:
    return {
        "field_name": name,
        "type": "string",
        "role": "aux",
        "display_order": display_order,
    }


def _minimal_valid_fields() -> list[dict]:
    """Return a minimal valid field list (one Core field, no rationale_note)."""
    return [_make_core_enum()]


def _result_codes(result) -> list[str]:
    """Extract issue codes from a ValidationResult."""
    return [i.code for i in result.issues]


# ── Field ID Generation ────────────────────────────────────────────────────


class TestFieldIdGeneration:
    def test_field_ids_auto_generated_as_uuid4(self):
        fields = [_make_core_enum()]
        result = validate_and_derive(fields)
        assert result.save_allowed
        for f in result.processed_fields:
            assert UUID4_RE.match(f["field_id"]), f"Bad field_id: {f['field_id']}"

    def test_creation_assigns_fresh_field_ids(self):
        fields = [_make_core_enum()]
        fields[0]["field_id"] = "client-supplied-id"
        result = validate_and_derive(fields)
        assert result.save_allowed
        # The user-supplied field should have been overwritten
        core_field = [
            f for f in result.processed_fields if f["field_name"] == "category"
        ][0]
        assert core_field["field_id"] != "client-supplied-id"
        assert UUID4_RE.match(core_field["field_id"])

    def test_each_field_gets_unique_id(self):
        fields = [
            _make_core_enum("a", ["x", "y"], 1),
            _make_core_boolean("b", 2),
        ]
        result = validate_and_derive(fields)
        ids = [f["field_id"] for f in result.processed_fields]
        assert len(ids) == len(set(ids))


# ── Rationale Note ──────────────────────────────────────────────────────────


class TestRationaleNote:
    def test_disabled_when_missing(self):
        """Omitting rationale_note keeps it out of every derived contract."""
        fields = [_make_core_enum()]
        result = validate_and_derive(fields)
        assert result.save_allowed
        names = [f["field_name"] for f in result.processed_fields]
        assert "rationale_note" not in names
        assert "rationale_note" not in result.generation_order
        assert "rationale_note" not in result.derived_json_schema["properties"]

    def test_enabled_field_receives_backend_identity(self):
        fields = [
            _make_aux_string("rationale_note", display_order=0),
            _make_core_enum(),
        ]
        result = validate_and_derive(fields)
        rn = [
            f for f in result.processed_fields if f["field_name"] == "rationale_note"
        ][0]
        assert rn["type"] == "string"
        assert rn["role"] == "aux"
        assert UUID4_RE.match(rn["field_id"])

    def test_has_lowest_display_order(self):
        fields = [
            _make_aux_string("rationale_note", display_order=99),
            _make_aux_string("obs", display_order=0),
            _make_core_enum("cat", ["a", "b"], display_order=1),
        ]
        result = validate_and_derive(fields)
        rn = [
            f for f in result.processed_fields if f["field_name"] == "rationale_note"
        ][0]
        other_orders = [
            f["display_order"]
            for f in result.processed_fields
            if f["field_name"] != "rationale_note"
        ]
        assert rn["display_order"] < min(other_orders)

    def test_existing_rationale_note_display_order_enforced(self):
        """If user provides rationale_note with a high display_order, it's lowered."""
        fields = [
            {
                "field_name": "rationale_note",
                "type": "string",
                "role": "aux",
                "display_order": 99,
            },
            _make_core_enum("cat", ["a", "b"], display_order=1),
        ]
        result = validate_and_derive(fields)
        assert result.save_allowed
        rn = [
            f for f in result.processed_fields if f["field_name"] == "rationale_note"
        ][0]
        core = [f for f in result.processed_fields if f["field_name"] == "cat"][0]
        assert rn["display_order"] < core["display_order"]

    def test_wrong_role_rejected(self):
        fields = [
            {
                "field_name": "rationale_note",
                "type": "string",
                "role": "core",
                "display_order": 0,
            },
            _make_core_enum(),
        ]
        result = validate_and_derive(fields)
        assert RATIONALE_NOTE_WRONG_ROLE in _result_codes(result)

    def test_wrong_type_rejected(self):
        fields = [
            {
                "field_name": "rationale_note",
                "type": "boolean",
                "role": "aux",
                "display_order": 0,
            },
            _make_core_enum(),
        ]
        result = validate_and_derive(fields)
        assert RATIONALE_NOTE_WRONG_TYPE in _result_codes(result)

    def test_user_defined_rationale_note_as_core_rejected(self):
        """A user-defined field named 'rationale_note' with role=core is an error."""
        fields = [
            {
                "field_name": "rationale_note",
                "type": "string",
                "role": "core",
                "display_order": 0,
            },
            _make_core_enum(),
        ]
        result = validate_and_derive(fields)
        assert not result.save_allowed
        assert RATIONALE_NOTE_WRONG_ROLE in _result_codes(result)


# ── Generation Order ────────────────────────────────────────────────────────


class TestGenerationOrder:
    def test_rationale_note_absent_when_disabled(self):
        fields = [_make_core_enum()]
        result = validate_and_derive(fields)
        assert result.generation_order == ["category"]

    def test_aux_before_core(self):
        fields = [
            _make_aux_string("obs", display_order=5),
            _make_core_enum("cat", ["a", "b"], display_order=1),
        ]
        result = validate_and_derive(fields)
        order = result.generation_order
        obs_idx = order.index("obs")
        cat_idx = order.index("cat")
        assert obs_idx < cat_idx

    def test_enabled_rationale_note_first(self):
        fields = [
            _make_aux_string("obs", display_order=5),
            _make_aux_string("rationale_note", display_order=9),
            _make_core_enum("cat", ["a", "b"], display_order=1),
        ]
        result = validate_and_derive(fields)
        assert result.generation_order == ["rationale_note", "obs", "cat"]

    def test_within_role_sorted_by_display_order(self):
        fields = [
            _make_core_enum("z_field", ["a", "b"], display_order=10),
            _make_core_boolean("a_field", display_order=5),
            _make_aux_string("y_aux", display_order=3),
            _make_aux_string("x_aux", display_order=1),
        ]
        result = validate_and_derive(fields)
        order = result.generation_order
        # Aux sorted by display_order: x_aux(1), y_aux(3)
        aux_section = order[:2]
        assert aux_section == ["x_aux", "y_aux"]
        # Core sorted by display_order: a_field(5), z_field(10)
        core_section = order[2:]
        assert core_section == ["a_field", "z_field"]

    def test_tie_breaking_by_field_name(self):
        """When display_order ties, field_name breaks the tie (deterministic)."""
        fields = [
            _make_core_enum("banana", ["a", "b"], display_order=1),
            _make_core_boolean("apple", display_order=1),
        ]
        result = validate_and_derive(fields)
        core_section = [
            n for n in result.generation_order if n not in ("rationale_note",)
        ]
        # Within core, same display_order → alphabetical by field_name
        assert core_section == ["apple", "banana"]


# ── JSON Schema Derivation ──────────────────────────────────────────────────


class TestJsonSchemaDerivation:
    def test_enum_maps_to_string_with_enum_constraint(self):
        fields = [_make_core_enum("cat", ["good", "bad"])]
        result = validate_and_derive(fields)
        prop = result.derived_json_schema["properties"]["cat"]
        assert prop == {"type": "string", "enum": ["good", "bad"]}

    def test_enum_set_maps_to_array(self):
        fields = [
            {
                "field_name": "tags",
                "type": "enum_set",
                "role": "core",
                "allowed_values": ["a", "b", "c"],
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        prop = result.derived_json_schema["properties"]["tags"]
        assert prop == {
            "type": "array",
            "items": {"type": "string", "enum": ["a", "b", "c"]},
            "uniqueItems": True,
        }


class TestStripGuidedDecodingUnsupportedKeys:
    """The guided-decoding schema drops keywords NIM grammar can't compile.

    A hosted NIM (e.g. stepfun-ai/step-3.7-flash, where this was measured) 400s
    the whole call with ``Grammar error: Unimplemented keys: ["uniqueItems"]`` when
    an enum_set's ``uniqueItems`` reaches ``response_format``. The stored schema
    keeps it (semantic contract); the grammar request must not carry it.
    """

    def test_enum_set_uniqueitems_is_stripped(self):
        derived = validate_and_derive(
            [
                {
                    "field_name": "tags",
                    "type": "enum_set",
                    "role": "core",
                    "allowed_values": ["a", "b", "c"],
                    "display_order": 1,
                }
            ]
        ).derived_json_schema
        cleaned = strip_guided_decoding_unsupported_keys(derived)
        assert "uniqueItems" not in cleaned["properties"]["tags"]
        # The enum constraint (which grammar CAN compile) is preserved.
        assert cleaned["properties"]["tags"]["items"]["enum"] == ["a", "b", "c"]
        # The derived schema is not mutated in place.
        assert derived["properties"]["tags"]["uniqueItems"] is True

    def test_supported_constraints_survive(self):
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "code": {"type": "string", "minLength": 2, "maxLength": 8},
                "cat": {"type": "string", "enum": ["x", "y"]},
            },
            "required": ["score"],
            "additionalProperties": False,
            "x-generation-order": ["score", "code", "cat"],
        }
        assert strip_guided_decoding_unsupported_keys(schema) == schema

    def test_recurses_into_nested_arrays(self):
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                }
            },
        }
        cleaned = strip_guided_decoding_unsupported_keys(schema)
        assert "uniqueItems" not in cleaned["properties"]["outer"]
        assert "uniqueItems" not in cleaned["properties"]["outer"]["items"]

    def test_boolean_maps_correctly(self):
        fields = [_make_core_boolean("flag")]
        result = validate_and_derive(fields)
        prop = result.derived_json_schema["properties"]["flag"]
        assert prop == {"type": "boolean"}

    def test_integer_maps_with_min_max(self):
        fields = [
            {
                "field_name": "score",
                "type": "integer",
                "role": "core",
                "minimum": 0,
                "maximum": 10,
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        prop = result.derived_json_schema["properties"]["score"]
        assert prop == {"type": "integer", "minimum": 0, "maximum": 10}

    def test_integer_without_constraints(self):
        fields = [
            {
                "field_name": "count",
                "type": "integer",
                "role": "core",
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        prop = result.derived_json_schema["properties"]["count"]
        assert prop == {"type": "integer"}

    def test_string_maps_with_length_constraints(self):
        fields = [
            {
                "field_name": "desc",
                "type": "string",
                "role": "core",
                "min_length": 1,
                "max_length": 100,
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        prop = result.derived_json_schema["properties"]["desc"]
        assert prop == {"type": "string", "minLength": 1, "maxLength": 100}

    def test_string_without_constraints(self):
        fields = [
            {"field_name": "desc", "type": "string", "role": "core", "display_order": 1}
        ]
        result = validate_and_derive(fields)
        prop = result.derived_json_schema["properties"]["desc"]
        assert prop == {"type": "string"}

    def test_required_contains_only_core_fields(self):
        fields = [
            _make_aux_string("obs", display_order=0),
            _make_core_enum("cat", ["a", "b"], display_order=1),
            _make_core_boolean("flag", display_order=2),
        ]
        result = validate_and_derive(fields)
        required = result.derived_json_schema["required"]
        assert "cat" in required
        assert "flag" in required
        assert "rationale_note" not in required
        assert "obs" not in required

    def test_additional_properties_false(self):
        fields = [_make_core_enum()]
        result = validate_and_derive(fields)
        assert result.derived_json_schema["additionalProperties"] is False

    def test_x_generation_order_extension_present(self):
        fields = [_make_core_enum()]
        result = validate_and_derive(fields)
        assert "x-generation-order" in result.derived_json_schema
        assert (
            result.derived_json_schema["x-generation-order"] == result.generation_order
        )

    def test_properties_ordered_by_generation_order(self):
        fields = [
            _make_aux_string("observation", display_order=0),
            _make_core_enum("category", ["a", "b"], display_order=1),
            _make_core_boolean("confirmed", display_order=2),
        ]
        result = validate_and_derive(fields)
        prop_keys = list(result.derived_json_schema["properties"].keys())
        assert prop_keys == result.generation_order


# ── Schema Hash ─────────────────────────────────────────────────────────────


class TestSchemaHash:
    def test_deterministic_on_same_input(self):
        fields = [_make_core_enum("cat", ["x", "y"])]
        r1 = validate_and_derive(fields)
        r2 = validate_and_derive(list(fields))
        assert r1.schema_hash == r2.schema_hash

    def test_different_on_different_input(self):
        r1 = validate_and_derive([_make_core_enum("a", ["x", "y"])])
        r2 = validate_and_derive([_make_core_enum("b", ["x", "y"])])
        assert r1.schema_hash != r2.schema_hash

    def test_hash_is_hex_sha256(self):
        result = validate_and_derive([_make_core_enum()])
        assert len(result.schema_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.schema_hash)


# ── Validation Issues ───────────────────────────────────────────────────────


class TestValidationIssues:
    def test_no_core_fields(self):
        fields = [_make_aux_string("obs")]
        result = validate_and_derive(fields)
        assert NO_CORE_FIELDS in _result_codes(result)
        assert not result.save_allowed

    def test_missing_field_name(self):
        fields = [{"type": "boolean", "role": "core", "display_order": 1}]
        result = validate_and_derive(fields)
        assert MISSING_FIELD_NAME in _result_codes(result)

    def test_empty_field_name(self):
        fields = [
            {"field_name": "", "type": "boolean", "role": "core", "display_order": 1}
        ]
        result = validate_and_derive(fields)
        assert MISSING_FIELD_NAME in _result_codes(result)

    def test_duplicate_field_name(self):
        fields = [
            _make_core_enum("cat", ["a", "b"], 1),
            _make_core_boolean("cat", 2),
        ]
        result = validate_and_derive(fields)
        assert DUPLICATE_FIELD_NAME in _result_codes(result)

    def test_duplicate_field_name_across_core_and_aux(self):
        """Field names are unique across the combined Core + Aux namespace, not per-section."""
        fields = [
            _make_core_enum("cat", ["a", "b"], 1),
            _make_aux_string("cat", 2),
        ]
        result = validate_and_derive(fields)
        assert DUPLICATE_FIELD_NAME in _result_codes(result)

    def test_invalid_field_name_starts_with_digit(self):
        fields = [
            {
                "field_name": "1bad",
                "type": "boolean",
                "role": "core",
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        assert INVALID_FIELD_NAME in _result_codes(result)

    def test_invalid_field_name_contains_space(self):
        fields = [
            {
                "field_name": "has space",
                "type": "boolean",
                "role": "core",
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        assert INVALID_FIELD_NAME in _result_codes(result)

    def test_field_name_too_long(self):
        long_name = "a" * 65
        fields = [
            {
                "field_name": long_name,
                "type": "boolean",
                "role": "core",
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        assert FIELD_NAME_TOO_LONG in _result_codes(result)

    def test_missing_type(self):
        fields = [{"field_name": "x", "role": "core", "display_order": 1}]
        result = validate_and_derive(fields)
        assert MISSING_TYPE in _result_codes(result)

    def test_unsupported_type(self):
        fields = [
            {"field_name": "x", "type": "float", "role": "core", "display_order": 1}
        ]
        result = validate_and_derive(fields)
        assert MISSING_TYPE in _result_codes(result)

    def test_enum_too_few_values(self):
        fields = [
            {
                "field_name": "x",
                "type": "enum",
                "role": "core",
                "allowed_values": ["only_one"],
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        assert ENUM_TOO_FEW_VALUES in _result_codes(result)

    def test_enum_empty_value(self):
        fields = [
            {
                "field_name": "x",
                "type": "enum",
                "role": "core",
                "allowed_values": ["a", "  "],
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        assert ENUM_EMPTY_VALUE in _result_codes(result)

    def test_enum_duplicate_value_after_trim(self):
        fields = [
            {
                "field_name": "x",
                "type": "enum",
                "role": "core",
                "allowed_values": ["a", " a "],
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        assert ENUM_DUPLICATE_VALUE in _result_codes(result)

    def test_enum_set_same_constraints(self):
        fields = [
            {
                "field_name": "x",
                "type": "enum_set",
                "role": "core",
                "allowed_values": ["only"],
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        assert ENUM_TOO_FEW_VALUES in _result_codes(result)

    def test_min_exceeds_max(self):
        fields = [
            {
                "field_name": "x",
                "type": "integer",
                "role": "core",
                "minimum": 10,
                "maximum": 5,
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        assert MIN_EXCEEDS_MAX in _result_codes(result)

    def test_minlength_exceeds_maxlength(self):
        fields = [
            {
                "field_name": "x",
                "type": "string",
                "role": "core",
                "min_length": 10,
                "max_length": 5,
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        assert MINLENGTH_EXCEEDS_MAXLENGTH in _result_codes(result)

    def test_field_path_indices_are_per_section(self):
        """field_path indexes each field within its own
        section (core[i] / aux[j]) in submission order — not by position in
        the full list. The guidance builder attaches issues to rows by this
        path, so a mixed core/aux submission must yield section-local indices.
        """
        fields = [
            _make_aux_string("rationale_note", 0),  # aux[0]
            _make_core_enum("cat", ["a", "b"], 1),  # core[0]
            {  # core[1] — invalid name
                "field_name": "1bad",
                "type": "boolean",
                "role": "core",
                "display_order": 2,
            },
            _make_aux_string("bad name", 3),  # aux[1] — invalid name
        ]
        result = validate_and_derive(fields)
        paths = {
            (i.code, i.field_path)
            for i in result.issues
            if i.code == INVALID_FIELD_NAME
        }
        assert (INVALID_FIELD_NAME, "core[1].name") in paths
        assert (INVALID_FIELD_NAME, "aux[1].name") in paths


# ── Field Name Edge Cases ───────────────────────────────────────────────────


class TestFieldNameEdgeCases:
    """Names like type, id, class are valid (nothing is reserved beyond rationale_note)."""

    def test_type_is_valid_name(self):
        fields = [
            {
                "field_name": "type",
                "type": "boolean",
                "role": "core",
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        assert result.save_allowed

    def test_id_is_valid_name(self):
        fields = [
            {"field_name": "id", "type": "boolean", "role": "core", "display_order": 1}
        ]
        result = validate_and_derive(fields)
        assert result.save_allowed

    def test_class_is_valid_name(self):
        fields = [
            {
                "field_name": "class",
                "type": "boolean",
                "role": "core",
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        assert result.save_allowed

    def test_underscore_prefix_valid(self):
        fields = [
            {
                "field_name": "_internal",
                "type": "boolean",
                "role": "core",
                "display_order": 1,
            }
        ]
        result = validate_and_derive(fields)
        assert result.save_allowed

    def test_max_length_64_is_valid(self):
        name = "a" * 64
        fields = [
            {"field_name": name, "type": "boolean", "role": "core", "display_order": 1}
        ]
        result = validate_and_derive(fields)
        assert result.save_allowed

    def test_boolean_no_additional_constraints(self):
        """Boolean type has no additional constraint fields — always valid."""
        fields = [_make_core_boolean()]
        result = validate_and_derive(fields)
        assert result.save_allowed

    def test_valid_result_has_all_derivations(self):
        """A valid schema produces all derived artifacts."""
        fields = [_make_core_enum()]
        result = validate_and_derive(fields)
        assert result.save_allowed
        assert result.derived_json_schema is not None
        assert result.generation_order is not None
        assert result.schema_hash is not None
        assert result.processed_fields is not None


# ── Save Allowed Logic ──────────────────────────────────────────────────────


class TestSaveAllowed:
    def test_save_allowed_when_valid(self):
        result = validate_and_derive([_make_core_enum()])
        assert result.save_allowed is True

    def test_save_not_allowed_when_errors(self):
        """A schema with no Core fields fails the NO_CORE_FIELDS structural check and blocks saving."""
        result = validate_and_derive([])
        assert result.save_allowed is False
        assert NO_CORE_FIELDS in _result_codes(result)

    def test_derivation_skipped_when_errors(self):
        """When the NO_CORE_FIELDS structural check fails, no derived artifacts are produced."""
        result = validate_and_derive([])
        assert NO_CORE_FIELDS in _result_codes(result)
        assert result.derived_json_schema is None
        assert result.generation_order is None
        assert result.schema_hash is None


# ── SCHEMA_COMPILE_FAILURE ────────────────────────────────────────


class TestSchemaCompileFailure:
    """SCHEMA_COMPILE_FAILURE is detected on an internally inconsistent schema."""

    def test_derivation_error_produces_compile_failure(self):
        """If JSON Schema derivation raises, the error is caught and reported."""
        fields = [_make_core_enum()]
        with patch(
            "vlm_feedback_loop.services.schema_core._derive_json_schema",
            side_effect=ValueError("synthetic derivation failure"),
        ):
            result = validate_and_derive(fields)
        assert not result.save_allowed
        assert SCHEMA_COMPILE_FAILURE in _result_codes(result)
        assert "synthetic derivation failure" in result.issues[-1].message

    def test_compile_failure_clears_derived_artifacts(self):
        """On compile failure, derived_json_schema and schema_hash remain None."""
        fields = [_make_core_enum()]
        with patch(
            "vlm_feedback_loop.services.schema_core._derive_json_schema",
            side_effect=RuntimeError("boom"),
        ):
            result = validate_and_derive(fields)
        assert result.derived_json_schema is None
        assert result.schema_hash is None

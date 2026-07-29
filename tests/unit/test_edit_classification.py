# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for edit classification (services/edit_classification.py).

Pure logic tests — no database, no HTTP.  Covers the edit classification
boundary: in-place vs semantic, rename maps, enum value rename detection.
"""

from __future__ import annotations

from vlm_feedback_loop.services.edit_classification import (
    EditClassificationResult,
    classify_edits,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _field(
    field_id: str,
    name: str,
    ftype: str = "boolean",
    role: str = "core",
    allowed_values: list[str] | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    display_order: int = 0,
) -> dict:
    f: dict = {
        "field_id": field_id,
        "field_name": name,
        "type": ftype,
        "role": role,
        "display_order": display_order,
    }
    if allowed_values is not None:
        f["allowed_values"] = allowed_values
    if minimum is not None:
        f["minimum"] = minimum
    if maximum is not None:
        f["maximum"] = maximum
    if min_length is not None:
        f["min_length"] = min_length
    if max_length is not None:
        f["max_length"] = max_length
    return f


def _change_types(result: EditClassificationResult) -> list[str]:
    return [c.change_type for c in result.changes]


def _classifications(result: EditClassificationResult) -> list[str]:
    return [c.classification for c in result.changes]


# ── Field Matching ──────────────────────────────────────────────────────────


class TestFieldMatching:
    def test_identical_schemas_no_changes(self):
        old = [_field("f1", "cat", "enum", "core", ["a", "b"])]
        new = [_field("f1", "cat", "enum", "core", ["a", "b"])]
        result = classify_edits(old, new)
        assert result.changes == []
        assert not result.has_semantic_changes

    def test_fields_matched_by_field_id(self):
        old = [_field("f1", "old_name")]
        new = [_field("f1", "new_name")]
        result = classify_edits(old, new)
        assert len(result.changes) == 1
        assert result.changes[0].change_type == "field_rename"

    def test_unmatched_old_field_is_removal(self):
        old = [_field("f1", "cat"), _field("f2", "dog")]
        new = [_field("f1", "cat")]
        result = classify_edits(old, new)
        assert "remove_core_field" in _change_types(result)

    def test_unmatched_new_field_is_addition(self):
        old = [_field("f1", "cat")]
        new = [_field("f1", "cat"), _field("f2", "dog")]
        result = classify_edits(old, new)
        assert "add_core_field" in _change_types(result)

    def test_rationale_note_change_follows_aux_policy(self):
        """The optional reserved field can change without Core invalidation."""
        old = [_field("rn", "rationale_note", "string", "aux")]
        new = [_field("rn", "rationale_note", "string", "aux", display_order=99)]
        result = classify_edits(old, new)
        assert _change_types(result) == ["display_order_change"]
        assert _classifications(result) == ["in_place"]

    def test_rationale_note_toggle_follows_aux_policy(self):
        old = [
            _field("core", "category"),
            _field("rn", "rationale_note", "string", "aux"),
        ]
        disabled = [_field("core", "category")]
        removed = classify_edits(old, disabled)
        added = classify_edits(disabled, old)
        assert _change_types(removed) == ["remove_aux_field"]
        assert _change_types(added) == ["add_aux_field"]
        assert not removed.has_semantic_changes
        assert not added.has_semantic_changes


# ── In-Place Classifications ────────────────────────────────────────────────


class TestInPlaceClassification:
    def test_field_rename_is_in_place(self):
        """Core field rename via field_id does not trigger invalidation."""
        old = [_field("f1", "damage_type", "enum", "core", ["a", "b"])]
        new = [_field("f1", "defect_type", "enum", "core", ["a", "b"])]
        result = classify_edits(old, new)
        assert "in_place" in _classifications(result)
        assert not result.has_semantic_changes
        assert result.changes[0].change_type == "field_rename"
        assert result.field_renames == {"damage_type": "defect_type"}

    def test_exact_1to1_enum_value_rename_is_in_place(self):
        """Exact 1:1 enum value rename does not trigger invalidation."""
        old = [_field("f1", "cat", "enum", "core", ["damaged", "ok"])]
        new = [_field("f1", "cat", "enum", "core", ["severe_damage", "ok"])]
        result = classify_edits(old, new)
        assert "in_place" in _classifications(result)
        assert not result.has_semantic_changes
        assert result.changes[0].change_type == "enum_value_rename"
        assert result.enum_value_renames == {"cat": {"damaged": "severe_damage"}}

    def test_display_order_change_is_in_place(self):
        """Presentation metadata edits don't trigger invalidation."""
        old = [_field("f1", "cat", display_order=1)]
        new = [_field("f1", "cat", display_order=5)]
        result = classify_edits(old, new)
        assert "in_place" in _classifications(result)
        assert not result.has_semantic_changes

    def test_add_aux_field_is_in_place(self):
        old = [_field("f1", "cat")]
        new = [_field("f1", "cat"), _field("f2", "notes", "string", "aux")]
        result = classify_edits(old, new)
        assert "in_place" in _classifications(result)
        assert not result.has_semantic_changes
        assert "add_aux_field" in _change_types(result)

    def test_remove_aux_field_is_in_place(self):
        old = [_field("f1", "cat"), _field("f2", "notes", "string", "aux")]
        new = [_field("f1", "cat")]
        result = classify_edits(old, new)
        assert "in_place" in _classifications(result)
        assert not result.has_semantic_changes
        assert "remove_aux_field" in _change_types(result)


# ── Semantic Classifications ────────────────────────────────────────────────


class TestSemanticClassification:
    def test_add_core_field_is_semantic(self):
        old = [_field("f1", "cat")]
        new = [_field("f1", "cat"), _field("f2", "severity", "integer", "core")]
        result = classify_edits(old, new)
        assert result.has_semantic_changes
        assert "add_core_field" in _change_types(result)

    def test_remove_core_field_is_semantic(self):
        old = [_field("f1", "cat"), _field("f2", "severity")]
        new = [_field("f1", "cat")]
        result = classify_edits(old, new)
        assert result.has_semantic_changes
        assert "remove_core_field" in _change_types(result)

    def test_type_change_is_semantic(self):
        old = [_field("f1", "cat", "enum", "core", ["a", "b"])]
        new = [_field("f1", "cat", "string", "core")]
        result = classify_edits(old, new)
        assert result.has_semantic_changes
        assert "type_change" in _change_types(result)

    def test_integer_constraint_change_is_semantic(self):
        """Integer constraint change on a Core field triggers invalidation."""
        old = [_field("f1", "score", "integer", "core", minimum=0, maximum=10)]
        new = [_field("f1", "score", "integer", "core", minimum=0, maximum=20)]
        result = classify_edits(old, new)
        assert result.has_semantic_changes
        assert "constraint_change" in _change_types(result)

    def test_string_constraint_change_is_semantic(self):
        """String constraint change on a Core field triggers invalidation."""
        old = [_field("f1", "desc", "string", "core", min_length=1, max_length=100)]
        new = [_field("f1", "desc", "string", "core", min_length=1, max_length=200)]
        result = classify_edits(old, new)
        assert result.has_semantic_changes

    def test_enum_add_value_is_semantic(self):
        """Adding an enum value is a semantic change."""
        old = [_field("f1", "cat", "enum", "core", ["a", "b"])]
        new = [_field("f1", "cat", "enum", "core", ["a", "b", "c"])]
        result = classify_edits(old, new)
        assert result.has_semantic_changes
        assert "allowed_value_change" in _change_types(result)

    def test_enum_remove_value_is_semantic(self):
        """Removing an enum value is a semantic change."""
        old = [_field("f1", "cat", "enum", "core", ["a", "b", "c"])]
        new = [_field("f1", "cat", "enum", "core", ["a", "b"])]
        result = classify_edits(old, new)
        assert result.has_semantic_changes

    def test_core_to_aux_role_change_is_semantic(self):
        """Demoting Core to Aux is a semantic change."""
        old = [_field("f1", "cat", "boolean", "core")]
        new = [_field("f1", "cat", "boolean", "aux")]
        result = classify_edits(old, new)
        assert result.has_semantic_changes
        assert "role_change" in _change_types(result)

    def test_aux_to_core_role_change_is_semantic(self):
        """Promoting Aux to Core is a semantic change."""
        old = [_field("f1", "obs", "string", "aux")]
        new = [_field("f1", "obs", "string", "core")]
        result = classify_edits(old, new)
        assert result.has_semantic_changes
        assert "role_change" in _change_types(result)


# ── Enum Value Rename Detection ─────────────────────────────────────────────


class TestEnumValueRenameDetection:
    def test_single_value_rename_detected(self):
        old = [_field("f1", "cat", "enum", "core", ["good", "bad"])]
        new = [_field("f1", "cat", "enum", "core", ["good", "poor"])]
        result = classify_edits(old, new)
        assert not result.has_semantic_changes
        assert result.enum_value_renames == {"cat": {"bad": "poor"}}

    def test_two_values_swapped_is_semantic(self):
        """If 2+ values differ it's not a simple 1:1 rename."""
        old = [_field("f1", "cat", "enum", "core", ["a", "b", "c"])]
        new = [_field("f1", "cat", "enum", "core", ["x", "y", "c"])]
        result = classify_edits(old, new)
        assert result.has_semantic_changes  # 2 removed + 2 added, not 1:1

    def test_add_and_remove_different_count_is_semantic(self):
        old = [_field("f1", "cat", "enum", "core", ["a", "b"])]
        new = [_field("f1", "cat", "enum", "core", ["a", "c", "d"])]
        result = classify_edits(old, new)
        assert result.has_semantic_changes

    def test_enum_set_value_rename(self):
        old = [_field("f1", "tags", "enum_set", "core", ["x", "y"])]
        new = [_field("f1", "tags", "enum_set", "core", ["x", "z"])]
        result = classify_edits(old, new)
        assert not result.has_semantic_changes
        assert result.enum_value_renames == {"tags": {"y": "z"}}

    def test_no_value_change_when_same_set(self):
        old = [_field("f1", "cat", "enum", "core", ["a", "b"])]
        new = [_field("f1", "cat", "enum", "core", ["a", "b"])]
        result = classify_edits(old, new)
        assert result.changes == []


# ── Mixed Edits ─────────────────────────────────────────────────────────────


class TestMixedEdits:
    def test_in_place_and_semantic_together_yields_semantic(self):
        """If any change is semantic, the whole result is semantic."""
        old = [
            _field("f1", "cat", "enum", "core", ["a", "b"]),
            _field("f2", "score", "integer", "core", minimum=0, maximum=10),
        ]
        new = [
            _field("f1", "category", "enum", "core", ["a", "b"]),  # rename (in-place)
            _field(
                "f2", "score", "integer", "core", minimum=0, maximum=20
            ),  # constraint (semantic)
        ]
        result = classify_edits(old, new)
        assert result.has_semantic_changes
        assert "in_place" in _classifications(result)

    def test_rename_map_built_from_in_place_renames(self):
        old = [_field("f1", "old_name")]
        new = [_field("f1", "new_name")]
        result = classify_edits(old, new)
        assert result.field_renames == {"old_name": "new_name"}

    def test_multiple_renames(self):
        old = [_field("f1", "a"), _field("f2", "b")]
        new = [_field("f1", "alpha"), _field("f2", "beta")]
        result = classify_edits(old, new)
        assert result.field_renames == {"a": "alpha", "b": "beta"}


# ── Change Summary ──────────────────────────────────────────────────────────


class TestChangeSummary:
    def test_summary_documents_all_changes(self):
        old = [_field("f1", "cat", "enum", "core", ["a", "b"])]
        new = [
            _field(
                "f1", "cat", "enum", "core", ["a", "b", "c"]
            ),  # add value (semantic)
        ]
        result = classify_edits(old, new)
        summary = result.change_summary
        assert summary["total_changes"] == 1
        assert len(summary["semantic_changes"]) == 1
        assert len(summary["in_place_changes"]) == 0

    def test_summary_separates_in_place_and_semantic(self):
        old = [
            _field("f1", "cat", "enum", "core", ["a", "b"]),
            _field("f2", "score", "integer", "core", minimum=0, maximum=10),
        ]
        new = [
            _field("f1", "category", "enum", "core", ["a", "b"]),  # rename
            _field(
                "f2", "score", "integer", "core", minimum=0, maximum=20
            ),  # constraint
        ]
        result = classify_edits(old, new)
        summary = result.change_summary
        assert len(summary["in_place_changes"]) >= 1
        assert len(summary["semantic_changes"]) >= 1


# ── Aux Field Constraint Changes ────────────────────────────────────────────


class TestAuxConstraintChanges:
    def test_aux_constraint_change_is_in_place(self):
        """Constraint changes on Aux fields are in-place, not semantic."""
        old = [
            _field("f1", "cat", "boolean", "core"),
            _field("f2", "note", "string", "aux", max_length=100),
        ]
        new = [
            _field("f1", "cat", "boolean", "core"),
            _field("f2", "note", "string", "aux", max_length=200),
        ]
        result = classify_edits(old, new)
        assert "in_place" in _classifications(result)
        assert not result.has_semantic_changes

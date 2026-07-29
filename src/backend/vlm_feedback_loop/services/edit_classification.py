# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edit classification: compare old vs new SchemaCore and classify changes.

Pure domain module — no database or HTTP dependencies. Implements the
SchemaCore edit policy (in-place vs semantic change lists, the decision
rule) and the schema evolution trigger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

# ── Change types ────────────────────────────────────────────────────────────

ChangeType = Literal[
    "field_rename",
    "enum_value_rename",
    "display_order_change",
    "add_aux_field",
    "add_core_field",
    "remove_core_field",
    "remove_aux_field",
    "type_change",
    "constraint_change",
    "allowed_value_change",
    "role_change",
]

Classification = Literal["in_place", "semantic"]

# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class FieldChange:
    """A single detected change between old and new schema versions."""

    field_id: str
    change_type: ChangeType
    classification: Classification
    detail: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class EditClassificationResult:
    """Result of classifying edits between two schema versions."""

    changes: list[FieldChange] = field(default_factory=list[FieldChange])
    has_semantic_changes: bool = False

    # Rename maps for in-place propagation
    field_renames: dict[str, str] = field(default_factory=dict[str, str])
    # Maps old_field_name → new_field_name

    enum_value_renames: dict[str, dict[str, str]] = field(
        default_factory=dict[str, dict[str, str]]
    )
    # Maps field_name → {old_value: new_value}

    change_summary: dict[str, Any] = field(default_factory=dict[str, Any])


# ── Public API ──────────────────────────────────────────────────────────────


def classify_edits(
    old_fields: list[dict[str, Any]],
    new_fields: list[dict[str, Any]],
) -> EditClassificationResult:
    """Compare old and new SchemaCore fields, classify each change.

    Fields are matched by ``field_id``.  A field in old but not new is a
    removal; in new but not old is an addition; in both is checked for
    renames, type changes, constraint changes, etc.

    The optional reserved ``rationale_note`` field follows the normal Aux
    policy: adding or removing it is an in-place change and never triggers
    semantic Core schema evolution.
    """
    result = EditClassificationResult()

    old_map = _build_field_map(old_fields)
    new_map = _build_field_map(new_fields)

    old_ids = set(old_map.keys())
    new_ids = set(new_map.keys())

    # Removals: in old but not new
    for fid in sorted(old_ids - new_ids):
        old_f = old_map[fid]
        change = _classify_field_removal(old_f, fid)
        result.changes.append(change)

    # Additions: in new but not old
    for fid in sorted(new_ids - old_ids):
        new_f = new_map[fid]
        change = _classify_field_addition(new_f, fid)
        result.changes.append(change)

    # Modifications: in both (matched by field_id)
    for fid in sorted(old_ids & new_ids):
        old_f = old_map[fid]
        new_f = new_map[fid]
        changes = _classify_field_modifications(old_f, new_f, fid)
        result.changes.extend(changes)

    # Aggregate flag
    result.has_semantic_changes = any(
        c.classification == "semantic" for c in result.changes
    )

    # Build rename maps from in_place changes
    for c in result.changes:
        if c.change_type == "field_rename" and c.classification == "in_place":
            old_name = c.detail.get("old_name", "")
            new_name = c.detail.get("new_name", "")
            if old_name and new_name:
                result.field_renames[old_name] = new_name

        if c.change_type == "enum_value_rename" and c.classification == "in_place":
            fname = c.detail.get("field_name", "")
            old_val = c.detail.get("old_value", "")
            new_val = c.detail.get("new_value", "")
            if fname and old_val and new_val:
                result.enum_value_renames.setdefault(fname, {})[old_val] = new_val

    # Build change summary
    result.change_summary = _build_change_summary(result.changes)

    return result


# ── Internal helpers ────────────────────────────────────────────────────────


def _build_field_map(fields: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a field_id → field dict lookup."""
    return {f["field_id"]: f for f in fields if "field_id" in f}


def _classify_field_removal(old_field: dict[str, Any], field_id: str) -> FieldChange:
    """Classify a field that exists in old but not new."""
    role = old_field.get("role", "core")
    # Removing a Core field is semantic; removing an Aux field is in-place
    if role == "core":
        return FieldChange(
            field_id=field_id,
            change_type="remove_core_field",
            classification="semantic",
            detail={"field_name": old_field.get("field_name", "")},
        )
    else:
        return FieldChange(
            field_id=field_id,
            change_type="remove_aux_field",
            classification="in_place",
            detail={"field_name": old_field.get("field_name", "")},
        )


def _classify_field_addition(new_field: dict[str, Any], field_id: str) -> FieldChange:
    """Classify a field that exists in new but not old."""
    role = new_field.get("role", "core")
    # Adding a Core field is semantic; adding an Aux field is in-place
    if role == "core":
        return FieldChange(
            field_id=field_id,
            change_type="add_core_field",
            classification="semantic",
            detail={"field_name": new_field.get("field_name", "")},
        )
    else:
        return FieldChange(
            field_id=field_id,
            change_type="add_aux_field",
            classification="in_place",
            detail={"field_name": new_field.get("field_name", "")},
        )


def _classify_field_modifications(
    old_field: dict[str, Any],
    new_field: dict[str, Any],
    field_id: str,
) -> list[FieldChange]:
    """Compare two versions of the same field (matched by field_id).

    Handles: rename, type change, role change, constraint changes,
    allowed_value changes (including exact 1:1 enum renames),
    display_order changes.
    """
    changes: list[FieldChange] = []
    old_role = old_field.get("role", "core")
    new_role = new_field.get("role", "core")
    old_name = old_field.get("field_name", "")
    new_name = new_field.get("field_name", "")
    # Use the new name for display purposes in detail dicts
    display_name = new_name or old_name

    # Role change (Core ↔ Aux) — always semantic
    if old_role != new_role:
        changes.append(
            FieldChange(
                field_id=field_id,
                change_type="role_change",
                classification="semantic",
                detail={
                    "field_name": display_name,
                    "old_role": old_role,
                    "new_role": new_role,
                },
            )
        )
        # Role change is already semantic; skip further checks on this field
        # since the field's semantics have fundamentally changed
        return changes

    # Field rename (in-place for both Core and Aux)
    if old_name != new_name:
        changes.append(
            FieldChange(
                field_id=field_id,
                change_type="field_rename",
                classification="in_place",
                detail={"old_name": old_name, "new_name": new_name},
            )
        )

    # Type change — semantic for Core fields
    old_type = old_field.get("type", "")
    new_type = new_field.get("type", "")
    if old_type != new_type:
        changes.append(
            FieldChange(
                field_id=field_id,
                change_type="type_change",
                classification="semantic" if old_role == "core" else "in_place",
                detail={
                    "field_name": display_name,
                    "old_type": old_type,
                    "new_type": new_type,
                },
            )
        )
        # Type changed, skip constraint/value checks (they don't apply across types)
        return changes

    # Constraint changes (semantic for Core, in-place for Aux)
    constraint_changes = _classify_constraint_changes(
        old_field, new_field, field_id, display_name, old_role
    )
    changes.extend(constraint_changes)

    # Allowed-value changes for enum/enum_set
    if old_type in ("enum", "enum_set"):
        value_changes = _classify_enum_value_changes(
            old_field, new_field, field_id, display_name, old_role
        )
        changes.extend(value_changes)

    # Display order change (in-place — presentation metadata)
    old_order = old_field.get("display_order", 0)
    new_order = new_field.get("display_order", 0)
    if old_order != new_order:
        changes.append(
            FieldChange(
                field_id=field_id,
                change_type="display_order_change",
                classification="in_place",
                detail={
                    "field_name": display_name,
                    "old_order": old_order,
                    "new_order": new_order,
                },
            )
        )

    return changes


def _classify_constraint_changes(
    old_field: dict[str, Any],
    new_field: dict[str, Any],
    field_id: str,
    field_name: str,
    role: str,
) -> list[FieldChange]:
    """Compare integer minimum/maximum and string min_length/max_length constraints."""
    changes: list[FieldChange] = []
    ftype = old_field.get("type", "")

    constraints = {
        "integer": ("minimum", "maximum"),
        "string": ("min_length", "max_length"),
    }.get(ftype, ())

    for constraint in constraints:
        old_val = old_field.get(constraint)
        new_val = new_field.get(constraint)
        if old_val != new_val:
            changes.append(
                FieldChange(
                    field_id=field_id,
                    change_type="constraint_change",
                    classification="semantic" if role == "core" else "in_place",
                    detail={
                        "field_name": field_name,
                        "constraint": constraint,
                        "old_value": old_val,
                        "new_value": new_val,
                    },
                )
            )

    return changes


def _classify_enum_value_changes(
    old_field: dict[str, Any],
    new_field: dict[str, Any],
    field_id: str,
    field_name: str,
    role: str,
) -> list[FieldChange]:
    """Detect enum/enum_set value changes.

    Exact 1:1 rename: old and new have the same count and the symmetric
    difference has exactly 2 elements (one removed, one added).  That pair
    is a rename → in_place.

    Any other difference (add/remove/merge/split) → semantic for Core.
    """
    changes: list[FieldChange] = []
    old_vals_raw: Any = old_field.get("allowed_values", []) or []
    new_vals_raw: Any = new_field.get("allowed_values", []) or []
    old_vals: list[Any] = (
        cast("list[Any]", old_vals_raw) if isinstance(old_vals_raw, list) else []
    )
    new_vals: list[Any] = (
        cast("list[Any]", new_vals_raw) if isinstance(new_vals_raw, list) else []
    )

    old_set: set[Any] = set(old_vals)
    new_set: set[Any] = set(new_vals)

    if old_set == new_set:
        return changes  # No value changes

    removed = old_set - new_set
    added = new_set - old_set

    # Exact 1:1 rename: same count, exactly one removed and one added
    if len(old_vals) == len(new_vals) and len(removed) == 1 and len(added) == 1:
        old_val = removed.pop()
        new_val = added.pop()
        changes.append(
            FieldChange(
                field_id=field_id,
                change_type="enum_value_rename",
                classification="in_place",
                detail={
                    "field_name": field_name,
                    "old_value": old_val,
                    "new_value": new_val,
                },
            )
        )
    else:
        # Any other difference is a semantic change for Core fields
        changes.append(
            FieldChange(
                field_id=field_id,
                change_type="allowed_value_change",
                classification="semantic" if role == "core" else "in_place",
                detail={
                    "field_name": field_name,
                    "removed": sorted(removed),
                    "added": sorted(added),
                },
            )
        )

    return changes


def _build_change_summary(changes: list[FieldChange]) -> dict[str, Any]:
    """Build a human-readable change summary for schema_change_summary."""
    summary: dict[str, Any] = {
        "total_changes": len(changes),
        "semantic_changes": [],
        "in_place_changes": [],
    }
    for c in changes:
        entry = {
            "change_type": c.change_type,
            "field_id": c.field_id,
            **c.detail,
        }
        if c.classification == "semantic":
            summary["semantic_changes"].append(entry)
        else:
            summary["in_place_changes"].append(entry)
    return summary

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SchemaCore type system: validation, derivation, and canonical JSON Schema.

Pure domain module — no database or HTTP dependencies.  Every downstream
consumer (labeling, evaluation, export, training) depends on this contract.

Covers SchemaCore validation, JSON Schema derivation and its
generation_order consumers, and draft validation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Final, Literal, cast

from vlm_feedback_loop.db.base import generate_uuid4

# ── Constants ───────────────────────────────────────────────────────────────

FIELD_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
FIELD_NAME_MAX_LEN = 64
RESERVED_FIELD_NAME = "rationale_note"

VALID_FIELD_TYPES = frozenset({"enum", "enum_set", "boolean", "integer", "string"})

# ── Issue codes ─────────────────────────────────────────────────────────────

NO_CORE_FIELDS = "NO_CORE_FIELDS"
MISSING_FIELD_NAME = "MISSING_FIELD_NAME"
DUPLICATE_FIELD_NAME = "DUPLICATE_FIELD_NAME"
INVALID_FIELD_NAME = "INVALID_FIELD_NAME"
FIELD_NAME_TOO_LONG = "FIELD_NAME_TOO_LONG"
MISSING_TYPE = "MISSING_TYPE"
ENUM_TOO_FEW_VALUES = "ENUM_TOO_FEW_VALUES"
ENUM_EMPTY_VALUE = "ENUM_EMPTY_VALUE"
ENUM_DUPLICATE_VALUE = "ENUM_DUPLICATE_VALUE"
MIN_EXCEEDS_MAX = "MIN_EXCEEDS_MAX"
MINLENGTH_EXCEEDS_MAXLENGTH = "MINLENGTH_EXCEEDS_MAXLENGTH"
RATIONALE_NOTE_WRONG_ROLE = "RATIONALE_NOTE_WRONG_ROLE"
RATIONALE_NOTE_WRONG_TYPE = "RATIONALE_NOTE_WRONG_TYPE"
SCHEMA_COMPILE_FAILURE = "SCHEMA_COMPILE_FAILURE"

# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class SchemaIssue:
    """A single validation issue found during SchemaCore validation.

    ``field_path`` addresses the offending field as ``{section}[{i}].{attr}``
    (e.g. ``core[2].name``, ``aux[0].allowed_values``), where ``i`` is the
    field's index *within its section* (core/aux) in submission order. The
    guidance builder UI uses this path to attach issues to rows, so the
    per-section indexing is part of the response contract.
    """

    severity: Literal["error", "warning"]
    code: str
    message: str
    field_path: str | None = None


@dataclass
class ValidationResult:
    """Result of ``validate_and_derive``.

    When ``save_allowed`` is True, ``processed_fields``, ``derived_json_schema``,
    ``generation_order``, and ``schema_hash`` are guaranteed non-None.
    """

    issues: list[SchemaIssue] = field(default_factory=list[SchemaIssue])
    derived_json_schema: dict[str, Any] | None = None
    generation_order: list[str] | None = None
    schema_hash: str | None = None
    save_allowed: bool = False
    processed_fields: list[dict[str, Any]] | None = None


# ── Public API ──────────────────────────────────────────────────────────────


def validate_and_derive(
    fields: list[dict[str, Any]],
) -> ValidationResult:
    """Validate SchemaCore fields and derive the canonical JSON Schema.

    This is the **single entry point** used by both ``create_guidance`` and
    ``validate_draft`` (both must run the same ``validate_and_derive`` function).

    Guidance creation owns field IDs and assigns fresh UUID4s before the
    shared validate/derive pipeline runs.
    """
    return _validate_and_derive(_assign_field_ids(fields))


def validate_and_derive_edit(
    old_fields: list[dict[str, Any]],
    new_fields: list[dict[str, Any]],
) -> ValidationResult:
    """Validate SchemaCore fields for an edit, preserving existing field_ids.

    Like ``validate_and_derive`` but uses ``_assign_field_ids_for_edit``
    to preserve field_ids that match the old schema.  New fields get fresh
    UUIDs.  This enables edit classification by field_id continuity.
    """
    return _validate_and_derive(_assign_field_ids_for_edit(old_fields, new_fields))


def _validate_and_derive(
    processed: list[dict[str, Any]],
) -> ValidationResult:
    """Shared validate/derive pipeline behind both public entry points.

    ``processed`` is the field list with ``field_id`` values already
    assigned (fresh for create, continuity-preserving for edit).

    Steps (in order):
    1. If ``rationale_note`` is present, validate its reserved shape.
    2. Validate all fields (name, type, per-type constraints, at least one Core).
    3. If no errors → derive generation_order → JSON Schema → schema_hash.
    """
    result = ValidationResult()

    # 1. Validate the optional reserved rationale_note field.
    processed, rn_issues = _validate_rationale_note(processed)
    result.issues.extend(rn_issues)

    # 2. Validate all fields
    field_issues = _validate_fields(processed)
    result.issues.extend(field_issues)

    # Store processed fields for response even when there are errors
    result.processed_fields = processed

    # 3. Derive if no errors
    has_errors = any(i.severity == "error" for i in result.issues)
    result.save_allowed = not has_errors

    if not has_errors:
        try:
            generation_order = _derive_generation_order(processed)
            result.generation_order = generation_order

            derived = _derive_json_schema(processed, generation_order)
            result.derived_json_schema = derived

            result.schema_hash = _compute_schema_hash(derived)
        except Exception as exc:
            # Safety net: catches contradictions that individual field-level
            # rules might miss (surfaced as SCHEMA_COMPILE_FAILURE).
            result.issues.append(
                SchemaIssue(
                    severity="error",
                    code=SCHEMA_COMPILE_FAILURE,
                    message=f"Schema cannot be compiled (internal inconsistency). See details: {exc}",
                )
            )
            result.save_allowed = False

    return result


# ── Internal helpers ────────────────────────────────────────────────────────


def _assign_field_ids(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep-copy fields and assign a fresh UUID4 ``field_id`` to each.

    Creation always assigns system-owned identifiers. This also makes the
    helper safe when deriving from an internal field list that already has
    persisted identifiers.
    """
    out: list[dict[str, Any]] = []
    for f in fields:
        copy = dict(f)
        copy["field_id"] = generate_uuid4()
        out.append(copy)
    return out


def _assign_field_ids_for_edit(
    old_fields: list[dict[str, Any]],
    new_fields: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve field_ids from old schema where they match; assign fresh for new.

    If a new field supplies a ``field_id`` that exists in old_fields, preserve
    it (the field existed before — this enables edit classification by field_id
    continuity).  Otherwise, generate a fresh UUID4.
    """
    old_ids = {f["field_id"] for f in old_fields if "field_id" in f}
    out: list[dict[str, Any]] = []
    for f in new_fields:
        copy = dict(f)
        supplied_id = copy.get("field_id")
        if supplied_id and supplied_id in old_ids:
            copy["field_id"] = supplied_id
        else:
            copy["field_id"] = generate_uuid4()
        out.append(copy)
    return out


def rationale_note_enabled(fields: list[dict[str, Any]]) -> bool:
    """Return whether Guidance opts into the reserved rationale field."""
    return any(f.get("field_name") == RESERVED_FIELD_NAME for f in fields)


def _validate_rationale_note(
    fields: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[SchemaIssue]]:
    """Validate the reserved ``rationale_note`` field when enabled.

    Presence is the Guidance-level opt-in. If present, the field must remain
    role=aux and type=string and receives the lowest ``display_order``. If
    absent, rationale generation and review are disabled.

    Returns the (possibly reordered) field list and any issues found.
    """
    issues: list[SchemaIssue] = []

    existing_idx: int | None = None
    for i, f in enumerate(fields):
        if f.get("field_name") == RESERVED_FIELD_NAME:
            existing_idx = i
            break

    if existing_idx is not None:
        rn = fields[existing_idx]
        # Per-section path (see SchemaIssue docstring): index of rationale_note
        # among the fields of its own section, in submission order.
        section = "aux" if rn.get("role") == "aux" else "core"
        section_idx = sum(
            1
            for f in fields[:existing_idx]
            if ("aux" if f.get("role") == "aux" else "core") == section
        )
        path = f"{section}[{section_idx}]"
        if rn.get("role") != "aux":
            issues.append(
                SchemaIssue(
                    severity="error",
                    code=RATIONALE_NOTE_WRONG_ROLE,
                    message="`rationale_note` must be an Aux field.",
                    field_path=f"{path}.role",
                )
            )
        if rn.get("type") != "string":
            issues.append(
                SchemaIssue(
                    severity="error",
                    code=RATIONALE_NOTE_WRONG_TYPE,
                    message="`rationale_note` must be type String.",
                    field_path=f"{path}.type",
                )
            )
        _enforce_lowest_display_order(fields, existing_idx)
    return fields, issues


def _enforce_lowest_display_order(fields: list[dict[str, Any]], rn_idx: int) -> None:
    """Ensure ``rationale_note`` has the lowest ``display_order``."""
    rn = fields[rn_idx]
    other_orders = [
        f.get("display_order", 0) for i, f in enumerate(fields) if i != rn_idx
    ]
    if other_orders:
        min_other = min(other_orders)
        current = rn.get("display_order", 0)
        if current >= min_other:
            rn["display_order"] = min_other - 1


def _validate_fields(fields: list[dict[str, Any]]) -> list[SchemaIssue]:
    """Validate all field-level constraints.

    Checks: name, type, per-type constraints, uniqueness, at least one Core.

    ``field_path`` indices are per-section (see SchemaIssue docstring): the
    n-th core field is ``core[n-1]`` regardless of how many aux fields
    precede it in the submitted list, matching the builder UI's row order.
    """
    issues: list[SchemaIssue] = []
    seen_names: dict[str, int] = {}  # name -> first index
    section_counters = {"core": 0, "aux": 0}
    has_core = False

    for idx, f in enumerate(fields):
        field_name = f.get("field_name", "")
        field_type = f.get("type", "")
        role = f.get("role", "core")
        section = "aux" if role == "aux" else "core"
        path = f"{section}[{section_counters[section]}]"
        section_counters[section] += 1

        # -- Name validation --
        if not field_name:
            issues.append(
                SchemaIssue(
                    severity="error",
                    code=MISSING_FIELD_NAME,
                    message="Field name is required.",
                    field_path=f"{path}.name",
                )
            )
        else:
            if len(field_name) > FIELD_NAME_MAX_LEN:
                issues.append(
                    SchemaIssue(
                        severity="error",
                        code=FIELD_NAME_TOO_LONG,
                        message="Field name must be 64 characters or fewer.",
                        field_path=f"{path}.name",
                    )
                )
            elif not FIELD_NAME_RE.match(field_name):
                issues.append(
                    SchemaIssue(
                        severity="error",
                        code=INVALID_FIELD_NAME,
                        message="Use only letters, numbers, and underscores. Must not start with a number.",
                        field_path=f"{path}.name",
                    )
                )

            # Duplicate check (case-sensitive)
            if field_name in seen_names:
                issues.append(
                    SchemaIssue(
                        severity="error",
                        code=DUPLICATE_FIELD_NAME,
                        message=f"Duplicate field name: `{field_name}`.",
                        field_path=f"{path}.name",
                    )
                )
            else:
                seen_names[field_name] = idx

        # -- Type validation --
        if not field_type:
            issues.append(
                SchemaIssue(
                    severity="error",
                    code=MISSING_TYPE,
                    message="Select a type.",
                    field_path=f"{path}.type",
                )
            )
        elif field_type not in VALID_FIELD_TYPES:
            issues.append(
                SchemaIssue(
                    severity="error",
                    code=MISSING_TYPE,
                    message=f"Unsupported type: `{field_type}`. Use one of: enum, enum_set, boolean, integer, string.",
                    field_path=f"{path}.type",
                )
            )
        else:
            issues.extend(_validate_type_constraints(f, field_type, path))

        if role == "core":
            has_core = True

    # At least one Core field required
    if not has_core:
        issues.append(
            SchemaIssue(
                severity="error",
                code=NO_CORE_FIELDS,
                message="Add at least one Core field (required for evaluation).",
            )
        )

    return issues


def _validate_type_constraints(
    f: dict[str, Any], field_type: str, path: str
) -> list[SchemaIssue]:
    """Validate per-type constraints for a single field."""
    issues: list[SchemaIssue] = []

    if field_type in ("enum", "enum_set"):
        allowed_raw: Any = f.get("allowed_values")
        if not allowed_raw or not isinstance(allowed_raw, list):
            issues.append(
                SchemaIssue(
                    severity="error",
                    code=ENUM_TOO_FEW_VALUES,
                    message="Add at least two allowed values.",
                    field_path=f"{path}.allowed_values",
                )
            )
        else:
            allowed = cast("list[Any]", allowed_raw)
            # Check for empty strings
            for i, v in enumerate(allowed):
                if not isinstance(v, str) or not v.strip():
                    issues.append(
                        SchemaIssue(
                            severity="error",
                            code=ENUM_EMPTY_VALUE,
                            message="Allowed values cannot be empty strings.",
                            field_path=f"{path}.allowed_values[{i}]",
                        )
                    )

            # Check count (after acknowledging empties)
            if len(allowed) < 2:
                issues.append(
                    SchemaIssue(
                        severity="error",
                        code=ENUM_TOO_FEW_VALUES,
                        message="Add at least two allowed values.",
                        field_path=f"{path}.allowed_values",
                    )
                )

            # Check uniqueness after trim
            trimmed: list[str] = [v.strip() for v in allowed if isinstance(v, str)]
            seen: set[str] = set()
            for v in trimmed:
                if v in seen:
                    issues.append(
                        SchemaIssue(
                            severity="error",
                            code=ENUM_DUPLICATE_VALUE,
                            message=f"Duplicate value: `{v}`.",
                            field_path=f"{path}.allowed_values",
                        )
                    )
                    break  # One duplicate error per field is sufficient
                seen.add(v)

    elif field_type == "integer":
        minimum = f.get("minimum")
        maximum = f.get("maximum")
        if minimum is not None and maximum is not None and minimum > maximum:
            issues.append(
                SchemaIssue(
                    severity="error",
                    code=MIN_EXCEEDS_MAX,
                    message="Min must be ≤ Max.",
                    field_path=f"{path}.minimum",
                )
            )

    elif field_type == "string":
        # The reserved rationale_note carries no length constraints, so this
        # check naturally no-ops for it.
        min_len = f.get("min_length")
        max_len = f.get("max_length")
        if min_len is not None and max_len is not None and min_len > max_len:
            issues.append(
                SchemaIssue(
                    severity="error",
                    code=MINLENGTH_EXCEEDS_MAXLENGTH,
                    message="minLength must be ≤ maxLength.",
                    field_path=f"{path}.min_length",
                )
            )

    return issues


def _derive_generation_order(fields: list[dict[str, Any]]) -> list[str]:
    """Derive the canonical generation order.

    Order: optional rationale_note → remaining Aux by display_order → Core by
    display_order.
    """
    rationale: list[str] = []
    aux: list[
        tuple[int, str, str]
    ] = []  # (display_order, field_name, field_id) for stable sort
    core: list[tuple[int, str, str]] = []

    for f in fields:
        name = f["field_name"]
        order = f.get("display_order", 0)
        fid = f.get("field_id", "")
        role = f.get("role", "core")

        if name == RESERVED_FIELD_NAME:
            rationale.append(name)
        elif role == "aux":
            aux.append((order, name, fid))
        else:
            core.append((order, name, fid))

    # Sort by display_order, then field_name for deterministic tie-breaking
    aux.sort(key=lambda t: (t[0], t[1]))
    core.sort(key=lambda t: (t[0], t[1]))

    return rationale + [t[1] for t in aux] + [t[1] for t in core]


def _derive_json_schema(
    fields: list[dict[str, Any]], generation_order: list[str]
) -> dict[str, Any]:
    """Derive the canonical JSON Schema from SchemaCore.

    Properties are ordered by ``generation_order``.  ``required`` contains
    only Core field names.  Includes ``x-generation-order`` extension.
    """
    field_map: dict[str, dict[str, Any]] = {f["field_name"]: f for f in fields}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name in generation_order:
        f = field_map[name]
        prop = _field_to_json_schema_property(f)
        properties[name] = prop

        if f.get("role") == "core":
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
        "x-generation-order": generation_order,
    }
    return schema


def _field_to_json_schema_property(f: dict[str, Any]) -> dict[str, Any]:
    """Map a single SchemaCore field to its JSON Schema property."""
    ft = f["type"]

    if ft == "boolean":
        return {"type": "boolean"}

    if ft == "integer":
        prop: dict[str, Any] = {"type": "integer"}
        if f.get("minimum") is not None:
            prop["minimum"] = f["minimum"]
        if f.get("maximum") is not None:
            prop["maximum"] = f["maximum"]
        return prop

    if ft == "string":
        prop = {"type": "string"}
        if f.get("min_length") is not None:
            prop["minLength"] = f["min_length"]
        if f.get("max_length") is not None:
            prop["maxLength"] = f["max_length"]
        return prop

    if ft == "enum":
        return {"type": "string", "enum": list(f.get("allowed_values", []))}

    if ft == "enum_set":
        return {
            "type": "array",
            "items": {"type": "string", "enum": list(f.get("allowed_values", []))},
            "uniqueItems": True,
        }

    # Should not reach here — type validated upstream
    return {"type": "string"}


#: JSON Schema keywords the NVIDIA-hosted NIM guided-decoding grammar backend
#: does not implement. Present inside a ``response_format`` json_schema, any of
#: these fails the WHOLE Teacher call with
#: ``400 Grammar error: Unimplemented keys: [...]`` — measured on
#: stepfun-ai/step-3.7-flash (the then-seeded default Teacher) 2026-07-22
#: against a project whose schema had an ``enum_set`` field. ``uniqueItems`` is emitted by
#: every ``enum_set`` field but carries no validation weight (enum_set output is
#: deduped in the exact-match normalizer, and nothing here runs jsonschema over
#: the derived schema), so stripping it for the grammar request is loss-free.
_GUIDED_DECODING_UNSUPPORTED_KEYS: Final[frozenset[str]] = frozenset({"uniqueItems"})


def strip_guided_decoding_unsupported_keys(schema: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy ``schema`` with grammar-unsupported keywords removed.

    The derived schema is the full JSON-Schema semantic contract; the schema
    sent to guided decoding is constrained to what the NIM grammar backend can
    compile. This bridges the two so an ``enum_set`` field (or any future
    constraint the backend can't grammar-compile) doesn't 400 the Teacher call.
    Recurses through nested ``properties``/``items``. See
    :data:`_GUIDED_DECODING_UNSUPPORTED_KEYS`.
    """
    return cast("dict[str, Any]", _strip_unsupported_keys(schema))


def _strip_unsupported_keys(node: Any) -> Any:
    if isinstance(node, dict):
        items = cast("dict[str, Any]", node)
        return {
            k: _strip_unsupported_keys(v)
            for k, v in items.items()
            if k not in _GUIDED_DECODING_UNSUPPORTED_KEYS
        }
    if isinstance(node, list):
        elems = cast("list[Any]", node)
        return [_strip_unsupported_keys(v) for v in elems]
    return node


def _compute_schema_hash(derived_json_schema: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of the derived JSON Schema."""
    canonical = json.dumps(derived_json_schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def place_rationale_last(
    schema: dict[str, Any],
    generation_order: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Return the production output schema with ``rationale_note`` last."""
    properties = schema.get("properties", {})
    has_rationale = RESERVED_FIELD_NAME in properties or (
        RESERVED_FIELD_NAME in generation_order
    )
    if not has_rationale:
        return schema, generation_order

    new_order = [n for n in generation_order if n != RESERVED_FIELD_NAME]
    new_properties = {n: v for n, v in properties.items() if n != RESERVED_FIELD_NAME}
    new_order.append(RESERVED_FIELD_NAME)
    if RESERVED_FIELD_NAME in properties:
        new_properties[RESERVED_FIELD_NAME] = properties[RESERVED_FIELD_NAME]

    new_schema = dict(schema)
    new_schema["properties"] = new_properties
    new_schema["x-generation-order"] = new_order

    return new_schema, new_order

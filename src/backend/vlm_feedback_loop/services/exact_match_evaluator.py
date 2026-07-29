# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Exact Match evaluator.

Single backend-canonical implementation of label correctness logic.
Consumed by:
  - Proposal validation
  - Evaluation scoring
  - Batch Labeling validation
  - TAO per-sample re-scoring

Per-type normalization, per-field matching, and aggregate derived metrics.
All functions are synchronous, stateless, deterministic.  No database or
I/O dependencies — only imports ``RESERVED_FIELD_NAME`` from schema_core.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, cast

# Markdown code-fence opening: ```json / ```JSON / ``` (optional language tag),
# followed by an end-of-line. Matches the common LLM pattern where a model
# wraps its JSON output in a fenced block despite being asked for raw JSON.
_OPEN_FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_\-]*\s*\n")

# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class FieldValidationResult:
    """Result of validating and normalizing a single field value."""

    field_name: str
    field_type: str  # enum | enum_set | boolean | integer | string
    role: str  # core | aux
    valid: bool
    normalized_value: Any  # normalized value, or None if invalid
    error: str | None  # human-readable error if invalid
    normalization_steps: list[str] = field(default_factory=list[str])


@dataclass
class SchemaValidationReport:
    """Full validation report for a model response against SchemaCore."""

    schema_valid_core: bool
    core_errors: list[str]
    aux_errors: list[str]
    field_results: list[FieldValidationResult]
    normalized_json: dict[str, Any] | None  # None on parse failure
    parse_error: str | None
    truncation_attributed_schema_invalid: bool


@dataclass
class FieldMatchResult:
    """Per-field comparison between predicted and ground-truth values."""

    field_name: str
    matched: bool
    predicted: Any
    ground_truth: Any


@dataclass
class PerValueMetrics:
    """Precision, recall, F1 for a single categorical value."""

    precision: float
    recall: float
    f1: float


@dataclass
class AggregateMetrics:
    """Aggregate accuracy metrics across a set of examples."""

    overall_exact_match_rate: float
    example_count: int
    per_core_field_match_rate: dict[str, float]
    per_value_metrics: dict[str, dict[str, PerValueMetrics]]
    # Macro-F1 = unweighted mean of per-value F1. ``per_field_macro_f1``
    # averages across a field's values; ``overall_macro_f1`` averages across
    # categorical core fields. Macro-averaging exposes minority-class collapse
    # that Exact Match and the majority-weighted field match-rate both hide —
    # the load-bearing metric for the ICL diagnostic. Defaulted so the
    # zero-example early return and any external constructor stay valid.
    per_field_macro_f1: dict[str, float] = field(default_factory=dict[str, float])
    overall_macro_f1: float = 0.0


# ── Per-type normalizers ─────────────────────────────────────────────────────


def _normalize_enum(
    raw_value: Any,
    field_def: dict[str, Any],
) -> FieldValidationResult:
    """Normalize an enum field: trim/lowercase, match against allowed_values."""
    name = field_def["field_name"]
    role = field_def.get("role", "core")
    steps: list[str] = []
    allowed = field_def.get("allowed_values", [])

    if raw_value is None:
        return FieldValidationResult(
            field_name=name,
            field_type="enum",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}' is missing (null)",
            normalization_steps=["value is None"],
        )

    if not isinstance(raw_value, str):
        return FieldValidationResult(
            field_name=name,
            field_type="enum",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}': expected string, got {type(raw_value).__name__}",
            normalization_steps=[
                f"rejected non-string type {type(raw_value).__name__}"
            ],
        )

    trimmed = raw_value.strip()
    if trimmed != raw_value:
        steps.append(f"trimmed whitespace: '{raw_value}' → '{trimmed}'")
    lowered = trimmed.lower()
    if lowered != trimmed:
        steps.append(f"lowercased: '{trimmed}' → '{lowered}'")

    # Match against allowed_values (case-insensitive)
    canonical_map = {v.strip().lower(): v for v in allowed}
    if lowered in canonical_map:
        canonical = canonical_map[lowered]
        if canonical != raw_value:
            steps.append(f"matched canonical value: '{canonical}'")
        return FieldValidationResult(
            field_name=name,
            field_type="enum",
            role=role,
            valid=True,
            normalized_value=canonical,
            error=None,
            normalization_steps=steps,
        )

    return FieldValidationResult(
        field_name=name,
        field_type="enum",
        role=role,
        valid=False,
        normalized_value=None,
        error=f"Field '{name}': value '{raw_value}' not in allowed values {allowed}",
        normalization_steps=steps + [f"no match in allowed values: {allowed}"],
    )


def _normalize_enum_set(
    raw_value: Any,
    field_def: dict[str, Any],
) -> FieldValidationResult:
    """Normalize an enum_set field: normalize each element, dedupe, sort."""
    name = field_def["field_name"]
    role = field_def.get("role", "core")
    steps: list[str] = []
    allowed = field_def.get("allowed_values", [])

    if raw_value is None:
        return FieldValidationResult(
            field_name=name,
            field_type="enum_set",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}' is missing (null)",
            normalization_steps=["value is None"],
        )

    if not isinstance(raw_value, list):
        return FieldValidationResult(
            field_name=name,
            field_type="enum_set",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}': expected array, got {type(raw_value).__name__}",
            normalization_steps=[f"rejected non-list type {type(raw_value).__name__}"],
        )

    canonical_map = {v.strip().lower(): v for v in allowed}
    normalized: list[str] = []
    errors: list[str] = []

    raw_list = cast("list[Any]", raw_value)
    for i, elem in enumerate(raw_list):
        if not isinstance(elem, str):
            errors.append(f"element [{i}]: expected string, got {type(elem).__name__}")
            continue
        trimmed = elem.strip()
        lowered = trimmed.lower()
        if lowered in canonical_map:
            normalized.append(canonical_map[lowered])
        else:
            errors.append(f"element '{elem}' not in allowed values")

    if errors:
        return FieldValidationResult(
            field_name=name,
            field_type="enum_set",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}': {'; '.join(errors)}",
            normalization_steps=steps + errors,
        )

    # Deduplicate and sort
    before_dedup = len(normalized)
    deduped = sorted(set(normalized))
    if len(deduped) < before_dedup:
        steps.append(f"deduplicated: {before_dedup} → {len(deduped)} elements")
    steps.append(f"sorted: {deduped}")

    return FieldValidationResult(
        field_name=name,
        field_type="enum_set",
        role=role,
        valid=True,
        normalized_value=deduped,
        error=None,
        normalization_steps=steps,
    )


def _normalize_boolean(
    raw_value: Any,
    field_def: dict[str, Any],
) -> FieldValidationResult:
    """Normalize a boolean field: ONLY JSON true/false accepted.

    String proxies ("true", "false", "yes", "no") and numeric proxies
    (1, 0) are schema-invalid by design.  This is the single
    backend-canonical boolean normalizer.
    """
    name = field_def["field_name"]
    role = field_def.get("role", "core")

    if raw_value is None:
        return FieldValidationResult(
            field_name=name,
            field_type="boolean",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}' is missing (null)",
            normalization_steps=["value is None"],
        )

    # Strict type check: must be exactly bool, not int subclass trickery
    # In Python, bool is a subclass of int, so isinstance(True, int) is True.
    # We check bool FIRST to distinguish True/False from 1/0.
    if isinstance(raw_value, bool):
        return FieldValidationResult(
            field_name=name,
            field_type="boolean",
            role=role,
            valid=True,
            normalized_value=raw_value,
            error=None,
            normalization_steps=["accepted JSON boolean"],
        )

    # Reject integer proxies (1, 0) — they are NOT canonical booleans
    if isinstance(raw_value, int):
        return FieldValidationResult(
            field_name=name,
            field_type="boolean",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}': numeric proxy {raw_value} is not a valid boolean; use JSON true/false",
            normalization_steps=[f"rejected numeric proxy: {raw_value}"],
        )

    # Reject string proxies ("true", "false", "yes", "no")
    if isinstance(raw_value, str):
        return FieldValidationResult(
            field_name=name,
            field_type="boolean",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}': string '{raw_value}' is not a valid boolean; use JSON true/false",
            normalization_steps=[f"rejected string proxy: '{raw_value}'"],
        )

    return FieldValidationResult(
        field_name=name,
        field_type="boolean",
        role=role,
        valid=False,
        normalized_value=None,
        error=f"Field '{name}': expected boolean, got {type(raw_value).__name__}",
        normalization_steps=[f"rejected type {type(raw_value).__name__}"],
    )


def _normalize_integer(
    raw_value: Any,
    field_def: dict[str, Any],
) -> FieldValidationResult:
    """Normalize an integer field: validate type and range.

    Accepts int directly.  Accepts float only if whole number (3.0 → 3).
    Out-of-range is invalid (not clamped).
    """
    name = field_def["field_name"]
    role = field_def.get("role", "core")
    steps: list[str] = []

    if raw_value is None:
        return FieldValidationResult(
            field_name=name,
            field_type="integer",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}' is missing (null)",
            normalization_steps=["value is None"],
        )

    # bool is subclass of int in Python — reject it
    if isinstance(raw_value, bool):
        return FieldValidationResult(
            field_name=name,
            field_type="integer",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}': boolean is not a valid integer",
            normalization_steps=["rejected boolean as integer"],
        )

    int_value: int | None = None

    if isinstance(raw_value, int):
        int_value = raw_value
    elif isinstance(raw_value, float):
        # Pyright flags float == int as always-False, but Python's `__eq__` returns True
        # when the float represents a whole number (e.g., 3.0 == 3) — needed for whole-number floats.
        if math.isfinite(raw_value) and raw_value == int(raw_value):  # pyright: ignore[reportUnnecessaryComparison]
            int_value = int(raw_value)
            steps.append(f"converted float {raw_value} → int {int_value}")
        else:
            return FieldValidationResult(
                field_name=name,
                field_type="integer",
                role=role,
                valid=False,
                normalized_value=None,
                error=f"Field '{name}': float {raw_value} is not a whole number",
                normalization_steps=[f"rejected non-whole float: {raw_value}"],
            )
    else:
        return FieldValidationResult(
            field_name=name,
            field_type="integer",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}': expected integer, got {type(raw_value).__name__}",
            normalization_steps=[f"rejected type {type(raw_value).__name__}"],
        )

    # Validate range (out-of-range = invalid, not clamped)
    minimum = field_def.get("minimum")
    maximum = field_def.get("maximum")

    if minimum is not None and int_value < minimum:
        return FieldValidationResult(
            field_name=name,
            field_type="integer",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}': value {int_value} below minimum {minimum}",
            normalization_steps=steps + [f"out of range: {int_value} < {minimum}"],
        )

    if maximum is not None and int_value > maximum:
        return FieldValidationResult(
            field_name=name,
            field_type="integer",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}': value {int_value} above maximum {maximum}",
            normalization_steps=steps + [f"out of range: {int_value} > {maximum}"],
        )

    return FieldValidationResult(
        field_name=name,
        field_type="integer",
        role=role,
        valid=True,
        normalized_value=int_value,
        error=None,
        normalization_steps=steps,
    )


def _normalize_string(
    raw_value: Any,
    field_def: dict[str, Any],
) -> FieldValidationResult:
    """Normalize a string field: trim, validate length constraints."""
    name = field_def["field_name"]
    role = field_def.get("role", "core")
    steps: list[str] = []

    if raw_value is None:
        return FieldValidationResult(
            field_name=name,
            field_type="string",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}' is missing (null)",
            normalization_steps=["value is None"],
        )

    if not isinstance(raw_value, str):
        return FieldValidationResult(
            field_name=name,
            field_type="string",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}': expected string, got {type(raw_value).__name__}",
            normalization_steps=[f"rejected type {type(raw_value).__name__}"],
        )

    trimmed = raw_value.strip()
    if trimmed != raw_value:
        steps.append(f"trimmed whitespace: len {len(raw_value)} → {len(trimmed)}")

    min_length = field_def.get("min_length")
    max_length = field_def.get("max_length")

    if min_length is not None and len(trimmed) < min_length:
        return FieldValidationResult(
            field_name=name,
            field_type="string",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}': length {len(trimmed)} below minLength {min_length}",
            normalization_steps=steps + [f"too short: {len(trimmed)} < {min_length}"],
        )

    if max_length is not None and len(trimmed) > max_length:
        return FieldValidationResult(
            field_name=name,
            field_type="string",
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Field '{name}': length {len(trimmed)} above maxLength {max_length}",
            normalization_steps=steps + [f"too long: {len(trimmed)} > {max_length}"],
        )

    return FieldValidationResult(
        field_name=name,
        field_type="string",
        role=role,
        valid=True,
        normalized_value=trimmed,
        error=None,
        normalization_steps=steps,
    )


# ── Dispatcher ───────────────────────────────────────────────────────────────

_NORMALIZERS = {
    "enum": _normalize_enum,
    "enum_set": _normalize_enum_set,
    "boolean": _normalize_boolean,
    "integer": _normalize_integer,
    "string": _normalize_string,
}


def normalize_ground_truth(
    label_json: dict[str, Any],
    fields: list[dict[str, Any]],
) -> dict[str, Any]:
    """Best-effort canonicalization of a stored Verified label.

    Labels saved before the save-time normalization boundary (or written
    by external tools) may hold un-normalized values, while
    :func:`match_fields` assumes both sides are canonical. Values that
    normalize cleanly are canonicalized; values that fail are left as
    stored — a legacy quirk is never made worse at read time.
    """
    out = dict(label_json)
    for fdef in fields:
        fname = fdef.get("field_name")
        if not fname or fname not in out:
            continue
        res = normalize_field_value(out[fname], fdef)
        if res.valid:
            out[fname] = res.normalized_value
    return out


def normalize_field_value(
    raw_value: Any,
    field_def: dict[str, Any],
) -> FieldValidationResult:
    """Normalize and validate a single field value against its SchemaCore definition."""
    ftype = field_def.get("type", "string")
    normalizer = _NORMALIZERS.get(ftype)
    if normalizer is None:
        name = field_def.get("field_name", "unknown")
        role = field_def.get("role", "core")
        return FieldValidationResult(
            field_name=name,
            field_type=ftype,
            role=role,
            valid=False,
            normalized_value=None,
            error=f"Unknown field type '{ftype}'",
            normalization_steps=[f"unknown type: {ftype}"],
        )
    return normalizer(raw_value, field_def)


# ── Proposal validation ─────────────────────────────────────────────────────


def strip_code_fence(text: str) -> str:
    """Strip a surrounding Markdown code fence if present.

    Handles the two common LLM-output shapes where the model wraps JSON in
    a fenced block despite being asked for raw JSON:

        ```json
        {"foo": 1}
        ```

        ```
        {"foo": 1}
        ```

    Only strips when BOTH an opening fence (with a newline after the
    optional language tag) AND a trailing ``` are present. If either is
    missing, returns the input unchanged so a legitimately malformed
    response still surfaces as a parse error rather than being silently
    altered. A valid JSON document with no fence is unaffected.

    The canonical JSON fence stripper — proposal validation and TAO
    rescoring both route through it. (ai_assist_client's rationale
    cleaner is deliberately looser: it cleans PROSE, not JSON, and must
    tolerate missing closing fences.)
    """
    open_match = _OPEN_FENCE_RE.match(text)
    if open_match is None:
        return text
    body = text[open_match.end() :]
    # Closing fence must appear at the end, ignoring trailing whitespace.
    rstripped = body.rstrip()
    if not rstripped.endswith("```"):
        return text
    return rstripped[:-3].rstrip()


def validate_proposal(
    raw_content: str | None,
    fields: list[dict[str, Any]],
    finish_reason: str | None = None,
) -> SchemaValidationReport:
    """Validate and normalize a model response against SchemaCore.

    Parameters
    ----------
    raw_content:
        The raw string content from the model response.  May be None on
        timeout/endpoint error.
    fields:
        The SchemaCore field definitions from the Guidance envelope.
    finish_reason:
        The ``finish_reason`` from the model response (e.g. ``"stop"``,
        ``"length"``).

    Returns
    -------
    SchemaValidationReport with separated Core/Aux errors, normalized JSON,
    and truncation attribution.
    """
    # Handle None/empty content (timeout, endpoint error, etc.)
    if raw_content is None or raw_content.strip() == "":
        core_fields = [f for f in fields if f.get("role") == "core"]
        core_errors = [
            f"Field '{f['field_name']}': no model response to validate"
            for f in core_fields
        ]
        return SchemaValidationReport(
            schema_valid_core=False,
            core_errors=core_errors,
            aux_errors=[],
            field_results=[],
            normalized_json=None,
            parse_error="No model response content",
            truncation_attributed_schema_invalid=(finish_reason == "length"),
        )

    # Parse JSON. Strip a surrounding Markdown code fence first — when a
    # Teacher is running in prompt-only mode (no `response_format=json_schema`
    # probe confirmed yet) some models wrap their output in ```json … ```
    # despite being instructed to emit raw JSON. Stripping recovers the
    # common case; if the fence is malformed or absent, the text passes
    # through unchanged and the error surfaces normally.
    content_to_parse = strip_code_fence(raw_content)
    try:
        parsed = json.loads(content_to_parse)
    except (json.JSONDecodeError, TypeError) as exc:
        core_fields = [f for f in fields if f.get("role") == "core"]
        core_errors = [
            f"Field '{f['field_name']}': JSON parse failure" for f in core_fields
        ]
        return SchemaValidationReport(
            schema_valid_core=False,
            core_errors=core_errors,
            aux_errors=[],
            field_results=[],
            normalized_json=None,
            parse_error=f"Invalid JSON: {exc}",
            truncation_attributed_schema_invalid=(finish_reason == "length"),
        )

    if not isinstance(parsed, dict):
        core_fields = [f for f in fields if f.get("role") == "core"]
        core_errors = [
            f"Field '{f['field_name']}': response is not a JSON object"
            for f in core_fields
        ]
        return SchemaValidationReport(
            schema_valid_core=False,
            core_errors=core_errors,
            aux_errors=[],
            field_results=[],
            normalized_json=None,
            parse_error=f"Expected JSON object, got {type(parsed).__name__}",
            truncation_attributed_schema_invalid=(finish_reason == "length"),
        )

    parsed_dict = cast("dict[str, Any]", parsed)
    # Validate each field
    field_map = {f["field_name"]: f for f in fields}
    field_results: list[FieldValidationResult] = []
    core_errors: list[str] = []
    aux_errors: list[str] = []
    normalized: dict[str, Any] = {}

    for f in fields:
        fname = f["field_name"]
        role = f.get("role", "core")
        raw: Any = parsed_dict.get(fname)

        if fname not in parsed_dict:
            # Missing field
            if role == "core":
                error = f"Field '{fname}': required Core field is missing"
                core_errors.append(error)
                field_results.append(
                    FieldValidationResult(
                        field_name=fname,
                        field_type=f["type"],
                        role=role,
                        valid=False,
                        normalized_value=None,
                        error=error,
                        normalization_steps=["missing from response"],
                    )
                )
            else:
                warning = f"Field '{fname}': optional Aux field is missing"
                aux_errors.append(warning)
                field_results.append(
                    FieldValidationResult(
                        field_name=fname,
                        field_type=f["type"],
                        role=role,
                        valid=True,
                        normalized_value=None,
                        error=None,
                        normalization_steps=["missing aux field (acceptable)"],
                    )
                )
            continue

        result = normalize_field_value(raw, f)
        field_results.append(result)

        if result.valid:
            normalized[fname] = result.normalized_value
        elif role == "core":
            core_errors.append(result.error or f"Field '{fname}': invalid")
        else:
            aux_errors.append(result.error or f"Field '{fname}': invalid (aux)")

    # Warn about extra fields not in schema
    schema_names = set(field_map.keys())
    for key in parsed_dict:
        if key not in schema_names:
            aux_errors.append(f"Extra field '{key}' not in schema")

    schema_valid_core = len(core_errors) == 0
    truncation_attributed = finish_reason == "length" and not schema_valid_core

    return SchemaValidationReport(
        schema_valid_core=schema_valid_core,
        core_errors=core_errors,
        aux_errors=aux_errors,
        field_results=field_results,
        normalized_json=normalized if schema_valid_core else None,
        parse_error=None,
        truncation_attributed_schema_invalid=truncation_attributed,
    )


# ── Field matching (for evaluation) ─────────────────────────────────────────


def match_fields(
    predicted_json: dict[str, Any],
    ground_truth_json: dict[str, Any],
    fields: list[dict[str, Any]],
) -> list[FieldMatchResult]:
    """Compare predicted vs ground-truth label on Core fields.

    Both inputs should already be normalized.  Only Core fields are
    compared; Aux fields are ignored for evaluation.
    """
    results: list[FieldMatchResult] = []

    for f in fields:
        if f.get("role") != "core":
            continue
        fname = f["field_name"]
        predicted = predicted_json.get(fname)
        truth = ground_truth_json.get(fname)

        if f["type"] == "enum_set" and predicted is not None and truth is not None:
            # Set equality (order-independent). Guarded on both values being
            # present: a missing prediction (the schema-invalid sentinel
            # ``predicted_json={}``) must stay a miss even against an empty
            # ground-truth set — coercing None to [] made garbage responses
            # exact-match legitimate "no defects" labels.
            matched = set(predicted) == set(truth)
        else:
            matched = predicted == truth

        results.append(
            FieldMatchResult(
                field_name=fname,
                matched=matched,
                predicted=predicted,
                ground_truth=truth,
            )
        )

    return results


# ── Aggregate metrics ────────────────────────────────────────────────────────


def compute_aggregate_metrics(
    all_results: list[list[FieldMatchResult]],
    core_fields: list[dict[str, Any]],
) -> AggregateMetrics:
    """Compute aggregate accuracy metrics across examples.

    Parameters
    ----------
    all_results:
        One ``list[FieldMatchResult]`` per example (from ``match_fields``).
    core_fields:
        Core field definitions (for type information).
    """
    n = len(all_results)
    if n == 0:
        return AggregateMetrics(
            overall_exact_match_rate=0.0,
            example_count=0,
            per_core_field_match_rate={},
            per_value_metrics={},
        )

    # Overall Exact Match: fraction where ALL core fields match
    exact_matches = sum(1 for results in all_results if all(r.matched for r in results))
    overall_rate = exact_matches / n

    # Per-core-field match rate
    field_match_counts: dict[str, int] = defaultdict(int)
    field_total_counts: dict[str, int] = defaultdict(int)

    for results in all_results:
        for r in results:
            field_total_counts[r.field_name] += 1
            if r.matched:
                field_match_counts[r.field_name] += 1

    per_field_rate: dict[str, float] = {}
    for fname, total in field_total_counts.items():
        per_field_rate[fname] = field_match_counts[fname] / total if total > 0 else 0.0

    # Per-value metrics for categorical Core fields (enum, enum_set, boolean)
    per_value_metrics: dict[str, dict[str, PerValueMetrics]] = {}

    categorical_types = {"enum", "enum_set", "boolean"}

    for f in core_fields:
        ftype = f["type"]
        fname = f["field_name"]
        if ftype not in categorical_types:
            continue

        # Collect all values for this field
        if ftype == "boolean":
            all_values = ["true", "false"]
        elif ftype in ("enum", "enum_set"):
            all_values = [v.strip().lower() for v in f.get("allowed_values", [])]
        else:
            continue

        # Count TP, FP, FN per value
        tp: dict[str, int] = defaultdict(int)
        fp: dict[str, int] = defaultdict(int)
        fn: dict[str, int] = defaultdict(int)

        for results in all_results:
            for r in results:
                if r.field_name != fname:
                    continue

                if ftype == "enum_set":
                    pred_set = {
                        str(v).strip().lower()
                        for v in cast("list[Any]", r.predicted or [])
                    }
                    truth_set = {
                        str(v).strip().lower()
                        for v in cast("list[Any]", r.ground_truth or [])
                    }
                    for val in all_values:
                        in_pred = val in pred_set
                        in_truth = val in truth_set
                        if in_pred and in_truth:
                            tp[val] += 1
                        elif in_pred and not in_truth:
                            fp[val] += 1
                        elif not in_pred and in_truth:
                            fn[val] += 1
                elif ftype == "boolean":
                    pred_str = (
                        str(r.predicted).lower() if r.predicted is not None else ""
                    )
                    truth_str = (
                        str(r.ground_truth).lower()
                        if r.ground_truth is not None
                        else ""
                    )
                    for val in all_values:
                        in_pred = pred_str == val
                        in_truth = truth_str == val
                        if in_pred and in_truth:
                            tp[val] += 1
                        elif in_pred and not in_truth:
                            fp[val] += 1
                        elif not in_pred and in_truth:
                            fn[val] += 1
                else:  # enum
                    pred_norm = (
                        str(r.predicted).strip().lower()
                        if r.predicted is not None
                        else ""
                    )
                    truth_norm = (
                        str(r.ground_truth).strip().lower()
                        if r.ground_truth is not None
                        else ""
                    )
                    for val in all_values:
                        in_pred = pred_norm == val
                        in_truth = truth_norm == val
                        if in_pred and in_truth:
                            tp[val] += 1
                        elif in_pred and not in_truth:
                            fp[val] += 1
                        elif not in_pred and in_truth:
                            fn[val] += 1

        value_metrics: dict[str, PerValueMetrics] = {}
        for val in all_values:
            precision = (
                tp[val] / (tp[val] + fp[val]) if (tp[val] + fp[val]) > 0 else 0.0
            )
            recall = tp[val] / (tp[val] + fn[val]) if (tp[val] + fn[val]) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            # Use canonical value (original case) for display
            display_val = val
            if ftype in ("enum", "enum_set"):
                for orig in f.get("allowed_values", []):
                    if orig.strip().lower() == val:
                        display_val = orig
                        break
            value_metrics[display_val] = PerValueMetrics(
                precision=precision,
                recall=recall,
                f1=f1,
            )
        per_value_metrics[fname] = value_metrics

    # Macro-F1: per field = unweighted mean of its per-value F1; overall =
    # mean across categorical core fields. Unweighted on purpose so a
    # collapsed minority class drags the score down regardless of base rate.
    per_field_macro_f1: dict[str, float] = {}
    for fname, value_metrics in per_value_metrics.items():
        per_field_macro_f1[fname] = (
            sum(m.f1 for m in value_metrics.values()) / len(value_metrics)
            if value_metrics
            else 0.0
        )
    overall_macro_f1 = (
        sum(per_field_macro_f1.values()) / len(per_field_macro_f1)
        if per_field_macro_f1
        else 0.0
    )

    return AggregateMetrics(
        overall_exact_match_rate=overall_rate,
        example_count=n,
        per_core_field_match_rate=per_field_rate,
        per_value_metrics=per_value_metrics,
        per_field_macro_f1=per_field_macro_f1,
        overall_macro_f1=overall_macro_f1,
    )

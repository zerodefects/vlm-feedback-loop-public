// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Test-double emulation of ``POST /v1/projects/{id}/guidance:validate_draft``.
 *
 * The guidance builder is backend-driven: the AUTHORITATIVE validation rules
 * live only in the backend (``services/schema_core.validate_and_derive``,
 * pinned by ``tests/unit/test_schema_core.py``). This fake reproduces the
 * response contract — issue codes, messages, and per-section ``field_path``
 * addressing (``core[i].name`` / ``aux[j].allowed_values``) — for the draft
 * states the page tests exercise, so they can run without a live server.
 * It is a wire mock, not product logic; if the backend contract changes,
 * update this fake alongside the backend tests.
 */

import type {
  DraftValidationRequest,
  DraftValidationResponse,
  SchemaIssueResponse,
} from "@/types/guidance";

const FIELD_NAME_RE = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
const FIELD_NAME_MAX_LEN = 64;

export function fakeValidateDraft(
  body: DraftValidationRequest,
): DraftValidationResponse {
  const issues: SchemaIssueResponse[] = [];
  const fields = body.schema;

  if (!fields.some((f) => f.role === "core")) {
    issues.push({
      severity: "error",
      code: "NO_CORE_FIELDS",
      message: "Add at least one Core field (required for evaluation).",
    });
  }

  const counters = { core: 0, aux: 0 };
  const seen = new Set<string>();
  for (const f of fields) {
    const section = f.role === "aux" ? "aux" : "core";
    const path = `${section}[${counters[section]}]`;
    counters[section] += 1;

    const name = f.field_name;
    if (!name) {
      issues.push({
        severity: "error",
        code: "MISSING_FIELD_NAME",
        message: "Field name is required.",
        field_path: `${path}.name`,
      });
    } else {
      if (name.length > FIELD_NAME_MAX_LEN) {
        issues.push({
          severity: "error",
          code: "FIELD_NAME_TOO_LONG",
          message: "Field name must be 64 characters or fewer.",
          field_path: `${path}.name`,
        });
      } else if (!FIELD_NAME_RE.test(name)) {
        issues.push({
          severity: "error",
          code: "INVALID_FIELD_NAME",
          message:
            "Use only letters, numbers, and underscores. Must not start with a number.",
          field_path: `${path}.name`,
        });
      }
      if (seen.has(name)) {
        issues.push({
          severity: "error",
          code: "DUPLICATE_FIELD_NAME",
          message: `Duplicate field name: \`${name}\`.`,
          field_path: `${path}.name`,
        });
      } else {
        seen.add(name);
      }
    }

    if (f.type === "enum" || f.type === "enum_set") {
      const vals = f.allowed_values ?? [];
      if (vals.length < 2) {
        issues.push({
          severity: "error",
          code: "ENUM_TOO_FEW_VALUES",
          message: "Add at least two allowed values.",
          field_path: `${path}.allowed_values`,
        });
      }
      vals.forEach((v, i) => {
        if (!v.trim()) {
          issues.push({
            severity: "error",
            code: "ENUM_EMPTY_VALUE",
            message: "Allowed values cannot be empty strings.",
            field_path: `${path}.allowed_values[${i}]`,
          });
        }
      });
      const trimmed = vals.map((v) => v.trim());
      const dup = trimmed.find((v, i) => trimmed.indexOf(v) !== i);
      if (dup !== undefined) {
        issues.push({
          severity: "error",
          code: "ENUM_DUPLICATE_VALUE",
          message: `Duplicate value: \`${dup}\`.`,
          field_path: `${path}.allowed_values`,
        });
      }
    }

    if (
      f.type === "integer" &&
      f.minimum != null &&
      f.maximum != null &&
      f.minimum > f.maximum
    ) {
      issues.push({
        severity: "error",
        code: "MIN_EXCEEDS_MAX",
        message: "Min must be ≤ Max.",
        field_path: `${path}.minimum`,
      });
    }
    if (
      f.type === "string" &&
      f.min_length != null &&
      f.max_length != null &&
      f.min_length > f.max_length
    ) {
      issues.push({
        severity: "error",
        code: "MINLENGTH_EXCEEDS_MAXLENGTH",
        message: "minLength must be ≤ maxLength.",
        field_path: `${path}.min_length`,
      });
    }
  }

  const saveAllowed = issues.length === 0;
  return {
    issues,
    derived_json_schema: saveAllowed
      ? {
          type: "object",
          properties: Object.fromEntries(fields.map((f) => [f.field_name, {}])),
          required: fields.filter((f) => f.role === "core").map((f) => f.field_name),
        }
      : null,
    schema_hash: saveAllowed ? "fake-hash" : null,
    save_allowed: saveAllowed,
  };
}

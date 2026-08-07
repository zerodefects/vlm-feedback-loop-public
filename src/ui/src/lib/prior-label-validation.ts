// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SchemaFieldResponse } from "@/types/guidance";

/** Normalize an enum token the way the backend does: `strip().lower()`. */
function normalizeEnumToken(value: string): string {
  return value.trim().toLowerCase();
}

/**
 * Check if a prior value is valid under the current field schema.
 *
 * Mirrors exact_match_evaluator per type: enum/enum_set match allowed
 * values case-insensitively after trimming; booleans are strict JSON
 * booleans (no "true"/1 proxies); integers must be whole numbers within
 * [minimum, maximum]; strings are trimmed then length-checked.
 */
export function isPriorValueSchemaValid(
  field: SchemaFieldResponse,
  priorValue: unknown,
): boolean {
  if (priorValue == null) return false;

  switch (field.type) {
    case "enum": {
      if (typeof priorValue !== "string") return false;
      const allowed = (field.allowed_values ?? []).map(normalizeEnumToken);
      return allowed.includes(normalizeEnumToken(priorValue));
    }

    case "enum_set": {
      if (!Array.isArray(priorValue)) return false;
      const allowed = new Set((field.allowed_values ?? []).map(normalizeEnumToken));
      return priorValue.every(
        (value) => typeof value === "string" && allowed.has(normalizeEnumToken(value)),
      );
    }

    case "boolean":
      return typeof priorValue === "boolean";

    case "integer": {
      // Backend accepts whole-number floats (3.0 → 3); JS numbers make the
      // two indistinguishable, so Number.isInteger covers both.
      if (typeof priorValue !== "number" || !Number.isInteger(priorValue)) {
        return false;
      }
      if (field.minimum != null && priorValue < field.minimum) return false;
      if (field.maximum != null && priorValue > field.maximum) return false;
      return true;
    }

    case "string": {
      if (typeof priorValue !== "string") return false;
      const trimmed = priorValue.trim();
      if (field.min_length != null && trimmed.length < field.min_length) return false;
      if (field.max_length != null && trimmed.length > field.max_length) return false;
      return true;
    }

    default:
      return true;
  }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Drift guard for the guidance-validation wire mock.
 *
 * fake-validate-draft.ts re-implements the backend's validation issue
 * codes so page tests can run without a server. If the backend renames or
 * removes a code, the fake would keep page tests green against a contract
 * that no longer exists — this guard forces every code the fake emits to
 * match the backend's canonical set, so a rename becomes a conscious
 * two-sided edit.
 */

import { describe, it, expect } from "vitest";

import fakeSource from "@/test/fake-validate-draft.ts?raw";

// Canonical issue codes, copied by hand from the backend source of truth:
// the "Issue codes" constants block in
// src/backend/vlm_feedback_loop/services/schema_core.py (behavior pinned by
// tests/unit/test_schema_core.py). When the backend adds or renames a code,
// update this list and the fake together.
const CANONICAL_ISSUE_CODES = new Set([
  "NO_CORE_FIELDS",
  "MISSING_FIELD_NAME",
  "DUPLICATE_FIELD_NAME",
  "INVALID_FIELD_NAME",
  "FIELD_NAME_TOO_LONG",
  "MISSING_TYPE",
  "ENUM_TOO_FEW_VALUES",
  "ENUM_EMPTY_VALUE",
  "ENUM_DUPLICATE_VALUE",
  "MIN_EXCEEDS_MAX",
  "MINLENGTH_EXCEEDS_MAXLENGTH",
  "RATIONALE_NOTE_WRONG_ROLE",
  "RATIONALE_NOTE_WRONG_TYPE",
  "SCHEMA_COMPILE_FAILURE",
]);

describe("fakeValidateDraft issue codes", () => {
  it("every issue code the fake can emit exists in the backend's canonical set", () => {
    const emitted = [...fakeSource.matchAll(/code:\s*"([A-Z0-9_]+)"/g)].map(
      (m) => m[1],
    );
    // If the source scan stops matching (e.g. the fake moves codes into
    // constants), the guard is void — fail loudly instead of passing on
    // an empty scan.
    expect(emitted.length).toBeGreaterThan(0);
    const unknown = [...new Set(emitted)].filter(
      (code) => !CANONICAL_ISSUE_CODES.has(code),
    );
    expect(unknown).toEqual([]);
  });
});

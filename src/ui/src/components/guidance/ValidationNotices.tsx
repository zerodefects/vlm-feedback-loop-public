// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Backend-validation notices rendered above the schema field sections on the
 * Create and Edit Guidance screens: the SCHEMA_COMPILE_FAILURE
 * banner and the validation-service failure line. Not gated on saveAttempted
 * — compile failures and outages are surfaced as soon as they are known.
 */

import { Text } from "@kui/react";
import type { SchemaIssueResponse } from "@/types/guidance";

interface ValidationNoticesProps {
  /** Ungated issue list from the form hook (`form.issues`). */
  issues: SchemaIssueResponse[];
  /** Non-null when the validate_draft call itself failed (`form.backendError`). */
  backendError: string | null;
}

export function ValidationNotices({ issues, backendError }: ValidationNoticesProps) {
  const compileFailure = issues.some((i) => i.code === "SCHEMA_COMPILE_FAILURE");
  return (
    <>
      {compileFailure && (
        <div
          className="mb-4 rounded-lg px-4 py-3 toast-error"
          data-testid="error-SCHEMA_COMPILE_FAILURE"
        >
          <Text kind="body/regular/sm" style={{ color: "inherit" }}>
            Schema cannot be compiled (internal inconsistency). Check field constraints.
          </Text>
        </div>
      )}
      {backendError && (
        <Text
          kind="body/regular/sm"
          className="mb-4"
          style={{ color: "var(--text-faint)", display: "block" }}
        >
          {backendError}
        </Text>
      )}
    </>
  );
}

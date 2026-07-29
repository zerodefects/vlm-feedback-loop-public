// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Inline validation-error rows for the Guidance builder — a red X icon and
 * message per error, with fix-it affordances for auto-correctable codes.
 * The row treatment is the shared <InlineError/> primitive (re-exported
 * here as ``ErrorRow`` for the builder call sites); InlineErrors maps a
 * field path's issues onto it.
 */

import { InlineError } from "@/components/InlineError";
import type { SchemaIssueResponse } from "@/types/guidance";

/** Single validation-error row: red X icon + message + optional fix-it slot. */
export { InlineError as ErrorRow };

interface InlineErrorsProps {
  issues: SchemaIssueResponse[];
  fieldPath: string;
  onFix?: (code: string) => void;
}

export function InlineErrors({ issues, fieldPath, onFix }: InlineErrorsProps) {
  // Match the exact path plus indexed sub-paths — the backend reports
  // per-value enum issues as e.g. "core[0].allowed_values[2]".
  const matching = issues.filter(
    (i) =>
      i.severity === "error" &&
      (i.field_path === fieldPath ||
        (i.field_path?.startsWith(`${fieldPath}[`) ?? false)),
  );
  if (matching.length === 0) return null;
  return (
    <div className="mt-1 space-y-1">
      {matching.map((issue) => (
        <InlineError
          key={`${issue.code}-${issue.field_path}`}
          message={issue.message}
          testId={`error-${issue.code}`}
        >
          {onFix && issue.code === "ENUM_DUPLICATE_VALUE" && (
            <button
              type="button"
              onClick={() => onFix(issue.code)}
              className="rounded px-1.5 py-0.5 text-sm underline hover:bg-white/10"
              data-testid={`fix-${issue.code}`}
            >
              Remove duplicates
            </button>
          )}
          {onFix && issue.code === "MIN_EXCEEDS_MAX" && (
            <button
              type="button"
              onClick={() => onFix(issue.code)}
              className="rounded px-1.5 py-0.5 text-sm underline hover:bg-white/10"
              data-testid={`fix-${issue.code}`}
            >
              Swap min/max
            </button>
          )}
        </InlineError>
      ))}
    </div>
  );
}

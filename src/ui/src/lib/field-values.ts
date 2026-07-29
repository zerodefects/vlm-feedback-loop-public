// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Equality check for label field values across the schema field types.
 *
 * Scalars compare with strict equality; arrays (enum_set values) compare
 * order-insensitively, since selection order carries no meaning. Used by
 * the proposal dirty-field tracker, where both sides come from the same
 * proposal baseline so strict comparison is correct. Display surfaces that
 * compare values from *different* origins (e.g., the prior-label hint's
 * agree/disagree indicator) layer backend-style normalization on top —
 * see PriorLabelHints; the backend (exact_match_evaluator) remains the
 * authority on canonical equality at save/eval time.
 */
export function fieldValuesEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    const sa = [...a].sort();
    const sb = [...b].sort();
    return sa.every((v, i) => v === sb[i]);
  }
  return false;
}

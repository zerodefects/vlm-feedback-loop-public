// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Validation status badge — shows "Valid", "N required" (pre-save), or
 * "N error(s)" (post-save). Uses KUI Badge.
 */

import { Badge } from "@kui/react";

interface StatusBadgeProps {
  errorCount: number;
  /** True once the SME has clicked Save (or the badge) at least once.
   *  Pre-save the badge uses neutral grey "N required" language; post-save
   *  it switches to live red "N error(s)". Mirrors the inline-error
   *  gating in useGuidanceForm.ts so the badge stops misrepresenting an
   *  un-touched blank form as broken. */
  saveAttempted?: boolean;
  /** When provided and errorCount > 0, the badge becomes a
   *  keyboard-accessible button that reveals inline errors and jumps the
   *  viewport to the first one. KUI Badge has no interactive mode; the
   *  raw <button> wrapper is the KUI-first rules' "custom interactive
   *  control" carve-out for small inline affordances that wrap a KUI
   *  primitive. */
  onClick?: () => void;
}

export function StatusBadge({
  errorCount,
  saveAttempted = false,
  onClick,
}: StatusBadgeProps) {
  if (errorCount === 0) {
    return (
      <Badge color="green" kind="outline" data-testid="status-badge">
        Schema: Valid
      </Badge>
    );
  }

  const badge = saveAttempted ? (
    <Badge color="red" kind="outline" data-testid="status-badge">
      Schema: {errorCount} error{errorCount !== 1 ? "s" : ""}
    </Badge>
  ) : (
    <Badge color="gray" kind="outline" data-testid="status-badge">
      Schema: {errorCount} required
    </Badge>
  );
  if (!onClick) return badge;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label="Jump to first error"
      data-testid="status-badge-button"
      style={{
        background: "none",
        border: 0,
        padding: 0,
        margin: 0,
        cursor: "pointer",
      }}
    >
      {badge}
    </button>
  );
}

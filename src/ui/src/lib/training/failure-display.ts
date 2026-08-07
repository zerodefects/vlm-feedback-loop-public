// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Split a classified TAO failure into its provider message and remediation.
 *
 * `classify_tao_failure` uses an em dash as the canonical delimiter. When it
 * is absent, the whole value is treated as the provider message.
 */
export function splitClassifiedFailure(err: string | null | undefined): {
  primary: string;
  hint: string | null;
} {
  if (!err) return { primary: "TAO job failed.", hint: null };
  const separatorIndex = err.indexOf(" — ");
  if (separatorIndex < 0) return { primary: err, hint: null };
  return {
    primary: err.slice(0, separatorIndex),
    hint: err.slice(separatorIndex + 3),
  };
}

const FAILURE_REASON_LABELS: Record<string, string> = {
  submission_interrupted:
    "Submission interrupted — the backend restarted before TAO confirmed this job.",
  student_nim_evaluation_failed:
    "Student NIM serving validation failed. Open Compare & Benchmark to inspect or retry deployment.",
  tao_evaluate_failed: "TAO quality evaluation failed.",
  tao_evaluate_canceled: "TAO quality evaluation was canceled.",
};

/** Convert bare-token error references into SME-facing text. */
export function humanizeFailureReason(reason: string): string {
  const normalized = reason.replace(/\\(['"])/g, "$1");
  const mapped = FAILURE_REASON_LABELS[normalized];
  if (mapped) return mapped;
  if (/^[a-z0-9]+(?:_[a-z0-9]+)+$/.test(normalized)) {
    const spaced = normalized.replace(/_/g, " ");
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
  }
  return normalized;
}

/** Remove durable chain bookkeeping IDs from the SME-facing halt reason. */
export function humanizeHaltedReason(reason: string): string {
  const normalized = humanizeFailureReason(reason);
  const withoutBookkeeping = normalized.replace(/\s+\(seq\s+\d+,\s*id=[^)]+\)/i, "");
  const concise = withoutBookkeeping.replace(
    /\s+reached terminal\s+['"]?([a-z_]+)['"]?$/i,
    (_match, status: string) => ` ${status.replace(/_/g, " ").toLowerCase()}`,
  );
  const prefixed = concise.startsWith("Chain halted")
    ? concise
    : `Chain halted: ${concise}`;
  return /[.!?]$/.test(prefixed) ? prefixed : `${prefixed}.`;
}

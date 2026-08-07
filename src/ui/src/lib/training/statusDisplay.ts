// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * TAOJob canonical status → display-label mapping for the Training
 * Job Monitor.
 *
 * The monitor enforces: raw canonical statuses from the backend MUST NOT
 * appear as badge text.  "succeeded" renders as **Completed**; a
 * ``failed`` job with ``chain_halted_reason != null`` renders as
 * **Halted** (distinct from plain **Failed**).  "Skipped" and "Pending"
 * MUST NEVER appear anywhere.
 *
 * Tone enum + ``TONE_STYLES`` live in ``@/lib/tone`` because multiple
 * domains (Training, Batch Run, etc.) share the same semantic colors.
 */

import type { TAOJobOutputsFetchStatus, TAOJobStatus } from "@/types/training";
import type { Tone } from "@/lib/tone";

export interface StatusDisplay {
  label: string;
  tone: Tone;
  /** Should a spinner accompany the badge? */
  spinner: boolean;
}

const TABLE: Readonly<Record<TAOJobStatus, StatusDisplay>> = {
  not_started: { label: "Not Started", tone: "neutral", spinner: false },
  submitting: { label: "Submitting", tone: "info", spinner: true },
  submitted: { label: "Submitted", tone: "info", spinner: false },
  // Queued may render as "info" or "neutral".  Picking "info"
  // aligns Queued with Submitted (both = pre-running TAO-accepted states)
  // and disambiguates from neutral terminal/pre-start siblings like
  // Canceled and Not Started, which would otherwise render an identical
  // pill chrome.
  queued: { label: "Queued", tone: "info", spinner: false },
  running: { label: "Running", tone: "success", spinner: true },
  paused: { label: "Paused", tone: "warning", spinner: false },
  succeeded: { label: "Completed", tone: "success", spinner: false }, // NOT "Succeeded"
  failed: { label: "Failed", tone: "error", spinner: false },
  canceled: { label: "Canceled", tone: "neutral", spinner: false },
  deleted: { label: "Deleted", tone: "subdued", spinner: false },
};

const AUTO_SKIP_REASON_PREFIX = "auto-skip:";

/** True when a canceled TAO job was deliberately omitted by chain policy. */
export function isAutoSkippedReason(reason: string | null | undefined): boolean {
  return (reason ?? "").startsWith(AUTO_SKIP_REASON_PREFIX);
}

/**
 * Resolve the visible badge representation for a TAOJob.
 *
 * @param status - canonical status from the backend (one of 10 values).
 * @param chainHaltedReason - when set on a ``failed`` job, the job did
 *   not run because an upstream sibling failed or was canceled; render
 *   as **Halted** (warning badge) rather than **Failed** (error badge).
 */
export function statusDisplay(
  status: TAOJobStatus,
  chainHaltedReason: string | null | undefined,
  outputsFetchStatus?: TAOJobOutputsFetchStatus | null,
): StatusDisplay {
  if (status === "canceled" && isAutoSkippedReason(chainHaltedReason)) {
    return { label: "Not Required", tone: "info", spinner: false };
  }
  if (status === "failed" && chainHaltedReason) {
    return { label: "Halted", tone: "warning", spinner: false };
  }
  if (status === "succeeded" && outputsFetchStatus === "failed") {
    return { label: "Artifact Failed", tone: "error", spinner: false };
  }
  if (
    status === "succeeded" &&
    (outputsFetchStatus === "pending" || outputsFetchStatus === "in_progress")
  ) {
    return { label: "Finalizing", tone: "info", spinner: true };
  }
  return TABLE[status];
}

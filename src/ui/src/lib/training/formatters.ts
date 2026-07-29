// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Pure formatting helpers for Training Job Monitor cards.
 *
 * Kept intentionally dependency-free so they unit-test cleanly and are
 * safe to call from any rendering path.
 */

/**
 * Parse an ISO-8601 timestamp, returning null if malformed.  The backend
 * persists all timestamps in UTC with a ``Z`` suffix, but the
 * browser's ``Date`` constructor accepts other shapes — a ``null``
 * result is easier to guard in the UI than an ``Invalid Date``.
 */
function parseTs(ts: string | null | undefined): Date | null {
  if (!ts) return null;
  const date = new Date(ts);
  return Number.isNaN(date.getTime()) ? null : date;
}

/**
 * Render a human-readable duration between two ISO timestamps.
 *
 * Examples:
 *   formatDuration("2026-04-17T10:00:00Z", "2026-04-17T11:45:32Z") → "1h 45m"
 *   formatDuration("2026-04-17T10:00:00Z", "2026-04-17T10:00:08Z") → "8s"
 *   formatDuration(null, "...") → null
 */
export function formatDuration(
  startedAt: string | null | undefined,
  completedAt: string | null | undefined,
): string | null {
  const start = parseTs(startedAt);
  const end = parseTs(completedAt);
  if (!start || !end) return null;
  const ms = end.getTime() - start.getTime();
  if (ms < 0) return null;
  return formatDurationSeconds(Math.round(ms / 1000));
}

/**
 * Render a human-readable remaining-time string for the Running state's
 * ETA label. Missing or invalid provider telemetry returns ``null`` so the
 * monitor can omit the field instead of inventing a placeholder.
 */
export function formatEta(seconds: number | null | undefined): string | null {
  if (
    seconds === null ||
    seconds === undefined ||
    !Number.isFinite(seconds) ||
    seconds < 0
  ) {
    return null;
  }
  return `ETA ${formatDurationSeconds(Math.round(seconds))}`;
}

/**
 * Core duration formatter.  Exported for direct use by tests.
 *
 * Output shapes (picked for readability, not SI accuracy):
 *   <60s           → "NNs"
 *   <60m           → "Nm" or "Nm Ss"
 *   <24h           → "Nh Nm"
 *   >=24h          → "Dd Nh"
 */
export function formatDurationSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) {
    return s === 0 ? `${m}m` : `${m}m ${s}s`;
  }
  const h = Math.floor(m / 60);
  const remM = m % 60;
  if (h < 24) {
    return remM === 0 ? `${h}h` : `${h}h ${remM}m`;
  }
  const d = Math.floor(h / 24);
  const remH = h % 24;
  return remH === 0 ? `${d}d` : `${d}d ${remH}h`;
}

/**
 * Render a succeeded/total count with "done" shortcut when complete.
 * Used by the Training Job Monitor's chain progress line: "8B: 6 of 6" →
 * "8B: done".
 */
export function formatChainProgress(done: number, total: number): string {
  if (total === 0) return "—";
  if (done === total) return "done";
  return `${done} of ${total}`;
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Date formatting utilities.
 *
 * The backend stores timestamps in UTC ISO 8601 with Z suffix.
 * The frontend converts to local time for display.  These functions
 * establish the display convention reused by every subsequent screen.
 */

const MINUTE = 60_000;
const HOUR = 3_600_000;
const DAY = 86_400_000;

/**
 * Format an ISO 8601 timestamp as a short date.
 *
 * Same year: "Mar 15"
 * Different year: "Mar 15, 2025"
 */
export function formatDate(iso: string): string {
  const date = new Date(iso);
  // Guard against an unparseable timestamp — otherwise toLocaleDateString
  // renders the literal "Invalid Date" in the UI.
  if (Number.isNaN(date.getTime())) return "—";
  const now = new Date();
  const sameYear = date.getFullYear() === now.getFullYear();

  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(sameYear ? {} : { year: "numeric" }),
  });
}

/**
 * Format an ISO 8601 timestamp as a relative time string.
 *
 * "just now"  (< 1 min)
 * "2 min ago" (< 1 hour)
 * "3 hours ago" (< 24 hours)
 * "yesterday"
 * Falls back to {@link formatDate} for older dates.
 */
export function formatRelativeTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  const diff = Date.now() - date.getTime();

  if (diff < MINUTE) return "just now";
  if (diff < HOUR) {
    const mins = Math.floor(diff / MINUTE);
    return `${mins} min ago`;
  }
  if (diff < DAY) {
    const hours = Math.floor(diff / HOUR);
    return `${hours} ${hours === 1 ? "hour" : "hours"} ago`;
  }
  if (diff < 2 * DAY) return "yesterday";
  return formatDate(iso);
}

/**
 * Render a compact month-day + wall-clock string for a UTC ISO timestamp
 * (e.g. "Mar 30, 16:15").  Browser's ``toLocaleString`` converts to the
 * viewer's timezone (timestamps display in local time).  Null /
 * malformed → null so the caller can omit the surrounding line.
 */
export function formatTimestamp(ts: string | null | undefined): string | null {
  if (!ts) return null;
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return null;
  const monthDay = d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
  });
  const time = d.toLocaleString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  return `${monthDay}, ${time}`;
}

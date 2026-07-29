// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, vi, afterEach } from "vitest";
import { formatDate, formatRelativeTime, formatTimestamp } from "@/lib/format-date";

describe("formatDate", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("formats a same-year date as 'Mon DD'", () => {
    vi.useFakeTimers({ now: new Date("2026-04-11T12:00:00Z") });
    const result = formatDate("2026-03-15T14:22:07Z");
    expect(result).toContain("Mar");
    expect(result).toContain("15");
    // Should NOT contain the year
    expect(result).not.toContain("2026");
  });

  it("formats a different-year date with year", () => {
    vi.useFakeTimers({ now: new Date("2026-04-11T12:00:00Z") });
    const result = formatDate("2025-12-01T10:00:00Z");
    expect(result).toContain("Dec");
    expect(result).toContain("1");
    expect(result).toContain("2025");
  });
});

describe("formatRelativeTime", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns "just now" for < 1 minute ago', () => {
    vi.useFakeTimers({ now: new Date("2026-04-11T12:00:30Z") });
    expect(formatRelativeTime("2026-04-11T12:00:00Z")).toBe("just now");
  });

  it('returns "N min ago" for < 1 hour', () => {
    vi.useFakeTimers({ now: new Date("2026-04-11T12:10:00Z") });
    expect(formatRelativeTime("2026-04-11T12:00:00Z")).toBe("10 min ago");
  });

  it('returns "N hours ago" for < 24 hours', () => {
    vi.useFakeTimers({ now: new Date("2026-04-11T15:00:00Z") });
    expect(formatRelativeTime("2026-04-11T12:00:00Z")).toBe("3 hours ago");
  });

  it('returns "1 hour ago" (singular)', () => {
    vi.useFakeTimers({ now: new Date("2026-04-11T13:00:00Z") });
    expect(formatRelativeTime("2026-04-11T12:00:00Z")).toBe("1 hour ago");
  });

  it('returns "yesterday" for 24-48 hours ago', () => {
    vi.useFakeTimers({ now: new Date("2026-04-12T12:00:00Z") });
    expect(formatRelativeTime("2026-04-11T12:00:00Z")).toBe("yesterday");
  });

  it("falls back to formatDate for older dates", () => {
    vi.useFakeTimers({ now: new Date("2026-04-11T12:00:00Z") });
    const result = formatRelativeTime("2026-03-15T14:22:07Z");
    expect(result).toContain("Mar");
    expect(result).toContain("15");
  });
});

describe("formatTimestamp", () => {
  it("renders an ISO timestamp as 'Mon D, HH:MM' in the local tz", () => {
    // Wall-clock varies by runner timezone, so assert the shape rather than
    // the exact string.  Pattern: "<Abbrev-Month> <1-2 digits>, <HH>:<MM>".
    const out = formatTimestamp("2026-04-17T10:00:00Z");
    expect(out).toMatch(/^[A-Z][a-z]{2} \d{1,2}, \d{2}:\d{2}$/);
  });

  it("returns null for null / undefined / malformed input", () => {
    expect(formatTimestamp(null)).toBeNull();
    expect(formatTimestamp(undefined)).toBeNull();
    expect(formatTimestamp("")).toBeNull();
    expect(formatTimestamp("not-a-date")).toBeNull();
  });
});

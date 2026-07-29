// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  formatChainProgress,
  formatDuration,
  formatDurationSeconds,
  formatEta,
} from "../formatters";

describe("formatDurationSeconds", () => {
  it("renders seconds under a minute", () => {
    expect(formatDurationSeconds(0)).toBe("0s");
    expect(formatDurationSeconds(45)).toBe("45s");
    expect(formatDurationSeconds(59)).toBe("59s");
  });

  it("renders minutes and seconds under an hour", () => {
    expect(formatDurationSeconds(60)).toBe("1m");
    expect(formatDurationSeconds(125)).toBe("2m 5s");
    expect(formatDurationSeconds(3599)).toBe("59m 59s");
  });

  it("renders hours and minutes under a day", () => {
    expect(formatDurationSeconds(3600)).toBe("1h");
    expect(formatDurationSeconds(6300)).toBe("1h 45m");
    expect(formatDurationSeconds(86399)).toBe("23h 59m");
  });

  it("renders days and hours for long runs", () => {
    expect(formatDurationSeconds(86400)).toBe("1d");
    expect(formatDurationSeconds(90000)).toBe("1d 1h");
  });
});

describe("formatDuration", () => {
  it("computes delta between two ISO timestamps", () => {
    expect(formatDuration("2026-04-17T10:00:00Z", "2026-04-17T11:45:00Z")).toBe(
      "1h 45m",
    );
    expect(formatDuration("2026-04-17T10:00:00Z", "2026-04-17T10:00:08Z")).toBe("8s");
  });

  it("returns null when either side is missing or invalid", () => {
    expect(formatDuration(null, "2026-04-17T10:00:00Z")).toBeNull();
    expect(formatDuration("2026-04-17T10:00:00Z", null)).toBeNull();
    expect(formatDuration(undefined, undefined)).toBeNull();
    expect(formatDuration("not-a-date", "2026-04-17T10:00:00Z")).toBeNull();
  });

  it("returns null when completedAt precedes startedAt", () => {
    expect(formatDuration("2026-04-17T12:00:00Z", "2026-04-17T10:00:00Z")).toBeNull();
  });
});

describe("formatEta", () => {
  it("prefixes with 'ETA' when value is present and positive", () => {
    expect(formatEta(2700)).toBe("ETA 45m");
    expect(formatEta(8)).toBe("ETA 8s");
    expect(formatEta(0)).toBe("ETA 0s");
  });

  it("returns null for missing or invalid values", () => {
    expect(formatEta(null)).toBeNull();
    expect(formatEta(undefined)).toBeNull();
    expect(formatEta(-3)).toBeNull();
    expect(formatEta(Number.NaN)).toBeNull();
    expect(formatEta(Number.POSITIVE_INFINITY)).toBeNull();
  });
});

describe("formatChainProgress", () => {
  it("renders 'done' when all jobs succeeded", () => {
    expect(formatChainProgress(6, 6)).toBe("done");
  });

  it("renders 'N of M' while jobs remain", () => {
    expect(formatChainProgress(0, 6)).toBe("0 of 6");
    expect(formatChainProgress(5, 6)).toBe("5 of 6");
  });

  it("handles the zero-total edge case", () => {
    expect(formatChainProgress(0, 0)).toBe("—");
  });
});

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { formatDeltaPoints, formatPct } from "../format-percent";

describe("formatPct", () => {
  it("renders a 0-1 rate as a rounded whole percent", () => {
    expect(formatPct(0.87)).toBe("87%");
    expect(formatPct(0)).toBe("0%");
    expect(formatPct(1)).toBe("100%");
  });

  it("rounds half-to-even, matching the backend's Python :.0% output", () => {
    // Ties must agree with backend-built messages that render the same
    // metric: 25/40 = 0.625 reads "62%" in both the client label and the
    // backend gate detail — never "62%" vs "63%" in the same card.
    expect(formatPct(0.625)).toBe("62%");
    expect(formatPct(0.875)).toBe("88%");
    expect(formatPct(0.005)).toBe("0%");
  });

  it("renders em-dash for missing or non-numeric values", () => {
    expect(formatPct(null)).toBe("—");
    expect(formatPct(undefined)).toBe("—");
    expect(formatPct(Number.NaN)).toBe("—");
  });
});

describe("formatDeltaPoints", () => {
  it("renders a signed whole-point delta with the pts suffix", () => {
    expect(formatDeltaPoints(0.07)).toBe("+7 pts");
    expect(formatDeltaPoints(-0.05)).toBe("-5 pts");
    expect(formatDeltaPoints(0)).toBe("+0 pts");
  });

  it("rounds ties half-to-even, matching formatPct", () => {
    expect(formatDeltaPoints(0.025)).toBe("+2 pts");
    expect(formatDeltaPoints(0.035)).toBe("+4 pts");
    expect(formatDeltaPoints(-0.025)).toBe("-2 pts");
  });
});

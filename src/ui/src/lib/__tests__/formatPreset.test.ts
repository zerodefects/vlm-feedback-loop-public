// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { titleCasePreset } from "../formatPreset";

describe("titleCasePreset", () => {
  it("title-cases simple single-word preset keys", () => {
    expect(titleCasePreset("precise")).toBe("Precise");
    expect(titleCasePreset("explore")).toBe("Explore");
    expect(titleCasePreset("balanced")).toBe("Balanced");
    expect(titleCasePreset("fast")).toBe("Fast");
  });

  it("renders snake_case preset keys as spaced Title Case", () => {
    // The canonical name is "High Detail" (the top bar abbreviates to
    // "High"); the separator must not survive as a literal underscore.
    expect(titleCasePreset("high_detail")).toBe("High Detail");
  });

  it("renders hyphen-separated preset keys as spaced Title Case", () => {
    expect(titleCasePreset("high-detail")).toBe("High Detail");
  });

  it("title-cases space-separated values word-by-word", () => {
    expect(titleCasePreset("high detail")).toBe("High Detail");
  });

  it("renders em-dash placeholder for null / empty", () => {
    expect(titleCasePreset(null)).toBe("—");
    expect(titleCasePreset(undefined)).toBe("—");
    expect(titleCasePreset("")).toBe("—");
  });

  it("passes through already-capitalized values unchanged", () => {
    expect(titleCasePreset("Precise")).toBe("Precise");
  });
});

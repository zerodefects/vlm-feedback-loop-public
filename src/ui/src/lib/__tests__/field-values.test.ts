// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { fieldValuesEqual } from "../field-values";

describe("fieldValuesEqual", () => {
  it("compares scalars with strict equality", () => {
    expect(fieldValuesEqual("rock", "rock")).toBe(true);
    expect(fieldValuesEqual("rock", "paper")).toBe(false);
    expect(fieldValuesEqual(3, 3)).toBe(true);
    expect(fieldValuesEqual(true, false)).toBe(false);
    expect(fieldValuesEqual(null, null)).toBe(true);
    // No coercion — a stringified number is not the number.
    expect(fieldValuesEqual("3", 3)).toBe(false);
  });

  it("compares enum_set arrays order-insensitively", () => {
    expect(fieldValuesEqual(["a", "b"], ["b", "a"])).toBe(true);
    expect(fieldValuesEqual(["a", "b"], ["a", "b"])).toBe(true);
    expect(fieldValuesEqual(["a"], ["a", "b"])).toBe(false);
    expect(fieldValuesEqual(["a", "b"], ["a", "c"])).toBe(false);
    expect(fieldValuesEqual([], [])).toBe(true);
  });

  it("does not equate an array with a scalar", () => {
    expect(fieldValuesEqual(["a"], "a")).toBe(false);
  });
});

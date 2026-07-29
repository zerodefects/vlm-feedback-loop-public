// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  findCriterion,
  formatAcceptLine,
  formatEvalLine,
  formatPoolLine,
  summariseCriterion,
} from "../readinessFormatters";
import type { GateCriterion } from "@/types/evaluation";

function makeCriterion(partial: Partial<GateCriterion>): GateCriterion {
  return {
    criterion_name: "accept_rate",
    passed: false,
    current_value: 0,
    threshold: 0.8,
    message: "",
    details: null,
    ...partial,
  };
}

describe("summariseCriterion", () => {
  it("returns pass when criterion.passed is true", () => {
    const c = makeCriterion({ passed: true, message: "All good." });
    const line = summariseCriterion(c, "Label");
    expect(line.status).toBe("pass");
    expect(line.label).toBe("Label");
    expect(line.detail).toBe("All good.");
  });

  it("returns pending when details.blocked_by is set", () => {
    const c = makeCriterion({
      passed: false,
      details: { blocked_by: "overall_exact_match" },
      message: "Depends on evaluation results.",
    });
    expect(summariseCriterion(c, "x").status).toBe("pending");
  });

  it("returns fail otherwise", () => {
    const c = makeCriterion({ passed: false, details: null, message: "bad" });
    expect(summariseCriterion(c, "x").status).toBe("fail");
  });
});

describe("findCriterion", () => {
  it("returns matching criterion or null", () => {
    const arr = [makeCriterion({ criterion_name: "accept_rate" })];
    expect(findCriterion(arr, "accept_rate")).not.toBeNull();
    expect(findCriterion(arr, "nope")).toBeNull();
  });
});

describe("formatPoolLine", () => {
  it("uses criterion threshold when available", () => {
    const line = formatPoolLine(
      [
        makeCriterion({
          criterion_name: "min_test_pool_size",
          current_value: 8,
          threshold: 20,
          passed: false,
          message: "Need 12 more.",
        }),
      ],
      8,
    );
    expect(line.status).toBe("fail");
    expect(line.label).toContain("Test Pool: 8 / 20 min");
  });

  it("falls back to verified count when criterion absent", () => {
    const line = formatPoolLine([], 3);
    expect(line.status).toBe("pending");
    expect(line.detail).toContain("3 Verified");
  });
});

describe("formatAcceptLine", () => {
  it("formats percentages", () => {
    const line = formatAcceptLine([
      makeCriterion({
        criterion_name: "accept_rate",
        current_value: 0.67,
        threshold: 0.8,
        passed: false,
        message: "Low accept.",
      }),
    ]);
    expect(line.label).toContain("67%");
    expect(line.label).toContain("need 80%");
  });
});

describe("formatEvalLine", () => {
  it("is pending when the backend marks no_completed_run structurally", () => {
    const line = formatEvalLine([
      makeCriterion({
        criterion_name: "overall_exact_match",
        current_value: 0,
        threshold: 0.8,
        passed: false,
        // Deliberately NOT the shipped backend copy: the pending state
        // keys on details.no_completed_run, so a message copy-edit must
        // not flip the Readiness card.
        message: "Evaluation has not been run.",
        details: { no_completed_run: true },
      }),
    ]);
    expect(line.status).toBe("pending");
    expect(line.detail).toMatch(/No evaluation run yet/);
  });

  it("is fail (not pending) for a genuine 0% Exact-Match run", () => {
    // current_value=0 alone can't distinguish "no eval" from a real 0%
    // result; only details.no_completed_run does. A real failing 0% run
    // must show as fail, not be masked as "not run yet".
    const line = formatEvalLine([
      makeCriterion({
        criterion_name: "overall_exact_match",
        current_value: 0,
        threshold: 0.8,
        passed: false,
        message:
          "Model accuracy: 0% overall (need 80%). Continue labeling or refine Guidance.",
      }),
    ]);
    expect(line.status).toBe("fail");
  });

  it("is pass when overall_exact_match is above threshold", () => {
    const line = formatEvalLine([
      makeCriterion({
        criterion_name: "overall_exact_match",
        current_value: 0.85,
        threshold: 0.8,
        passed: true,
        message: "Passed.",
      }),
    ]);
    expect(line.status).toBe("pass");
    // The label uses the backend's plain-language "Model accuracy" naming,
    // never the raw metric name — it must agree with the gate messages.
    expect(line.label).toContain("Model accuracy: 85%");
  });

  it("is fail when criterion failed but a run has happened", () => {
    const line = formatEvalLine([
      makeCriterion({
        criterion_name: "overall_exact_match",
        current_value: 0.72,
        threshold: 0.8,
        passed: false,
        message: "Model accuracy too low.",
      }),
    ]);
    expect(line.status).toBe("fail");
  });
});

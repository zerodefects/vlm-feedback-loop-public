// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { TAO_JOB_STATUSES, type TAOJobStatus } from "@/types/training";

import { statusDisplay } from "../statusDisplay";

describe("statusDisplay — training-job display-label mapping", () => {
  it.each([
    ["not_started", "Not Started", "neutral", false],
    ["submitting", "Submitting", "info", true],
    ["submitted", "Submitted", "info", false],
    // Queued uses "info" so it doesn't collapse into the same pill
    // chrome as Canceled / Not Started.
    ["queued", "Queued", "info", false],
    ["running", "Running", "success", true],
    ["paused", "Paused", "warning", false],
    ["succeeded", "Completed", "success", false],
    ["failed", "Failed", "error", false],
    ["canceled", "Canceled", "neutral", false],
    ["deleted", "Deleted", "subdued", false],
  ] as Array<[TAOJobStatus, string, string, boolean]>)(
    "%s (no halt) → '%s' (tone=%s, spinner=%s)",
    (status, expectedLabel, expectedTone, expectedSpinner) => {
      const d = statusDisplay(status, null);
      expect(d.label).toBe(expectedLabel);
      expect(d.tone).toBe(expectedTone);
      expect(d.spinner).toBe(expectedSpinner);
    },
  );

  it("succeeded renders as 'Completed', not 'Succeeded'", () => {
    expect(statusDisplay("succeeded", null).label).toBe("Completed");
    expect(statusDisplay("succeeded", null).label).not.toBe("Succeeded");
  });

  it("distinguishes Blueprint artifact finalization from TAO completion", () => {
    expect(statusDisplay("succeeded", null, "in_progress")).toEqual({
      label: "Finalizing",
      tone: "info",
      spinner: true,
    });
    expect(statusDisplay("succeeded", null, "failed")).toEqual({
      label: "Artifact Failed",
      tone: "error",
      spinner: false,
    });
    expect(statusDisplay("succeeded", null, "completed").label).toBe("Completed");
  });

  it("failed + chain_halted_reason renders as 'Halted' (warning), not 'Failed'", () => {
    const d = statusDisplay(
      "failed",
      "Chain halted: evaluate (seq 2, id=xyz) reached terminal 'failed'",
    );
    expect(d.label).toBe("Halted");
    expect(d.tone).toBe("warning");
  });

  it("distinguishes an intentional evaluation omission from cancellation", () => {
    expect(
      statusDisplay("canceled", "auto-skip: action=evaluate auto-skipped"),
    ).toEqual({ label: "Not Required", tone: "info", spinner: false });
  });

  it("failed with empty-string halt reason stays 'Failed'", () => {
    // An empty string is falsy — treat as no halt.
    const d = statusDisplay("failed", "");
    expect(d.label).toBe("Failed");
    expect(d.tone).toBe("error");
  });

  it("non-failed statuses ignore chain_halted_reason", () => {
    const d = statusDisplay("running", "some reason");
    expect(d.label).toBe("Running");
    expect(d.tone).toBe("success");
  });

  it("'Skipped' and 'Pending' never appear as badge labels", () => {
    const forbidden = new Set(["Skipped", "Pending"]);
    for (const s of TAO_JOB_STATUSES) {
      expect(forbidden.has(statusDisplay(s, null).label)).toBe(false);
      expect(forbidden.has(statusDisplay(s, "some halt reason").label)).toBe(false);
    }
  });

  it("canonical status strings never appear verbatim as display labels", () => {
    // Each TAO_JOB_STATUSES value is lowercase snake_case; no display label
    // should match it exactly (case-sensitive comparison).
    for (const s of TAO_JOB_STATUSES) {
      expect(statusDisplay(s, null).label).not.toBe(s);
    }
  });
});

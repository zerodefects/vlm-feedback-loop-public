// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { TAO_JOB_STATUSES } from "@/types/training";
import { statusDisplay } from "@/lib/training/statusDisplay";

import { TrainingStatusBadge } from "../TrainingStatusBadge";

describe("TrainingStatusBadge", () => {
  // Expected labels are derived from the product mapping (statusDisplay);
  // the literal status→label contract is pinned in
  // lib/training/__tests__/statusDisplay.test.ts. This case verifies the
  // badge renders that mapping, not the mapping itself.
  it.each(TAO_JOB_STATUSES.map((s) => [s, statusDisplay(s, null).label] as const))(
    "renders %s as '%s'",
    (status, expectedLabel) => {
      render(<TrainingStatusBadge status={status} />);
      expect(screen.getByTestId("training-status-badge").textContent).toContain(
        expectedLabel,
      );
    },
  );

  it("renders failed + chain_halted_reason as 'Halted' not 'Failed'", () => {
    render(
      <TrainingStatusBadge
        status="failed"
        chainHaltedReason="Chain halted: evaluate (seq 2, id=abc)"
      />,
    );
    const txt = screen.getByTestId("training-status-badge").textContent ?? "";
    expect(txt).toContain("Halted");
    expect(txt).not.toContain("Failed");
  });

  it("never renders the canonical status string verbatim for any status", () => {
    // Canonical values are lowercase snake_case ("submitting", "not_started",
    // "succeeded"). Display labels are Title Case ("Submitting",
    // "Not Started", "Completed"). The labels MAY share letters when
    // lower-cased, but the verbatim canonical string MUST NOT appear in
    // the rendered badge text.
    for (const s of TAO_JOB_STATUSES) {
      const { unmount } = render(<TrainingStatusBadge status={s} />);
      const text = screen.getByTestId("training-status-badge").textContent ?? "";
      expect(text).not.toContain(s);
      // "Succeeded" must never appear — we render "Completed" instead.
      expect(text).not.toMatch(/Succeeded/);
      // "Skipped" and "Pending" forbidden everywhere.
      expect(text).not.toMatch(/Skipped/);
      expect(text).not.toMatch(/Pending/);
      unmount();
    }
  });

  it("exposes status and halt flag via data attributes", () => {
    render(<TrainingStatusBadge status="failed" chainHaltedReason="reason" />);
    const badge = screen.getByTestId("training-status-badge");
    expect(badge.getAttribute("data-status")).toBe("failed");
    expect(badge.getAttribute("data-halted")).toBe("true");
  });
});

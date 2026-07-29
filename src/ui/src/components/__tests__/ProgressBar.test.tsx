// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { ProgressBar } from "../ProgressBar";

describe("ProgressBar", () => {
  it("sets aria-valuenow to the rounded percent", () => {
    render(<ProgressBar percent={42.6} ariaLabel="x" />);
    const bar = screen.getByRole("progressbar");
    expect(bar.getAttribute("aria-valuenow")).toBe("43");
  });

  it("clamps percent below 0 to 0", () => {
    render(<ProgressBar percent={-10} ariaLabel="x" fillTestId="fill" />);
    expect(screen.getByTestId("fill").style.width).toBe("0%");
  });

  it("clamps percent above 100 to 100", () => {
    render(<ProgressBar percent={150} ariaLabel="x" fillTestId="fill" />);
    expect(screen.getByTestId("fill").style.width).toBe("100%");
  });

  it("renders a visible track outline at 0% via glass-border-subtle", () => {
    render(<ProgressBar percent={0} ariaLabel="x" />);
    const bar = screen.getByRole("progressbar");
    // The track MUST carry an explicit border so the pill is visible at
    // 0% (the empty state). We don't assert an exact rgba value
    // because JSDOM resolves CSS vars to the literal string; we just check
    // that a non-empty border rule is present on the element.
    expect(bar.style.border).toMatch(/1px solid/);
  });

  it("switches fill color for the paused variant", () => {
    render(
      <ProgressBar percent={20} variant="paused" fillTestId="fill" ariaLabel="x" />,
    );
    expect(screen.getByTestId("fill").style.backgroundColor).toMatch(
      /warning-amber|#f59e0b/,
    );
  });
});

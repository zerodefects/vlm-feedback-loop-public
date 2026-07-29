// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for PathChoiceCard (F-amendment NIM-FTU-Local-Peer).
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { PathChoiceCard } from "@/components/PathChoiceCard";

describe("PathChoiceCard", () => {
  it("renders title + meta + children", () => {
    render(
      <PathChoiceCard
        kind="primary"
        eyebrow="RECOMMENDED FOR YOUR GPU"
        title="Run Cosmos Reason2 8B locally"
        meta="56+ GB GPU memory minimum."
        testId="card-primary"
      >
        <button type="button">Deploy locally</button>
      </PathChoiceCard>,
    );

    expect(screen.getByText(/RECOMMENDED FOR YOUR GPU/)).toBeInTheDocument();
    expect(screen.getByText(/Run Cosmos Reason2 8B locally/)).toBeInTheDocument();
    expect(screen.getByText(/56\+ GB/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Deploy locally/i })).toBeInTheDocument();
  });

  it("primary combines the base glass card with elevation; peer uses the base", () => {
    const { rerender } = render(
      <PathChoiceCard kind="primary" title="Primary" testId="card-primary" />,
    );
    const primary = screen.getByTestId("card-primary");
    expect(primary.className).toContain("glass-card");
    expect(primary.className).toContain("glass-card--elevated");

    rerender(<PathChoiceCard kind="peer" title="Peer" testId="card-peer" />);
    const peer = screen.getByTestId("card-peer");
    expect(peer.className).toContain("glass-card");
    expect(peer.className).not.toContain("glass-card--elevated");
  });

  it("omits eyebrow / meta / children when not provided", () => {
    const { container } = render(<PathChoiceCard kind="peer" title="Title only" />);
    // Nothing but the title renders — no eyebrow, no meta line, and no
    // stray "undefined" text for the omitted props.
    expect(screen.getByText(/Title only/)).toBeInTheDocument();
    expect(container.textContent).toBe("Title only");
  });
});

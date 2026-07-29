// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for IngestionCountRow — pinpointed coverage of the
 * "Errors: 450" scroll-box behavior.
 *
 * The failure detail list must live in a fixed-height
 * ``overflow-y: auto`` container: rendered as inline sibling <Text>
 * lines instead, 450 errors blow out the page vertically. This test
 * pins that behavior at the smallest possible level.
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { IngestionCountRow } from "@/components/ingest/IngestionCountRow";
import type { FailureItem } from "@/components/ingest/IngestionProgress";

describe("IngestionCountRow", () => {
  it("renders only the header when items list is empty", () => {
    render(<IngestionCountRow label="Errors" count={0} tone="error" />);
    expect(screen.getByText("Errors: 0")).toBeInTheDocument();
    // No detail container when there are no items.
    expect(screen.queryByTestId("count-row-items-errors")).toBeNull();
  });

  it("renders a short detail list inline without bordered chrome", () => {
    const items: FailureItem[] = [
      { name: "a.jpg", reason: "validation failed" },
      { name: "b.jpg", reason: "validation failed" },
    ];
    render(<IngestionCountRow label="Errors" count={2} items={items} tone="error" />);
    const container = screen.getByTestId("count-row-items-errors");
    expect(container).toBeInTheDocument();
    // Small lists (<= 4) render without a visible border / elevated bg.
    const style = container.getAttribute("style") ?? "";
    expect(style).not.toContain("1px solid");
  });

  it("renders a long error list in a scrollable, bounded container", () => {
    const items: FailureItem[] = Array.from({ length: 450 }, (_, i) => ({
      name: `img_${i}.png`,
      reason: "Image validation produced no image",
    }));
    render(<IngestionCountRow label="Errors" count={450} items={items} tone="error" />);

    expect(screen.getByText("Errors: 450")).toBeInTheDocument();

    const container = screen.getByTestId("count-row-items-errors");
    // Tailwind ``max-h-48 overflow-y-auto`` classes are the load-bearing
    // pair that bound the height + enable scrolling. Assert both.
    const className = container.className;
    expect(className).toContain("max-h-48");
    expect(className).toContain("overflow-y-auto");
    // Long lists (> 4) get a visible border + elevated background so
    // the scroll area reads as a contained surface.
    const style = container.getAttribute("style") ?? "";
    expect(style).toContain("1px solid");
    // All 450 items are still in the DOM — scrolling, not truncation.
    const rows = container.querySelectorAll("*");
    // 450 Text components → at least 450 DOM nodes inside the container.
    expect(rows.length).toBeGreaterThanOrEqual(450);
  });
});

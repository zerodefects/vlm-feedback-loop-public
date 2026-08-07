// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { chartGroupKey } from "@/lib/chart-group";

import { GroupedBarChart } from "../GroupedBarChart";
import type { ChartGroup, ChartSeries } from "../GroupedBarChart";

const VALUES = [
  "beans",
  "cake",
  "candy",
  "cereal",
  "chips",
  "chocolate",
  "coffee",
  "corn",
  "fish",
  "flour",
  "honey",
  "jam",
  "juice",
  "milk",
  "nuts",
  "oil",
  "pasta",
  "rice",
  "soda",
  "spices",
  "sugar",
  "tea",
  "tomato_sauce",
  "vinegar",
  "water",
];

function denseFixture(): { groups: ChartGroup[]; series: ChartSeries[] } {
  const groups = VALUES.map((label) => ({ label, cluster: "product_category" }));
  const series = ["Teacher", "Super", "Nano FP8", "Nano BF16"].map(
    (label, seriesIndex) => ({
      label,
      color: `rgb(${seriesIndex}, 0, 0)`,
      values: Object.fromEntries(groups.map((group) => [chartGroupKey(group), 0.8])),
    }),
  );
  return { groups, series };
}

describe("GroupedBarChart", () => {
  it("keeps a 25-value four-series comparison inside the desktop chart canvas", () => {
    const fixture = denseFixture();
    render(
      <GroupedBarChart
        title="Per-value F1"
        groups={fixture.groups}
        series={fixture.series}
      />,
    );

    const svg = screen.getByRole("img", { name: "Per-value F1" });
    const svgWidth = Number(svg.getAttribute("width"));
    expect(svgWidth).toBeLessThanOrEqual(1056);
    expect(screen.queryByTestId("chart-scroll-guidance")).toBeNull();

    const lastBar = screen.getByTestId("chart-bar-Nano BF16-product_category::water");
    const lastBarRight =
      Number(lastBar.getAttribute("x")) + Number(lastBar.getAttribute("width"));
    expect(lastBarRight).toBeLessThanOrEqual(svgWidth);

    const denseLabel = screen.getByTestId(
      "chart-group-label-product_category::tomato_sauce",
    );
    expect(denseLabel).toHaveAttribute(
      "transform",
      expect.stringContaining("rotate(-40"),
    );
    expect(denseLabel).toHaveTextContent("tomato sauce");
  });

  it("makes wider comparison matrices explicitly keyboard-scrollable", () => {
    const fixture = denseFixture();
    const extraSeries = ["Student 5", "Student 6", "Student 7"].map((label, index) => ({
      label,
      color: `rgb(0, ${index}, 0)`,
      values: Object.fromEntries(
        fixture.groups.map((group) => [chartGroupKey(group), 0.7]),
      ),
    }));
    render(
      <GroupedBarChart
        title="Per-value Recall"
        groups={fixture.groups}
        series={[...fixture.series, ...extraSeries]}
      />,
    );

    const panel = screen.getByTestId("grouped-bar-chart");
    expect(panel).toHaveAttribute("tabindex", "0");
    expect(panel).toHaveAccessibleName(
      "Per-value Recall. Scroll horizontally to view all chart values.",
    );
    expect(screen.getByTestId("chart-scroll-guidance")).toHaveTextContent(
      "Scroll horizontally to view all chart values.",
    );
  });
});

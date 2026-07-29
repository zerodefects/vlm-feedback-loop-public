// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Color palette for the Compare & Benchmark grouped bar chart.
 *
 * The Teacher uses NVIDIA brand green so it's visually anchored as the
 * accuracy baseline; Student variants cycle through a stable accessible
 * palette tuned for the dark theme. Values are CSS hex with reasonable
 * contrast against the glass-card background.
 */

export const CHART_TEACHER_COLOR = "#76b900"; // NVIDIA green

const CHART_VARIANT_PALETTE: ReadonlyArray<string> = [
  "#3aa0ff", // sky blue
  "#f97373", // coral red
  "#c084fc", // amethyst
  "#facc15", // amber
  "#34d399", // mint
  "#fb923c", // tangerine
  "#22d3ee", // cyan
  "#f472b6", // pink
];

/**
 * Deterministic color lookup for a Student variant by index — wraps the
 * palette if there are more variants than colors. The Teacher always
 * uses ``CHART_TEACHER_COLOR``.
 */
export function variantColor(index: number): string {
  return CHART_VARIANT_PALETTE[index % CHART_VARIANT_PALETTE.length];
}

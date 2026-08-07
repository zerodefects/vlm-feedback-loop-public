// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared 0–1 rate → whole-percent formatter ("0.87" → "87%").
 *
 * Used wherever evaluation/benchmark rates render as percentages
 * (Compare cards, Scale-Up readiness lines). Null / undefined / NaN
 * render as an em-dash so callers can pass optional metrics directly.
 *
 * Rounds half-to-even to match the backend's Python ``:.0%`` formatting,
 * so a client-rendered rate never disagrees with a backend-built message
 * showing the same metric (0.625 → "62%" on both sides, not "63%").
 */

// `roundingMode` is an ES2023 Intl option; the tsconfig lib is ES2022, so
// the options object needs an assertion until the lib target moves up.
const WHOLE_PERCENT = new Intl.NumberFormat("en-US", {
  style: "percent",
  maximumFractionDigits: 0,
  roundingMode: "halfEven",
} as Intl.NumberFormatOptions);

export function formatPct(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return WHOLE_PERCENT.format(v);
}

// Evaluation and benchmark results need enough precision for an SME to compare
// close runs without adding a trailing decimal to already-whole values.
const METRIC_PERCENT = new Intl.NumberFormat("en-US", {
  style: "percent",
  maximumFractionDigits: 1,
  roundingMode: "halfEven",
} as Intl.NumberFormatOptions);

/**
 * 0–1 result metric → percent with one meaningful decimal at most.
 *
 * Keep ``formatPct`` for readiness copy whose backend explanation is rounded
 * to whole percentages. Result surfaces use this formatter so a measured
 * 90.8333% score remains distinguishable from 91.2% instead of both reading
 * as 91%.
 */
export function formatMetricPct(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return METRIC_PERCENT.format(v);
}

// Preserve one decimal place when the underlying rates differ by a fractional
// percentage point. Whole-point deltas stay compact ("+7 pts"), while a real
// 2.5-point gap is never presented as 2 points merely because it lands on a
// half-even tie.
const DELTA_POINTS = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 1,
  useGrouping: false,
  roundingMode: "halfEven",
} as Intl.NumberFormatOptions);

/**
 * Signed rate delta → percentage points ("0.07" → "+7 pts").
 *
 * The delta between two 0–1 rates is percentage points, not a
 * percent-of-percent; the "pts" suffix keeps the semantic unambiguous
 * everywhere a run is compared against a previous one (evaluation
 * banner delta badge, Results panel comparison rows).
 */
export function formatDeltaPoints(delta: number): string {
  // `+ 0` folds Intl's "-0" (a tiny negative delta) into plain zero.
  const pts = Number(DELTA_POINTS.format(delta * 100)) + 0;
  return `${pts >= 0 ? "+" : ""}${DELTA_POINTS.format(pts)} pts`;
}

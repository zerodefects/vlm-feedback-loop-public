// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Scope + metric controls strip on the Compare & Benchmark screen.
 *
 * Holds the metric dropdown (Match rate · Per-value F1 / Precision /
 * Recall), the [Chart] toggle, and the [Benchmark All] / [Benchmark
 * Selected] action buttons. The component is
 * presentation-only — the parent owns the underlying state + handlers.
 */

import { Button, Text } from "@kui/react";

export type MetricSelection =
  | "match_rate"
  | "per_value_f1"
  | "per_value_precision"
  | "per_value_recall";

const METRIC_LABELS: Record<MetricSelection, string> = {
  match_rate: "Match rate",
  per_value_f1: "Per-value F1",
  per_value_precision: "Per-value Precision",
  per_value_recall: "Per-value Recall",
};

export interface CompareScopeBarProps {
  metricSelection: MetricSelection;
  onMetricChange: (next: MetricSelection) => void;
  chartOpen: boolean;
  onToggleChart: () => void;

  selectedCount: number;
  unbenchmarkedCount: number;
  onBenchmarkAll: () => void;
  onBenchmarkSelected: () => void;

  /** ``true`` while any variant is currently being benchmarked —
      disables the action buttons so the sequential queue isn't
      double-fired. */
  busy: boolean;
}

export function CompareScopeBar({
  metricSelection,
  onMetricChange,
  chartOpen,
  onToggleChart,
  selectedCount,
  unbenchmarkedCount,
  onBenchmarkAll,
  onBenchmarkSelected,
  busy,
}: CompareScopeBarProps) {
  return (
    // p-4 (not the p-6 content-card inset): this is a toolbar strip, not
    // a content card — the tighter padding keeps it reading as chrome.
    <div
      className="glass-card p-4 flex flex-wrap items-center gap-4 justify-between"
      data-testid="compare-scope-bar"
    >
      <div className="flex items-center gap-2">
        <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
          Per-field metric:
        </Text>
        <select
          className="glass-input px-3 py-1.5 text-sm"
          value={metricSelection}
          onChange={(e) => onMetricChange(e.target.value as MetricSelection)}
          data-testid="compare-metric-select"
        >
          {Object.entries(METRIC_LABELS).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <Button
          kind="secondary"
          onClick={onBenchmarkAll}
          disabled={busy || unbenchmarkedCount === 0}
          data-testid="benchmark-all-button"
        >
          Benchmark All
        </Button>
        <Button
          kind="secondary"
          onClick={onBenchmarkSelected}
          disabled={busy || selectedCount === 0}
          data-testid="benchmark-selected-button"
        >
          Benchmark Selected ({selectedCount})
        </Button>
        {/* Stable-label pressed pill (SegmentedControl's active treatment)
            rather than a kind-swapping button — the chart is an additive
            panel, not a Table|Chart view swap, so a single toggle pill is
            the right control. */}
        <Button
          kind="tertiary"
          className="px-3 py-1 text-xs font-medium transition-all"
          style={{
            borderRadius: 999,
            background: chartOpen ? "var(--accent-green-bg)" : "transparent",
            color: chartOpen ? "var(--accent-green)" : "var(--text-muted)",
            border: chartOpen
              ? "1px solid var(--accent-green-border)"
              : "1px solid transparent",
          }}
          onClick={onToggleChart}
          data-testid="chart-toggle-button"
          aria-pressed={chartOpen}
        >
          Chart
        </Button>
      </div>
    </div>
  );
}

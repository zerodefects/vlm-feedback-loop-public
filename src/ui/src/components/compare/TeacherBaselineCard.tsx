// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Teacher accuracy baseline card on the Compare & Benchmark screen.
 *
 * The Teacher is shown as an **accuracy baseline only** — no serving
 * metrics, since the Teacher may be hosted externally and serving
 * comparisons aren't apples-to-apples. Source is the most recent
 * completed Teacher evaluation run.
 */

import { Text } from "@kui/react";

import { CHART_TEACHER_COLOR } from "@/lib/chart-palette";
import { formatMetricPct } from "@/lib/format-percent";
import type { EvaluationMetrics, MetricsBucket } from "@/types/evaluation";
import type { SchemaFieldResponse } from "@/types/guidance";

import type { MetricSelection } from "./CompareScopeBar";
import { PerFieldMetricsBlock } from "./PerFieldMetricsBlock";

export interface TeacherBaselineCardProps {
  /** Friendly model name (e.g., ``mistralai/mistral-large-3-…``). */
  modelLabel: string;
  /** Metrics of the latest completed Teacher evaluation run. */
  metrics: EvaluationMetrics | null;
  /** ``"available"`` when a completed Teacher run exists. */
  baselineStatus: "available" | "not_available";
  /** Caption shown when the baseline's snapshotted Teacher or Guidance
      differs from the current project selection. */
  staleNote?: string | null;
  /** Schema fields from the active Guidance (used to walk per-value). */
  coreFields: SchemaFieldResponse[];
  /** Active metric selection from the page-level ScopeBar. */
  metricSelection: MetricSelection;
  perFieldMatchThreshold: number;
  minPerValueF1Threshold: number;
}

export function TeacherBaselineCard({
  modelLabel,
  metrics,
  baselineStatus,
  staleNote,
  coreFields,
  metricSelection,
  perFieldMatchThreshold,
  minPerValueF1Threshold,
}: TeacherBaselineCardProps) {
  const overall: MetricsBucket | null = metrics?.overall ?? null;

  return (
    <div
      className="glass-card glass-card--elevated p-6 flex flex-col gap-3"
      data-testid="teacher-baseline-card"
      style={{
        // Inset box-shadow renders a flat-edge accent stripe that is not
        // clipped by the card's 18 px outer radius (a bordered stripe would
        // be visually absorbed into the corner curve). Stacks on top of the
        // .glass-card--elevated shadow chain.
        boxShadow:
          `inset 4px 0 0 ${CHART_TEACHER_COLOR},` +
          " inset 0 1px 0 rgba(255, 255, 255, 0.09)," +
          " 0 8px 24px rgba(0, 0, 0, 0.35)",
      }}
    >
      <div className="flex items-baseline justify-between gap-3">
        <div className="flex items-baseline gap-2">
          <Text kind="label/bold/sm">Teacher (accuracy baseline)</Text>
          <Text
            kind="label/regular/xs"
            style={{ color: "var(--text-muted)" }}
            data-testid="teacher-baseline-model-label"
          >
            {modelLabel}
          </Text>
        </div>
        {staleNote && (
          <Text
            kind="label/regular/xs"
            style={{ color: "var(--warning-amber, #f59e0b)" }}
            data-testid="teacher-baseline-stale-note"
          >
            {staleNote}
          </Text>
        )}
      </div>

      {baselineStatus === "not_available" || !overall ? (
        <Text
          kind="body/regular/sm"
          style={{ color: "var(--text-muted)" }}
          data-testid="teacher-baseline-empty"
        >
          No completed Teacher evaluation yet — run an evaluation from the labeling
          screen to populate this baseline.
        </Text>
      ) : (
        <>
          <div className="flex items-baseline gap-2">
            <Text
              kind="label/regular/xs"
              className="section-eyebrow"
              style={{ color: "var(--text-muted)" }}
            >
              Exact Match
            </Text>
            <Text
              kind="label/bold/lg"
              style={{ color: "var(--accent-green)" }}
              data-testid="teacher-baseline-exact-match"
            >
              {formatMetricPct(overall.exact_match_rate)}
            </Text>
          </div>

          <PerFieldMetricsBlock
            overall={overall}
            coreFields={coreFields}
            metricSelection={metricSelection}
            perFieldMatchThreshold={perFieldMatchThreshold}
            minPerValueF1Threshold={minPerValueF1Threshold}
            data-testid-prefix="teacher-baseline"
          />
        </>
      )}
    </div>
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Hand-rolled SVG grouped bar chart for the Compare & Benchmark screen.
 *
 * Generic over groups (Core fields when ``Match rate`` is selected, or per-
 * value buckets when ``Per-value F1/Precision/Recall`` is selected) and
 * series (Teacher + Student variants). Hover reveals exact percentages
 * via SVG ``<title>`` (native tooltip).
 *
 * No charting library is added — package.json carries no chart dep and
 * the visual is simple enough that 150 LOC of explicit SVG keeps the
 * bundle lean and the styling consistent with the glass aesthetic.
 */

import { Text } from "@kui/react";

import { chartGroupKey } from "@/lib/chart-group";
import { formatPct } from "@/lib/format-percent";

export interface ChartSeries {
  label: string;
  color: string;
  /** Values keyed by ``chartGroupKey(group)``. Missing values render as
      zero-height bars (placeholder slot kept so groups stay aligned
      across series). */
  values: Record<string, number | null | undefined>;
}

export interface ChartGroup {
  /** Caption rendered under the group's bars. */
  label: string;
  /** Optional cluster caption rendered above ``label`` (e.g. the field
      name when ``label`` is an enum value). */
  cluster?: string | null;
}

export interface GroupedBarChartProps {
  title: string;
  groups: ChartGroup[];
  series: ChartSeries[];
  "data-testid"?: string;
}

const CHART_HEIGHT = 232;
const TOP_PAD = 24;
const BOTTOM_PAD = 76; // room for wrapped group + cluster captions
const LEFT_PAD = 36;
const RIGHT_PAD = 12;
const DEFAULT_GROUP_GAP = 16;
const DENSE_GROUP_GAP = 8;
const DENSE_GROUP_THRESHOLD = 12;
// Compare uses PageContainer's max-w-6xl canvas. After the page and chart
// card insets, 1056 px is available to the SVG at the assigned desktop
// viewport. Wider charts remain intentionally scrollable.
const COMPARE_CHART_CONTENT_WIDTH = 1056;
const BAR_GAP = 2;
// Cap individual bar width so single-group / few-bucket charts don't
// render their bars as flat 80–90 px color blocks (pathology:
// 1 Core field × 7 series at FLOOR_WIDTH=720 over-allocates
// width per bar). Leftover horizontal space is redistributed as a wider
// effective inter-bar gap so the cluster spreads across the inner-group
// span instead of clumping flush-left.
const MAX_BAR_WIDTH = 64;

export function GroupedBarChart({
  title,
  groups,
  series,
  "data-testid": testid,
}: GroupedBarChartProps) {
  if (groups.length === 0 || series.length === 0) {
    return (
      <div className="glass-card p-6" data-testid={testid ?? "grouped-bar-chart-empty"}>
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          No data to chart.
        </Text>
      </div>
    );
  }

  // Compute geometry. Dense categorical charts use tighter groups so a
  // common 25-value field with four comparison series fits the Compare
  // canvas. Do not cap a wider natural layout: doing so makes the final bars
  // and captions escape the SVG's own coordinate bounds. Larger series
  // matrices remain intentionally scrollable instead.
  const groupCount = groups.length;
  const seriesCount = series.length;
  const dense = groupCount > DENSE_GROUP_THRESHOLD;
  const groupGap = dense ? DENSE_GROUP_GAP : DEFAULT_GROUP_GAP;
  const minBarWidth = dense ? 5 : 8;
  const minBarsWidth =
    seriesCount * minBarWidth + BAR_GAP * Math.max(0, seriesCount - 1);
  const minGroupWidth = Math.max(dense ? 32 : 0, minBarsWidth);
  const naturalWidth = LEFT_PAD + RIGHT_PAD + groupCount * (minGroupWidth + groupGap);
  // A few groups still fill a useful 720 px plot. Dense layouts use their
  // full natural width, which is 1048 px for Freiburg's 25 values × four
  // series and therefore fits without clipping in the 1056 px canvas.
  const FLOOR_WIDTH = 720;
  const width = Math.max(FLOOR_WIDTH, naturalWidth);
  const usableWidth = width - LEFT_PAD - RIGHT_PAD;
  const groupWidth = usableWidth / groupCount;
  const innerGroupWidth = Math.max(minBarsWidth, groupWidth - groupGap);
  const naiveBarWidth = Math.max(
    minBarWidth,
    (innerGroupWidth - BAR_GAP * (seriesCount - 1)) / seriesCount,
  );
  const barWidth = Math.min(naiveBarWidth, MAX_BAR_WIDTH);
  // When the cap fires the bars no longer fill `innerGroupWidth`. Push
  // the leftover space into a wider inter-bar gap so the cluster spans
  // its allotted inner-group region and reads as a chart, not a stripe.
  const totalBarsWidth = barWidth * seriesCount;
  const effectiveBarGap =
    seriesCount > 1
      ? Math.max(BAR_GAP, (innerGroupWidth - totalBarsWidth) / (seriesCount - 1))
      : BAR_GAP;
  const plotHeight = CHART_HEIGHT - TOP_PAD - BOTTOM_PAD;

  // 0%-100% percent scale.
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <div
      className="glass-card p-6 flex flex-col gap-3 overflow-x-auto"
      data-testid={testid ?? "grouped-bar-chart"}
      tabIndex={width > COMPARE_CHART_CONTENT_WIDTH ? 0 : undefined}
      aria-label={
        width > COMPARE_CHART_CONTENT_WIDTH
          ? `${title}. Scroll horizontally to view all chart values.`
          : undefined
      }
    >
      <div className="flex items-center justify-between">
        <Text kind="label/bold/sm">{title}</Text>
        <div className="flex items-center gap-3 flex-wrap" data-testid="chart-legend">
          {series.map((s) => (
            <div
              key={s.label}
              className="flex items-center gap-1.5"
              data-testid={`chart-legend-${s.label}`}
            >
              <span
                aria-hidden="true"
                style={{
                  display: "inline-block",
                  width: 12,
                  height: 12,
                  borderRadius: 2,
                  backgroundColor: s.color,
                }}
              />
              <Text kind="label/regular/xs" style={{ color: "var(--text-secondary)" }}>
                {s.label}
              </Text>
            </div>
          ))}
        </div>
      </div>

      {width > COMPARE_CHART_CONTENT_WIDTH && (
        <Text
          kind="body/regular/xs"
          style={{ color: "var(--text-muted)" }}
          data-testid="chart-scroll-guidance"
        >
          Scroll horizontally to view all chart values.
        </Text>
      )}

      <svg
        width={width}
        height={CHART_HEIGHT}
        role="img"
        aria-label={title}
        style={{ overflow: "visible" }}
      >
        {/* Y-axis tick lines + labels (no axis line — keeps the chrome
            light against the glass background). */}
        {yTicks.map((t) => {
          const y = TOP_PAD + plotHeight * (1 - t);
          return (
            <g key={t}>
              <line
                x1={LEFT_PAD}
                x2={width - RIGHT_PAD}
                y1={y}
                y2={y}
                stroke="var(--text-muted)"
                strokeWidth={0.5}
                opacity={0.25}
              />
              <text
                x={LEFT_PAD - 6}
                y={y + 4}
                textAnchor="end"
                fontSize={10}
                fill="var(--text-muted)"
              >
                {formatPct(t)}
              </text>
            </g>
          );
        })}

        {/* Group bars + per-bar captions (no cluster repetition here —
            cluster captions are emitted once per run below). */}
        {groups.map((group, gIdx) => {
          const groupX = LEFT_PAD + gIdx * groupWidth + groupGap / 2;
          const labelLines = group.label.split("_");
          const labelX = groupX + innerGroupWidth / 2;
          const labelY = CHART_HEIGHT - BOTTOM_PAD + 14;
          const denseLabel = group.label.replaceAll("_", " ");
          return (
            <g key={`${group.cluster ?? ""}::${group.label}::${gIdx}`}>
              {series.map((s, sIdx) => {
                const v = s.values[chartGroupKey(group)];
                const value = typeof v === "number" ? v : 0;
                const clamped = Math.max(0, Math.min(1, value));
                const barH = clamped * plotHeight;
                const x = groupX + sIdx * (barWidth + effectiveBarGap);
                const y = TOP_PAD + plotHeight - barH;
                const pct = Math.round(value * 100);
                return (
                  <rect
                    key={`${s.label}::${gIdx}`}
                    x={x}
                    y={y}
                    width={barWidth}
                    height={barH}
                    fill={s.color}
                    rx={1.5}
                    data-testid={`chart-bar-${s.label}-${chartGroupKey(group)}`}
                    data-percent={pct}
                  >
                    <title>
                      {`${s.label} · ${
                        group.cluster ? `${group.cluster} / ` : ""
                      }${group.label}: ${pct}%`}
                    </title>
                  </rect>
                );
              })}
              {/* Per-bar group caption (e.g. enum value name in
                  Per-value mode, or Core field name in Match-rate mode). */}
              <text
                x={labelX}
                y={labelY}
                textAnchor="middle"
                fontSize={11}
                fill="var(--text-secondary)"
                data-testid={`chart-group-label-${chartGroupKey(group)}`}
                transform={dense ? `rotate(-40 ${labelX} ${labelY})` : undefined}
              >
                {dense
                  ? denseLabel
                  : labelLines.length === 1
                    ? group.label
                    : labelLines.map((line, lineIndex) => (
                        <tspan
                          key={`${line}-${lineIndex}`}
                          x={labelX}
                          dy={lineIndex === 0 ? 0 : 12}
                        >
                          {line}
                        </tspan>
                      ))}
              </text>
            </g>
          );
        })}

        {/* Cluster captions: one per run of consecutive groups sharing
            the same ``cluster`` value (e.g. a `primary_damage` cluster
            spans crush/dent/scratch/rip/tear/leak in Per-value mode).
            Match-rate mode has no clusters → this loop emits nothing.
            Each run gets a thin bracket-style underline so the cluster
            boundary is visible without a geometric inter-cluster gap. */}
        {computeClusterRuns(groups).map((run, rIdx) => {
          const startX = LEFT_PAD + run.startIdx * groupWidth + groupGap / 2;
          const endX = LEFT_PAD + (run.endIdx + 1) * groupWidth - groupGap / 2;
          const centerX = (startX + endX) / 2;
          const bracketY = CHART_HEIGHT - BOTTOM_PAD + 24;
          return (
            <g
              key={`cluster-${run.cluster}-${rIdx}`}
              data-testid={`chart-cluster-${run.cluster}`}
            >
              <line
                x1={startX}
                x2={endX}
                y1={bracketY}
                y2={bracketY}
                stroke="var(--text-muted)"
                strokeWidth={0.75}
                opacity={0.5}
              />
              <text
                x={centerX}
                y={bracketY + 14}
                textAnchor="middle"
                fontSize={11}
                fill="var(--text-muted)"
              >
                {run.cluster.replaceAll("_", " ")}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/**
 * Walk ``groups`` and emit one record per maximal run of consecutive
 * groups sharing the same non-null ``cluster`` value. Used by the chart
 * to render each cluster caption once instead of once per bar
 * (the `|--- primary_damage ---|` bracket band).
 *
 * In Match-rate mode every group has ``cluster === undefined`` and this
 * function returns an empty array. In Per-value mode adjacent groups
 * with the same field name collapse into one run.
 */
interface ClusterRun {
  cluster: string;
  startIdx: number;
  endIdx: number;
}

function computeClusterRuns(groups: ChartGroup[]): ClusterRun[] {
  const runs: ClusterRun[] = [];
  let current: ClusterRun | null = null;
  groups.forEach((g, idx) => {
    if (!g.cluster) {
      if (current) {
        runs.push(current);
        current = null;
      }
      return;
    }
    if (current && current.cluster === g.cluster) {
      current.endIdx = idx;
    } else {
      if (current) runs.push(current);
      current = { cluster: g.cluster, startIdx: idx, endIdx: idx };
    }
  });
  if (current) runs.push(current);
  return runs;
}

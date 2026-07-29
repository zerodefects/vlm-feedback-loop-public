// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Evaluation Results detail panel — slides in as an overlay on the right
 * side of the labeling screen.
 */

import { Button, Text } from "@kui/react";
import { AlertCircle, AlertTriangle, ChevronDown, ChevronUp, X } from "lucide-react";
import { Fragment, useState } from "react";

import type { CoverageGap, EvaluationRunResponse } from "@/types/evaluation";
import type { GuidanceResponse } from "@/types/guidance";
import type { ModelConfigResponse } from "@/types/nim";

import { formatTimestamp } from "@/lib/format-date";
import { formatDeltaPoints, formatPct } from "@/lib/format-percent";
import { titleCasePreset } from "@/lib/formatPreset";

import { safeMetrics } from "./metricsHelpers";

interface ResultsPanelProps {
  run: EvaluationRunResponse;
  onClose: () => void;
  /**
   * Optional lookups so the metadata subtitle renders human-readable values
   * (model name, guidance "v{N}") instead of raw IDs.
   * Both are already loaded on LabelingPage via React Query — passing them
   * avoids a second fetch and keeps this panel pure.
   */
  teacherConfigs?: ModelConfigResponse[];
  guidanceVersions?: GuidanceResponse[];
}

export function ResultsPanel({
  run,
  onClose,
  teacherConfigs,
  guidanceVersions,
}: ResultsPanelProps) {
  const [showPerValue, setShowPerValue] = useState(false);
  const metrics = safeMetrics(run.metrics);
  // On an incomplete run, collapse the Returning/New comparison
  // and mark per-field rates diagnostic. The 4-row block implies regression
  // semantics that partial data can't support.
  const isIncomplete = run.status === "incomplete";
  const showComparison =
    !isIncomplete &&
    run.returning_example_keys != null &&
    run.previous_overall_exact_match != null;

  return (
    <div
      className="fixed top-0 right-0 h-full w-[420px] glass-card z-50 overflow-y-auto"
      style={{
        borderRadius: 0,
        borderLeft: "1px solid rgba(255, 255, 255, 0.12)",
        boxShadow: "-8px 0 24px rgba(0, 0, 0, 0.5)",
      }}
      data-testid="results-panel"
    >
      <div
        className="flex items-center justify-between p-4 border-b"
        style={{ borderColor: "var(--glass-border)" }}
      >
        <Text kind="title/sm">Evaluation Results</Text>
        <Button kind="tertiary" onClick={onClose} data-testid="close-results">
          <X size={16} />
        </Button>
      </div>

      <div className="p-4 space-y-4">
        {/* ── Run metadata + config snapshot ────────────────────── */}
        {/* Renders e.g. "nvidia/cosmos-reason2-8b · Guidance v3 ·
           Precise". When the parent supplies the teacherConfigs/guidanceVersions
           lookups, render the human-readable model name and "v{N}" version
           label. Without the lookups (e.g., direct unit-test render), fall back
           to the truncated id so the panel still renders deterministically. */}
        {/* Faint tier (one step below the section eyebrows) so the first
           ACCURACY eyebrow reads as a distinct section start. */}
        <div
          style={{ color: "var(--text-faint)", fontSize: 12 }}
          data-testid="run-metadata"
        >
          <div>
            {/* Strip leading `pool-` if present so we render `Pool v4`
               instead of the redundant `Pool pool-v4` that the raw id
               shape produces. */}
            <MetadataSegments
              segments={[
                formatTimestamp(run.created_at),
                `Pool ${
                  run.pool_version_id?.replace(/^pool-/, "").slice(0, 8) ??
                  run.pool_version_id?.slice(0, 8)
                }`,
                metrics?.overall.example_count != null
                  ? `${metrics.overall.example_count} images`
                  : null,
              ]}
            />
          </div>
          <div className="mt-1">
            {(() => {
              const modelLabel =
                teacherConfigs?.find((m) => m.model_config_id === run.model_config_id)
                  ?.model_name ?? run.model_config_id?.slice(0, 20);
              const guidanceVersion = guidanceVersions?.find(
                (g) => g.guidance_id === run.guidance_id,
              )?.version_number;
              const guidanceLabel =
                guidanceVersion != null
                  ? `Guidance v${guidanceVersion}`
                  : `Guidance ${run.guidance_id?.slice(0, 8)}`;
              const preset = run.generation_preset_key
                ? titleCasePreset(run.generation_preset_key)
                : "Precise";
              const thinking = run.thinking_mode_effective
                ? titleCasePreset(run.thinking_mode_effective)
                : null;
              const visual = run.visual_budget_preset_key
                ? titleCasePreset(run.visual_budget_preset_key)
                : null;
              return (
                <MetadataSegments
                  segments={[
                    modelLabel,
                    guidanceLabel,
                    preset,
                    thinking ? `Thinking ${thinking}` : null,
                    visual ? `Visual ${visual}` : null,
                  ]}
                />
              );
            })()}
          </div>
        </div>

        {/* ── Incomplete warning ───────────────────────────────────── */}
        {run.status === "incomplete" &&
          (() => {
            // "Incomplete: N of M examples failed" — N is derived
            // from pool total minus successful-metric count. Fall back to
            // "some examples failed" when the derivation isn't possible OR
            // when the derived failure count is 0 (an `incomplete` status
            // with zero failures is logically inconsistent; mirrors the
            // EvaluationStrip incomplete-banner guard).
            const total = run.progress?.total;
            const succeeded = metrics?.overall.example_count;
            const failed =
              total != null && succeeded != null ? total - succeeded : null;
            return (
              <div
                className="glass-info flex items-center gap-2 px-3 py-2"
                style={{ color: "var(--warning-amber)" }}
                data-testid="incomplete-warning"
              >
                <AlertCircle size={14} />
                <Text kind="label/regular/xs">
                  {failed != null && failed > 0 && total != null
                    ? `Incomplete: ${failed} of ${total} examples failed. Results are diagnostic only.`
                    : "Incomplete: some examples failed. Results are diagnostic only."}
                </Text>
              </div>
            );
          })()}

        {/* ── Accuracy comparison ──────────────────────────────────── */}
        {metrics && (
          <>
            <div data-testid="accuracy-section">
              <Text
                kind="label/regular/xs"
                className="section-eyebrow"
                style={{
                  marginBottom: 8,
                  color: "var(--text-muted)",
                  display: "block",
                }}
              >
                {showComparison ? "ACCURACY COMPARISON" : "ACCURACY"}
              </Text>

              {/* The previous
                  Overall vs current Returning comparison is the core regression
                  signal and gets the bold treatment. New + Overall are
                  secondary context. On first runs (no comparison rows),
                  Overall stands alone and is bolded as the headline. */}
              {showComparison &&
                run.previous_overall_exact_match != null &&
                run.returning_example_keys && (
                  <>
                    <MetricLine
                      label="Previous"
                      rate={run.previous_overall_exact_match}
                      count={run.returning_example_keys.length}
                      bold
                    />
                    <MetricLine
                      label="Same images now"
                      rate={metrics.returning?.exact_match_rate}
                      count={metrics.returning?.example_count}
                      delta={
                        metrics.returning?.exact_match_rate != null
                          ? metrics.returning.exact_match_rate -
                            run.previous_overall_exact_match
                          : undefined
                      }
                      bold
                    />
                    {metrics.new && (
                      <MetricLine
                        label="New images"
                        rate={metrics.new.exact_match_rate}
                        count={metrics.new.example_count}
                      />
                    )}
                  </>
                )}

              <MetricLine
                label="Overall"
                rate={metrics.overall.exact_match_rate}
                count={metrics.overall.example_count}
                totalCount={isIncomplete ? (run.progress?.total ?? null) : null}
                bold={!showComparison}
              />
            </div>

            {/* ── Per-field match rates ────────────────────────────── */}
            <div data-testid="per-field-section">
              <Text
                kind="label/regular/xs"
                className="section-eyebrow"
                style={{
                  marginBottom: 8,
                  color: "var(--text-muted)",
                  display: "block",
                }}
              >
                {isIncomplete
                  ? "PER-FIELD MATCH RATES (diagnostic)"
                  : "PER-FIELD MATCH RATES"}
              </Text>
              {Object.entries(metrics.overall.per_field_match_rates).map(
                ([field, rate]) => (
                  <FieldBar key={field} field={field} rate={rate} />
                ),
              )}
            </div>

            {/* ── Per-value breakdown (expandable) ─────────────────── */}
            <div data-testid="per-value-section">
              <Button
                kind="tertiary"
                className="flex items-center gap-1 cursor-pointer"
                onClick={() => setShowPerValue(!showPerValue)}
                style={{ color: "var(--text-secondary)", padding: 0 }}
                data-testid="toggle-per-value"
              >
                {showPerValue ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                <Text kind="label/regular/xs">Per-value breakdown</Text>
              </Button>

              {showPerValue && (
                <div className="mt-2 space-y-3">
                  {/* Every allowed value of an enum or
                     enum-set Core field gets a row in the per-value table —
                     including values with no examples in the test pool, which
                     render as `—  —  —  (no examples)` so coverage gaps are
                     visible inline rather than only in the COVERAGE GAPS
                     section below. We pull `allowed_values` from the run's
                     active Guidance schema. */}
                  {Object.entries(metrics.overall.per_value_metrics).map(
                    ([field, values]) => {
                      const schemaField = guidanceVersions
                        ?.find((g) => g.guidance_id === run.guidance_id)
                        ?.schema_fields?.find((f) => f.field_name === field);
                      const allowed = schemaField?.allowed_values ?? [];
                      // Combine values with metrics + values from schema that
                      // have no examples. Preserve schema order; metric-only
                      // values (shouldn't normally occur but defensive) appear
                      // last.
                      const allValues = [
                        ...allowed,
                        ...Object.keys(values).filter((v) => !allowed.includes(v)),
                      ];
                      const display =
                        allValues.length > 0 ? allValues : Object.keys(values);
                      return (
                        <div key={field}>
                          <Text
                            kind="label/bold/xs"
                            style={{ color: "var(--text-secondary)" }}
                          >
                            {field}
                          </Text>
                          <table className="w-full mt-1" style={{ fontSize: 12 }}>
                            <thead>
                              <tr
                                className="section-eyebrow"
                                style={{
                                  color: "var(--text-muted)",
                                  fontSize: 10,
                                }}
                              >
                                <th className="text-left py-1">Value</th>
                                <th className="text-right py-1">F1</th>
                                <th className="text-right py-1">Prec</th>
                                <th className="text-right py-1">Recall</th>
                                <th className="text-left py-1 pl-3"></th>
                              </tr>
                            </thead>
                            <tbody>
                              {display.map((val) => {
                                const m = values[val];
                                if (m == null) {
                                  // Schema-allowed value with no test-pool examples —
                                  // emit the `—  —  —  (no examples)` row.
                                  return (
                                    <tr
                                      key={val}
                                      data-testid={`pv-${field}-${val}-empty`}
                                    >
                                      <td className="py-1">{val}</td>
                                      <td
                                        className="text-right"
                                        style={{ color: "var(--text-faint)" }}
                                      >
                                        —
                                      </td>
                                      <td
                                        className="text-right"
                                        style={{ color: "var(--text-faint)" }}
                                      >
                                        —
                                      </td>
                                      <td
                                        className="text-right"
                                        style={{ color: "var(--text-faint)" }}
                                      >
                                        —
                                      </td>
                                      <td
                                        className="text-left py-1 pl-3"
                                        style={{
                                          color: "var(--text-muted)",
                                          fontSize: 10,
                                        }}
                                      >
                                        (no examples)
                                      </td>
                                    </tr>
                                  );
                                }
                                const belowThreshold = m.f1 < 0.8;
                                return (
                                  <tr key={val} data-testid={`pv-${field}-${val}`}>
                                    <td className="py-1">{val}</td>
                                    <td
                                      className="text-right"
                                      style={
                                        belowThreshold
                                          ? { color: "var(--error-red-text)" }
                                          : undefined
                                      }
                                    >
                                      <span className="inline-flex items-center justify-end gap-1">
                                        {formatPct(m.f1)}
                                        {belowThreshold && (
                                          <AlertTriangle
                                            size={14}
                                            aria-label="below 80%"
                                            style={{ color: "var(--warning-amber)" }}
                                          />
                                        )}
                                      </span>
                                    </td>
                                    <td className="text-right">
                                      {formatPct(m.precision)}
                                    </td>
                                    <td className="text-right">
                                      {formatPct(m.recall)}
                                    </td>
                                    {/* An inline
                                       "below 80%" text suffix sits beside the icon
                                       so the affordance is screen-reader-legible
                                       without depending on the icon's aria-label. */}
                                    <td
                                      className="text-left py-1 pl-3"
                                      style={{ fontSize: 10 }}
                                    >
                                      {belowThreshold && (
                                        <Text
                                          kind="label/regular/xs"
                                          style={{ color: "var(--warning-amber)" }}
                                        >
                                          below 80%
                                        </Text>
                                      )}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      );
                    },
                  )}
                </div>
              )}
            </div>

            {/* ── Coverage gaps ────────────────────────────────────── */}
            {run.coverage_gaps && run.coverage_gaps.length > 0 && (
              <div data-testid="coverage-gaps">
                <Text
                  kind="label/regular/xs"
                  className="section-eyebrow"
                  style={{
                    marginBottom: 8,
                    color: "var(--text-muted)",
                    display: "block",
                  }}
                >
                  COVERAGE GAPS
                </Text>
                {run.coverage_gaps.map((gap: CoverageGap) => (
                  <div
                    key={gap.field_name}
                    className="flex items-start gap-2 mb-1"
                    style={{ color: "var(--warning-amber)", fontSize: 12 }}
                  >
                    <AlertCircle size={12} className="mt-0.5 shrink-0" />
                    <Text kind="label/regular/xs">
                      {gap.field_name}: no examples with{" "}
                      {gap.missing_values.map((v) => `"${v}"`).join(", ")}
                    </Text>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Small components ───────────────────────────────────────────────────────

function MetricLine({
  label,
  rate,
  count,
  totalCount,
  delta,
  bold,
}: {
  label: string;
  rate?: number | null;
  count?: number;
  totalCount?: number | null;
  delta?: number | null;
  bold?: boolean;
}) {
  if (rate == null) return null;
  const kind = bold ? "label/bold/sm" : ("label/regular/sm" as const);
  const countLabel =
    totalCount != null && count != null
      ? `${count} of ${totalCount} images`
      : `${count} images`;
  return (
    <div className="flex items-center justify-between py-1">
      <Text kind={kind}>{label}</Text>
      <div className="flex items-center gap-2">
        <Text kind={kind}>
          {formatPct(rate)} ({countLabel})
        </Text>
        {/* Fixed-width delta slot rendered on every row so the value
           column right-aligns at one edge whether or not a row carries
           a delta. */}
        <span style={{ minWidth: 52, textAlign: "right" }}>
          {delta != null && (
            <Text
              kind="label/regular/xs"
              style={{
                color: delta >= 0 ? "var(--accent-green)" : "var(--error-red-text)",
              }}
            >
              {formatDeltaPoints(delta)}
            </Text>
          )}
        </span>
      </div>
    </div>
  );
}

/**
 * "·"-separated metadata run-on line. Each token is one non-wrapping
 * span that carries its FOLLOWING separator, so a wrap breaks between
 * spans and the "·" ends the previous line (comma-style) instead of
 * dangling at the start of the next one.
 */
function MetadataSegments({
  segments,
}: {
  segments: Array<string | null | undefined>;
}) {
  const present = segments.filter((s): s is string => s != null);
  return (
    <>
      {present.map((segment, i) => (
        <Fragment key={i}>
          {i > 0 && " "}
          <span className="whitespace-nowrap">
            {segment}
            {i < present.length - 1 && " ·"}
          </span>
        </Fragment>
      ))}
    </>
  );
}

function FieldBar({ field, rate }: { field: string; rate: number }) {
  return (
    <div className="mb-2">
      <div className="flex items-center justify-between mb-1">
        <Text kind="label/regular/xs">{field}</Text>
        <Text kind="label/regular/xs">{formatPct(rate)}</Text>
      </div>
      <div className="h-2 rounded-full" style={{ background: "var(--bar-track)" }}>
        <div
          className="h-2 rounded-full"
          style={{
            width: `${Math.min(100, rate * 100)}%`,
            background: "var(--accent-green)",
            transition: "width 300ms",
          }}
        />
      </div>
    </div>
  );
}

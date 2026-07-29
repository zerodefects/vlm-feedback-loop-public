// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared per-field metrics block used by the Teacher baseline card and by
 * each Student variant card on the Compare & Benchmark screen.
 *
 * In ``Match rate`` mode, renders a one-line ``<field> <pct>`` row per
 * Core field. In ``Per-value F1 / Precision / Recall`` mode, expands
 * categorical Core fields (enum, enum_set, boolean) to show every value's
 * F1/precision/recall; integer/string fields collapse to their match rate
 * with a hint.
 *
 * The categorical-vs-non-categorical decision is driven by
 * ``SchemaFieldResponse.type`` from the active Guidance — that's the
 * single source of truth. Per-value buckets are read from
 * ``MetricsBucket.per_value_metrics`` populated by the canonical Exact
 * Match evaluator. Per-value lists wrap via CSS grid so
 * 25-bucket fields render without horizontal overflow.
 */

import { Text } from "@kui/react";

import { formatPct } from "@/lib/format-percent";
import type { MetricsBucket, PerValueMetric } from "@/types/evaluation";
import type { SchemaFieldResponse } from "@/types/guidance";

import type { MetricSelection } from "./CompareScopeBar";

const CATEGORICAL_TYPES: ReadonlySet<string> = new Set(["enum", "enum_set", "boolean"]);

const PER_VALUE_THRESHOLD = 0.8; // below-80% values get the flag color

export interface PerFieldMetricsBlockProps {
  overall: MetricsBucket;
  coreFields: SchemaFieldResponse[];
  metricSelection: MetricSelection;
  /** Used as the ``data-testid`` prefix for sub-elements so different
      cards on the same page don't share IDs. */
  "data-testid-prefix"?: string;
}

export function PerFieldMetricsBlock({
  overall,
  coreFields,
  metricSelection,
  "data-testid-prefix": prefix = "per-field",
}: PerFieldMetricsBlockProps) {
  const sortedCore = [...coreFields].sort((a, b) => a.display_order - b.display_order);

  if (metricSelection === "match_rate") {
    return (
      <div className="flex flex-col gap-1" data-testid={`${prefix}-block-match-rate`}>
        <Text
          kind="label/regular/xs"
          className="section-eyebrow"
          style={{ color: "var(--text-muted)" }}
        >
          Per-field
        </Text>
        {/* Two max-content columns keep each value adjacent to its label
            instead of stretching the pair across the full card width. */}
        <div
          className="grid items-baseline gap-x-6 gap-y-0.5 w-fit"
          style={{ gridTemplateColumns: "max-content max-content" }}
        >
          {sortedCore.map((field) => {
            const rate = overall.per_field_match_rates?.[field.field_name];
            return (
              <div
                key={field.field_name}
                style={{ display: "contents" }}
                data-testid={`${prefix}-row-${field.field_name}`}
              >
                <Text kind="body/regular/sm">{field.field_name}</Text>
                <Text
                  kind="body/regular/sm"
                  style={{
                    textAlign: "right",
                    color:
                      rate != null && rate < PER_VALUE_THRESHOLD
                        ? "var(--warning-amber, #f59e0b)"
                        : "var(--text-primary)",
                  }}
                >
                  {formatPct(rate)}
                </Text>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // Per-value mode (F1 / Precision / Recall)
  const valueKey: keyof PerValueMetric =
    metricSelection === "per_value_f1"
      ? "f1"
      : metricSelection === "per_value_precision"
        ? "precision"
        : "recall";

  return (
    <div
      className="flex flex-col gap-2"
      data-testid={`${prefix}-block-per-value`}
      data-metric-key={valueKey}
    >
      <Text
        kind="label/regular/xs"
        className="section-eyebrow"
        style={{ color: "var(--text-muted)" }}
      >
        Per-field
      </Text>
      {sortedCore.map((field) => {
        const fieldRate = overall.per_field_match_rates?.[field.field_name];
        const isCategorical = CATEGORICAL_TYPES.has(field.type);
        const perValue = overall.per_value_metrics?.[field.field_name] ?? {};
        const valueLabels = isCategorical ? collectValueLabels(field, perValue) : [];

        return (
          <div
            key={field.field_name}
            className="flex flex-col gap-1"
            data-testid={`${prefix}-row-${field.field_name}`}
            data-categorical={isCategorical}
          >
            <div className="flex items-baseline gap-6 w-fit">
              <Text kind="body/regular/sm">{field.field_name}</Text>
              <div className="flex items-baseline gap-3">
                <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
                  match rate
                </Text>
                <Text
                  kind="body/regular/sm"
                  style={{
                    color:
                      fieldRate != null && fieldRate < PER_VALUE_THRESHOLD
                        ? "var(--warning-amber, #f59e0b)"
                        : "var(--text-primary)",
                  }}
                >
                  {formatPct(fieldRate)}
                </Text>
              </div>
            </div>

            {isCategorical && valueLabels.length > 0 && (
              <div
                className="ml-4 grid gap-1"
                style={{
                  // At 25 allowed values the list wraps to multiple
                  // lines instead of overflowing horizontally.
                  gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
                }}
                data-testid={`${prefix}-row-${field.field_name}-values`}
              >
                {valueLabels.map((value) => {
                  const m = perValue[value];
                  const v = m ? m[valueKey] : null;
                  const flagged = v != null && v < PER_VALUE_THRESHOLD;
                  return (
                    <div
                      key={value}
                      className="flex items-baseline gap-2 truncate"
                      data-testid={`${prefix}-value-${field.field_name}-${value}`}
                      title={`${value}: ${formatPct(v)} ${valueKey}`}
                    >
                      <Text
                        kind="label/regular/xs"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {value}
                      </Text>
                      <Text
                        kind="label/regular/xs"
                        style={{
                          color: flagged
                            ? "var(--warning-amber, #f59e0b)"
                            : "var(--text-primary)",
                        }}
                      >
                        {formatPct(v)}
                      </Text>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Non-categorical fields (integer/string) have no per-value
                breakdown. The field row renders alone with no explanatory
                caption — the absence of indented value rows is itself the
                signal. We attach a hidden ``data-testid`` marker so tests
                can still detect non-categorical rows without the visual
                noise of a repeated boilerplate caption (which compounds
                when a schema has multiple integer/string Core fields). */}
            {!isCategorical && (
              <span
                aria-hidden="true"
                data-testid={`${prefix}-row-${field.field_name}-noncategorical`}
                hidden
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

/**
 * Collect the value labels to display for a categorical field. Boolean
 * fields are anchored to ``["true", "false"]``; enum/enum_set fields use
 * the field's declared ``allowed_values`` so the row order is stable
 * across cards (``allowed_values`` is the canonical list).
 */
function collectValueLabels(
  field: SchemaFieldResponse,
  perValue: Record<string, PerValueMetric>,
): string[] {
  if (field.type === "boolean") {
    return ["true", "false"];
  }
  const declared = field.allowed_values ?? [];
  if (declared.length > 0) return declared;
  // Fallback: derive from observed metrics if allowed_values is missing.
  return Object.keys(perValue);
}

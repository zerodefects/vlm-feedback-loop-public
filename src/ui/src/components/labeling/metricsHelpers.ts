// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  EvaluationMetrics,
  MetricsBucket,
  PerValueMetric,
} from "@/types/evaluation";

interface SafeMetricsBucket {
  exact_match_rate: number | null;
  example_count: number;
  per_field_match_rates: Record<string, number>;
  per_value_metrics: Record<string, Record<string, PerValueMetric>>;
}

export interface SafeMetrics {
  overall: SafeMetricsBucket;
  returning: SafeMetricsBucket | null;
  new: SafeMetricsBucket | null;
}

function safeBucket(bucket: MetricsBucket | null | undefined): SafeMetricsBucket {
  return {
    exact_match_rate:
      typeof bucket?.exact_match_rate === "number" ? bucket.exact_match_rate : null,
    example_count: typeof bucket?.example_count === "number" ? bucket.example_count : 0,
    per_field_match_rates: bucket?.per_field_match_rates ?? {},
    per_value_metrics: bucket?.per_value_metrics ?? {},
  };
}

export function safeMetrics(
  metrics: EvaluationMetrics | null | undefined,
): SafeMetrics | null {
  if (!metrics) return null;
  return {
    overall: safeBucket(metrics.overall),
    returning: metrics.returning ? safeBucket(metrics.returning) : null,
    new: metrics.new ? safeBucket(metrics.new) : null,
  };
}

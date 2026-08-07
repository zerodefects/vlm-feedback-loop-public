// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Types for evaluation runs, triggers, and the Scale-Up gate. */

// ── Evaluation Run ─────────────────────────────────────────────────────────

export interface EvaluationRunResponse {
  run_id: string;
  run_type: string;
  status: string; // queued | running | canceling | completed | incomplete | canceled | failed
  status_reason: string | null;
  pool_version_id: string | null;
  guidance_id: string | null;
  model_config_id: string | null;
  icl_mode: string | null;
  evaluation_source: string | null;
  generation_preset_key: string | null;
  thinking_mode_effective: string | null;
  visual_budget_preset_key: string | null;
  structured_generation_mode_effective: string | null;
  inference_contract: Record<string, unknown> | null;
  icl_eligible_count_at_start: number | null;
  icl_eligible_count_at_completion: number | null;

  /** Set on Student evaluation runs (quality/serving), null on Teacher
      runs — the Compare screen's Teacher-baseline discriminator. */
  student_model_config_id: string | null;
  progress: { processed: number; total: number } | null;
  metrics: EvaluationMetrics | null;
  previous_pool_version: number | null;
  returning_example_keys: string[] | null;
  new_example_keys: string[] | null;
  previous_overall_exact_match: number | null;
  coverage_gaps: CoverageGap[] | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface EvaluationMetrics {
  overall: MetricsBucket;
  returning: MetricsBucket | null;
  new: MetricsBucket | null;
  benchmarks?: ServingBenchmarkResult[];
  benchmark_workload?: ServingBenchmarkWorkload;
}

export interface ServingBenchmarkResult {
  concurrency: number;
  status?: "passed" | "failed";
  latency_p50_ms: number | null;
  latency_p90_ms: number | null;
  latency_p99_ms: number | null;
  request_throughput_rps?: number | null;
  attempted_request_count?: number;
  successful_request_count?: number;
  failed_request_count?: number;
  failure_rate?: number | null;
  input_tokens_mean?: number | null;
  output_tokens_mean?: number | null;
  driver?: string;
  driver_version?: string;
  failure_reason?: string | null;
}

export interface ServingBenchmarkWorkload {
  version?: string;
  workload_hash?: string;
  pool_id?: string;
  pool_member_count?: number;
  selected_count?: number;
  selection_policy?: string;
  guidance_id?: string;
  guidance_schema_hash?: string;
  prompt_hash?: string | null;
  inference_contract?: Record<string, unknown>;
  visual_budget_params?: Record<string, unknown>;
  output_limit_mode?: string;
  kv_cache_reuse?: string;
  driver?: { name?: string; version?: string };
  tokenizer?: string;
  application_version?: string;
  code_revision?: string | null;
  created_at?: string;
}

export interface MetricsBucket {
  exact_match_rate: number;
  example_count: number;
  per_field_match_rates: Record<string, number>;
  per_value_metrics: Record<string, Record<string, PerValueMetric>>;
}

export interface PerValueMetric {
  precision: number;
  recall: number;
  f1: number;
}

export interface CoverageGap {
  field_name: string;
  field_type: string;
  missing_values: string[];
}

export interface EvaluationRunListResponse {
  items: EvaluationRunResponse[];
  next_cursor: string | null;
}

export interface EvaluationRunCreateResponse {
  run_id: string;
  run_type: string;
  status: string;
  pool_version: number;
  guidance_id: string | null;
  model_config_id: string | null;
  generation_preset_key: string | null;
  thinking_mode_effective: string | null;
  visual_budget_preset_key: string | null;
  structured_generation_mode_effective: string | null;
  evaluation_source: string | null;
  icl_mode: string | null;
  created_at: string;
  superseded_run_id: string | null;
}

// ── Trigger Status ─────────────────────────────────────────────────────────

interface TriggerInfo {
  is_active: boolean;
  dismissed: boolean;
  message: string;
  context: Record<string, unknown> | null;
}

export interface TriggerStatusResponse {
  auto_evaluate_enabled: boolean;
  first_pool_threshold: TriggerInfo;
  configuration_change: TriggerInfo;
  icl_growth: TriggerInfo;
  updated_at: string;
}

// ── Scale-Up Gate ──────────────────────────────────────────────────────────

export interface GateCriterion {
  criterion_name: string;
  passed: boolean;
  current_value: number;
  threshold: number;
  message: string;
  details: Record<string, unknown> | null;
}

export interface ScaleUpGateResponse {
  gate_status: "not_ready" | "ready";
  criteria: GateCriterion[];
  evaluated_at: string;
}

// ── NIM benchmark SSE event payloads ──────────────────────────────────────

/**
 * Stage names emitted on ``nim_benchmark_progress`` events. Mirror
 * STAGE_* constants in ``services/student_nim_lifecycle.py``. Display
 * labels live in ``@/lib/nim-stage-labels`` so the UI prose stays in
 * one place.
 */
export type NimBenchmarkStage =
  | "preflight"
  | "docker_run"
  | "health_poll"
  | "smoke_inference"
  | "registering_endpoint"
  | "evaluation"
  | "benchmark"
  | "stopping";

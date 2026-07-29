// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NIM benchmark stage → UI prose label.
 *
 * Backend ``services/student_nim_lifecycle.py`` emits a small set of stage
 * strings on ``nim_benchmark_progress`` events. The Compare & Benchmark
 * screen shows the SME prose strings instead ("Starting container...",
 * "Waiting for NIM ready", etc.). Keep the mapping in one place so any
 * prose tweak stays single-source.
 */

import type { NimBenchmarkStage } from "@/types/evaluation";

// The base
// label is the descriptive part the SME sees first; the
// `BenchmarkStageStrip` component appends a dynamic suffix `(N / M)` for
// `evaluation` and `(c=N)` for `benchmark`, then a global trailing `...`
// to indicate "in progress". Keep the base strings WITHOUT trailing `...`
// so the dynamic-suffix appending stays clean.
export const NIM_STAGE_LABELS: Record<NimBenchmarkStage, string> = {
  preflight: "Preflight (Docker / GPU / NGC checks)",
  docker_run: "Starting NIM container",
  health_poll: "Waiting for NIM ready",
  smoke_inference: "Smoke inference",
  registering_endpoint: "Registering endpoint",
  evaluation: "Running evaluation",
  benchmark: "Running latency benchmarks",
  stopping: "Stopping container",
};

/**
 * Ordered stage progression the SME walks through during a single
 * benchmark run. Useful for rendering a step indicator.
 */
export const NIM_STAGE_ORDER: ReadonlyArray<NimBenchmarkStage> = [
  "preflight",
  "docker_run",
  "health_poll",
  "smoke_inference",
  "registering_endpoint",
  "evaluation",
  "benchmark",
  "stopping",
];

/**
 * Format the elapsed-time hint for a benchmark stage.
 * ``health_poll`` shows progress against the backend startup budget; other
 * stages just show seconds. The budget is served by the backend
 * (EnvironmentResponse.nim_startup_timeout_s = NIM_STARTUP_TIMEOUT_S) so the
 * label always matches the real deadline; the default below is only a
 * fallback for callers with no environment payload and mirrors the backend
 * default in _defaults.py.
 */
const FALLBACK_HEALTH_POLL_BUDGET_S = 1200;

export function formatStageElapsed(
  stage: NimBenchmarkStage,
  elapsedMs: number,
  healthPollBudgetS: number = FALLBACK_HEALTH_POLL_BUDGET_S,
): string {
  const elapsedS = Math.floor(elapsedMs / 1000);
  if (stage === "health_poll") {
    return `${elapsedS}s / ${healthPollBudgetS}s`;
  }
  return `${elapsedS}s`;
}

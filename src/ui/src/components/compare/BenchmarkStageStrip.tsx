// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Stage progression strip for an active NIM benchmark.
 *
 * Renders the prose strings from ``@/lib/nim-stage-labels`` with the
 * elapsed-time hint for the current stage. The component is a pure
 * presentation layer — the parent owns the SSE subscription + queue
 * state and passes the latest stage payload down.
 */

import { Text } from "@kui/react";
import { MiniSpinner } from "@/components/common/MiniSpinner";
import {
  formatStageElapsed,
  NIM_STAGE_LABELS,
  NIM_STAGE_ORDER,
} from "@/lib/nim-stage-labels";
import type { NimBenchmarkStage } from "@/types/evaluation";

export interface BenchmarkStageStripProps {
  stage: NimBenchmarkStage;
  elapsedMs: number;
  /** Optional ``concurrency`` payload from ``nim_benchmark_progress``;
      shown as ``c=N`` while the benchmark stage is active. */
  concurrency?: number | null;
  /** Optional ``processed/total`` for the ``evaluation`` stage. */
  evaluationProgress?: { processed: number; total: number } | null;
  /** Backend-served NIM startup budget (EnvironmentResponse
      .nim_startup_timeout_s); shown as the health_poll denominator. */
  startupBudgetS?: number;
  "data-testid"?: string;
}

export function BenchmarkStageStrip({
  stage,
  elapsedMs,
  concurrency,
  evaluationProgress,
  startupBudgetS,
  "data-testid": testid,
}: BenchmarkStageStripProps) {
  const label = NIM_STAGE_LABELS[stage];
  const elapsed = formatStageElapsed(stage, elapsedMs, startupBudgetS);

  // The "Running evaluation" stage gets a (N / M) suffix.
  // The "Running latency benchmarks" stage gets a (c=N) suffix when the
  // backend reports its current concurrency level.
  // Label, dynamic suffix, and
  // elapsed/budget hint render as separate text slots — not as one
  // parenthesised prose blob — so the SME can parse "what stage" vs
  // "what concurrency" at a glance. The label slot below carries the
  // canonical name; the suffix slot below carries the dynamic
  // (N/M) / (c=N) part in muted text.
  let suffix: string | null = null;
  if (stage === "evaluation" && evaluationProgress) {
    suffix = `(${evaluationProgress.processed} / ${evaluationProgress.total})`;
  } else if (stage === "benchmark" && concurrency != null) {
    suffix = `(c=${concurrency})`;
  }

  const stageIndex = NIM_STAGE_ORDER.indexOf(stage);

  return (
    <div
      className="flex flex-col gap-2"
      data-testid={testid ?? "benchmark-stage-strip"}
      data-stage={stage}
    >
      <div className="flex items-center gap-2">
        <MiniSpinner />
        <Text
          kind="body/regular/sm"
          style={{ color: "var(--text-primary)" }}
          data-testid="benchmark-stage-current-label"
        >
          {label}
          {/* Trailing ellipsis communicates "in progress" — kept on the
              label slot since it's part of the stage state, not the
              dynamic suffix. */}
          ...
        </Text>
        {suffix && (
          <Text
            kind="label/regular/xs"
            style={{ color: "var(--text-muted)" }}
            data-testid="benchmark-stage-suffix"
          >
            {suffix}
          </Text>
        )}
        <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
          {elapsed}
        </Text>
      </div>

      {/* A small dot row showing progression through the canonical
          stage list. Past stages light up; the current stage is
          highlighted; future stages are dim. Useful for the SME to
          gauge how far along the lifecycle is. */}
      <div className="flex items-center gap-1">
        {NIM_STAGE_ORDER.map((s, idx) => {
          const passed = idx < stageIndex;
          const current = idx === stageIndex;
          return (
            <span
              key={s}
              aria-hidden="true"
              data-testid={`benchmark-stage-dot-${s}`}
              data-state={passed ? "passed" : current ? "current" : "future"}
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: 999,
                backgroundColor: passed
                  ? "var(--text-secondary)"
                  : current
                    ? "var(--accent-green)"
                    : "var(--text-muted)",
                opacity: passed ? 0.65 : current ? 1 : 0.35,
                transition: "background-color 140ms ease",
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

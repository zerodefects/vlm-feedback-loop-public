// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Latency × concurrency matrix for a Student variant card: one row per
 * concurrency level, columns p50 / p90 / p99. Reads the
 * ``metrics.benchmarks`` table the NIM benchmark lifecycle writes onto
 * the Student's serving evaluation run.
 */

import type { EvaluationRunResponse } from "@/types/evaluation";

export interface ServingMatrixProps {
  /** The Student's serving evaluation run
      (``student.serving_evaluation_run_id``). */
  run: EvaluationRunResponse | null;
  /** Concurrency levels emitted by the backend latency sweep
      (``settings.STUDENT_LATENCY_TEST_CONCURRENCIES``, default ``[1, 8, 24]``). */
  concurrencies: number[];
}

interface BenchmarkSlot {
  concurrency: number;
  latency_p50_ms: number | null;
  latency_p90_ms: number | null;
  latency_p99_ms: number | null;
}

function readBenchmarks(run: EvaluationRunResponse | null): BenchmarkSlot[] {
  if (!run) return [];
  // Benchmarks live in the run's metadata under ``metrics.benchmarks`` or
  // a sibling ``serving_metrics`` field. The run-record metrics shape
  // carries them under ``metrics.benchmarks`` for NIM-source runs.
  const metrics = run.metrics as
    | (Record<string, unknown> & {
        benchmarks?: Array<{
          concurrency: number;
          latency_p50_ms?: number | null;
          latency_p90_ms?: number | null;
          latency_p99_ms?: number | null;
        }>;
      })
    | null;
  const arr = metrics?.benchmarks ?? [];
  return arr.map((b) => ({
    concurrency: b.concurrency,
    latency_p50_ms: b.latency_p50_ms ?? null,
    latency_p90_ms: b.latency_p90_ms ?? null,
    latency_p99_ms: b.latency_p99_ms ?? null,
  }));
}

function findSlot(benchmarks: BenchmarkSlot[], c: number): BenchmarkSlot | null {
  return benchmarks.find((b) => b.concurrency === c) ?? null;
}

function formatLatency(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "—";
  // Render seconds throughout (e.g. ``0.3s``, ``0.5s``, ``1.2s``) —
  // mixing ``ms`` and ``s`` units in adjacent rows of the same column
  // would force the SME to mentally normalise. Use two decimal places
  // below 100 ms (so ``0.15s`` keeps useful precision) and one decimal
  // place above so larger numbers stay terse.
  const seconds = ms / 1000;
  return ms < 100 ? `${seconds.toFixed(2)}s` : `${seconds.toFixed(1)}s`;
}

/** One right-aligned latency value cell. */
function LatencyCell({
  testId,
  value,
}: {
  testId: string;
  value: number | null | undefined;
}) {
  return (
    <td data-testid={testId} style={{ padding: "2px 16px", textAlign: "right" }}>
      {formatLatency(value)}
    </td>
  );
}

export function ServingMatrix({ run, concurrencies }: ServingMatrixProps) {
  const benchmarks = readBenchmarks(run);

  return (
    <div className="flex flex-col gap-1" data-testid="serving-matrix">
      {/* The parent (StudentVariantCard) already renders a "Serving:"
          eyebrow above this matrix; a second "Latency (by concurrency):"
          eyebrow here would orphan-stack two muted xs labels for no
          additional information (the table's own "concurrency" column +
          p50/p90/p99 sub-headers communicate the same thing). */}
      {/* The table sizes to content (no w-full) so the concurrency label
          sits adjacent to its latency values instead of stretching the
          pair across the full card width. */}
      <table className="text-sm" style={{ borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--glass-border-subtle)" }}>
            <th
              className="section-eyebrow"
              style={{
                textAlign: "left",
                padding: "2px 8px 2px 0",
                fontWeight: 400,
                fontSize: 12,
                color: "var(--text-muted)",
              }}
            >
              concurrency
            </th>
            <th
              colSpan={3}
              className="section-eyebrow"
              style={{
                // Centered across the three value columns so the header
                // reads as a group label; right alignment would park it
                // over p99 and leave p50 visually unlabeled.
                textAlign: "center",
                padding: "2px 8px",
                fontWeight: 400,
                fontSize: 12,
                color: "var(--text-muted)",
              }}
              data-testid="serving-matrix-header"
            >
              Latency (p50 / p90 / p99)
            </th>
          </tr>
        </thead>
        <tbody>
          {concurrencies.map((c) => {
            const slot = findSlot(benchmarks, c);
            return (
              <tr key={c} data-testid={`serving-matrix-row-c${c}`}>
                <td
                  style={{
                    padding: "2px 8px 2px 0",
                    color: "var(--text-secondary)",
                  }}
                >
                  c={c}
                </td>
                <LatencyCell
                  testId={`serving-matrix-cell-c${c}-p50`}
                  value={slot?.latency_p50_ms}
                />
                <LatencyCell
                  testId={`serving-matrix-cell-c${c}-p90`}
                  value={slot?.latency_p90_ms}
                />
                <LatencyCell
                  testId={`serving-matrix-cell-c${c}-p99`}
                  value={slot?.latency_p99_ms}
                />
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

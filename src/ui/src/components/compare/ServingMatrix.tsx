// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { Button, Text } from "@kui/react";

import type {
  EvaluationRunResponse,
  ServingBenchmarkResult,
  ServingBenchmarkWorkload,
} from "@/types/evaluation";

export interface ServingMatrixProps {
  run: EvaluationRunResponse | null;
  concurrencies: number[];
}

function findSlot(
  benchmarks: ServingBenchmarkResult[],
  concurrency: number,
): ServingBenchmarkResult | null {
  return benchmarks.find((row) => row.concurrency === concurrency) ?? null;
}

function formatMilliseconds(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : `${Math.round(value)} ms`;
}

function formatRps(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(2);
}

function formatFailure(row: ServingBenchmarkResult | null): string {
  if (!row) return "—";
  const rate = row.failure_rate;
  if (rate != null && Number.isFinite(rate)) return `${(rate * 100).toFixed(1)}%`;
  if (row.failed_request_count != null && row.attempted_request_count) {
    return `${((row.failed_request_count / row.attempted_request_count) * 100).toFixed(1)}%`;
  }
  return row.status === "passed" ? "0.0%" : "—";
}

function ValueText({ children }: { children: string }) {
  return (
    <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
      {children}
    </Text>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
        {label}
      </Text>
      <Text
        kind="body/regular/xs"
        style={{ color: "var(--text-secondary)", textAlign: "right" }}
      >
        {value}
      </Text>
    </div>
  );
}

function WorkloadDetails({ workload }: { workload: ServingBenchmarkWorkload | null }) {
  if (!workload) {
    return (
      <Text kind="body/regular/xs" style={{ color: "var(--warning-amber, #f59e0b)" }}>
        Legacy synthetic benchmark — workload provenance was not recorded.
      </Text>
    );
  }
  const contract = workload.inference_contract?.output_field_mode;
  const driver = workload.driver;
  const driverName =
    driver?.name?.toLowerCase() === "aiperf" ? "AIPerf" : (driver?.name ?? "AIPerf");
  return (
    <div
      className="glass-card-inner flex flex-col gap-1.5"
      style={{ padding: 12 }}
      data-testid="serving-workload-details"
    >
      <Detail
        label="Workload"
        value={`${workload.selected_count ?? "—"} real Test Pool images${
          workload.pool_member_count != null ? ` of ${workload.pool_member_count}` : ""
        }`}
      />
      <Detail label="Prompt" value={`Guidance schema · ${contract ?? "unknown"}`} />
      <Detail label="Output limit" value="Uncapped request (EOS/server policy)" />
      <Detail label="KV cache reuse" value={workload.kv_cache_reuse ?? "unknown"} />
      <Detail label="Driver" value={`${driverName} ${driver?.version ?? ""}`.trim()} />
      <Detail
        label="Workload hash"
        value={(workload.workload_hash ?? "—").slice(0, 16)}
      />
    </div>
  );
}

export function ServingMatrix({ run, concurrencies }: ServingMatrixProps) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const benchmarks = run?.metrics?.benchmarks ?? [];
  const workload = run?.metrics?.benchmark_workload ?? null;

  return (
    <div className="flex flex-col gap-2" data-testid="serving-matrix">
      <div style={{ overflowX: "auto" }}>
        <table className="text-sm" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--glass-border-subtle)" }}>
              {["Concurrency", "p50", "p90", "p99", "RPS", "Failures"].map(
                (heading) => (
                  <th
                    key={heading}
                    style={{
                      textAlign: heading === "Concurrency" ? "left" : "right",
                      padding: "3px 10px",
                      fontWeight: 400,
                      whiteSpace: "nowrap",
                    }}
                  >
                    <Text
                      kind="label/regular/xs"
                      className="section-eyebrow"
                      style={{ color: "var(--text-muted)" }}
                    >
                      {heading}
                    </Text>
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {concurrencies.map((concurrency) => {
              const row = findSlot(benchmarks, concurrency);
              const values = [
                formatMilliseconds(row?.latency_p50_ms),
                formatMilliseconds(row?.latency_p90_ms),
                formatMilliseconds(row?.latency_p99_ms),
                formatRps(row?.request_throughput_rps),
                formatFailure(row),
              ];
              return (
                <tr
                  key={concurrency}
                  data-testid={`serving-matrix-row-c${concurrency}`}
                >
                  <td style={{ padding: "3px 10px" }}>
                    <ValueText>{`c=${concurrency}`}</ValueText>
                  </td>
                  {values.map((value, index) => (
                    <td
                      key={`${concurrency}-${index}`}
                      data-testid={
                        index < 3
                          ? `serving-matrix-cell-c${concurrency}-p${[50, 90, 99][index]}`
                          : undefined
                      }
                      style={{ padding: "3px 10px", textAlign: "right" }}
                    >
                      <ValueText>{value}</ValueText>
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div>
        <Button
          kind="tertiary"
          onClick={() => setDetailsOpen((open) => !open)}
          aria-expanded={detailsOpen}
          data-testid="serving-workload-toggle"
          style={{ borderRadius: 999 }}
        >
          {detailsOpen ? "Hide workload details" : "Workload details"}
        </Button>
      </div>
      {detailsOpen && <WorkloadDetails workload={workload} />}
    </div>
  );
}

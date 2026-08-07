// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

/**
 * QualityFailedVariantCard.
 *
 * Compact notice for a Student whose ``quality_status="failed"``. These
 * variants used to be silently filtered off the Compare page — a
 * serving-validated Student could vanish with no explanation and no
 * remediation affordance (``:rerescore`` was API-only). This card keeps
 * them visible with:
 *
 *   - why the variant is excluded from comparison (its recorded quality
 *     failure reason, from ``nim_preflight_details.quality_failure_reason``,
 *     falling back to a generic line);
 *   - a [Re-score quality] button replaying the canonical rescore
 *     (POST ``:rerescore``); backend 409/400 details render inline so a
 *     genuinely un-rescorable Student explains itself honestly.
 *   - a [Deploy for serving validation] action while serving is not yet
 *     validated. This runs the normal Student NIM lifecycle; the backend may
 *     promote quality only when the TAO failure matches its narrow,
 *     authoritative loader-gap classifier. Arbitrary TAO failures remain
 *     failed even when the checkpoint serves successfully.
 *
 * Deliberately NOT a full StudentVariantCard: a failed variant has no
 * quality metrics to drill into, and rendering the twin-shell layout
 * around absent data would imply comparability that doesn't exist.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button, Spinner, Text } from "@kui/react";
import { AlertTriangle } from "lucide-react";

import { parseApiErrorDetail } from "@/api/client";
import { rerescoreStudentModel } from "@/api/students";
import { studentModelKeys } from "@/api/query-keys";
import { BenchmarkStageStrip } from "@/components/compare/BenchmarkStageStrip";
import { humanizeFailureReason } from "@/lib/training/failure-display";
import { quantizationDisplayName } from "@/lib/model-display";
import type { NimBenchmarkStage } from "@/types/evaluation";
import type { StudentModel } from "@/types/training";

function failureReason(student: StudentModel): string | null {
  const details = student.nim_preflight_details;
  const reason = details?.["quality_failure_reason"];
  if (typeof reason === "string" && reason.trim().length > 0) return reason;
  return null;
}

export function QualityFailedVariantCard({
  projectId,
  student,
  baseModelLabel,
  benchmarkStage,
  benchmarkStartupBudgetS,
  benchmarkElapsedMs,
  benchmarkConcurrency,
  benchmarkEvalProgress,
  benchmarkError,
  benchmarkEmphasis,
  busy,
  onBenchmark,
}: {
  projectId: string;
  student: StudentModel;
  baseModelLabel: string;
  benchmarkStage: NimBenchmarkStage | null;
  benchmarkStartupBudgetS: number;
  benchmarkElapsedMs: number;
  benchmarkConcurrency: number | null;
  benchmarkEvalProgress: { processed: number; total: number } | null;
  benchmarkError: string | null;
  benchmarkEmphasis: "primary" | "secondary";
  busy: boolean;
  onBenchmark: () => void;
}) {
  const queryClient = useQueryClient();

  const rerescoreMut = useMutation({
    mutationFn: () => rerescoreStudentModel(projectId, student.student_model_id),
    onSettled: () => {
      // Success may promote quality_status (card disappears into the
      // comparison list); failure keeps the card with the error inline.
      void queryClient.invalidateQueries({
        queryKey: studentModelKeys.list(projectId),
      });
    },
  });

  const variantLabel = `${baseModelLabel} · ${quantizationDisplayName(student.quantization_method)}`;
  const serverPending = student.serving_status === "pending";
  const benchmarking = benchmarkStage !== null || serverPending;
  const canValidateServing = student.serving_status !== "validated";
  const recordedFailureReason = failureReason(student);

  return (
    <div
      className="rounded-[14px] border px-5 py-4"
      style={{
        background: "rgba(239, 68, 68, 0.06)",
        borderColor: "rgba(239, 68, 68, 0.25)",
      }}
      data-testid="quality-failed-variant-card"
    >
      <div className="flex items-center gap-2">
        <AlertTriangle
          size={15}
          strokeWidth={2.25}
          style={{ color: "var(--text-error, #ef4444)" }}
        />
        <Text kind="body/bold/md" style={{ color: "var(--text-primary)" }}>
          {variantLabel}
        </Text>
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          — excluded from comparison: quality evaluation failed
          {student.serving_status === "validated" ? " (serving is validated)" : ""}
        </Text>
      </div>
      <div className="mt-1 flex items-start justify-between gap-4">
        {recordedFailureReason && (
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--text-muted)" }}
            data-testid="quality-failure-reason"
          >
            {humanizeFailureReason(recordedFailureReason)}
          </Text>
        )}
        {!benchmarking && (
          <div className="flex shrink-0 items-center gap-2">
            <Button
              kind="secondary"
              onClick={() => rerescoreMut.mutate()}
              disabled={rerescoreMut.isPending || busy}
              data-testid="rerescore-button"
            >
              {rerescoreMut.isPending ? "Re-scoring…" : "Re-score quality"}
            </Button>
            {canValidateServing && (
              <Button
                kind={benchmarkEmphasis}
                className={
                  benchmarkEmphasis === "primary" ? "nvidia-green-button" : undefined
                }
                onClick={onBenchmark}
                disabled={busy || rerescoreMut.isPending}
                data-testid={`failed-quality-deploy-${student.student_model_id}`}
              >
                Deploy for serving validation
              </Button>
            )}
          </div>
        )}
      </div>
      {canValidateServing && !benchmarking && (
        <Text
          kind="body/regular/xs"
          className="mt-2"
          style={{ color: "var(--text-muted)" }}
          data-testid="quality-failed-nim-fallback-help"
        >
          NIM validation tests the packaged checkpoint. Persisted lineage determines
          whether a clean result validates quality or serving only.
        </Text>
      )}
      {benchmarking && benchmarkStage && (
        <div className="mt-3">
          <BenchmarkStageStrip
            stage={benchmarkStage}
            elapsedMs={benchmarkElapsedMs}
            concurrency={benchmarkConcurrency}
            evaluationProgress={benchmarkEvalProgress}
            startupBudgetS={benchmarkStartupBudgetS}
            data-testid="quality-failed-benchmark-stage"
          />
        </div>
      )}
      {benchmarking && !benchmarkStage && (
        <div
          className="mt-3 flex items-center gap-2"
          data-testid="quality-failed-benchmark-reconciled"
        >
          <Spinner size="small" aria-label="Serving validation in progress" />
          <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
            Serving validation is running. Live stage details resume when events arrive.
          </Text>
        </div>
      )}
      {benchmarkError && (
        <Text
          kind="body/regular/sm"
          className="mt-2"
          style={{ color: "var(--text-error, #ef4444)" }}
          data-testid="quality-failed-benchmark-error"
        >
          {benchmarkError}
        </Text>
      )}
      {rerescoreMut.isError && (
        <Text
          kind="body/regular/sm"
          style={{ color: "var(--text-error, #ef4444)" }}
          data-testid="rerescore-error"
        >
          {parseApiErrorDetail(rerescoreMut.error) ?? "Re-score request failed."}
        </Text>
      )}
    </div>
  );
}

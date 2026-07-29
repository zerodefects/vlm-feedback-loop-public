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
 *
 * Deliberately NOT a full StudentVariantCard: a failed variant has no
 * quality metrics to drill into, and rendering the twin-shell layout
 * around absent data would imply comparability that doesn't exist.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button, Text } from "@kui/react";
import { AlertTriangle } from "lucide-react";

import { parseApiErrorDetail } from "@/api/client";
import { rerescoreStudentModel } from "@/api/students";
import { studentModelKeys } from "@/api/query-keys";
import type { StudentModel } from "@/types/training";

function failureReason(student: StudentModel): string {
  const details = student.nim_preflight_details;
  const reason = details?.["quality_failure_reason"];
  if (typeof reason === "string" && reason.trim().length > 0) return reason;
  return "No failure reason recorded on this Student.";
}

export function QualityFailedVariantCard({
  projectId,
  student,
  baseModelLabel,
}: {
  projectId: string;
  student: StudentModel;
  baseModelLabel: string;
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

  const variantLabel = student.quantization_method
    ? `${baseModelLabel} · ${student.quantization_method}`
    : baseModelLabel;

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
      <div className="mt-1 flex items-center justify-between gap-4">
        <Text
          kind="body/regular/sm"
          style={{ color: "var(--text-muted)" }}
          data-testid="quality-failure-reason"
        >
          {failureReason(student)}
        </Text>
        <Button
          kind="secondary"
          onClick={() => rerescoreMut.mutate()}
          disabled={rerescoreMut.isPending}
          data-testid="rerescore-button"
        >
          {rerescoreMut.isPending ? "Re-scoring…" : "Re-score quality"}
        </Button>
      </div>
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

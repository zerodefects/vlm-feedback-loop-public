// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** API client for evaluation runs, triggers, and the Scale-Up gate. */

import { apiFetch } from "@/api/client";
import type {
  EvaluationRunCreateResponse,
  EvaluationRunListResponse,
  EvaluationRunResponse,
  ScaleUpGateResponse,
  TriggerStatusResponse,
} from "@/types/evaluation";

// ── Evaluation Runs ────────────────────────────────────────────────────────

export function createEvaluationRun(
  projectId: string,
  body: { icl_mode?: string; structured_generation_mode?: string | null },
): Promise<EvaluationRunCreateResponse> {
  return apiFetch(`/projects/${projectId}/evaluation_runs`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getEvaluationRun(
  projectId: string,
  runId: string,
): Promise<EvaluationRunResponse> {
  return apiFetch(`/projects/${projectId}/evaluation_runs/${runId}`);
}

export function listEvaluationRuns(
  projectId: string,
  params?: {
    status?: string;
    basis?: "gate" | "benchmark";
    limit?: number;
    cursor?: string;
  },
): Promise<EvaluationRunListResponse> {
  const qs = new URLSearchParams();
  if (params?.status) qs.set("status", params.status);
  if (params?.basis) qs.set("basis", params.basis);
  if (params?.limit) qs.set("limit", String(params.limit));
  if (params?.cursor) qs.set("cursor", params.cursor);
  const suffix = qs.toString() ? `?${qs}` : "";
  return apiFetch(`/projects/${projectId}/evaluation_runs${suffix}`);
}

export function cancelEvaluationRun(
  projectId: string,
  runId: string,
): Promise<{ run_id: string; status: string; cancel_requested_at: string }> {
  return apiFetch(`/projects/${projectId}/evaluation_runs/${runId}:cancel`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

// ── Trigger Status ─────────────────────────────────────────────────────────

export function fetchTriggerStatus(projectId: string): Promise<TriggerStatusResponse> {
  return apiFetch(`/projects/${projectId}/evaluation_trigger_status`);
}

export function dismissTrigger(
  projectId: string,
  triggerType: string,
): Promise<{ trigger_type: string; dismissed: boolean }> {
  return apiFetch(`/projects/${projectId}/evaluation_trigger_status:dismiss`, {
    method: "POST",
    body: JSON.stringify({ trigger_type: triggerType }),
  });
}

// ── Scale-Up Gate ──────────────────────────────────────────────────────────

export function fetchScaleUpGate(projectId: string): Promise<ScaleUpGateResponse> {
  return apiFetch(`/projects/${projectId}/scaleup_gate`);
}

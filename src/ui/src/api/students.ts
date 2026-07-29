// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * API wrappers for the StudentModel endpoints consumed by the
 * Compare & Benchmark screen.
 *
 * Endpoints:
 *   - GET  /v1/projects/{id}/student_models                  (list)
 *   - POST /v1/projects/{id}/student_models/{id}:deploy_nim
 *   - POST /v1/projects/{id}/student_models/{id}:deployment_handoff
 *          (gated dual + Inference Contract; returns 409 if any
 *           gate fails)
 *
 * The ``:deployment_handoff`` endpoint is the **only** path that runs the
 * readiness + contract-parity gates; the generic ``:generate`` action-request endpoint
 * dispatches to a gate-less renderer for ``deployment_handoff``. The
 * Compare screen's ``[Request Production Deployment]`` button MUST go
 * through this function. (The ``[Deploy for serving validation]`` button is the
 * sibling fallback affordance for the failed-preflight case and uses the
 * generic ``:generate`` endpoint with the gate-less ``student_nim_deploy``
 * AR type.)
 */

import { apiFetch } from "@/api/client";
import type { ActionRequestGenerateResponse } from "@/types/nim";
import type {
  DeployNimRequest,
  DeployNimResponse,
  StudentModelListResponse,
} from "@/types/training";

export function listStudentModels(
  projectId: string,
  params?: { limit?: number; cursor?: string },
): Promise<StudentModelListResponse> {
  const qs = new URLSearchParams();
  if (params?.limit) qs.set("limit", String(params.limit));
  if (params?.cursor) qs.set("cursor", params.cursor);
  const suffix = qs.toString() ? `?${qs}` : "";
  return apiFetch<StudentModelListResponse>(
    `/projects/${projectId}/student_models${suffix}`,
  );
}

export function deployNim(
  projectId: string,
  studentModelId: string,
  body: DeployNimRequest,
): Promise<DeployNimResponse> {
  return apiFetch<DeployNimResponse>(
    `/projects/${projectId}/student_models/${studentModelId}:deploy_nim`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

/**
 * Replay the canonical quality rescore for a ``quality_status="failed"``
 * Student (the Compare page's remediation affordance; POST ``:rerescore``).
 *
 * Backend guards: 409 unless the Student is currently ``failed``; 400 when
 * there is no paired evaluate TAOJob to replay. On success the response
 * carries the updated ``quality_status`` (may stay ``failed`` if the
 * predictions genuinely don't rescore).
 */
export function rerescoreStudentModel(
  projectId: string,
  studentModelId: string,
): Promise<{ quality_status?: string | null; run_id?: string | null }> {
  return apiFetch(`/projects/${projectId}/student_models/${studentModelId}:rerescore`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

/**
 * Generate the ``deployment_handoff`` Action Request via the Student-scoped
 * endpoint that enforces all four gates (dual readiness + Inference Contract parity).
 *
 * Throws ``ApiError`` with status 409 and a body containing one of:
 *   - ``conflict: quality_status_not_validated``
 *   - ``conflict: serving_status_not_validated``
 *   - ``conflict: serving_evaluation_run_missing``
 *   - ``conflict: INFERENCE_CONTRACT_MISMATCH``
 *
 * The response shape matches the generic Action Request response so it
 * can flow through ``ActionRequestPanel`` unchanged.
 */
export function requestDeploymentHandoff(
  projectId: string,
  studentModelId: string,
): Promise<ActionRequestGenerateResponse> {
  return apiFetch<ActionRequestGenerateResponse>(
    `/projects/${projectId}/student_models/${studentModelId}:deployment_handoff`,
    { method: "POST", body: JSON.stringify({}) },
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * API wrappers for the Student Training and Training Job Monitor screens.
 *
 * Consumes:
 *   - POST /v1/projects/{id}/training_suites
 *   - GET  /v1/projects/{id}/training_suites/{id}
 *   - POST /v1/projects/{id}/training_suites/{id}:cancel
 *   - POST /v1/projects/{id}/training_presets:resolve
 *   - GET  /v1/projects/{id}/tao_jobs/{id}
 *   - POST /v1/projects/{id}/tao_jobs/{id}:cancel
 *   - GET  /v1/projects/{id}/model_configs?eligible_role=student_base
 *          (reuses the existing fetchModelConfigs wrapper)
 */

import { apiFetch } from "@/api/client";
import { fetchModelConfigs } from "@/api/model-configs";
import type { ModelConfigListResponse } from "@/types/nim";
import type {
  TAOJob,
  TAOBaseProvisioningRun,
  TrainingPreflightResponse,
  TrainingPresetResolveResponse,
  TrainingSuite,
  TrainingSuiteCancelResponse,
  TrainingSuiteCreateRequest,
  TrainingSuiteListResponse,
} from "@/types/training";

// ── Training preflight / data preview ─────────────────────────────────────

export function runTrainingPreflight(
  projectId: string,
  studentBaseModelConfigIds: string[],
  includeAutoLabeled = true,
  enableLora = true,
  quantizationSchemes: string[] = ["FP8_DYNAMIC"],
): Promise<TrainingPreflightResponse> {
  return apiFetch<TrainingPreflightResponse>(
    `/projects/${projectId}/training_preflight`,
    {
      method: "POST",
      body: JSON.stringify({
        student_base_model_config_ids: studentBaseModelConfigIds,
        include_auto_labeled: includeAutoLabeled,
        enable_lora: enableLora,
        quantization_schemes: quantizationSchemes,
      }),
    },
  );
}

// ── First-use TAO base provisioning ────────────────────────────────────────

export function startTAOBaseProvisioning(
  projectId: string,
  studentBaseModelConfigIds: string[],
): Promise<TAOBaseProvisioningRun> {
  return apiFetch<TAOBaseProvisioningRun>(
    `/projects/${projectId}/tao_base_experiment_provisioning`,
    {
      method: "POST",
      body: JSON.stringify({
        student_base_model_config_ids: studentBaseModelConfigIds,
      }),
    },
  );
}

export function getTAOBaseProvisioning(
  projectId: string,
  provisioningRunId: string,
): Promise<TAOBaseProvisioningRun> {
  return apiFetch<TAOBaseProvisioningRun>(
    `/projects/${projectId}/tao_base_experiment_provisioning/${provisioningRunId}`,
  );
}

// ── Training suites ───────────────────────────────────────────────────────

export function createTrainingSuite(
  projectId: string,
  body: TrainingSuiteCreateRequest,
): Promise<TrainingSuite> {
  return apiFetch<TrainingSuite>(`/projects/${projectId}/training_suites`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getTrainingSuite(
  projectId: string,
  trainingSuiteId: string,
): Promise<TrainingSuite> {
  return apiFetch<TrainingSuite>(
    `/projects/${projectId}/training_suites/${trainingSuiteId}`,
  );
}

export function listTrainingSuites(
  projectId: string,
  limit = 100,
): Promise<TrainingSuiteListResponse> {
  return apiFetch<TrainingSuiteListResponse>(
    `/projects/${projectId}/training_suites?limit=${limit}`,
  );
}

export function cancelTrainingSuite(
  projectId: string,
  trainingSuiteId: string,
): Promise<TrainingSuiteCancelResponse> {
  return apiFetch<TrainingSuiteCancelResponse>(
    `/projects/${projectId}/training_suites/${trainingSuiteId}:cancel`,
    {
      method: "POST",
      body: JSON.stringify({}),
    },
  );
}

// ── Read-only preset resolution ───────────────────────────────────────────

export function resolveTrainingPresets(
  projectId: string,
  studentBaseModelConfigIds: string[],
): Promise<TrainingPresetResolveResponse> {
  return apiFetch<TrainingPresetResolveResponse>(
    `/projects/${projectId}/training_presets:resolve`,
    {
      method: "POST",
      body: JSON.stringify({
        student_base_model_config_ids: studentBaseModelConfigIds,
      }),
    },
  );
}

// ── TAO jobs ──────────────────────────────────────────────────────────────

export function getTAOJob(
  projectId: string,
  taoJobId: string,
  refresh = false,
): Promise<TAOJob> {
  const suffix = refresh ? "?refresh=true" : "";
  return apiFetch<TAOJob>(`/projects/${projectId}/tao_jobs/${taoJobId}${suffix}`);
}

export function cancelTAOJob(projectId: string, taoJobId: string): Promise<TAOJob> {
  return apiFetch<TAOJob>(`/projects/${projectId}/tao_jobs/${taoJobId}:cancel`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

// ── Student-base model configs (thin convenience wrapper) ─────────────────

export function listStudentBaseModelConfigs(
  projectId: string,
): Promise<ModelConfigListResponse> {
  return fetchModelConfigs(projectId, "student_base");
}

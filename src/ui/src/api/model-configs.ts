// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * API functions for model config management and project updates.
 */

import { apiFetch } from "@/api/client";
import type { ProjectResponse } from "@/types/project";
import type { ModelConfigListResponse } from "@/types/nim";

export function fetchModelConfigs(
  projectId: string,
  eligibleRole?: string,
): Promise<ModelConfigListResponse> {
  const params = new URLSearchParams();
  if (eligibleRole) params.set("eligible_role", eligibleRole);
  const qs = params.toString();
  return apiFetch<ModelConfigListResponse>(
    `/projects/${projectId}/model_configs${qs ? `?${qs}` : ""}`,
  );
}

export function updateProject(
  projectId: string,
  body: Record<string, unknown>,
): Promise<ProjectResponse> {
  return apiFetch<ProjectResponse>(`/projects/${projectId}`, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
}

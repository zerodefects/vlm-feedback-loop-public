// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared lookup for the active Teacher's model config and the active
 * Guidance version — the pair every scale-up-era config summary prints
 * (Batch Pre-Run, Scale-Up Hub).
 *
 * Returns RAW values (config, name, guidance); label formatting and
 * fallbacks stay per-page because they deliberately differ. The optional
 * ``role`` param switches to the role-filtered model-config list —
 * pass the same role value on every screen so warm-cache sharing is
 * preserved.
 */

import { useQuery } from "@tanstack/react-query";

import { fetchGuidance } from "@/api/guidance";
import { fetchModelConfigs } from "@/api/model-configs";
import { guidanceKeys, modelConfigKeys } from "@/api/query-keys";
import type { ProjectResponse } from "@/types/project";
import type { GuidanceResponse } from "@/types/guidance";
import type { ModelConfigResponse } from "@/types/nim";

export interface TeacherAndGuidance {
  /** Catalog entry matching ``project.teacher_model_config_id``. */
  teacherConfig: ModelConfigResponse | null;
  /** Convenience: ``teacherConfig.model_name``. */
  teacherName: string | null;
  /** The project's active Guidance, when one exists and has loaded. */
  activeGuidance: GuidanceResponse | null;
}

export function useTeacherAndGuidance(
  projectId: string,
  project: ProjectResponse,
  role?: string,
): TeacherAndGuidance {
  const { data: modelConfigsData } = useQuery({
    queryKey: modelConfigKeys.list(projectId, role),
    queryFn: () => fetchModelConfigs(projectId, role),
  });
  const teacherConfig =
    modelConfigsData?.items.find(
      (m) => m.model_config_id === project.teacher_model_config_id,
    ) ?? null;

  const { data: activeGuidance } = useQuery({
    queryKey: guidanceKeys.detail(projectId, project.active_guidance_id ?? ""),
    queryFn: () =>
      project.active_guidance_id
        ? fetchGuidance(projectId, project.active_guidance_id)
        : Promise.resolve(null),
    enabled: Boolean(project.active_guidance_id),
  });

  return {
    teacherConfig,
    teacherName: teacherConfig?.model_name ?? null,
    activeGuidance: activeGuidance ?? null,
  };
}

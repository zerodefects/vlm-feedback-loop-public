// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * React Query key factories.
 *
 * `projectKeys` is shared with the SSE store (`stores/sse-store.ts`), which
 * invalidates project caches on reconnect and terminal events; the other
 * factories scope page-level caches. Only factories with live consumers are
 * kept — add new ones when a component needs them, not speculatively.
 */

export const projectKeys = {
  all: ["projects"] as const,
  // Active and archived lists are cached separately so toggling
  // "Show archived" on the Project List doesn't smear the two responses.
  list: (includeArchived?: boolean) =>
    ["projects", "list", includeArchived ?? false] as const,
  detail: (projectId: string) => ["project", projectId] as const,
};

export const environmentKeys = {
  assessment: () => ["environment", "assessment"] as const,
};

export const modelConfigKeys = {
  list: (projectId: string, role?: string) =>
    ["modelConfigs", projectId, "list", role] as const,
};

export const nimEndpointKeys = {
  list: (projectId: string) => ["nimEndpoints", projectId, "list"] as const,
};

// Local-NIM deployment polling — shared between NIMSetupGatePage and
// LocalDeployBanner so their pollers hit the same cache entry.
export const localNimKeys = {
  deployments: (projectId: string) => ["local-nim-deployments", projectId] as const,
};

export const guidanceKeys = {
  all: (projectId: string) => ["guidance", projectId] as const,
  list: (projectId: string) => ["guidance", projectId, "list"] as const,
  detail: (projectId: string, guidanceId: string) =>
    ["guidance", projectId, guidanceId] as const,
  iclCount: (projectId: string) => ["guidance", projectId, "iclCount"] as const,
  // Schema refinement reminder state — invalidated after each label
  // save and after a dismiss so the banner tracks the backend's decision.
  reminderStatus: (projectId: string) =>
    ["guidance", projectId, "reminderStatus"] as const,
};

export const evaluationKeys = {
  all: (projectId: string) => ["evaluations", projectId] as const,
  list: (projectId: string) => ["evaluations", projectId, "list"] as const,
  // Compare page's completed-runs window. Distinct from ``list`` — that
  // key is owned by EvaluationStrip with different query params; sharing
  // it would let two queryFns race for one cache entry.
  completedList: (projectId: string) =>
    ["evaluations", projectId, "list", "completed"] as const,
  detail: (projectId: string, runId: string) =>
    ["evaluations", projectId, runId] as const,
  triggerStatus: (projectId: string) =>
    ["evaluations", projectId, "triggerStatus"] as const,
  gate: (projectId: string) => ["evaluations", projectId, "gate"] as const,
};

export const batchKeys = {
  detail: (projectId: string, runId: string) => ["batch", projectId, runId] as const,
  export: (projectId: string, exportId: string) =>
    ["batch", projectId, "export", exportId] as const,
  exportList: (projectId: string) => ["batch", projectId, "exports"] as const,
};

export const trainingKeys = {
  preflight: (
    projectId: string,
    modelConfigIds: string[],
    includeAutoLabeled: boolean,
    enableLora = true,
    quantizationSchemes: string[] = ["FP8_DYNAMIC"],
  ) =>
    [
      "training",
      projectId,
      "preflight",
      [...modelConfigIds].sort().join(","),
      includeAutoLabeled,
      enableLora,
      [...quantizationSchemes].sort().join(","),
    ] as const,
  presets: (projectId: string, modelConfigIds: string[]) =>
    ["training", projectId, "presets", [...modelConfigIds].sort().join(",")] as const,
  studentBases: (projectId: string) => ["training", projectId, "studentBases"] as const,
  baseProvisioning: (projectId: string, provisioningRunId: string) =>
    ["training", projectId, "baseProvisioning", provisioningRunId] as const,
  suites: (projectId: string) => ["training", projectId, "suites"] as const,
  suite: (projectId: string, trainingSuiteId: string) =>
    ["training", projectId, "suite", trainingSuiteId] as const,
  job: (projectId: string, taoJobId: string) =>
    ["training", projectId, "job", taoJobId] as const,
};

// Compare & Benchmark page caches.
export const studentModelKeys = {
  list: (projectId: string) => ["studentModels", projectId, "list"] as const,
};

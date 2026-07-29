// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Project API functions consumed by React Query hooks.
 */

import { apiFetch } from "@/api/client";
import type {
  ProjectCreateRequest,
  ProjectListResponse,
  ProjectResponse,
} from "@/types/project";

export function fetchProject(projectId: string): Promise<ProjectResponse> {
  return apiFetch<ProjectResponse>(`/projects/${projectId}`);
}

export function createProject(body: ProjectCreateRequest): Promise<ProjectResponse> {
  return apiFetch<ProjectResponse>("/projects", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// Backend page-size cap is 200 (`projects.py`, `le=200`). We always
// request the max page size and walk `next_cursor` so the UI sees every
// project in one logical fetch — the Project List does not paginate. Hard ceiling
// on iterations protects against a malformed loop response that returns
// the same cursor forever.
const PROJECT_LIST_PAGE_SIZE = 200;
const PROJECT_LIST_MAX_PAGES = 50; // 10 000 projects — well past any v1 use case

export async function fetchProjectList(
  includeArchived?: boolean,
): Promise<ProjectListResponse> {
  const items: ProjectListResponse["items"] = [];
  let cursor: string | null = null;
  // Workspace-global flag, identical on every page — carry the latest
  // page's value through the aggregate result.
  let hasArchived = false;
  for (let page = 0; page < PROJECT_LIST_MAX_PAGES; page++) {
    const params = new URLSearchParams();
    params.set("limit", String(PROJECT_LIST_PAGE_SIZE));
    if (cursor) params.set("cursor", cursor);
    if (includeArchived) params.set("include_archived", "true");
    const resp = await apiFetch<ProjectListResponse>(`/projects?${params.toString()}`);
    if (resp.items) items.push(...resp.items);
    // `?? false` tolerates a backend that predates the field.
    hasArchived = resp.has_archived ?? false;
    if (!resp.next_cursor) {
      return { items, next_cursor: null, has_archived: hasArchived };
    }
    // A cursor that doesn't advance is a backend pagination bug — the loop
    // would otherwise burn all its pages re-fetching the same window.
    if (resp.next_cursor === cursor) {
      console.warn(
        `fetchProjectList: pagination cursor did not advance (${cursor}); ` +
          `stopping after ${items.length} items to avoid a fetch loop.`,
      );
      return { items, next_cursor: null, has_archived: hasArchived };
    }
    cursor = resp.next_cursor;
  }
  // Reached the safety ceiling. Return what we have rather than throwing —
  // a partial list is more useful than a hard error — but log it: hitting
  // 10 000 projects in v1 (single-user) almost certainly means a paginator
  // bug, and a silent partial list would read as "that's all of them".
  console.warn(
    `fetchProjectList: hit the ${PROJECT_LIST_MAX_PAGES}-page ceiling ` +
      `(${items.length} items) with more pages remaining — list may be truncated.`,
  );
  return { items, next_cursor: cursor, has_archived: hasArchived };
}

export function archiveProject(projectId: string): Promise<ProjectResponse> {
  return apiFetch<ProjectResponse>(`/projects/${projectId}:archive`, {
    method: "POST",
  });
}

export function unarchiveProject(projectId: string): Promise<ProjectResponse> {
  return apiFetch<ProjectResponse>(`/projects/${projectId}:unarchive`, {
    method: "POST",
  });
}

/**
 * Request body for ``POST /v1/projects/{id}:mark_setup_completed`` — must
 * match the backend ``MarkSetupCompletedRequest`` schema.
 */
export interface MarkSetupCompletedRequest {
  auto_skip: boolean;
  teacher_mode: string;
  embedding_mode: string;
  embedding_provider: string;
  /**
   * Names of local NIM models the
   * FTUE just kicked off via ``:deploy`` at gate-confirm time. Empty
   * when the SME walked the hosted-only path. Persisted verbatim on
   * the ``setup_completed`` AuditEvent.
   */
  local_deploy_queued?: string[];
}

export interface MarkSetupCompletedResponse {
  transitioned: boolean;
  project: ProjectResponse;
}

/**
 * Stamp ``setup_completed_at`` for a project — idempotent.
 *
 * Called by the FTU pages (NIMSetupGatePage at gate-confirm,
 * ConfirmDefaultsPage auto-skip + manual Save) so the
 * ProjectIndexRedirect gate flips on subsequent project opens.
 * NIMConnectionPage (settings, edit mode) also calls it as a no-op.
 */
export function markSetupCompleted(
  projectId: string,
  body: MarkSetupCompletedRequest,
): Promise<MarkSetupCompletedResponse> {
  return apiFetch<MarkSetupCompletedResponse>(
    `/projects/${projectId}:mark_setup_completed`,
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  );
}

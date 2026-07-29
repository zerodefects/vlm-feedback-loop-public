// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * API client functions for Guidance endpoints.
 */

import { apiFetch } from "@/api/client";
import type {
  GuidanceCreateRequest,
  GuidanceResponse,
  GuidanceListResponse,
  DraftValidationRequest,
  DraftValidationResponse,
  GuidanceEditRequest,
  EditPreviewResponse,
  EditExecuteResponse,
  IclCountResponse,
} from "@/types/guidance";

/** Create a new immutable Guidance version. */
export async function createGuidance(
  projectId: string,
  body: GuidanceCreateRequest,
): Promise<GuidanceResponse> {
  return apiFetch<GuidanceResponse>(`/projects/${projectId}/guidance`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** Validate a draft schema without persisting. */
export async function validateDraft(
  projectId: string,
  body: DraftValidationRequest,
): Promise<DraftValidationResponse> {
  return apiFetch<DraftValidationResponse>(
    `/projects/${projectId}/guidance:validate_draft`,
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  );
}

/** Fetch a single Guidance version by ID. */
export async function fetchGuidance(
  projectId: string,
  guidanceId: string,
): Promise<GuidanceResponse> {
  return apiFetch<GuidanceResponse>(`/projects/${projectId}/guidance/${guidanceId}`);
}

/** List Guidance versions (newest-first, cursor pagination). */
export async function listGuidances(
  projectId: string,
  cursor?: string,
  limit?: number,
): Promise<GuidanceListResponse> {
  const params = new URLSearchParams();
  if (cursor) params.set("cursor", cursor);
  if (limit != null) params.set("limit", String(limit));
  const qs = params.toString();
  return apiFetch<GuidanceListResponse>(
    `/projects/${projectId}/guidance${qs ? `?${qs}` : ""}`,
  );
}

/** Preview or execute a Guidance edit. */
export async function editGuidancePreview(
  projectId: string,
  body: Omit<GuidanceEditRequest, "dry_run">,
): Promise<EditPreviewResponse> {
  return apiFetch<EditPreviewResponse>(`/projects/${projectId}/guidance:edit`, {
    method: "POST",
    body: JSON.stringify({ ...body, dry_run: true }),
  });
}

export async function editGuidanceExecute(
  projectId: string,
  body: Omit<GuidanceEditRequest, "dry_run">,
): Promise<EditExecuteResponse> {
  return apiFetch<EditExecuteResponse>(`/projects/${projectId}/guidance:edit`, {
    method: "POST",
    body: JSON.stringify({ ...body, dry_run: false }),
  });
}

// ── ICL count ───────────────────────────────────────────────────────────────

/** Fetch the ICL-eligible Edit count for the active Guidance. */
export async function fetchIclCount(projectId: string): Promise<IclCountResponse> {
  return apiFetch<IclCountResponse>(`/projects/${projectId}/guidance:icl_count`);
}

// ── Schema refinement reminders ─────────────────────────────────────────────

import type { ReminderStatusResponse, ReminderDismissResponse } from "@/types/guidance";

/** GET /v1/projects/{id}/guidance:reminder_status — backend-owned reminder state. */
export async function fetchReminderStatus(
  projectId: string,
): Promise<ReminderStatusResponse> {
  return apiFetch<ReminderStatusResponse>(
    `/projects/${projectId}/guidance:reminder_status`,
  );
}

/** POST /v1/projects/{id}/guidance:dismiss_reminder — increments the dismissed counter. */
export async function dismissReminder(
  projectId: string,
): Promise<ReminderDismissResponse> {
  return apiFetch<ReminderDismissResponse>(
    `/projects/${projectId}/guidance:dismiss_reminder`,
    { method: "POST" },
  );
}

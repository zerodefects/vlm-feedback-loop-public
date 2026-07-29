// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * API client functions for the Interactive Labeling workflow.
 *
 * Endpoints: review selector, proposals, label save, skip,
 * restore omitted, and image serving URL helper.
 */

import { apiFetch } from "@/api/client";
import type {
  ProposalRequest,
  ProposalResponse,
  ReviewSelectorNextResponse,
  LabelSaveRequest,
  LabelSaveResponse,
  SkipResponse,
  RestoreOmittedResponse,
  RationaleRegenerationRequest,
  RationaleRegenerationResponse,
} from "@/types/labeling";

/** GET /v1/projects/{id}/review_selector/next */
export function fetchNextReviewItem(
  projectId: string,
): Promise<ReviewSelectorNextResponse> {
  return apiFetch<ReviewSelectorNextResponse>(
    `/projects/${projectId}/review_selector/next`,
  );
}

/** POST /v1/projects/{id}/proposals */
export function createProposal(
  projectId: string,
  body: ProposalRequest,
): Promise<ProposalResponse> {
  return apiFetch<ProposalResponse>(`/projects/${projectId}/proposals`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** POST /v1/projects/{id}/labels */
export function saveLabel(
  projectId: string,
  body: LabelSaveRequest,
): Promise<LabelSaveResponse> {
  return apiFetch<LabelSaveResponse>(`/projects/${projectId}/labels`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** POST /v1/projects/{id}/examples/{key}:skip */
export function skipExample(
  projectId: string,
  exampleKey: string,
): Promise<SkipResponse> {
  return apiFetch<SkipResponse>(`/projects/${projectId}/examples/${exampleKey}:skip`, {
    method: "POST",
  });
}

/** POST /v1/projects/{id}/examples:restore_omitted */
export function restoreOmitted(projectId: string): Promise<RestoreOmittedResponse> {
  return apiFetch<RestoreOmittedResponse>(
    `/projects/${projectId}/examples:restore_omitted`,
    { method: "POST" },
  );
}

/** POST /v1/projects/{id}/examples/{key}:regenerate_rationale */
export function regenerateRationale(
  projectId: string,
  exampleKey: string,
  body: RationaleRegenerationRequest,
): Promise<RationaleRegenerationResponse> {
  return apiFetch<RationaleRegenerationResponse>(
    `/projects/${projectId}/examples/${exampleKey}:regenerate_rationale`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

/**
 * Build the URL for the image serving endpoint.
 *
 * Returns a relative URL that works through the Vite dev proxy
 * and same-origin in production.  Used as `<img src={imageUrl(...)}>`.
 */
export function imageUrl(projectId: string, exampleKey: string): string {
  return `/v1/projects/${projectId}/examples/${exampleKey}/image`;
}

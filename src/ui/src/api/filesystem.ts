// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * API client functions for filesystem browse, scan, and image ingestion.
 */

import { apiFetch } from "@/api/client";
import type {
  BrowseResponse,
  IngestRequest,
  IngestResponse,
  ScanResponse,
} from "@/types/filesystem";

/**
 * Browse a directory on the backend host's filesystem.
 * Deployment-scoped (no project_id).
 */
export function browseFilesystem(
  path?: string,
  showFiles = true,
  imageFormatsOnly = true,
): Promise<BrowseResponse> {
  const params = new URLSearchParams();
  if (path) params.set("path", path);
  params.set("show_files", String(showFiles));
  params.set("image_formats_only", String(imageFormatsOnly));
  return apiFetch<BrowseResponse>(`/filesystem/browse?${params}`);
}

/**
 * Recursively scan a directory for supported images.
 * Deployment-scoped (no project_id in path).
 */
export function scanDirectory(
  path: string,
  recursive = true,
  projectId?: string | null,
): Promise<ScanResponse> {
  return apiFetch<ScanResponse>("/filesystem/scan", {
    method: "POST",
    body: JSON.stringify({
      path,
      recursive,
      project_id: projectId ?? null,
    }),
  });
}

/**
 * Batch-ingest images into a project.
 * Project-scoped.
 */
export function ingestExamples(
  projectId: string,
  body: IngestRequest,
): Promise<IngestResponse> {
  return apiFetch<IngestResponse>(`/projects/${projectId}/examples:ingest`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

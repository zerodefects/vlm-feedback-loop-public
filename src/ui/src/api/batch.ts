// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** API client for batch labeling runs, dataset exports, and schema-invalid manifests. */

import { apiFetch } from "@/api/client";
import type {
  BatchLabelRunCreateResponse,
  BatchLabelRunResponse,
  DatasetExportResponse,
  SchemaInvalidManifestResponse,
} from "@/types/batch";

// ── Batch Label Runs ──────────────────────────────────────────────────────

export function createBatchLabelRun(
  projectId: string,
  body: {
    include_auto_labeled?: boolean;
    run_limit?: number | null;
    structured_generation_mode?: string | null;
    ingested_after?: string | null;
    ingested_before?: string | null;
  },
): Promise<BatchLabelRunCreateResponse> {
  return apiFetch(`/projects/${projectId}/batch_label_runs`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getBatchLabelRun(
  projectId: string,
  runId: string,
): Promise<BatchLabelRunResponse> {
  return apiFetch(`/projects/${projectId}/batch_label_runs/${runId}`);
}

export function resumeBatchLabelRun(
  projectId: string,
  runId: string,
): Promise<{ run_id: string; status: string }> {
  return apiFetch(`/projects/${projectId}/batch_label_runs/${runId}:resume`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function cancelBatchLabelRun(
  projectId: string,
  runId: string,
): Promise<{ run_id: string; status: string; cancel_requested_at: string }> {
  return apiFetch(`/projects/${projectId}/batch_label_runs/${runId}:cancel`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

// ── Schema-Invalid Manifest ───────────────────────────────────────────────

export function getSchemaInvalidManifest(
  projectId: string,
  runId: string,
): Promise<SchemaInvalidManifestResponse> {
  return apiFetch(
    `/projects/${projectId}/batch_label_runs/${runId}/schema_invalid_manifest`,
  );
}

// ── Dataset Exports ───────────────────────────────────────────────────────

export function listDatasetExports(
  projectId: string,
  limit = 5,
): Promise<{ items: DatasetExportResponse[]; next_cursor: string | null }> {
  return apiFetch(`/projects/${projectId}/dataset_exports?limit=${limit}`);
}

export function getDatasetExport(
  projectId: string,
  datasetExportId: string,
): Promise<DatasetExportResponse> {
  return apiFetch(`/projects/${projectId}/dataset_exports/${datasetExportId}`);
}

export function createDatasetExport(
  projectId: string,
  body: {
    dataset_intent: string;
    label_tier_filter?: string;
    export_field_mode?: string;
    batch_label_run_id?: string | null;
  },
): Promise<DatasetExportResponse> {
  return apiFetch(`/projects/${projectId}/dataset_exports`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

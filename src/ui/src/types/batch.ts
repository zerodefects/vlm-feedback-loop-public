// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Types for batch labeling runs, dataset exports, and schema-invalid manifests. */

export interface CommonErrorEntry {
  code: string;
  count: number;
  sample: string | null;
}

// ── Batch Label Run ───────────────────────────────────────────────────────

export interface BatchLabelRunResponse {
  run_id: string;
  run_type: string;
  status: string; // queued | running | paused | canceling | completed | canceled | failed
  status_reason: string | null;
  paused_reason: string | null;
  circuit_breaker_threshold: number | null;

  // Config snapshot
  guidance_id: string | null;
  guidance_version_number: number | null;
  model_config_id: string | null;
  model_name: string | null;
  generation_preset_key: string | null;
  thinking_mode_effective: string | null;
  visual_budget_preset_key: string | null;
  structured_generation_mode_effective: string | null;
  icl_mode: string | null;

  // Progress
  progress: { processed: number; total: number } | null;

  // Per-outcome counters
  examples_succeeded: number;
  examples_schema_invalid: number;
  examples_timeout: number;
  examples_endpoint_error: number;
  examples_total: number;

  // Top-N aggregated error signatures (the run-status screen's "Common errors:" block)
  common_errors: CommonErrorEntry[];

  // Timestamps
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  cancel_requested_at: string | null;

  recovered_from_restart: boolean;
}

export interface BatchLabelRunCreateResponse {
  run_id: string;
  run_type: string;
  status: string;
  guidance_id: string | null;
  model_config_id: string | null;
  generation_preset_key: string | null;
  thinking_mode_effective: string | null;
  visual_budget_preset_key: string | null;
  structured_generation_mode_effective: string | null;
  icl_mode: string | null;
  examples_total: number;
  created_at: string;
}

// ── Schema-Invalid Manifest ───────────────────────────────────────────────

interface SchemaInvalidExample {
  example_key: string;
  validation_errors_core: string[];
  inference_invocation_id: string;
}

export interface SchemaInvalidManifestResponse {
  batch_label_run_id: string;
  schema_invalid_examples: SchemaInvalidExample[];
  total_count: number;
}

// ── Dataset Export ─────────────────────────────────────────────────────────

// The full export record — create, get, and list items share this shape.
// The archive builds in a backend background task: the create response
// arrives with status "running" and null artifact_refs/manifest_ref; both
// populate when the record reaches "completed" (poll GET or follow the
// export_* SSE events).
export interface DatasetExportResponse {
  dataset_export_id: string;
  dataset_intent: string;
  export_field_mode: string;
  label_tier_filter: string;
  guidance_id: string;
  selection_definition_snapshot: Record<string, unknown>;
  example_count: number;
  status: "running" | "completed" | "failed";
  status_reason: string | null;
  progress: { images_written: number; images_total: number } | null;
  started_at: string | null;
  completed_at: string | null;
  artifact_refs: { archive_path: string; checksum_sha256: string } | null;
  manifest_ref: string | null;
  created_at: string;
}

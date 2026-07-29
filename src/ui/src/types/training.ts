// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Types for the Student Training and Training Job Monitor screens.
 *
 * Mirrors the backend Pydantic schemas in:
 *   - src/backend/vlm_feedback_loop/schemas/tao_job.py
 *   - src/backend/vlm_feedback_loop/schemas/training_suite.py
 *   - src/backend/vlm_feedback_loop/schemas/student_model.py
 *
 * TAOJob statuses are the 10 canonical backend values.  The UI
 * maps them to display labels via src/lib/training/statusDisplay.ts — raw
 * canonical strings never appear as badge text.
 */

// ── TAOJob state machine ───────────────────────────────────────────────────

export const TAO_JOB_STATUSES = [
  "not_started",
  "submitting",
  "submitted",
  "queued",
  "running",
  "paused",
  "succeeded",
  "failed",
  "canceled",
  "deleted",
] as const;
export type TAOJobStatus = (typeof TAO_JOB_STATUSES)[number];

export const TERMINAL_TAO_STATUSES: ReadonlySet<TAOJobStatus> = new Set([
  "succeeded",
  "failed",
  "canceled",
  "deleted",
]);

export type TAOJobAction = "train" | "evaluate" | "quantize" | "inference";

// ── Training presets / export field modes / quantization schemes ──────────

export type TrainingPreset = "quick" | "standard" | "high_quality" | "max_quality";
export type QuantizationScheme = "FP8_DYNAMIC" | "W8A8" | "W8A16" | "W4A16";
export type ExportFieldMode = "all" | "aux_and_core" | "core_only";

// ── Training preflight / authoritative data preview ───────────────────────

export type TrainingPreflightCheckName =
  | "tao_reachable"
  | "tao_job_timeout_supported"
  | "tao_workspace_reachable"
  | "tao_base_experiment_ready"
  | "student_base_role"
  | "verified_train_examples";

export interface TrainingPreflightCheck {
  check_name: TrainingPreflightCheckName;
  passed: boolean;
  message: string;
  model_config_id: string | null;
  provisioning_required: boolean;
  remediation: string | null;
}

export interface TrainingDataSummary {
  verified_training_count: number;
  test_pool_count: number;
  auto_labeled_eligible_count: number;
  auto_labeled_included_count: number;
  excluded_test_pool_count: number;
  excluded_auto_labeled_count: number;
  usable_training_count: number;
}

export interface TrainingPreflightResponse {
  status: "passed" | "failed";
  checks: TrainingPreflightCheck[];
  data_summary: TrainingDataSummary;
  resolved_presets: Record<string, Record<TrainingPreset, ResolvedTrainingPatch>>;
}

// ── TAOJob record ─────────────────────────────────────────────────────────

export interface TAOJobProgress {
  epoch_current?: number | null;
  epoch_total?: number | null;
  eta_seconds?: number | null;
  metrics_latest?: Record<string, number | string> | null;
  metrics_history_ref?: string | null;
}

export interface TAOJobOutputArtifact {
  name?: string;
  artifact_ref?: string | null;
  tao_file_path?: string | null;
  kind?: string;
  uri?: string;
  checksum?: string | null;
}

export interface TAOJobOutputs {
  artifacts?: TAOJobOutputArtifact[] | null;
  logs_ref?: string | null;
  metrics_ref?: string | null;
  tao_job_metadata_ref?: string | null;
  [key: string]: unknown;
}

export interface TAOJob {
  tao_job_id: string;
  project_id: string;
  status: TAOJobStatus;
  tao_status_raw: string | null;

  action: TAOJobAction;
  training_backend: string;
  training_policy_type: string | null;

  student_base_model_config_id: string;
  dataset_export_ids: string[];

  job_config: Record<string, unknown>;
  tao_create_job_request: Record<string, unknown>;
  tao_external_job_id: string | null;

  progress: TAOJobProgress | null;
  outputs: TAOJobOutputs | null;

  parent_tao_job_id: string | null;
  chain_id: string | null;
  chain_sequence: number | null;
  chain_halted_reason: string | null;

  preflight_result: Record<string, unknown> | null;
  error_ref: string | null;
  poll_error_ref: string | null;

  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  last_polled_at: string | null;
}

// ── Training suite ────────────────────────────────────────────────────────

export interface TrainingSuiteJob {
  tao_job_id: string;
  action: TAOJobAction;
  chain_sequence: number;
  status: TAOJobStatus;
  tao_external_job_id: string | null;
  chain_halted_reason: string | null;
}

export interface TrainingSuiteChain {
  chain_id: string;
  student_base_model_config_id: string;
  base_model_name: string;
  jobs: TrainingSuiteJob[];
}

export interface TrainingSuite {
  training_suite_id: string;
  project_id: string;
  idempotency_key: string;
  guidance_id: string;

  training_preset: TrainingPreset;
  export_field_mode: ExportFieldMode;
  include_auto_labeled: boolean;
  quantization_schemes: QuantizationScheme[];

  training_dataset_export_id: string | null;
  evaluation_dataset_export_id: string | null;
  selected_student_base_model_config_ids: string[];

  chain_ids_ordered: string[];
  chains: TrainingSuiteChain[];

  provisioning_run_id: string | null;
  provisioning_model_names: string[];
  setup_error_ref: string | null;

  status: TrainingSuiteStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export type TrainingSuiteStatus =
  | "provisioning"
  | "preparing"
  | "initialized"
  | "running"
  | "completed"
  | "failed"
  | "canceled";

export const TERMINAL_TRAINING_SUITE_STATUSES: ReadonlySet<TrainingSuiteStatus> =
  new Set(["completed", "failed", "canceled"]);

export interface TrainingSuiteListResponse {
  items: TrainingSuite[];
  next_cursor: string | null;
}

export interface TrainingSuiteCancelFailure {
  tao_job_id: string;
  error: string;
}

export interface TrainingSuiteCancelResponse {
  training_suite: TrainingSuite;
  jobs_canceled: number;
  jobs_already_terminal: number;
  setup_tasks_canceled: number;
  remote_cancel_failures: TrainingSuiteCancelFailure[];
}

export interface TrainingSuiteCreateRequest {
  student_base_model_config_ids: string[];
  training_preset: TrainingPreset;
  include_auto_labeled: boolean;
  export_field_mode: ExportFieldMode;
  quantization_schemes: QuantizationScheme[];
  idempotency_key: string;
}

export interface ResolvedTrainingPatch {
  train: {
    epoch: number;
    resume: boolean;
    ckpt: {
      enable_checkpoint: boolean;
      save_freq_in_epoch: number;
      max_keep: number;
      export_safetensors: boolean;
    };
  };
}

export interface TrainingPresetResolveResponse {
  /** Server-resolved training patches: model_config_id → preset → patch. */
  resolved_presets: Record<string, Record<TrainingPreset, ResolvedTrainingPatch>>;
}

// ── First-use TAO base provisioning ────────────────────────────────────────

export type TAOBaseProvisioningStatus = "queued" | "running" | "succeeded" | "failed";

export interface TAOBaseProvisioningFailure {
  target: string;
  error: string;
}

export interface TAOBaseProvisioningRun {
  provisioning_run_id: string;
  project_id: string;
  requested_model_config_ids: string[];
  requested_model_names: string[];
  status: TAOBaseProvisioningStatus;
  registered: string[];
  already_registered: string[];
  failures: TAOBaseProvisioningFailure[];
  error_ref: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
}

// ── Student model ──────────────────────────────────────────────────────────
// Mirrors backend StudentModelResponse (schemas/student_model.py).

export type CheckpointPackagingStatus = "pending" | "validated" | "failed";
export type QualityStatus = "pending" | "validated" | "partial" | "failed";
export type ServingStatus = "pending" | "validated" | "failed" | "not_attempted";

export interface StudentModel {
  student_model_id: string;
  project_id: string;
  student_base_model_config_id: string;
  tao_job_id: string;
  guidance_id: string;
  dataset_export_ids: string[];
  training_preset: TrainingPreset;
  lora_config: Record<string, unknown>;
  created_at: string;

  // Checkpoint packaging
  checkpoint_packaging_status: CheckpointPackagingStatus;
  nim_checkpoint_ref: string | null;

  // Two-part readiness
  quality_status: QualityStatus;
  quality_evaluation_run_id: string | null;
  serving_status: ServingStatus;
  serving_evaluation_run_id: string | null;

  // NIM deployment state
  nim_preflight_status: string | null;
  nim_preflight_details: Record<string, unknown> | null;
  nim_preflight_at: string | null;
  nim_deployment_mode: string | null;
  nim_container_id: string | null;
  nim_endpoint_url: string | null;

  // Deployment metadata
  nim_vlm_release_version: string | null;
  nim_model_profile_requested: string | null;
  nim_model_profile_selected: string | null;
  nim_profile_metadata: Record<string, unknown> | null;
  gpu_type: string | null;
  gpu_count: number | null;

  // Quantization provenance
  quantization_method: string | null;
  quantize_tao_job_id: string | null;
}

export interface StudentModelListResponse {
  items: StudentModel[];
  next_cursor: string | null;
}

// ── :deploy_nim request/response ────────────────────────────────────────────

export interface DeployNimRequest {
  /** ``null`` triggers local Docker orchestration; a URL registers an
      already-deployed external endpoint instead. */
  nim_endpoint_url?: string | null;
  nim_container_image?: string | null;
  nim_release_version?: string | null;
  gpu_assignment?: string | null;
  auth_mode?: "none" | "bearer";
}

export interface DeployNimResponse {
  student_model_id: string;
  nim_deployment_mode: "local" | "external";
  serving_status: "pending";
  task_id: string;
  created_at: string;
}

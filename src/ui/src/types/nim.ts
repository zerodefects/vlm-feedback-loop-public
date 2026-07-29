// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * TypeScript mirrors of backend Pydantic schemas for NIM endpoints,
 * model configs, local NIM deployment, and environment assessment.
 */

// ── Literal unions ──────────────────────────────────────────────────────────

export type AuthMode = "bearer" | "none";
export type ProbeKind = "models" | "embeddings";
export type RecommendedMode = "hosted" | "local" | "none";
export type CapabilityStatus = "unknown" | "supported" | "unsupported";

// ── Environment assessment (GET /v1/environment) ────────────────────────────

export interface GpuInfo {
  name: string;
  memory_total_gb: number;
  compute_capability?: number | null;
}

export interface LocalDeployableModel {
  model_name: string;
  nim_container_image: string;
  gpu_memory_minimum_gb: number;
  compute_capability_minimum?: number | null;
  fits: boolean;
}

export interface MissingPrerequisite {
  check: string;
  install_hint: string;
}

export interface EmbeddingDeploymentSummary {
  model_name: string;
  nim_container_image: string;
  gpu_memory_minimum_gb: number;
  fits: boolean;
  provider: string;
}

export interface ActiveLocalNimResident {
  project_id: string;
  project_name: string;
  local_nim_deployment_id: string;
  role: string;
  model_name: string | null;
  nim_container_image: string;
  gpu_assignment: string;
  status: string;
}

export interface EnvironmentResponse {
  hosted_nim_available: boolean;
  local_deploy_available: boolean;
  docker_available: boolean;
  nvidia_toolkit_available: boolean;
  nvidia_api_key_configured: boolean;
  ngc_api_key_configured: boolean;
  gpus: GpuInfo[];
  local_deployable_models: LocalDeployableModel[];
  embedding_deployment: EmbeddingDeploymentSummary;
  missing_prerequisites: MissingPrerequisite[];
  recommended_teacher_mode: RecommendedMode;
  recommended_embedding_mode: RecommendedMode;
  /** Backend-authoritative NIM startup budget (NIM_STARTUP_TIMEOUT_S) shown
      in benchmark stage labels. */
  nim_startup_timeout_s: number;
  /** Backend-authoritative benchmark concurrency sweep
      (STUDENT_LATENCY_TEST_CONCURRENCIES) shown as Compare-table columns. */
  student_latency_test_concurrencies: number[];
  /** Backend-authoritative seeded hosted default Teacher
      (DEFAULT_TEACHER_MODEL). Confirm Defaults preselects this when a
      project's stored teacher id is null/orphaned — the UI must not
      hardcode a model name here. */
  default_teacher_model_name: string;
  /**
   * Populated when a teacher-eligible locally-deployable model fits
   * the detected GPU (quality-first among eligible models: Omni on supported
   * ≥80 GB cc≥9.0 GPUs, CR3-Nano at ≥56 GB, or Cosmos Reason2 2B at
   * 36–55 GB). Null when no GPU or no teacher fits. The frontend uses
   * these to render the setup-choice screen's local primary card
   * (Case A: no key + GPU + fit) or hybrid peer card (Case B: key +
   * GPU + fit).
   */
  recommended_local_teacher_model_name?: string | null;
  recommended_local_teacher_image?: string | null;
  recommended_local_teacher_gpu_memory_minimum_gb?: number | null;
  /** Blueprint-managed NIMs currently occupying host GPUs. */
  active_local_nim_residents?: ActiveLocalNimResident[];
  /**
   * When false, the NIM Configuration UI hides
   * the "Save to ~/.vlm_feedback_loop/.env" checkbox and ``POST
   * /v1/secrets:set`` returns 403 on ``persist=true``. Production /
   * container deployments typically set this to false (the .env file is
   * managed externally — by the operator, k8s secrets, etc.).
   */
  allow_secret_persist: boolean;
}

// ── Connection testing (POST /v1/nim/test_connection) ───────────────────────

export interface ConnectionTestRequest {
  base_url: string;
  auth_mode?: AuthMode;
  credential_transient?: string | null;
  probe_kind?: ProbeKind;
}

export interface ConnectionTestResponse {
  success: boolean;
  models?: string[] | null;
  error?: string | null;
}

// ── Model configs ───────────────────────────────────────────────────────────

export type UnavailableReason =
  | "no_nvidia_api_key"
  | "hosted_not_compatible"
  | "endpoint_unhealthy"
  | "local_not_running"
  | "endpoint_missing"
  | "unknown_endpoint_mode";

export interface ModelAvailability {
  available: boolean;
  reason: UnavailableReason | null;
}

export interface ModelConfigResponse {
  model_config_id: string;
  project_id: string;
  endpoint_id: string;
  model_name: string;
  context_window_tokens: number;
  eligible_roles: string[];
  supports_image_input: boolean;
  structured_generation_support: CapabilityStatus;
  thinking_toggle_mode: string;
  thinking_toggle_support: CapabilityStatus;
  visual_budget_mode: string;
  visual_budget_support: CapabilityStatus;
  model_quantization: string | null;
  nim_model_profile: string | null;
  nim_profile_metadata: Record<string, unknown> | null;
  local_deploy_metadata: Record<string, unknown> | null;
  tao_base_experiment_id?: string | null;
  tao_base_experiment_pull_status?: string | null;
  hosted_compatible: boolean;
  availability: ModelAvailability;
  created_at: string;
}

export interface ModelConfigListResponse {
  items: ModelConfigResponse[];
  next_cursor?: string | null;
}

// (ModelConfigResponse mirrors only the fields the UI consumes; the API
// also carries e.g. max_images_per_request and default_icl_max_examples.)

// ── Local NIM deployment ────────────────────────────────────────────────────

export interface PreflightCheck {
  check_name: string;
  passed: boolean;
  diagnostic: string;
}

export interface PreflightResponse {
  all_passed: boolean;
  checks: PreflightCheck[];
  docker_run_command?: string | null;
  resolved_port?: number | null;
  gpu_assignment?: string | null;
}

export interface LocalNimPreflightRequest {
  role: "teacher" | "embedding";
  model_config_id?: string | null;
  nim_container_image?: string | null;
  gpu_assignment?: string | null;
}

export interface LocalNimDeployRequest {
  role: "teacher" | "embedding";
  model_config_id?: string | null;
  nim_container_image?: string | null;
  gpu_assignment?: string | null;
  preferred_port?: number | null;
  replace_resident?: boolean;
  /** Teacher-only: select this model after the NIM passes verification. */
  activate_on_success?: boolean;
}

export interface LocalNimDeploymentResponse {
  local_nim_deployment_id: string;
  project_id: string;
  model_config_id: string;
  role: string;
  nim_container_image: string;
  container_name: string;
  container_id: string | null;
  host_port: number;
  endpoint_url: string;
  gpu_assignment: string;
  status: string;
  status_reason: string | null;
  activate_on_success?: boolean;
  deployed_at: string | null;
  stopped_at: string | null;
  created_at: string;
  // False when the project's active config for this role no longer
  // references this deployment's model config (stale failure evidence).
  matches_active_role_config?: boolean;
}

export interface LocalNimDeployResponse {
  /** Null when an already-running Teacher resident was reused. */
  deployment: LocalNimDeploymentResponse | null;
  preflight: PreflightResponse;
  disposition?: "queued" | "reused";
  resident?: ActiveLocalNimResident | null;
}

export interface LocalNimGpuConflict {
  code: "gpu_occupied" | "gpu_exhausted" | "resident_starting";
  message: string;
  can_replace: boolean;
  matches_requested_model: boolean;
  resident: ActiveLocalNimResident | null;
}

export interface LocalNimDeploymentListResponse {
  items: LocalNimDeploymentResponse[];
}

// ── Action Requests ─────────────────────────────────────────────────────────

export interface ActionRequestGenerateRequest {
  request_type: string;
  context?: Record<string, unknown> | null;
}

export interface ActionRequestGenerateResponse {
  request_type: string;
  generated_at: string;
  project_name: string;
  technical_requirements: Record<string, unknown>;
  current_environment: Record<string, unknown>;
  rendered_text: string;
}

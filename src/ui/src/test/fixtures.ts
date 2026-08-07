// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared, fully-typed API-response fixtures.
 *
 * Building fixtures through these factories keeps every fixture assignable
 * to the real wire type — when the type gains a required field,
 * `pnpm typecheck` fails here instead of pages silently reading `undefined`
 * in tests.
 *
 * Defaults describe a post-setup, labeling-ready project (guidance active,
 * hosted teacher selected, no student, backend default thresholds).
 */

import type { ProjectCounts, ProjectResponse } from "@/types/project";
import type { EnvironmentResponse } from "@/types/nim";

function makeProjectCounts(overrides: Partial<ProjectCounts> = {}): ProjectCounts {
  return {
    verified: 0,
    unlabeled: 0,
    auto_labeled: 0,
    omitted: 0,
    pending_relabel: 0,
    prior_relabeled: 0,
    ...overrides,
  };
}

export function makeProjectResponse(
  overrides: Partial<ProjectResponse> = {},
): ProjectResponse {
  return {
    project_id: "test-pid",
    name: "Test Project",
    description: null,
    project_dir: "/tmp/workspace/projects/test-pid",
    created_at: "2026-04-13T10:00:00Z",
    updated_at: "2026-04-13T10:00:00Z",
    counts: makeProjectCounts(),
    teacher_model_config_id: "mc-teacher",
    active_guidance_id: "g-1",
    active_student_model_config_id: null,
    labeling_generation_preset_key: "precise",
    thinking_default_on: true,
    visual_budget_preset_key: "balanced",
    structured_generation_mode_default: "auto",
    rationale_anti_anchoring: true,
    auto_evaluate_enabled: false,
    icl_recommendation_dismissed_at_count: 0,
    export_field_mode: "all",
    embedding_provider: "none",
    embedding_model_id: null,
    embedding_dim: null,
    embedding_endpoint_id: null,
    phash_algorithm: "dct_phash_64",
    schema_refinement_reminders_dismissed: 0,
    schema_change_context_example_key: null,
    test_pool_fraction: 0.4,
    scaleup_exact_match_threshold: 0.8,
    scaleup_per_field_match_threshold: 0.8,
    scaleup_min_per_value_f1_threshold: 0.6,
    scaleup_accept_rate_threshold: 0.8,
    scaleup_accept_rate_window: 50,
    scaleup_min_test_pool_size: 60,
    archived_at: null,
    setup_completed_at: "2026-04-13T10:00:00Z",
    ...overrides,
  };
}

/**
 * Defaults describe a bare host: no keys, no Docker/toolkit, no GPUs, the
 * embedding NIM catalog entry present but not deployable. Timeout/concurrency
 * values mirror the backend defaults (`NIM_STARTUP_TIMEOUT_S`,
 * `STUDENT_LATENCY_TEST_CONCURRENCIES`).
 */
export function makeEnvironmentResponse(
  overrides: Partial<EnvironmentResponse> = {},
): EnvironmentResponse {
  return {
    hosted_nim_available: false,
    local_deploy_available: false,
    docker_available: false,
    nvidia_toolkit_available: false,
    nvidia_api_key_configured: false,
    ngc_api_key_configured: false,
    gpus: [],
    local_deployable_models: [],
    embedding_deployment: {
      model_name: "nvidia/llama-nemotron-embed-vl-1b-v2",
      nim_container_image: "nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0",
      gpu_memory_minimum_gb: 24,
      fits: false,
      provider: "none",
    },
    missing_prerequisites: [],
    recommended_teacher_mode: "none",
    recommended_embedding_mode: "none",
    nim_startup_timeout_s: 1200,
    student_latency_test_concurrencies: [1, 8, 24],
    default_teacher_model_name: "stepfun-ai/step-3.7-flash",
    recommended_local_teacher_model_name: null,
    recommended_local_teacher_image: null,
    recommended_local_teacher_gpu_memory_minimum_gb: null,
    allow_secret_persist: true,
    ...overrides,
  };
}

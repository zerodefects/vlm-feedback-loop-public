// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * TypeScript mirrors of backend Pydantic response schemas.
 * Minimal shape — only the fields the UI actually consumes.
 */

export interface ProjectCounts {
  verified: number;
  unlabeled: number;
  auto_labeled: number;
  omitted: number;
  pending_relabel: number;
  /**
   * Verified examples that still carry a ``prior_verified_label_ref`` — i.e.
   * priors the SME has already re-labeled after a semantic Core change.
   * Powers the labeling screen's "Prior labels: N of M re-labeled" progress strip
   * (M = prior_relabeled + pending_relabel).
   */
  prior_relabeled: number;
}

export interface ProjectListItem {
  project_id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  counts: ProjectCounts;
  /**
   * ISO 8601 string when the project is soft-archived; null when
   * active. Drives the "Show archived" toggle and per-card
   * archived-state styling on the Project List.
   */
  archived_at: string | null;
  /**
   * ISO 8601 string once the SME has acknowledged onboarding. Null on
   * freshly-created projects until they transition through the NIM setup
   * and model-defaults screens. Drives ProjectIndexRedirect's
   * setup-routing gate.
   */
  setup_completed_at: string | null;
}

export interface ProjectListResponse {
  items: ProjectListItem[];
  next_cursor: string | null;
  /**
   * Workspace-global: true when at least one project is archived,
   * regardless of the include_archived filter. Backed by a cheap
   * marker-file scan server-side; drives the "Show archived" affordance
   * without an archived-inclusive fetch.
   */
  has_archived: boolean;
}

export interface ProjectCreateRequest {
  name: string;
  description?: string | null;
}

export interface ProjectResponse {
  project_id: string;
  name: string;
  description: string | null;
  project_dir: string;
  created_at: string;
  updated_at: string;

  /**
   * Example-state counts. Mirrors backend ``ProjectResponse``'s
   * ``counts: ProjectCounts`` so a single project-detail fetch powers
   * the Student Training screen's Training Data card without pulling
   * the full project list.
   */
  counts: ProjectCounts;

  // Selection pointers
  teacher_model_config_id: string;
  active_guidance_id: string | null;
  active_student_model_config_id: string | null;

  // Generation Controls
  labeling_generation_preset_key: string;
  thinking_default_on: boolean;
  visual_budget_preset_key: string;
  structured_generation_mode_default: string;

  // Rationale
  rationale_anti_anchoring: boolean;

  // Evaluation
  auto_evaluate_enabled: boolean;
  icl_recommendation_dismissed_at_count: number;

  // Export
  export_field_mode: string;

  // Embedding
  embedding_provider: string;
  embedding_model_id: string | null;
  embedding_dim: number | null;
  embedding_endpoint_id: string | null;

  // pHash
  phash_algorithm: string;

  // Schema refinement reminders
  schema_refinement_reminders_dismissed: number;
  schema_change_context_example_key: string | null;

  // Test Pool
  test_pool_fraction: number;

  // Scale-Up Gate
  scaleup_exact_match_threshold: number;
  scaleup_per_field_match_threshold: number;
  scaleup_min_per_value_f1_threshold: number;
  scaleup_accept_rate_threshold: number;
  scaleup_accept_rate_window: number;
  scaleup_min_test_pool_size: number;

  /** ISO 8601 string when archived; null when active. */
  archived_at: string | null;
  /**
   * ISO 8601 string once the SME has acknowledged onboarding for this
   * project. When null, ProjectIndexRedirect routes to /setup.
   */
  setup_completed_at: string | null;
}

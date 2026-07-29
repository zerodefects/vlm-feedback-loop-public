// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * TypeScript mirrors of backend Pydantic schemas for the labeling workflow.
 *
 * Source: src/backend/vlm_feedback_loop/schemas/proposal.py,
 *         review_selector.py, label.py
 */

// ── Proposal ─────────────────────────────────────────────────────────────────

export type InvocationStatus =
  | "success"
  | "schema_invalid"
  | "timeout"
  | "endpoint_error"
  | "rate_limited";

export interface ProposalRequest {
  example_key: string;
  teacher_model_config_id_override?: string | null;
  guidance_id_override?: string | null;
  generation_preset_key_override?: string | null;
  thinking_mode_override?: "on" | "off" | null;
  visual_budget_preset_key_override?: string | null;
  retry_of_inference_invocation_id?: string | null;
  use_existing_label?: boolean;
}

export interface ProposalResponse {
  inference_invocation_id: string;
  example_key: string;
  proposal_json: Record<string, unknown> | null;
  schema_valid_core: boolean;
  validation_errors_core: string[];
  validation_errors_aux: string[];
  invocation_status: InvocationStatus;
  latency_ms_end_to_end: number | null;
  // Inline ICL image injection — per-invocation counts.
  // Invariant: ``icl_images_attached_count === icl_example_keys_used.length``
  // for every successful proposal: every retained ICL example is image-grounded.
  icl_images_attached_count: number;
  icl_example_keys_used: string[];
  used_existing_label: boolean;
}

// ── Review Selector ──────────────────────────────────────────────────────────

export interface ReviewSelectorNextResponse {
  example_key: string | null;
  example_state: string | null;
  has_existing_label: boolean;
  selection_mode: string;
  queue_empty: boolean;
  /** Backend filesystem path of the selected example — feeds the
   * missing-image diagnostic ("Expected:" line). Null when queue_empty. */
  storage_ref: string | null;
  /** JSON-serialized PriorLabelSnapshot for re-labeling after a semantic
   * schema change; null when the example has no prior label. */
  prior_verified_label_ref: string | null;
}

// ── Label Save ───────────────────────────────────────────────────────────────

export type RationaleSource =
  | "teacher_proposal"
  | "sme_edited"
  | "teacher_regenerated_approved";

export interface LabelSaveRequest {
  example_key: string;
  inference_invocation_id: string;
  label_json: Record<string, unknown>;
  rationale_source?: RationaleSource | null;
  rationale_regeneration_invocation_id?: string | null;
}

export interface LabelSaveResponse {
  example_key: string;
  label_status: string;
  verified_outcome: string; // "Accept" | "Edit"
  verified_at: string;
  edited_core_fields: string[];
  edited_aux_fields: string[];
  pool_assignment: string | null;
}

// ── Skip ─────────────────────────────────────────────────────────────────────

export interface SkipResponse {
  example_key: string;
  state: string; // "Omitted"
  omitted_at: string;
}

// ── Restore Omitted ──────────────────────────────────────────────────────────

export interface RestoreOmittedResponse {
  restored_count: number;
}

// ── Prior-Label Snapshot (from schema evolution) ─────────────────────────────

export interface PriorLabelSnapshot {
  label_json: Record<string, unknown>;
  vlm_proposal_json: Record<string, unknown>;
  edited_core_fields: string[];
  edited_aux_fields: string[];
  verified_outcome: string; // "Accept" | "Edit"
  rationale_note?: string;
  guidance_id: string;
}

// ── Rationale Regeneration ───────────────────────────────────────────────────

export interface RationaleRegenerationRequest {
  teacher_model_config_id?: string | null;
}

export interface RationaleRegenerationResponse {
  inference_invocation_id: string;
  rationale_note: string;
  invocation_status: string; // "success" | "timeout" | "endpoint_error"
}

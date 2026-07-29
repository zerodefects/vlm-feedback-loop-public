// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * TypeScript mirrors of backend Guidance Pydantic schemas.
 *
 * Source: src/backend/vlm_feedback_loop/schemas/guidance.py
 */

// ── Field types ─────────────────────────────────────────────────────────────

export type FieldType = "enum" | "enum_set" | "boolean" | "integer" | "string";
export type FieldRole = "core" | "aux";

// ── Request types ───────────────────────────────────────────────────────────

/** A single SchemaCore field definition (create/validate/local state). */
export interface SchemaFieldInput {
  field_name: string;
  type: FieldType;
  role: FieldRole;
  allowed_values?: string[] | null;
  minimum?: number | null;
  maximum?: number | null;
  min_length?: number | null;
  max_length?: number | null;
  display_order: number;
}

/** Edit requests preserve backend-issued identity for existing fields. */
export interface SchemaFieldEditInput extends SchemaFieldInput {
  field_id?: string | null;
}

/** POST /v1/projects/{id}/guidance — wire key is "schema", not "schema_fields". */
export interface GuidanceCreateRequest {
  description: string;
  schema: SchemaFieldInput[];
  rules: string;
}

/** POST /v1/projects/{id}/guidance:validate_draft */
export interface DraftValidationRequest {
  description: string;
  schema: SchemaFieldInput[];
  rules: string;
}

// ── Response types ──────────────────────────────────────────────────────────

/** A single SchemaCore field in a response — includes system-generated field_id. */
export interface SchemaFieldResponse {
  field_id: string;
  field_name: string;
  type: string;
  role: string;
  allowed_values?: string[] | null;
  minimum?: number | null;
  maximum?: number | null;
  min_length?: number | null;
  max_length?: number | null;
  display_order: number;
}

/** A validation issue from draft validation or create. */
export interface SchemaIssueResponse {
  severity: string;
  code: string;
  message: string;
  field_path?: string | null;
}

/** Full Guidance record for API responses. */
export interface GuidanceResponse {
  guidance_id: string;
  project_id: string;
  version_number: number;
  description: string;
  schema_fields: SchemaFieldResponse[];
  rules: string;
  derived_json_schema: Record<string, unknown>;
  generation_order: string[];
  schema_hash: string;
  created_at: string;
  semantic_core_change_from_guidance_id?: string | null;
  schema_change_summary?: Record<string, unknown> | null;
}

/** Paginated list of Guidance records. */
export interface GuidanceListResponse {
  items: GuidanceResponse[];
  next_cursor?: string | null;
}

/** Response from the draft validation endpoint. */
export interface DraftValidationResponse {
  issues: SchemaIssueResponse[];
  derived_json_schema?: Record<string, unknown> | null;
  schema_hash?: string | null;
  save_allowed: boolean;
}

// ── Edit types ──────────────────────────────────────────────────────────────

/** POST /v1/projects/{id}/guidance:edit */
export interface GuidanceEditRequest {
  description: string;
  schema: SchemaFieldEditInput[];
  rules: string;
  dry_run: boolean;
  schema_change_context_example_key?: string | null;
}

/** A single detected change between schema versions. */
export interface FieldChangeResponse {
  field_id: string;
  change_type: string;
  classification: string; // "in_place" | "semantic"
  detail?: Record<string, unknown>;
}

/** Response from dry_run=true: classification + affected counts. */
export interface EditPreviewResponse {
  edit_type: string; // "in_place" | "semantic" | "no_change"
  changes: FieldChangeResponse[];
  verified_count: number;
  auto_labeled_count: number;
  change_summary: Record<string, unknown>;
}

/** Response from dry_run=false: new guidance + mutation results. */
export interface EditExecuteResponse {
  guidance: GuidanceResponse;
  edit_type: string;
  verified_reverted_count: number;
  auto_labeled_reverted_count: number;
  changes: FieldChangeResponse[];
}

// ── ICL count ───────────────────────────────────────────────────────────────

/** Response from GET /v1/projects/{id}/guidance:icl_count — the ICL-eligible
 * Edit count (Verified Edits, non-pool, current Guidance). */
export interface IclCountResponse {
  eligible_count: number;
}

// ── Schema refinement reminders ─────────────────────────────────────────────
// The backend owns reminder eligibility (thresholds from Settings, the
// higher-of-two rule, suppression once Guidance was edited past v1); the
// UI only renders what `active_reminder` says.

/** Response from GET /v1/projects/{id}/guidance:reminder_status. */
export interface ReminderStatusResponse {
  /** 1 or 2 for the reminder that should show; null when none is active. */
  active_reminder: number | null;
  verified_count: number;
  threshold_1: number;
  threshold_2: number;
  dismissed_count: number;
}

/** Response from POST /v1/projects/{id}/guidance:dismiss_reminder. */
export interface ReminderDismissResponse {
  dismissed_count: number;
}

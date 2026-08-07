// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared field utilities for Create and Edit Guidance screens.
 *
 * Pure functions — no React dependencies.
 */

import type {
  SchemaFieldInput,
  SchemaFieldEditInput,
  SchemaFieldResponse,
  FieldType,
  FieldRole,
  FieldChangeResponse,
} from "@/types/guidance";
import { RATIONALE_NOTE_FIELD_NAME } from "@/lib/guidance-templates";

// ── Client-side field identity ──────────────────────────────────────────────

/** Extends SchemaFieldInput with a local React key. Never sent to backend. */
export interface ClientField extends SchemaFieldEditInput {
  _clientId: string;
}

export function stampClientIds(fields: SchemaFieldInput[]): ClientField[] {
  return fields.map((f) => ({ ...f, _clientId: crypto.randomUUID() }));
}

/** Strip _clientId before sending to backend. */
export function stripClientIds(fields: ClientField[]): SchemaFieldEditInput[] {
  return fields.map(({ _clientId, ...rest }) => rest);
}

/** Strip both client and persisted identities for the create-shaped draft API. */
export function stripDraftIds(fields: ClientField[]): SchemaFieldInput[] {
  return fields.map(({ _clientId, field_id: _fieldId, ...rest }) => rest);
}

/**
 * Convert schema fields from the response shape (``schema_fields``) to the
 * input shape mutations send back (``schema``). Used by the Edit Guidance
 * screen, which resubmits a loaded Guidance's fields.
 */
export function responseFieldsToInput(
  fields: SchemaFieldResponse[],
): SchemaFieldEditInput[] {
  return fields.map((f) => ({
    field_name: f.field_name,
    type: f.type as FieldType,
    role: f.role as FieldRole,
    allowed_values: f.allowed_values,
    minimum: f.minimum,
    maximum: f.maximum,
    min_length: f.min_length,
    max_length: f.max_length,
    display_order: f.display_order,
    field_id: f.field_id,
  }));
}

export function recalcDisplayOrders(fields: ClientField[]): ClientField[] {
  const core = [...fields.filter((f) => f.role === "core")];
  const aux = [...fields.filter((f) => f.role === "aux")];
  core.sort((a, b) => a.display_order - b.display_order);
  aux.sort((a, b) => {
    if (a.field_name === RATIONALE_NOTE_FIELD_NAME) return -1;
    if (b.field_name === RATIONALE_NOTE_FIELD_NAME) return 1;
    return a.display_order - b.display_order;
  });
  const result: ClientField[] = [];
  core.forEach((f, i) => result.push({ ...f, display_order: i }));
  aux.forEach((f, i) => result.push({ ...f, display_order: i }));
  return result;
}

// ── Backend field_path addressing ───────────────────────────────────────────

/**
 * Map each field's client id to the ``field_path`` prefix the backend uses
 * for it in validation issues — ``core[i]`` / ``aux[j]``, indexed per section
 * in submission order (see ``services/schema_core.py::SchemaIssue``).
 *
 * Pure presentation glue: it addresses builder rows so backend-reported
 * issues can be rendered inline. The validation rules themselves live only
 * in the backend (`guidance:validate_draft`).
 */
export function computeFieldPathPrefixes(fields: ClientField[]): Map<string, string> {
  const counters = { core: 0, aux: 0 };
  const prefixes = new Map<string, string>();
  for (const f of fields) {
    const section = f.role === "aux" ? "aux" : "core";
    prefixes.set(f._clientId, `${section}[${counters[section]}]`);
    counters[section] += 1;
  }
  return prefixes;
}

// ── Example output generator ────────────────────────────────────────────────

export function generateExampleOutput(fields: ClientField[]): Record<string, unknown> {
  const example: Record<string, unknown> = {};
  const sorted = [...fields].sort((a, b) => {
    if (a.field_name === RATIONALE_NOTE_FIELD_NAME) return -1;
    if (b.field_name === RATIONALE_NOTE_FIELD_NAME) return 1;
    if (a.role === "aux" && b.role === "core") return -1;
    if (a.role === "core" && b.role === "aux") return 1;
    return a.display_order - b.display_order;
  });
  for (const f of sorted) {
    if (f.field_name === RATIONALE_NOTE_FIELD_NAME) {
      example[f.field_name] = "visible dent on front panel";
      continue;
    }
    switch (f.type) {
      case "enum":
        example[f.field_name] = f.allowed_values?.[0] ?? "";
        break;
      case "enum_set":
        example[f.field_name] = f.allowed_values?.slice(0, 1) ?? [];
        break;
      case "boolean":
        example[f.field_name] = false;
        break;
      case "integer": {
        const min = f.minimum ?? 0;
        const max = f.maximum ?? 10;
        example[f.field_name] = Math.floor((min + max) / 2);
        break;
      }
      case "string":
        example[f.field_name] = "";
        break;
    }
  }
  return example;
}

// ── Semantic change description (confirmation dialog) ──────────────────────

/**
 * Produce a human-readable sentence describing semantic schema changes for the
 * confirmation dialog. Returns empty string when no semantic changes exist.
 *
 * The sentence pattern is "Changing [specific thing] changes what a
 * correct answer looks like."
 */
export function describeSemanticChanges(changes: FieldChangeResponse[]): string {
  const semantic = changes.filter((c) => c.classification === "semantic");
  if (semantic.length === 0) return "";

  const fragments = semantic.map((c) => {
    const name = (c.detail?.field_name as string) ?? "";
    switch (c.change_type) {
      case "type_change":
        return `the type of "${name}"`;
      case "constraint_change":
        return `the ${(c.detail?.constraint as string) ?? "constraints"} for "${name}"`;
      case "allowed_value_change":
        return `the allowed values for "${name}"`;
      case "add_core_field":
        return `adding Core field "${name}"`;
      case "remove_core_field":
        return `removing Core field "${name}"`;
      case "role_change":
        return `the role of "${name}"`;
      default:
        return `"${name}"`;
    }
  });

  const joined =
    fragments.length === 1
      ? fragments[0]
      : fragments.length === 2
        ? `${fragments[0]} and ${fragments[1]}`
        : `${fragments.slice(0, -1).join(", ")}, and ${fragments[fragments.length - 1]}`;

  return `Changing ${joined} changes what a correct answer looks like.`;
}

// ── Field type labels ───────────────────────────────────────────────────────

export const FIELD_TYPE_OPTIONS: { value: FieldType; label: string }[] = [
  { value: "enum", label: "Enum" },
  { value: "enum_set", label: "Enum Set" },
  { value: "boolean", label: "Boolean" },
  { value: "integer", label: "Integer" },
  { value: "string", label: "String" },
];

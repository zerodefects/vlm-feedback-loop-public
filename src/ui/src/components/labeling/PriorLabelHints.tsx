// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-field prior-label annotations for re-labeling after schema
 * evolution.
 *
 * Shows: prior value, "you edited" badge, VLM agree/disagree indicator,
 * schema-invalid flag, and [Adopt prior] button.
 */

import { Button, Text } from "@kui/react";
import { Check, X as XIcon, AlertTriangle, ArrowLeft } from "lucide-react";

import { fieldValuesEqual } from "@/lib/field-values";
import type { SchemaFieldResponse } from "@/types/guidance";

interface PriorLabelFieldHintProps {
  field: SchemaFieldResponse;
  priorValue: unknown;
  currentProposalValue: unknown;
  wasEdited: boolean;
  isSchemaValid: boolean;
  onAdoptPrior: () => void;
}

export function PriorLabelFieldHint({
  field,
  priorValue,
  currentProposalValue,
  wasEdited,
  isSchemaValid,
  onAdoptPrior,
}: PriorLabelFieldHintProps) {
  if (priorValue === undefined) return null;

  const agrees = priorAgreesWithProposal(field, priorValue, currentProposalValue);
  const priorDisplay = formatValue(priorValue);

  return (
    <div
      className="flex items-center flex-wrap gap-1.5 mt-1 text-xs"
      data-testid={`prior-hint-${field.field_name}`}
    >
      {/* Prior value */}
      <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
        Prior: {priorDisplay}
      </Text>

      {/* "you edited" badge */}
      {wasEdited && (
        <span
          className="glass-pill"
          style={{ fontSize: 10, padding: "2px 6px" }}
          data-testid={`prior-edited-${field.field_name}`}
        >
          <Text kind="label/regular/xs">you edited</Text>
        </span>
      )}

      {/* VLM agree/disagree */}
      {agrees ? (
        <span
          className="flex items-center gap-0.5"
          style={{ color: "var(--accent-green)" }}
        >
          <Check size={10} /> <Text kind="label/regular/xs">Agrees</Text>
        </span>
      ) : (
        <span
          className="flex items-center gap-0.5"
          style={{ color: "var(--warning-amber)" }}
        >
          <XIcon size={10} /> <Text kind="label/regular/xs">Disagrees</Text>
        </span>
      )}

      {/* Schema-invalid flag */}
      {!isSchemaValid && (
        <span
          className="flex items-center gap-0.5 text-error"
          data-testid={`prior-invalid-${field.field_name}`}
        >
          <AlertTriangle size={10} />{" "}
          <Text kind="label/regular/xs">schema-invalid</Text>
        </span>
      )}

      {/* [Adopt prior] button — only when schema-valid */}
      {isSchemaValid && (
        <Button
          kind="tertiary"
          size="tiny"
          onClick={onAdoptPrior}
          data-testid={`adopt-prior-${field.field_name}`}
          className="inline-flex items-center gap-0.5"
          style={{ fontSize: 10 }}
        >
          <ArrowLeft size={10} /> Adopt prior
        </Button>
      )}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────
//
// DISPLAY-ONLY heuristics. These mirror the backend's per-field
// normalization (`services/exact_match_evaluator.py` `_normalize_*`, the
// single canonical validator) just closely enough to annotate the hint row;
// the backend remains authoritative at save time — nothing here gates or
// transforms what gets persisted.

/** Normalize an enum token the way the backend does: `strip().lower()`. */
function normalizeEnumToken(value: string): string {
  return value.trim().toLowerCase();
}

/**
 * Field-type-aware agree/disagree for the hint row: compares the prior
 * value against the VLM proposal after backend-style normalization, so a
 * prior "Dent " agrees with a proposed "dent" exactly when the backend
 * would canonicalize both to the same value. Falls back to the shared
 * strict `fieldValuesEqual` for types with no textual normalization.
 */
function priorAgreesWithProposal(
  field: SchemaFieldResponse,
  priorValue: unknown,
  proposalValue: unknown,
): boolean {
  switch (field.type) {
    case "enum":
      if (typeof priorValue === "string" && typeof proposalValue === "string") {
        return normalizeEnumToken(priorValue) === normalizeEnumToken(proposalValue);
      }
      return fieldValuesEqual(priorValue, proposalValue);

    case "enum_set":
      if (
        Array.isArray(priorValue) &&
        Array.isArray(proposalValue) &&
        priorValue.every((v) => typeof v === "string") &&
        proposalValue.every((v) => typeof v === "string")
      ) {
        // Backend normalizes enum_set to a deduped sorted set — compare as sets.
        const a = new Set((priorValue as string[]).map(normalizeEnumToken));
        const b = new Set((proposalValue as string[]).map(normalizeEnumToken));
        return a.size === b.size && [...a].every((v) => b.has(v));
      }
      return fieldValuesEqual(priorValue, proposalValue);

    case "string":
      if (typeof priorValue === "string" && typeof proposalValue === "string") {
        return priorValue.trim() === proposalValue.trim();
      }
      return fieldValuesEqual(priorValue, proposalValue);

    default:
      return fieldValuesEqual(priorValue, proposalValue);
  }
}

function formatValue(value: unknown): string {
  if (value == null) return "—";
  if (Array.isArray(value)) return `[${value.join(", ")}]`;
  return String(value);
}

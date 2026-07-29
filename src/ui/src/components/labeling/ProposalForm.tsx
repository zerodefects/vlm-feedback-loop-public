// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Editable proposal form for the labeling screen.
 *
 * Renders Core fields first, then Aux fields with secondary styling.
 * The optional rationale note is handled separately by LabelingPage.
 * Reports current values and dirty-field set to parent.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Text } from "@kui/react";

import { fieldValuesEqual } from "@/lib/field-values";
import { RATIONALE_NOTE_FIELD_NAME } from "@/lib/guidance-templates";
import { PriorLabelFieldHint, isPriorValueSchemaValid } from "./PriorLabelHints";
import type { SchemaFieldResponse } from "@/types/guidance";
import type { PriorLabelSnapshot } from "@/types/labeling";

// ── Props ────────────────────────────────────────────────────────────────────

interface ProposalFormProps {
  schemaFields: SchemaFieldResponse[];
  proposalJson: Record<string, unknown>;
  onValuesChange: (values: Record<string, unknown>, dirtyFields: Set<string>) => void;
  disabled?: boolean;
  /** Increment to force-reset form values to proposalJson (used by [Reset]). */
  resetKey?: number;
  /** Prior-label snapshot — shown as annotations when re-labeling after schema change. */
  priorLabel?: PriorLabelSnapshot | null;
  /** Called when the user clicks [Adopt prior] on a field. */
  onAdoptPrior?: (fieldName: string, value: unknown) => void;
}

// ── Component ────────────────────────────────────────────────────────────────

export function ProposalForm({
  schemaFields,
  proposalJson,
  onValuesChange,
  disabled = false,
  resetKey = 0,
  priorLabel = null,
  onAdoptPrior,
}: ProposalFormProps) {
  // Current form values — initialized from proposalJson.
  const [values, setValues] = useState<Record<string, unknown>>(() => ({
    ...proposalJson,
  }));
  const originalRef = useRef<Record<string, unknown>>({ ...proposalJson });

  // Reset form when proposal changes (new example or retry)
  const prevProposalRef = useRef(proposalJson);
  if (proposalJson !== prevProposalRef.current) {
    prevProposalRef.current = proposalJson;
    setValues({ ...proposalJson });
    originalRef.current = { ...proposalJson };
  }

  // Parent-triggered reset via resetKey (used by [Reset] button)
  const prevResetKeyRef = useRef(resetKey);
  if (resetKey !== prevResetKeyRef.current) {
    prevResetKeyRef.current = resetKey;
    setValues({ ...proposalJson });
  }

  // Separate and sort fields
  const { coreFields, auxFields } = useMemo(() => {
    const core = schemaFields
      .filter((f) => f.role === "core")
      .sort((a, b) => a.display_order - b.display_order);
    const aux = schemaFields
      .filter((f) => f.role === "aux")
      .sort((a, b) => a.display_order - b.display_order);
    return { coreFields: core, auxFields: aux };
  }, [schemaFields]);

  // Compute dirty fields and report up
  useEffect(() => {
    const dirty = new Set<string>();
    for (const f of schemaFields) {
      const orig = originalRef.current[f.field_name];
      const curr = values[f.field_name];
      if (!fieldValuesEqual(orig, curr)) {
        dirty.add(f.field_name);
      }
    }
    onValuesChange(values, dirty);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values]);

  function handleChange(fieldName: string, value: unknown) {
    setValues((prev) => ({ ...prev, [fieldName]: value }));
  }

  // One field row — editor plus the prior-label hint when re-labeling
  // after a schema change. Shared by the Core and Aux sections.
  function renderField(field: SchemaFieldResponse) {
    return (
      <div key={field.field_id}>
        <FieldEditor
          field={field}
          value={values[field.field_name]}
          onChange={(v) => handleChange(field.field_name, v)}
          disabled={disabled}
        />
        {priorLabel && priorLabel.label_json[field.field_name] !== undefined && (
          <PriorLabelFieldHint
            field={field}
            priorValue={priorLabel.label_json[field.field_name]}
            currentProposalValue={proposalJson[field.field_name]}
            wasEdited={
              priorLabel.edited_core_fields.includes(field.field_name) ||
              priorLabel.edited_aux_fields.includes(field.field_name)
            }
            isSchemaValid={isPriorValueSchemaValid(
              field,
              priorLabel.label_json[field.field_name],
            )}
            onAdoptPrior={() => {
              handleChange(field.field_name, priorLabel.label_json[field.field_name]);
              onAdoptPrior?.(field.field_name, priorLabel.label_json[field.field_name]);
            }}
          />
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 fade-in" data-testid="proposal-form">
      {/* Core Fields */}
      <div className="border-accent-core pl-4">
        <Text
          kind="label/bold/xs"
          style={{ color: "var(--text-primary)", display: "block" }}
          className="section-eyebrow mb-3"
        >
          Core Fields
        </Text>
        <div className="flex flex-col gap-3">
          {coreFields.map((field) => renderField(field))}
        </div>
      </div>

      {/* Aux Fields — rationale_note is handled by the dedicated review
          panel when enabled. Computed before the JSX so the heading +
          container are suppressed entirely when there's nothing to show. */}
      {(() => {
        const visibleAuxFields = auxFields.filter(
          (f) => f.field_name !== RATIONALE_NOTE_FIELD_NAME,
        );
        if (visibleAuxFields.length === 0) return null;
        return (
          <div className="border-accent-aux pl-4">
            <Text
              kind="label/bold/xs"
              style={{ color: "var(--text-primary)", display: "block" }}
              className="section-eyebrow mb-3"
            >
              Aux Fields
            </Text>
            <div className="flex flex-col gap-3">
              {visibleAuxFields.map((field) => renderField(field))}
            </div>
          </div>
        );
      })()}
    </div>
  );
}

// ── Field Editor ─────────────────────────────────────────────────────────────

interface FieldEditorProps {
  field: SchemaFieldResponse;
  value: unknown;
  onChange: (value: unknown) => void;
  disabled: boolean;
}

function FieldEditor({ field, value, onChange, disabled }: FieldEditorProps) {
  const label = (
    <Text
      kind="label/regular/sm"
      style={{ color: "var(--text-secondary)" }}
      className="mb-1 block"
    >
      {field.field_name}
    </Text>
  );

  switch (field.type) {
    case "enum":
      return (
        <div data-testid={`field-${field.field_name}`}>
          {label}
          <select
            className="glass-input w-full px-3 py-2 text-sm"
            value={String(value ?? "")}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
            data-testid={`field-input-${field.field_name}`}
          >
            <option value="" disabled>
              Select...
            </option>
            {(field.allowed_values ?? []).map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>
      );

    case "enum_set": {
      const selected = Array.isArray(value) ? (value as string[]) : [];
      return (
        <div data-testid={`field-${field.field_name}`}>
          {label}
          <div className="flex flex-wrap gap-2">
            {(field.allowed_values ?? []).map((v) => {
              const checked = selected.includes(v);
              return (
                <label
                  key={v}
                  className={`glass-pill cursor-pointer select-none ${checked ? "green" : ""}`}
                  style={disabled ? { opacity: 0.5, pointerEvents: "none" } : undefined}
                >
                  <input
                    type="checkbox"
                    className="sr-only"
                    checked={checked}
                    onChange={() => {
                      const next = checked
                        ? selected.filter((s) => s !== v)
                        : [...selected, v];
                      onChange(next);
                    }}
                    disabled={disabled}
                  />
                  <span className="text-xs">{v}</span>
                </label>
              );
            })}
          </div>
        </div>
      );
    }

    case "boolean":
      return (
        <div data-testid={`field-${field.field_name}`}>
          {label}
          <label
            className="flex items-center gap-2 cursor-pointer"
            style={disabled ? { opacity: 0.5, pointerEvents: "none" } : undefined}
          >
            <input
              type="checkbox"
              checked={value === true}
              onChange={(e) => onChange(e.target.checked)}
              disabled={disabled}
              className="glass-input"
              data-testid={`field-input-${field.field_name}`}
            />
            <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
              {value === true ? "true" : "false"}
            </Text>
          </label>
        </div>
      );

    case "integer":
      return (
        <div data-testid={`field-${field.field_name}`}>
          {label}
          <input
            type="number"
            className="glass-input w-full px-3 py-2 text-sm"
            value={value != null ? String(value) : ""}
            min={field.minimum ?? undefined}
            max={field.maximum ?? undefined}
            onChange={(e) => {
              const v = e.target.value;
              onChange(v === "" ? null : Number(v));
            }}
            disabled={disabled}
            data-testid={`field-input-${field.field_name}`}
          />
        </div>
      );

    case "string":
    default:
      return (
        <div data-testid={`field-${field.field_name}`}>
          {label}
          <input
            type="text"
            className="glass-input w-full px-3 py-2 text-sm"
            value={String(value ?? "")}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
            data-testid={`field-input-${field.field_name}`}
          />
        </div>
      );
  }
}

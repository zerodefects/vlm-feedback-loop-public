// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Single editable row in the Schema builder: field name, type selector,
 * per-type constraint editors, reorder/move-section/delete controls, and
 * inline validation errors for the field's path.
 */

import { useEffect, useState } from "react";
import { Select, Text } from "@kui/react";
import {
  GripVertical,
  ChevronUp,
  ChevronDown,
  ArrowRightLeft,
  Trash2,
} from "lucide-react";

import type {
  SchemaFieldInput,
  FieldType,
  SchemaIssueResponse,
} from "@/types/guidance";
import { RATIONALE_NOTE_FIELD_NAME } from "@/lib/guidance-templates";
import type { ClientField } from "./field-utils";
import { FIELD_TYPE_OPTIONS } from "./field-utils";
import { IconBtn } from "./IconBtn";
import { MarkerIcon } from "./MarkerIcon";
import { EnumValueEditor } from "./EnumValueEditor";
import { InlineErrors } from "./InlineErrors";

/** Props for {@link FieldRow}. */
export interface FieldRowProps {
  field: ClientField;
  isFirst: boolean;
  isLast: boolean;
  issues: SchemaIssueResponse[];
  /** Backend field_path prefix for this row (e.g., "core[0]", "aux[2]") —
   *  see field-utils.ts::computeFieldPathPrefixes. */
  pathPrefix: string;
  showMarkers?: boolean;
  markerTooltip?: string;
  onRename: (name: string) => void;
  onChangeType: (type: FieldType) => void;
  onChangeConstraints: (patch: Partial<SchemaFieldInput>) => void;
  onDelete: () => void;
  onMoveSection: () => void;
  onReorder: (direction: "up" | "down") => void;
  onFixit: (code: string) => void;
}

export function FieldRow({
  field,
  isFirst,
  isLast,
  issues,
  pathPrefix,
  showMarkers,
  markerTooltip,
  onRename,
  onChangeType,
  onChangeConstraints,
  onDelete,
  onMoveSection,
  onReorder,
  onFixit,
}: FieldRowProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const hasMinLenError = issues.some(
    (i) => i.field_path === `${pathPrefix}.min_length`,
  );
  const hasMaxLenError = issues.some(
    (i) => i.field_path === `${pathPrefix}.max_length`,
  );
  const hasNameError = issues.some((i) => i.field_path === `${pathPrefix}.name`);
  const hasMinError = issues.some((i) => i.field_path === `${pathPrefix}.minimum`);
  const hasMaxError = issues.some((i) => i.field_path === `${pathPrefix}.maximum`);
  // The INVALID_FIELD_NAME error message is essentially identical to the
  // always-on helper text, so when the error is showing we suppress the helper
  // to avoid the two near-duplicate lines stacking. Other name errors
  // (MISSING / DUPLICATE / TOO_LONG) carry different information, so the
  // helper stays visible and adds the format rule as complementary context.
  const hasInvalidNameError = issues.some(
    (i) => i.field_path === `${pathPrefix}.name` && i.code === "INVALID_FIELD_NAME",
  );
  // Auto-open the advanced expander when a string length error is revealed so
  // the error message isn't hidden behind a collapsed control.
  useEffect(() => {
    if (hasMinLenError) setAdvancedOpen(true);
  }, [hasMinLenError]);

  return (
    <div
      className="px-3 py-2"
      style={{
        background: "var(--block-bg)",
        border: "1px solid var(--glass-border)",
        borderRadius: "var(--glass-radius-sm)",
      }}
      data-testid={`field-row-${field.field_name}`}
    >
      <div className="flex items-center gap-2">
        <GripVertical
          size={14}
          style={{ color: "var(--text-faint)", cursor: "grab" }}
          aria-hidden
        />

        <input
          type="text"
          aria-label={`Field name: ${field.field_name || "new field"}`}
          value={field.field_name}
          onChange={(e) => onRename(e.target.value)}
          placeholder="field_name"
          className="glass-input rounded px-2 py-1 text-sm font-medium"
          style={{ minWidth: 200, width: 220 }}
          aria-invalid={hasNameError ? "true" : undefined}
          data-testid={`field-name-input-${field._clientId}`}
        />

        <span className="relative inline-flex items-center">
          {showMarkers && <MarkerIcon overlay tooltip={markerTooltip} />}
          <Select
            aria-label={`Field type for ${field.field_name || "new field"}`}
            items={FIELD_TYPE_OPTIONS.map((opt) => ({
              value: opt.value,
              children: opt.label,
            }))}
            value={field.type}
            onValueChange={(val) => onChangeType(val as FieldType)}
            size="small"
            data-testid={`type-select-${field._clientId}`}
          />
        </span>

        <span
          className={`glass-pill uppercase tracking-wide ${field.role === "core" ? "green" : ""}`}
        >
          <Text kind="label/regular/xs">{field.role}</Text>
        </span>

        <div className="flex-1" />

        <div className="flex items-center gap-0.5">
          <IconBtn
            onClick={() => onReorder("up")}
            disabled={isFirst}
            label="Move field up"
            testId={`move-up-${field._clientId}`}
          >
            <ChevronUp size={14} style={{ color: "var(--text-muted)" }} />
          </IconBtn>
          <IconBtn
            onClick={() => onReorder("down")}
            disabled={isLast}
            label="Move field down"
            testId={`move-down-${field._clientId}`}
          >
            <ChevronDown size={14} style={{ color: "var(--text-muted)" }} />
          </IconBtn>
          <span className="relative inline-flex items-center">
            {showMarkers && <MarkerIcon overlay tooltip={markerTooltip} />}
            <IconBtn
              onClick={onMoveSection}
              label={field.role === "core" ? "Move to Aux" : "Move to Core"}
              testId={`move-section-${field._clientId}`}
            >
              <ArrowRightLeft size={14} style={{ color: "var(--text-muted)" }} />
            </IconBtn>
          </span>
          <span className="relative inline-flex items-center">
            {showMarkers && <MarkerIcon overlay tooltip={markerTooltip} />}
            <IconBtn
              onClick={onDelete}
              label="Delete field"
              testId={`delete-field-${field._clientId}`}
            >
              <Trash2 size={14} style={{ color: "var(--text-muted)" }} />
            </IconBtn>
          </span>
        </div>
      </div>

      <div className="mt-1 pl-6">
        <InlineErrors
          issues={issues}
          fieldPath={`${pathPrefix}.name`}
          onFix={onFixit}
        />
        {!hasInvalidNameError && (
          <Text
            kind="label/regular/xs"
            style={{ color: "var(--text-faint)", display: "block" }}
            data-testid={`field-name-helper-${field._clientId}`}
          >
            Letters, numbers, and underscores only. Must not start with a number.
          </Text>
        )}
      </div>

      <div className="mt-1.5 pl-6">
        {field.type === "enum" || field.type === "enum_set" ? (
          <>
            <EnumValueEditor
              values={field.allowed_values ?? []}
              onChange={(vals) => onChangeConstraints({ allowed_values: vals })}
              fieldClientId={field._clientId}
              showMarkers={showMarkers}
              markerTooltip={markerTooltip}
            />
            <InlineErrors
              issues={issues}
              fieldPath={`${pathPrefix}.allowed_values`}
              onFix={onFixit}
            />
          </>
        ) : field.type === "integer" ? (
          <>
            <div className="flex items-center gap-3 text-xs">
              {showMarkers && <MarkerIcon tooltip={markerTooltip} />}
              <label
                className="flex items-center gap-1"
                style={{ color: "var(--text-muted)" }}
              >
                <Text kind="label/regular/xs">Min:</Text>
                <input
                  type="number"
                  value={field.minimum ?? ""}
                  onChange={(e) => {
                    const v = e.target.value;
                    onChangeConstraints({ minimum: v === "" ? null : parseInt(v, 10) });
                  }}
                  className="glass-input w-20 rounded px-2 py-1"
                  aria-invalid={hasMinError ? "true" : undefined}
                  data-testid={`min-input-${field._clientId}`}
                />
              </label>
              <label
                className="flex items-center gap-1"
                style={{ color: "var(--text-muted)" }}
              >
                <Text kind="label/regular/xs">Max:</Text>
                <input
                  type="number"
                  value={field.maximum ?? ""}
                  onChange={(e) => {
                    const v = e.target.value;
                    onChangeConstraints({ maximum: v === "" ? null : parseInt(v, 10) });
                  }}
                  className="glass-input w-20 rounded px-2 py-1"
                  aria-invalid={hasMaxError ? "true" : undefined}
                  data-testid={`max-input-${field._clientId}`}
                />
              </label>
            </div>
            <InlineErrors
              issues={issues}
              fieldPath={`${pathPrefix}.minimum`}
              onFix={onFixit}
            />
          </>
        ) : field.type === "string" &&
          field.field_name !== RATIONALE_NOTE_FIELD_NAME ? (
          <div>
            <button
              type="button"
              onClick={() => setAdvancedOpen(!advancedOpen)}
              className="flex items-center gap-1 text-xs transition-colors hover:underline"
              style={{ color: "var(--text-muted)" }}
              data-testid={`advanced-expander-${field._clientId}`}
            >
              {advancedOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              Advanced constraints
            </button>
            {advancedOpen && (
              <>
                <div className="mt-1.5 flex items-center gap-3 text-xs">
                  {showMarkers && <MarkerIcon tooltip={markerTooltip} />}
                  <label
                    className="flex items-center gap-1"
                    style={{ color: "var(--text-muted)" }}
                  >
                    <Text kind="label/regular/xs">minLength:</Text>
                    <input
                      type="number"
                      min={0}
                      value={field.min_length ?? ""}
                      onChange={(e) => {
                        const v = e.target.value;
                        onChangeConstraints({
                          min_length: v === "" ? null : parseInt(v, 10),
                        });
                      }}
                      className="glass-input w-20 rounded px-2 py-1"
                      aria-invalid={hasMinLenError ? "true" : undefined}
                      data-testid={`minlength-input-${field._clientId}`}
                    />
                  </label>
                  <label
                    className="flex items-center gap-1"
                    style={{ color: "var(--text-muted)" }}
                  >
                    <Text kind="label/regular/xs">maxLength:</Text>
                    <input
                      type="number"
                      min={0}
                      value={field.max_length ?? ""}
                      onChange={(e) => {
                        const v = e.target.value;
                        onChangeConstraints({
                          max_length: v === "" ? null : parseInt(v, 10),
                        });
                      }}
                      className="glass-input w-20 rounded px-2 py-1"
                      aria-invalid={hasMaxLenError ? "true" : undefined}
                      data-testid={`maxlength-input-${field._clientId}`}
                    />
                  </label>
                </div>
                <InlineErrors
                  issues={issues}
                  fieldPath={`${pathPrefix}.min_length`}
                  onFix={onFixit}
                />
              </>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
}

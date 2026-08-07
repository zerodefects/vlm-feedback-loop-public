// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared card components used by both Create and Edit Guidance pages.
 *
 * Eliminates ~80 lines of JSX duplication per page.
 */

import { useState } from "react";
import { Switch, Text } from "@kui/react";
import { Plus, ChevronUp, ChevronDown } from "lucide-react";
import type { SchemaIssueResponse, DraftValidationResponse } from "@/types/guidance";
import type { ClientField } from "./field-utils";
import { generateExampleOutput } from "./field-utils";
import { FieldRow } from "./FieldRow";
import { ErrorRow } from "./InlineErrors";
import { MarkerIcon } from "./MarkerIcon";
import type { FieldType, FieldRole, SchemaFieldInput } from "@/types/guidance";

// ── Types ───────────────────────────────────────────────────────────────────

interface FieldHandlers {
  onRename: (clientId: string, name: string) => void;
  onChangeType: (clientId: string, type: FieldType) => void;
  onChangeConstraints: (clientId: string, patch: Partial<SchemaFieldInput>) => void;
  onDelete: (clientId: string) => void;
  onMoveSection: (clientId: string) => void;
  onReorder: (clientId: string, direction: "up" | "down") => void;
  onFixit: (clientId: string, code: string) => void;
  onAddField: (role: FieldRole) => void;
}

// ── Description Card ────────────────────────────────────────────────────────

interface DescriptionCardProps {
  description: string;
  onChange: (value: string) => void;
  descriptionRef?: React.RefObject<HTMLTextAreaElement | null>;
}

export function DescriptionCard({
  description,
  onChange,
  descriptionRef,
}: DescriptionCardProps) {
  return (
    <div className="glass-card p-6">
      <div className="flex items-center gap-2" style={{ marginBottom: 4 }}>
        <Text
          kind="label/bold/md"
          style={{ color: "var(--text-primary)" }}
          tabIndex={-1}
        >
          Task Description
        </Text>
        <Text kind="label/regular/xs" className="glass-pill">
          Optional
        </Text>
      </div>
      <Text
        kind="body/regular/sm"
        style={{ color: "var(--text-muted)", display: "block", marginBottom: 12 }}
      >
        Describe the task and what the model should focus on in each image.
      </Text>
      <textarea
        ref={descriptionRef}
        aria-label="Task description"
        value={description}
        onChange={(e) => onChange(e.target.value)}
        placeholder="You are inspecting manufactured parts for surface damage. Focus on cracks, discoloration, and deformation. Minor cosmetic wear is acceptable..."
        rows={4}
        className="glass-input w-full resize-y rounded-lg px-3 py-2 text-sm"
        style={{ minHeight: 100 }}
        data-testid="description-textarea"
      />
      <div className="mt-1 text-right" data-testid="char-counter">
        <Text kind="label/regular/xs" style={{ color: "var(--text-faint)" }}>
          {description.length} characters
        </Text>
      </div>
    </div>
  );
}

// ── Schema Card ─────────────────────────────────────────────────────────────

interface SchemaCardProps {
  coreFields: ClientField[];
  auxFields: ClientField[];
  allIssues: SchemaIssueResponse[];
  /** Backend field_path prefix per client id ("core[0]", "aux[1]", …) —
   *  computed by useGuidanceForm from the submitted field order. */
  pathPrefixes: ReadonlyMap<string, string>;
  handlers: FieldHandlers;
  rationaleEnabled: boolean;
  onRationaleEnabledChange: (enabled: boolean) => void;
  showMarkers?: boolean;
  markerTooltip?: string;
  /** Slot for banner content between explainer strip and fields. */
  bannerSlot?: React.ReactNode;
  /** Slot for page-specific content after core fields. */
  coreExtraSlot?: React.ReactNode;
  /** Slot for backend errors above field sections. */
  backendErrorSlot?: React.ReactNode;
}

export function SchemaCard({
  coreFields,
  auxFields,
  allIssues,
  pathPrefixes,
  handlers,
  rationaleEnabled,
  onRationaleEnabledChange,
  showMarkers,
  markerTooltip,
  bannerSlot,
  coreExtraSlot,
  backendErrorSlot,
}: SchemaCardProps) {
  function renderFieldRows(roleFields: ClientField[]) {
    return roleFields.map((f, idx) => (
      <FieldRow
        key={f._clientId}
        field={f}
        isFirst={idx === 0}
        isLast={idx === roleFields.length - 1}
        issues={allIssues}
        pathPrefix={pathPrefixes.get(f._clientId) ?? ""}
        showMarkers={showMarkers}
        markerTooltip={markerTooltip}
        onRename={(name) => handlers.onRename(f._clientId, name)}
        onChangeType={(type) => handlers.onChangeType(f._clientId, type)}
        onChangeConstraints={(patch) =>
          handlers.onChangeConstraints(f._clientId, patch)
        }
        onDelete={() => handlers.onDelete(f._clientId)}
        onMoveSection={() => handlers.onMoveSection(f._clientId)}
        onReorder={(dir) => handlers.onReorder(f._clientId, dir)}
        onFixit={(code) => handlers.onFixit(f._clientId, code)}
      />
    ));
  }

  return (
    <div className="glass-card p-6">
      <Text
        kind="label/bold/md"
        style={{ color: "var(--text-primary)", display: "block", marginBottom: 12 }}
        tabIndex={-1}
      >
        Schema
      </Text>

      {bannerSlot}
      {backendErrorSlot}

      {/* Core Fields */}
      <div className="mb-4 pl-4 border-accent-core">
        <div className="mb-2 flex items-center gap-2">
          <Text
            kind="label/bold/xs"
            style={{ color: "var(--text-primary)" }}
            className="section-eyebrow"
            tabIndex={-1}
          >
            Core Fields
          </Text>
          <Text kind="label/regular/xs" className="glass-pill green">
            Required
          </Text>
        </div>
        <div className="space-y-1.5">
          {coreFields.length > 0 ? (
            renderFieldRows(coreFields)
          ) : (
            <Text
              kind="body/regular/sm"
              style={{
                color: "var(--text-faint)",
                display: "block",
                padding: "8px 12px",
                fontStyle: "italic",
              }}
            >
              No core fields defined.
            </Text>
          )}
        </div>
        {allIssues.some((i) => i.code === "NO_CORE_FIELDS") && (
          <ErrorRow
            className="mt-1 px-3"
            message="Add at least one Core field (required for evaluation)."
            testId="error-NO_CORE_FIELDS"
          />
        )}
        {coreExtraSlot}
        <button
          type="button"
          onClick={() => handlers.onAddField("core")}
          className="glass-btn green mt-2"
          data-testid="add-core-field-btn"
        >
          {showMarkers && <MarkerIcon tooltip={markerTooltip} />} <Plus size={14} />{" "}
          <Text kind="body/regular/sm" style={{ color: "inherit" }}>
            Add Core Field
          </Text>
        </button>
      </div>

      {/* Aux Fields */}
      <div className="pl-4 border-accent-aux">
        <div className="mb-2 flex items-center gap-2">
          <Text
            kind="label/bold/xs"
            style={{ color: "var(--text-primary)" }}
            className="section-eyebrow"
            tabIndex={-1}
          >
            Aux Fields
          </Text>
          <Text kind="label/regular/xs" className="glass-pill">
            Optional
          </Text>
        </div>
        <Text
          kind="body/regular/sm"
          style={{ color: "var(--text-muted)", display: "block", marginBottom: 8 }}
        >
          Have the model describe what it sees (material type, object count) before
          deciding Core values.
        </Text>
        <div
          className="glass-info mb-3 flex items-center justify-between gap-4 px-4 py-3"
          data-testid="rationale-note-setting"
        >
          <div>
            <Text
              kind="label/semibold/sm"
              style={{ color: "var(--text-primary)", display: "block" }}
            >
              Rationale notes
            </Text>
            <Text
              kind="body/regular/sm"
              style={{ color: "var(--text-muted)", display: "block", marginTop: 2 }}
            >
              Optional. When enabled, the Teacher gives a brief explanation of visible
              evidence and the SME reviews the note after editing a label.
            </Text>
          </div>
          <Switch
            checked={rationaleEnabled}
            onCheckedChange={onRationaleEnabledChange}
            size="small"
            slotLabel={
              <Text kind="label/regular/xs" className="sr-only">
                Enable rationale notes
              </Text>
            }
            data-testid="rationale-note-toggle"
          />
        </div>
        <div className="space-y-1.5">{renderFieldRows(auxFields)}</div>
        <button
          type="button"
          onClick={() => handlers.onAddField("aux")}
          className="glass-btn muted mt-2"
          data-testid="add-aux-field-btn"
        >
          <Plus size={14} />{" "}
          <Text kind="body/regular/sm" style={{ color: "inherit" }}>
            Add Aux Field
          </Text>
        </button>
      </div>
    </div>
  );
}

// ── Rules Card ──────────────────────────────────────────────────────────────

interface RulesCardProps {
  rules: string;
  onChange: (value: string) => void;
}

export function RulesCard({ rules, onChange }: RulesCardProps) {
  return (
    <div className="glass-card p-6">
      <div className="flex items-center gap-2" style={{ marginBottom: 4 }}>
        <Text
          kind="label/bold/md"
          style={{ color: "var(--text-primary)" }}
          tabIndex={-1}
        >
          Rules & Edge Cases
        </Text>
        <Text kind="label/regular/xs" className="glass-pill">
          Optional
        </Text>
      </div>
      <Text
        kind="body/regular/sm"
        style={{ color: "var(--text-muted)", display: "block", marginBottom: 12 }}
      >
        How should the model handle ambiguous or tricky images? Edge cases tend to
        surface during labeling, so you can always add Rules later.
      </Text>
      <textarea
        aria-label="Rules and edge cases"
        value={rules}
        onChange={(e) => onChange(e.target.value)}
        placeholder="If damage is partially obscured, classify based on the visible portion only..."
        rows={4}
        className="glass-input w-full resize-y rounded-lg px-3 py-2 text-sm"
        style={{ minHeight: 100 }}
        data-testid="rules-textarea"
      />
    </div>
  );
}

// ── Previews ────────────────────────────────────────────────────────────────

interface PreviewsProps {
  fields: ClientField[];
  backendValidation: DraftValidationResponse | null;
  errorCount: number;
}

export function Previews({ fields, backendValidation, errorCount }: PreviewsProps) {
  const [jsonSchemaOpen, setJsonSchemaOpen] = useState(false);
  const [exampleOutputOpen, setExampleOutputOpen] = useState(false);

  return (
    <div className="glass-info p-4 space-y-2">
      <div>
        <button
          type="button"
          onClick={() => setJsonSchemaOpen(!jsonSchemaOpen)}
          className="flex items-center gap-2 transition-colors hover:underline"
          style={{ color: "var(--text-muted)" }}
          data-testid="json-schema-toggle"
        >
          {jsonSchemaOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}{" "}
          <Text kind="body/regular/sm" style={{ color: "inherit" }}>
            Derived JSON Schema
          </Text>
        </button>
        {jsonSchemaOpen && (
          <Text asChild kind="mono/sm">
            <pre
              className="mt-2 overflow-x-auto rounded-lg p-4 text-xs glass-terminal"
              data-testid="json-schema-preview"
            >
              {backendValidation?.derived_json_schema
                ? JSON.stringify(backendValidation.derived_json_schema, null, 2)
                : errorCount > 0
                  ? "Fix errors above to see the derived schema."
                  : "Loading..."}
            </pre>
          </Text>
        )}
      </div>
      <div>
        <button
          type="button"
          onClick={() => setExampleOutputOpen(!exampleOutputOpen)}
          className="flex items-center gap-2 transition-colors hover:underline"
          style={{ color: "var(--text-muted)" }}
          data-testid="example-output-toggle"
        >
          {exampleOutputOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}{" "}
          <Text kind="body/regular/sm" style={{ color: "inherit" }}>
            Example label output
          </Text>
        </button>
        {exampleOutputOpen && (
          <div className="mt-2">
            <Text
              kind="label/regular/xs"
              style={{ color: "var(--text-faint)", display: "block", marginBottom: 4 }}
            >
              Is this what I want the model to produce?
            </Text>
            <Text asChild kind="mono/sm">
              <pre
                className="overflow-x-auto rounded-lg p-4 text-xs glass-terminal"
                data-testid="example-output-preview"
              >
                {JSON.stringify(generateExampleOutput(fields), null, 2)}
              </pre>
            </Text>
          </div>
        )}
      </div>
    </div>
  );
}

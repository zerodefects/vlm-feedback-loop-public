// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared hook for Guidance form state, field mutations, and validation.
 *
 * Used by both CreateGuidancePage and EditGuidancePage so the handler and
 * validation logic lives in one place.
 *
 * Validation is backend-driven: the `guidance:validate_draft` endpoint
 * (services/schema_core.py) is the ONLY validator. Every draft change
 * triggers a debounced re-validation; a save attempt triggers an immediate
 * one. The hook never re-implements validation rules in TypeScript.
 */

import { useState, useCallback, useEffect, useRef } from "react";
import type {
  SchemaFieldInput,
  FieldType,
  FieldRole,
  DraftValidationResponse,
  SchemaIssueResponse,
} from "@/types/guidance";
import { validateDraft } from "@/api/guidance";
import { RATIONALE_NOTE_FIELD_NAME } from "@/lib/guidance-templates";
import {
  type ClientField,
  stripClientIds,
  recalcDisplayOrders,
  computeFieldPathPrefixes,
} from "./field-utils";

/** Debounce for backend draft validation after edits. */
const VALIDATE_DEBOUNCE_MS = 400;

export interface UseGuidanceFormOptions {
  projectId: string;
  initialDescription: string;
  initialFields: ClientField[];
  initialRules: string;
  /** Called on every edit — used by Create page to track hasUserEdited. */
  onEdited?: () => void;
}

export function useGuidanceForm({
  projectId,
  initialDescription,
  initialFields,
  initialRules,
  onEdited,
}: UseGuidanceFormOptions) {
  const [description, setDescription] = useState(initialDescription);
  const [fields, setFields] = useState<ClientField[]>(initialFields);
  const [rules, setRules] = useState(initialRules);
  const [backendValidation, setBackendValidation] =
    useState<DraftValidationResponse | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  // Inline field-level errors are hidden until the SME
  // clicks Save at least once. The header status badge stays live.
  const [saveAttempted, setSaveAttempted] = useState(false);
  // Latest-wins guard for in-flight validation responses.
  const validationSeq = useRef(0);
  // A passive debounced validation must never supersede the explicit
  // validation started by Save. Without this guard, selecting a template
  // and immediately clicking Save can make the still-scheduled debounce
  // increment validationSeq after the Save request starts; the Save result
  // is then discarded as stale and the button appears to do nothing.
  const saveValidationInFlight = useRef(false);

  // ── Derived ───────────────────────────────────────────────────────────
  const rationaleEnabled = fields.some(
    (f) => f.field_name === RATIONALE_NOTE_FIELD_NAME,
  );
  const issues: SchemaIssueResponse[] = backendValidation?.issues ?? [];
  const totalErrorCount = issues.filter((i) => i.severity === "error").length;
  // Inline-error gating: cards/rows render from this, not from the raw array.
  const displayIssues = saveAttempted ? issues : [];
  // Row addressing for backend field_path values ("core[0].name", ...).
  const pathPrefixes = computeFieldPathPrefixes(fields);
  const coreFields = fields
    .filter((f) => f.role === "core")
    .sort((a, b) => a.display_order - b.display_order);
  const auxFields = fields
    .filter((f) => f.role === "aux" && f.field_name !== RATIONALE_NOTE_FIELD_NAME)
    .sort((a, b) => a.display_order - b.display_order);

  // ── Helpers ───────────────────────────────────────────────────────────
  function markEdited() {
    onEdited?.();
  }
  function markSaveAttempted() {
    setSaveAttempted(true);
  }

  // ── Backend validation (debounced after edits, immediate on save) ─────
  /**
   * Validate the current draft against the backend immediately and store the
   * response. Returns true when the draft may be saved. Latest-wins: a
   * response is discarded when a newer validation started after it.
   */
  const validateNow = useCallback(
    async (mode: "save" | "background" = "save"): Promise<boolean> => {
      if (mode === "background" && saveValidationInFlight.current) return false;
      if (mode === "save") saveValidationInFlight.current = true;
      const seq = ++validationSeq.current;
      try {
        const result = await validateDraft(projectId, {
          description,
          schema: stripClientIds(fields),
          rules,
        });
        if (seq !== validationSeq.current) return false; // stale — superseded
        setBackendValidation(result);
        setBackendError(null);
        return result.save_allowed;
      } catch {
        if (seq === validationSeq.current) {
          setBackendError("Backend validation failed.");
        }
        return false;
      } finally {
        if (mode === "save") saveValidationInFlight.current = false;
      }
    },
    [projectId, description, fields, rules],
  );

  // Any draft change (typing, field mutations, template apply, Edit-page
  // populate) recreates validateNow and re-arms the debounce timer, so the
  // backend is consulted once per pause in editing — including on mount.
  useEffect(() => {
    const timer = setTimeout(() => {
      void validateNow("background");
    }, VALIDATE_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [validateNow]);

  // ── Text handlers ─────────────────────────────────────────────────────
  function handleDescriptionChange(value: string) {
    setDescription(value);
    markEdited();
  }
  function handleRulesChange(value: string) {
    setRules(value);
    markEdited();
  }

  // ── Fix-it handlers ───────────────────────────────────────────────────
  function handleFixit(clientId: string, code: string) {
    markEdited();
    setFields((prev) =>
      prev.map((f) => {
        if (f._clientId !== clientId) return f;
        if (code === "ENUM_DUPLICATE_VALUE")
          return {
            ...f,
            allowed_values: [...new Set((f.allowed_values ?? []).map((v) => v.trim()))],
          };
        if (code === "MIN_EXCEEDS_MAX")
          return { ...f, minimum: f.maximum, maximum: f.minimum };
        return f;
      }),
    );
  }

  // ── Field mutation handlers ───────────────────────────────────────────
  function handleAddField(role: FieldRole) {
    markEdited();
    setFields((prev) => {
      // Append AFTER the section's existing fields: with display_order 0 the
      // stable sort in recalcDisplayOrders would slot the new row at position
      // 1 (after the first 0-ordered field) instead of last.
      const maxOrder = Math.max(
        -1,
        ...prev.filter((f) => f.role === role).map((f) => f.display_order),
      );
      return recalcDisplayOrders([
        ...prev,
        {
          _clientId: crypto.randomUUID(),
          field_name: "",
          type: "string",
          role,
          display_order: maxOrder + 1,
        },
      ]);
    });
  }
  function handleRationaleEnabledChange(enabled: boolean) {
    markEdited();
    setFields((prev) => {
      const withoutRationale = prev.filter(
        (f) => f.field_name !== RATIONALE_NOTE_FIELD_NAME,
      );
      if (!enabled) return recalcDisplayOrders(withoutRationale);
      const minAuxOrder = Math.min(
        0,
        ...withoutRationale.filter((f) => f.role === "aux").map((f) => f.display_order),
      );
      return recalcDisplayOrders([
        ...withoutRationale,
        {
          _clientId: crypto.randomUUID(),
          field_name: RATIONALE_NOTE_FIELD_NAME,
          type: "string",
          role: "aux",
          display_order: minAuxOrder - 1,
        },
      ]);
    });
  }
  function handleDeleteField(clientId: string) {
    markEdited();
    setFields((prev) => {
      const t = prev.find((f) => f._clientId === clientId);
      if (!t || t.field_name === RATIONALE_NOTE_FIELD_NAME) return prev;
      return recalcDisplayOrders(prev.filter((f) => f._clientId !== clientId));
    });
  }
  function handleRenameField(clientId: string, newName: string) {
    markEdited();
    setFields((prev) =>
      prev.map((f) => (f._clientId === clientId ? { ...f, field_name: newName } : f)),
    );
  }
  function handleChangeType(clientId: string, newType: FieldType) {
    markEdited();
    setFields((prev) =>
      prev.map((f) =>
        f._clientId !== clientId
          ? f
          : {
              ...f,
              type: newType,
              allowed_values: newType === "enum" || newType === "enum_set" ? [] : null,
              minimum: null,
              maximum: null,
              min_length: null,
              max_length: null,
            },
      ),
    );
  }
  function handleChangeConstraints(clientId: string, patch: Partial<SchemaFieldInput>) {
    markEdited();
    setFields((prev) =>
      prev.map((f) => (f._clientId === clientId ? { ...f, ...patch } : f)),
    );
  }
  function handleMoveSection(clientId: string) {
    markEdited();
    setFields((prev) => {
      const t = prev.find((f) => f._clientId === clientId);
      if (!t || t.field_name === RATIONALE_NOTE_FIELD_NAME) return prev;
      return recalcDisplayOrders(
        prev.map((f) =>
          f._clientId === clientId
            ? { ...f, role: (f.role === "core" ? "aux" : "core") as FieldRole }
            : f,
        ),
      );
    });
  }
  function handleReorder(clientId: string, direction: "up" | "down") {
    markEdited();
    setFields((prev) => {
      const target = prev.find((f) => f._clientId === clientId);
      if (!target || target.field_name === RATIONALE_NOTE_FIELD_NAME) return prev;
      const group = prev
        .filter((f) => f.role === target.role)
        .sort((a, b) => a.display_order - b.display_order);
      const idx = group.findIndex((f) => f._clientId === clientId);
      const swapIdx = direction === "up" ? idx - 1 : idx + 1;
      if (
        swapIdx < 0 ||
        swapIdx >= group.length ||
        group[swapIdx].field_name === RATIONALE_NOTE_FIELD_NAME
      )
        return prev;
      return recalcDisplayOrders(
        prev.map((f) => {
          if (f._clientId === group[idx]._clientId)
            return { ...f, display_order: group[swapIdx].display_order };
          if (f._clientId === group[swapIdx]._clientId)
            return { ...f, display_order: group[idx].display_order };
          return f;
        }),
      );
    });
  }

  return {
    // State
    description,
    setDescription,
    fields,
    setFields,
    rules,
    setRules,
    backendValidation,
    backendError,

    // Derived (all issue data originates from the backend response).
    issues,
    totalErrorCount,
    // Inline-error-gated view (empty until saveAttempted). Pass this to the
    // field-row / card components; keep `issues` for summary surfaces
    // (status badge, aria-live region, compile-failure banner).
    displayIssues,
    pathPrefixes,
    saveAttempted,
    coreFields,
    auxFields,
    rationaleEnabled,

    // Handlers
    markEdited,
    markSaveAttempted,
    validateNow,
    handleDescriptionChange,
    handleRulesChange,
    handleFixit,
    handleAddField,
    handleRationaleEnabledChange,
    handleDeleteField,
    handleRenameField,
    handleChangeType,
    handleChangeConstraints,
    handleMoveSection,
    handleReorder,
  };
}

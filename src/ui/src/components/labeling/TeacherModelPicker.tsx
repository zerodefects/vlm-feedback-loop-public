// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Top-bar Teacher model picker.
 *
 * Persistent dropdown at the left edge of the labeling top bar. Populated
 * from catalog entries with ``teacher`` role. Selecting a different model
 * updates ``project.teacher_model_config_id`` via PATCH and takes effect
 * on the NEXT proposal — the current proposal is not re-run (the SME can
 * still Retry explicitly for a per-attempt override).
 *
 * Backend enforces the invariant that a teacher selection has the
 * ``teacher`` role and ``supports_image_input=true``; this component's
 * options list is also filtered to that role + to entries whose
 * ``availability.available`` is true so models that would error out
 * against the bound endpoint never appear.
 */

import { Text } from "@kui/react";
import { Link } from "react-router-dom";

import type { ModelConfigResponse } from "@/types/nim";

interface TeacherModelPickerProps {
  /** Catalog entries already filtered to `teacher` role. Per-entry availability
   *  is applied inside the component so unreachable models (e.g. NVCF-gated
   *  Cosmos on hosted, or local NIM with no GPU) are hidden. */
  teacherConfigs: ModelConfigResponse[];
  /** Currently active teacher — comes from `project.teacher_model_config_id`. */
  currentTeacherId: string | null;
  /** Called with the new model_config_id when the SME picks a different row. */
  onChange: (modelConfigId: string) => void;
  /** Prevent changes while another header setting or label action is saving. */
  disabled?: boolean;
  /** Project ID — used to build the empty-state CTA link to NIM Connection. */
  projectId: string;
}

export function TeacherModelPicker({
  teacherConfigs,
  currentTeacherId,
  onChange,
  disabled = false,
  projectId,
}: TeacherModelPickerProps) {
  // Production responses always carry ``availability`` (backend default
  // covers it); ``?? true`` is for test fixtures and any older mock that
  // doesn't include the field — treat as available rather than crash on
  // an undefined property access.
  const availableConfigs = teacherConfigs.filter(
    (mc) => mc.availability?.available ?? true,
  );
  const currentConfig = teacherConfigs.find(
    (mc) => mc.model_config_id === currentTeacherId,
  );
  const currentIsUnavailable =
    currentTeacherId !== null &&
    !availableConfigs.some((mc) => mc.model_config_id === currentTeacherId);

  // No models at all — likely the catalog hasn't loaded yet. Render a stable
  // top-bar slot without a dropdown so layout doesn't jump during hydration.
  if (teacherConfigs.length === 0) {
    return (
      <Text
        kind="label/bold/sm"
        style={{ color: "var(--text-secondary)" }}
        data-testid="teacher-model-picker-empty"
      >
        Teacher
      </Text>
    );
  }

  // Catalog loaded but every entry is unreachable from the current env /
  // endpoint state. Surface the recovery path inline rather than showing
  // an empty <select>.
  if (availableConfigs.length === 0 && currentTeacherId === null) {
    return (
      <div
        className="flex items-center gap-2"
        data-testid="teacher-model-picker-unavailable"
      >
        <Text kind="label/regular/sm" style={{ color: "var(--text-muted)" }}>
          Teacher
        </Text>
        <Text kind="label/regular/sm" style={{ color: "var(--text-secondary)" }}>
          No models available —{" "}
          <Link
            to={`/projects/${projectId}/settings/nim`}
            style={{ color: "var(--accent-green)", textDecoration: "underline" }}
            data-testid="teacher-model-picker-configure-link"
          >
            configure endpoint
          </Link>
        </Text>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2" data-testid="teacher-model-picker">
      <Text kind="label/regular/sm" style={{ color: "var(--text-muted)" }}>
        Teacher
      </Text>
      <select
        className="glass-input px-3 py-1.5 text-sm"
        style={{
          minWidth: 260,
          opacity: disabled ? 0.5 : 1,
          cursor: disabled ? "not-allowed" : undefined,
        }}
        value={currentTeacherId ?? ""}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        data-testid="teacher-model-picker-select"
        aria-label="Teacher model"
      >
        {/* Sentinel option only rendered when no teacher is currently
            assigned — keeps the select controlled without a React warning. */}
        {currentTeacherId === null && (
          <option value="" disabled>
            Select Teacher
          </option>
        )}
        {/* Keep the backend-authoritative selection visible even when its
            endpoint is currently unhealthy or the catalog entry was removed.
            Omitting the selected value makes a native <select> visually fall
            back to its first available option, which falsely names a
            different Teacher without changing project state. */}
        {currentIsUnavailable && (
          <option value={currentTeacherId} disabled>
            {currentConfig?.model_name ?? "Selected Teacher"} (unavailable)
          </option>
        )}
        {availableConfigs.map((mc) => (
          <option key={mc.model_config_id} value={mc.model_config_id}>
            {mc.model_name}
          </option>
        ))}
      </select>
      {currentIsUnavailable && (
        <Link
          to={`/projects/${projectId}/settings/nim`}
          style={{ color: "var(--accent-green)", textDecoration: "underline" }}
          data-testid="teacher-model-picker-configure-link"
        >
          <Text kind="label/regular/xs" style={{ color: "inherit" }}>
            Configure endpoint
          </Text>
        </Link>
      )}
    </div>
  );
}

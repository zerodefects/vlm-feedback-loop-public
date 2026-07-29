// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Wires the schema-builder callbacks exposed by the shared form hook
 * (useGuidanceForm) into the `handlers` shape SchemaCard expects. The
 * wiring is identical on the Create and Edit Guidance pages; keeping it
 * here means the two cannot drift.
 */

import type { useGuidanceForm } from "./useGuidanceForm";

/** Return value of the shared Guidance form hook. */
export type GuidanceForm = ReturnType<typeof useGuidanceForm>;

export function makeFieldHandlers(form: GuidanceForm) {
  return {
    onRename: form.handleRenameField,
    onChangeType: form.handleChangeType,
    onChangeConstraints: form.handleChangeConstraints,
    onDelete: form.handleDeleteField,
    onMoveSection: form.handleMoveSection,
    onReorder: form.handleReorder,
    onFixit: form.handleFixit,
    onAddField: form.handleAddField,
  };
}

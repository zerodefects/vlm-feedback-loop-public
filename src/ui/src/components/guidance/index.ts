// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Barrel export for shared Guidance field-builder components.
 *
 * One component per file, following RAG Blueprint convention.
 */

// Utilities (pure functions, no React)
export {
  stampClientIds,
  stripClientIds,
  responseFieldsToInput,
  recalcDisplayOrders,
  describeSemanticChanges,
} from "./field-utils";

// Components (one per file)
export { MarkerIcon } from "./MarkerIcon";
export { ErrorRow } from "./InlineErrors";
export { ValidationNotices } from "./ValidationNotices";

// Shared card components
export { DescriptionCard, SchemaCard, RulesCard, Previews } from "./GuidanceCards";

// Shared form hook
export { useGuidanceForm } from "./useGuidanceForm";

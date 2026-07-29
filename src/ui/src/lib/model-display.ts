// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Display-label helpers for model identifiers.
 *
 * The backend stores raw provider-prefixed model IDs like
 * ``nvidia/cosmos-reason2-8b`` so downstream systems get unambiguous
 * names; the UI renders human-readable forms. This module is the single
 * normalizer — do not add per-page "pretty model name" helpers.
 *
 * It also owns the Teacher-capability control-visibility rules
 * ({@link thinkingToggleVisible}, {@link visualBudgetVisible}) so every
 * screen that renders the Thinking / Visual Budget controls hides them
 * under the same conditions.
 */

/** Minimal capability shape the visibility helpers need. */
interface CapabilityFields {
  thinking_toggle_mode: string;
  thinking_toggle_support: string;
  visual_budget_support: string;
}

/**
 * Whether the Thinking ON/OFF control should render for a model config.
 *
 * Hidden unless the model declares a request-based toggle and its capability
 * probe has confirmed that the endpoint accepts it. Unknown/unsupported
 * capability, mode ``"none"``, mode ``"always_on_reasoning"``, and an
 * undefined config all hide the control.
 */
export function thinkingToggleVisible(mc: CapabilityFields | undefined): boolean {
  if (!mc) return false;
  return (
    mc.thinking_toggle_support === "supported" &&
    mc.thinking_toggle_mode !== "none" &&
    mc.thinking_toggle_mode !== "always_on_reasoning"
  );
}

/**
 * Whether the Visual Budget control should render for a model config.
 * Hidden when the model ignores the setting (``"unsupported"``) or the
 * config is undefined (lookup miss).
 */
export function visualBudgetVisible(mc: CapabilityFields | undefined): boolean {
  if (!mc) return false;
  return mc.visual_budget_support !== "unsupported";
}

/** Casing style for {@link formatModelDisplayName}. */
export type ModelNameCasing = "title" | "upper";

/**
 * Convert a raw model identifier to a human-readable display name.
 *
 * Rules:
 *  - Strip the leading ``<org>/`` prefix (``nvidia/``, ``meta/``, etc.);
 *    everything after the first slash is kept intact.
 *  - Replace hyphens with spaces so individual tokens read as words.
 *  - ``"title"`` (default): title-case each token, uppercasing size
 *    suffixes like ``8b`` → ``8B`` (training/compare card style).
 *  - ``"upper"``: uppercase the whole string (job-monitor chain
 *    section-heading style).
 *  - Null / undefined / empty → ``—``.
 *
 * Examples:
 *  - ``nvidia/cosmos-reason2-8b`` → ``Cosmos Reason2 8B`` (title)
 *  - ``nvidia/cosmos-reason2-8b`` → ``COSMOS REASON2 8B`` (upper)
 *  - ``meta/llama3-70b`` → ``Llama3 70B`` (title)
 *  - ``mistralai/mistral-large-3`` → ``MISTRAL LARGE 3`` (upper)
 */
export function formatModelDisplayName(
  modelName: string | null | undefined,
  casing: ModelNameCasing = "title",
): string {
  if (!modelName) return "—";
  const bare = modelName.includes("/")
    ? modelName.split("/").slice(1).join("/")
    : modelName;
  return bare
    .split("-")
    .map((part) => {
      if (casing === "upper") return part.toUpperCase();
      // Uppercase size suffixes like "8b", "2b", "70b".
      if (/^\d+[a-z]$/i.test(part)) return part.toUpperCase();
      // Title-case everything else, preserving embedded digits.
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(" ");
}

/**
 * Friendly name for the locally-deployable Teacher the backend recommends
 * (``environment.recommended_local_teacher_model_name``). Hardware gates
 * eligibility; the backend's measured quality policy decides the winner.
 * Shared by the FTUE setup chain so every screen names
 * the same model the backend actually picked.
 */
export function localTeacherDisplayName(modelName: string | null | undefined): string {
  if (modelName?.includes("nemotron-3-nano-omni")) {
    return "Nemotron 3 Nano Omni";
  }
  if (modelName?.includes("cosmos3-nano-reasoner")) return "Cosmos 3 Nano (Reasoner)";
  if (modelName?.includes("cosmos3-super-reasoner")) {
    return "Cosmos 3 Super (Reasoner)";
  }
  if (modelName?.endsWith("-8b")) return "Cosmos Reason2 8B";
  if (modelName?.endsWith("-2b")) return "Cosmos Reason2 2B";
  return modelName ? formatModelDisplayName(modelName) : "Local Teacher";
}

/**
 * Extract a short compact label (e.g. `8B`, `2B`) from a model identifier.
 * Used for tight-space chain progress lines like `8B: done  2B: 5 of 6`.
 *
 * Falls back to the full model name when no size suffix is present.
 */
export function shortBaseLabel(modelName: string | null | undefined): string {
  if (!modelName) return "—";
  const match = modelName.match(/(\d+\s*b)$/i);
  if (match) return match[1].replace(/\s+/g, "").toUpperCase();
  return modelName;
}

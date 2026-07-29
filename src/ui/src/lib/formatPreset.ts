// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Display-format helpers for preset enum keys.
 *
 * The backend stores Generation Controls / Visual Budget presets as
 * lowercase/snake-case enum keys (`precise`, `balanced`, `high_detail`).
 * The summary/confirmation screens (Batch
 * Pre-Run) show the canonical spaced Title Case name — e.g.
 * `high_detail` → "High Detail". (The labeling top bar abbreviates that
 * to "High" for space; these screens have room for the full name.) This
 * helper centralizes the transform so every screen renders consistently.
 * Separators (`_`, `-`, whitespace) collapse to a single space rather
 * than being preserved as literal underscores/hyphens.
 */

export function titleCasePreset(value: string | null | undefined): string {
  if (value == null || value === "") return "—";
  return value
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

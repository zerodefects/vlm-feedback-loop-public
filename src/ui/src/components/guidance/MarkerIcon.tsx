// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Label-invalidation marker (~) shown on semantic-change controls in Edit
 * Guidance.
 */

import { RefreshCw } from "lucide-react";

/** Two placements:
 *
 * - inline (default) — flows with text/flex content. `vertical-align:
 *   -0.125em` is the standard SVG-in-text hint: ignored when the span is a
 *   flex child (parent `items-center` aligns it), and centres the 12px glyph
 *   on x-height inside flowing Text prose (the Edit Guidance post-save
 *   banner).
 * - overlay — a smaller 10px glyph pinned to the top-right corner of a
 *   `position: relative` control wrapper, so the marker reads as an
 *   annotation on that control rather than an extra clickable button in the
 *   row.
 */
export function MarkerIcon({
  tooltip,
  overlay,
}: {
  tooltip?: string;
  overlay?: boolean;
}) {
  return (
    <span
      title={tooltip ?? "Changing this may invalidate existing labels."}
      className={
        overlay ? "absolute text-amber-400" : "inline-flex items-center text-amber-400"
      }
      style={overlay ? { top: -4, right: -4 } : { verticalAlign: "-0.125em" }}
      data-testid="invalidation-marker"
    >
      <RefreshCw size={overlay ? 10 : 12} />
    </span>
  );
}

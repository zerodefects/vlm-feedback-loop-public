// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ICL signal chip for the labeling top bar.
 *
 * Surfaces how many SME Edits are influencing the current proposal so the
 * SME has a live signal that the loop is firing. Reads
 * ``icl_example_keys_used`` from the latest ProposalResponse — which the
 * image-budget invariant ties to ``icl_images_attached_count`` (every
 * retained ICL example is image-grounded under inline ICL image
 * injection).
 *
 * Always rendered. No yellow/red state — informational only.
 */

import { Sparkles } from "lucide-react";
import { Text } from "@kui/react";

interface IclChipProps {
  /** Count of ICL example keys retained for the current proposal.
   * Pass ``null`` while the next proposal is selecting context. */
  count: number | null;
  /** True when there is no current or pending proposal, such as queue empty. */
  idle?: boolean;
}

export function IclChip({ count, idle = false }: IclChipProps) {
  if (idle) {
    return (
      <div className="flex items-center gap-1.5" data-testid="icl-chip-idle">
        <Sparkles size={12} style={{ color: "var(--text-muted)" }} />
        <Text kind="label/regular/sm" style={{ color: "var(--text-muted)" }}>
          ICL: no active proposal
        </Text>
      </div>
    );
  }

  if (count === null) {
    return (
      <div className="flex items-center gap-1.5" data-testid="icl-chip-pending">
        <Sparkles size={12} style={{ color: "var(--text-muted)" }} />
        <Text kind="label/regular/sm" style={{ color: "var(--text-muted)" }}>
          ICL: selecting context…
        </Text>
      </div>
    );
  }

  if (count <= 0) {
    // Cold-start state — muted to recede in the top bar.
    return (
      <div className="flex items-center gap-1.5" data-testid="icl-chip-coldstart">
        <Sparkles size={12} style={{ color: "var(--text-muted)" }} />
        <Text kind="label/regular/sm" style={{ color: "var(--text-muted)" }}>
          ICL: no edits yet
        </Text>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1.5" data-testid="icl-chip-active">
      <Sparkles size={12} style={{ color: "var(--accent-green)" }} />
      <Text kind="label/regular/sm" style={{ color: "var(--text-muted)" }}>
        ICL: {count} {count === 1 ? "edit" : "edits"} in context
      </Text>
    </div>
  );
}

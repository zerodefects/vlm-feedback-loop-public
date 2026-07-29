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
   * Pass ``proposal.icl_example_keys_used.length`` (or 0 when no proposal
   * is loaded yet). */
  count: number;
}

export function IclChip({ count }: IclChipProps) {
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

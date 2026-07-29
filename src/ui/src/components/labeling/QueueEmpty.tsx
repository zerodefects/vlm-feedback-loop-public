// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Queue-empty terminal state for the labeling screen.
 *
 * Shows when all images have been reviewed, with action buttons to
 * continue the workflow.
 */

import { Button, Spinner, Text } from "@kui/react";
import { CheckCircle2, Plus, RotateCcw, TrendingUp } from "lucide-react";

interface QueueEmptyProps {
  verified: number;
  unlabeled: number;
  omitted: number;
  onAddImages: () => void;
  onRestoreOmitted: () => void;
  restoringOmitted: boolean;
  onGoToScaleUp: () => void;
}

export function QueueEmpty({
  verified,
  unlabeled,
  omitted,
  onAddImages,
  onRestoreOmitted,
  restoringOmitted,
  onGoToScaleUp,
}: QueueEmptyProps) {
  return (
    <div
      className="flex flex-1 flex-col items-center justify-center gap-6"
      data-testid="queue-empty"
    >
      {/* Subdued icon chip above a bold heading — the Retail Blueprint
          empty-state ladder, so the terminal state reads as a designed
          surface rather than loose text in an empty card. */}
      <div
        style={{
          width: 48,
          height: 48,
          borderRadius: 14,
          background: "rgba(255, 255, 255, 0.06)",
          border: "1px solid var(--glass-border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <CheckCircle2 size={24} color="var(--text-muted)" />
      </div>
      <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
        All images have been reviewed.
      </Text>

      {/* Counts — one muted meta line; the status bar on the same screen
          already carries the full Verified/Unlabeled/Omitted pills. */}
      <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
        Verified {verified} &middot; Omitted {omitted} &middot; Unlabeled {unlabeled}
      </Text>

      {/* Helper text */}
      <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
        Add more images to continue labeling, or proceed to Scale-Up.
      </Text>

      {/* Action buttons — two rows, so the "continue labeling"
          actions (Add More / Restore) sit visually apart from the
          full-width "proceed downstream" action (Go to Scale-Up).
          A 2-column grid with a uniform per-cell min-width keeps the top
          row column-aligned regardless of label length, instead of letting
          it center its own intrinsic width (which produces a small visible
          row offset). Go to Scale-Up is the primary forward action. */}
      <div
        className="grid grid-cols-2 gap-3 [&>*]:w-full"
        style={{ gridAutoRows: "auto" }}
        data-testid="queue-actions"
      >
        {/* Inline styles rather than col-span/w-full utilities throughout:
            KUI base.css loads after the Tailwind layer, so its Button
            sizing beats utility classes on the button root. With no
            Restore button, Add More Images spans the row so the lone
            button doesn't sit off-center in the left cell. */}
        <div style={{ gridColumn: omitted > 0 ? "auto" : "1 / -1" }}>
          <Button
            kind="secondary"
            style={{ width: "100%" }}
            onClick={onAddImages}
            data-testid="queue-add-images-btn"
          >
            <Plus size={14} /> Add More Images
          </Button>
        </div>

        {omitted > 0 && (
          <Button
            kind="secondary"
            onClick={onRestoreOmitted}
            disabled={restoringOmitted}
            data-testid="restore-omitted-btn"
          >
            {restoringOmitted ? (
              <>
                <Spinner size="small" aria-label="Restoring" /> Restoring...
              </>
            ) : (
              <>
                <RotateCcw size={14} /> Restore {omitted} Omitted
              </>
            )}
          </Button>
        )}

        {/* Full-row primary CTA. */}
        <div style={{ gridColumn: "1 / -1" }}>
          <Button
            kind="primary"
            className="nvidia-green-button"
            style={{ width: "100%" }}
            onClick={onGoToScaleUp}
            data-testid="queue-scaleup-btn"
          >
            <TrendingUp size={14} /> Go to Scale-Up
          </Button>
        </div>
      </div>
    </div>
  );
}

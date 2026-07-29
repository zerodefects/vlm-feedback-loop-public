// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Ingestion progress display.
 */

import { Button, Text } from "@kui/react";
import { ArrowRight, Check } from "lucide-react";

import { ProgressBar } from "@/components/ProgressBar";
import { EmbeddingStatusLine } from "@/components/ingest/EmbeddingStatusLine";
import { IngestionCountRow } from "@/components/ingest/IngestionCountRow";

// Mid-ingest [Start labeling] threshold. Gating on bare
// ``accepted > 0`` would surface the affordance with
// only 1 accepted image — too sparse to actually label with. Five
// matches the EVAL_FIRST_POOL_SIZE precedent and ensures the SME has
// a real working set when they hop forward. Ingests with
// ``total < 5`` complete straight into the summary screen (which has
// its own [Start labeling]), so this threshold never strands an SME.
const START_LABELING_MIN_ACCEPTED = 5;

export interface FailureItem {
  name: string;
  reason: string;
}

interface IngestionProgressProps {
  sourcePath: string;
  total: number;
  processed: number;
  accepted: number;
  skippedItems: FailureItem[];
  errorItems: FailureItem[];
  /** Project's effective hosted or local embedding provider. */
  embeddingProvider: string;
  embeddingProviderSettling?: boolean;
  /**
   * If provided, a [Start labeling] CTA appears in the progress card
   * once ``START_LABELING_MIN_ACCEPTED`` images are accepted. The
   * backend's ``:ingest`` endpoint returns 202 with skeleton rows so
   * any accepted image is already available for labeling — the SME
   * doesn't need to wait for the whole meter to fill. Subsequent
   * batches continue to dispatch from the in-page loop even after the
   * user navigates away (the async loop's closure survives the page
   * unmount). Closing the browser tab mid-ingest does abandon any
   * batches not yet dispatched; a future Run-record-backed ingest
   * would survive that too, but is out of scope here.
   */
  onStartLabeling?: () => void;
}

export function IngestionProgress({
  sourcePath,
  total,
  processed,
  accepted,
  skippedItems,
  errorItems,
  embeddingProvider,
  embeddingProviderSettling = false,
  onStartLabeling,
}: IngestionProgressProps) {
  const pct = total > 0 ? Math.round((processed / total) * 100) : 0;

  return (
    <div
      className="glass-card glass-card--elevated p-6 flex flex-col"
      data-testid="ingestion-progress"
    >
      {/* title/md matches the page-level Ingest Images header so the
          ingesting state carries the same workflow-tier weight even though
          the page-level header is suppressed while ingestion is active. */}
      <Text
        kind="title/md"
        style={{ color: "var(--text-primary)", display: "block", marginBottom: 8 }}
      >
        Ingesting Images
      </Text>
      <Text
        kind="body/regular/sm"
        style={{ color: "var(--text-muted)", display: "block" }}
      >
        Source: {sourcePath}
      </Text>
      <Text
        kind="body/regular/xs"
        className="mt-1"
        style={{ color: "var(--text-muted)", display: "block" }}
      >
        Images are referenced in place, not copied.
      </Text>

      {/* Progress bar (shared primitive — see components/ProgressBar.tsx) */}
      <ProgressBar
        percent={pct}
        size="md"
        className="mt-4 mb-2"
        ariaLabel="Ingestion progress"
      />
      <Text
        kind="body/regular/sm"
        style={{ color: "var(--text-secondary)", display: "block" }}
      >
        {processed} of {total} &middot; {pct}%
      </Text>
      <div className="mt-1">
        <EmbeddingStatusLine
          provider={embeddingProvider}
          providerSettling={embeddingProviderSettling}
        />
      </div>

      {/* Running counts — three header rows (Accepted / Skipped / Errors)
          are always present so the top of the card stays stable. Detail
          rows for skipped and errored items are appended below the header
          row only when items arrive, which does add height as the batch
          progresses — the detail rows render inline below each count
          header. */}
      <div className="mt-4 flex flex-col gap-1">
        <IngestionCountRow label="Accepted" count={accepted} tone="primary" />
        <IngestionCountRow
          label="Skipped"
          count={skippedItems.length}
          items={skippedItems}
          tone="secondary"
        />
        <IngestionCountRow
          label="Errors"
          count={errorItems.length}
          items={errorItems}
          tone={errorItems.length > 0 ? "error" : "secondary"}
        />
      </div>

      {/* Ready-to-label zone. The progress bar above
          says "wait"; this primary CTA says "go". The framing here —
          top separator + check glyph + microcopy that names the live
          accepted count AND acknowledges the background continuation —
          collapses the apparent contradiction by making the dual-state
          card explicit. Without the framing the bare button reads like
          a half-built screen. */}
      {onStartLabeling !== undefined && accepted >= START_LABELING_MIN_ACCEPTED && (
        <div
          className="mt-6 pt-4 flex items-center justify-between gap-4"
          style={{ borderTop: "1px solid var(--glass-border-subtle)" }}
          data-testid="in-progress-ready-zone"
        >
          <div className="flex items-start gap-2 flex-1 min-w-0">
            <Check
              size={18}
              style={{
                color: "var(--accent-green)",
                marginTop: 2,
                flexShrink: 0,
              }}
              aria-hidden="true"
            />
            <div className="flex flex-col">
              <Text
                kind="body/regular/sm"
                style={{ color: "var(--text-primary)", display: "block" }}
              >
                {accepted} {accepted === 1 ? "image" : "images"} ready to label
              </Text>
              <Text
                kind="body/regular/xs"
                style={{ color: "var(--text-muted)", display: "block" }}
              >
                The rest will keep loading in the background.
              </Text>
            </div>
          </div>
          <Button
            kind="primary"
            className="nvidia-green-button"
            onClick={onStartLabeling}
            data-testid="in-progress-start-labeling"
          >
            Start labeling <ArrowRight size={14} />
          </Button>
        </div>
      )}
    </div>
  );
}

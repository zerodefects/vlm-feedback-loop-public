// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Ingestion completion summary.
 */

import { Button, Text } from "@kui/react";
import { ArrowRight } from "lucide-react";
import type { FailureItem } from "@/components/ingest/IngestionProgress";
import { EmbeddingStatusLine } from "@/components/ingest/EmbeddingStatusLine";
import { IngestionCountRow } from "@/components/ingest/IngestionCountRow";
import { InfoBanner } from "@/components/common/InfoBanner";

interface IngestionSummaryProps {
  totalProcessed: number;
  accepted: number;
  skippedItems: FailureItem[];
  errorItems: FailureItem[];
  warnings: string[];
  totalUnlabeled: number;
  /**
   * Project's effective embedding provider. Drives the dynamic
   * "what's actually running" line below the Unlabeled count so
   * no-keys SMEs aren't told embeddings are computing when they
   * aren't.
   */
  embeddingProvider: string;
  embeddingProviderSettling?: boolean;
  /** True when this batch came from the shipped 15-image RPS walkthrough. */
  isBundledSample?: boolean;
  onAddMore: () => void;
  onContinue: () => void;
}

export function IngestionSummary({
  totalProcessed,
  accepted,
  skippedItems,
  errorItems,
  warnings,
  totalUnlabeled,
  embeddingProvider,
  embeddingProviderSettling = false,
  isBundledSample = false,
  onAddMore,
  onContinue,
}: IngestionSummaryProps) {
  const showLowCountWarning = accepted < 150;

  return (
    <div
      className="glass-card glass-card--elevated p-6 flex flex-col"
      data-testid="ingestion-summary"
    >
      {/* title/md matches the page-level Ingest Images header so the
          completion state carries the same workflow-tier weight even
          though the page-level header is suppressed while the summary is
          active (see ImageIngestPage.tsx showPageHeader rule). */}
      <Text
        kind="title/md"
        style={{ color: "var(--text-primary)", display: "block", marginBottom: 8 }}
      >
        Ingestion Complete
      </Text>

      <Text
        kind="body/regular/sm"
        style={{ color: "var(--text-secondary)", display: "block" }}
      >
        {totalProcessed} images processed
      </Text>

      {/* Summary sub-block. All three count rows are always rendered so the block
          reads as a structured summary even when Skipped/Errors are 0;
          otherwise it collapses to a single-row strip and loses the
          multi-row rhythm Retail Catalog's nested panels establish. */}
      <div
        className="rounded-lg border p-4 mt-3 flex flex-col gap-1"
        style={{
          borderColor: "var(--glass-border-subtle)",
          backgroundColor: "var(--block-bg-elevated)",
        }}
      >
        <IngestionCountRow
          label="Accepted"
          count={accepted}
          tone="primary"
          suffix="images (now Unlabeled)"
        />
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

      {/* Size warnings — subsection sibling to the Accepted/Skipped/Errors
          block above. The label uses primary/strong typography so it reads
          as a distinct subsection header rather than blending into the
          secondary-weight count rows inside the bordered summary above. */}
      {warnings.length > 0 && (
        <div className="mt-4 flex flex-col">
          <Text
            kind="body/bold/sm"
            style={{ color: "var(--text-primary)", display: "block" }}
          >
            Warnings: {warnings.length}
          </Text>
          {warnings.map((w) => (
            <Text
              key={w}
              kind="body/regular/xs"
              className="ml-4"
              style={{ color: "var(--text-muted)", display: "block" }}
            >
              {w}
            </Text>
          ))}
        </div>
      )}

      {/* Low image count warning. The shared InfoBanner
          primitive supplies the .glass-info chrome so the banner reads as
          a distinct advisory affordance against the nested Accepted/
          Skipped/Errors summary block above — reusing that block's
          --block-bg-elevated here would blend the two together. */}
      {showLowCountWarning && (
        <InfoBanner
          tone="info"
          className="mt-4"
          body={
            isBundledSample
              ? "This 15-image bundled sample is a walkthrough, not a Scale-Up-ready dataset. We recommend at least 150 images so the default 40% allocation can fill the 60-image Test Pool. All 150 must become Verified to meet that mathematical minimum; model-quality needs may be higher."
              : "We recommend at least 150 images so the default 40% allocation can fill the 60-image Test Pool. All 150 must become Verified to meet that mathematical minimum; model-quality needs may be higher."
          }
          data-testid="low-count-warning"
        />
      )}

      <div className="mt-4 flex flex-col gap-1">
        <Text
          kind="body/regular/sm"
          style={{ color: "var(--text-primary)", display: "block" }}
        >
          Total Unlabeled: {totalUnlabeled}
        </Text>
        {/* Provider-driven copy: a hardcoded "CLIP embeddings computing
            in background..." line would be a lie for projects with
            embedding_provider="none". */}
        <EmbeddingStatusLine
          provider={embeddingProvider}
          providerSettling={embeddingProviderSettling}
          showUnavailableNotice
        />
      </div>

      <div className="flex justify-end gap-3 mt-6">
        <Button kind="secondary" onClick={onAddMore}>
          Add More Images
        </Button>
        <Button kind="primary" className="nvidia-green-button" onClick={onContinue}>
          Start labeling <ArrowRight size={14} />
        </Button>
      </div>
    </div>
  );
}

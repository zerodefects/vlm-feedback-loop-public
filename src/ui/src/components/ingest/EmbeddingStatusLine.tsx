// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Provider-aware embedding status shared by ingestion progress and its
 * completion summary.
 */

import { Text } from "@kui/react";

import { EmbeddingsUnavailableNotice } from "@/components/common/EmbeddingsUnavailableNotice";

interface EmbeddingStatusLineProps {
  provider: string;
  providerSettling?: boolean;
  /**
   * The progress card stays quiet when embeddings are unavailable; the
   * completion summary opts into the actionable Configure notice.
   */
  showUnavailableNotice?: boolean;
}

export function EmbeddingStatusLine({
  provider,
  providerSettling = false,
  showUnavailableNotice = false,
}: EmbeddingStatusLineProps): JSX.Element | null {
  if (providerSettling) {
    return (
      <Text
        kind="body/regular/xs"
        style={{ color: "var(--text-muted)", display: "block" }}
        data-testid="embedding-status-line"
      >
        Confirming the configured embedding provider…
      </Text>
    );
  }

  if (provider === "hosted_nvclip") {
    return (
      <Text
        kind="body/regular/xs"
        style={{ color: "var(--text-muted)", display: "block" }}
        data-testid="embedding-status-line"
      >
        Computing CLIP embeddings via NVIDIA hosted NIM…
      </Text>
    );
  }

  if (provider === "self_hosted_nvclip" || provider === "local_nvclip") {
    return (
      <Text
        kind="body/regular/xs"
        style={{ color: "var(--text-muted)", display: "block" }}
        data-testid="embedding-status-line"
      >
        Computing CLIP embeddings via local embedding NIM…
      </Text>
    );
  }

  return showUnavailableNotice ? (
    <EmbeddingsUnavailableNotice variant="summary" />
  ) : null;
}

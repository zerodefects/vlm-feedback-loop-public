// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * "Embeddings unavailable — diversity selection uses pHash." advisory
 * with a Configure CTA that routes to the project's NIM settings page
 * (``../settings/nim``), where the SME can paste a key and apply it
 * without a backend restart.
 *
 * One source for the copy, the navigation target, and the CTA testid
 * (``configure-embeddings-cta``); two context-appropriate skins:
 *   - ``variant="banner"``  — persistent glass-info notice on the
 *     labeling screen's inline-notices slot.
 *   - ``variant="summary"`` — compact status line under the Unlabeled
 *     count in the ingestion completion summary.
 */

import { Button, Text } from "@kui/react";
import { useNavigate } from "react-router-dom";

import { InfoBanner } from "@/components/common/InfoBanner";

const EMBEDDINGS_UNAVAILABLE_COPY =
  "Embeddings unavailable — diversity selection uses pHash.";

interface EmbeddingsUnavailableNoticeProps {
  variant: "banner" | "summary";
}

export function EmbeddingsUnavailableNotice({
  variant,
}: EmbeddingsUnavailableNoticeProps) {
  const navigate = useNavigate();
  const goToNimSettings = () => {
    navigate("../settings/nim");
  };

  if (variant === "banner") {
    return (
      <InfoBanner
        tone="info"
        align="center"
        role="status"
        body={EMBEDDINGS_UNAVAILABLE_COPY}
        data-testid="embeddings-unavailable-notice"
        actions={
          <Button
            kind="secondary"
            onClick={goToNimSettings}
            data-testid="configure-embeddings-cta"
          >
            Configure
          </Button>
        }
      />
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <Text
        kind="body/regular/xs"
        style={{ color: "var(--text-muted)", display: "block" }}
      >
        {EMBEDDINGS_UNAVAILABLE_COPY}
      </Text>
      <Button
        kind="tertiary"
        onClick={goToNimSettings}
        data-testid="configure-embeddings-cta"
      >
        Configure embeddings
      </Button>
    </div>
  );
}

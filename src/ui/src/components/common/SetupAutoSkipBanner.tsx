// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Acknowledgment banner for the fully-auto-skipped FTU setup chain.
 *
 * When every setup screen auto-skips, the SME lands on Ingest or
 * Labeling without having seen what was configured — including that
 * the hosted-NIM paths send images to build.nvidia.com.
 * ConfirmDefaultsPage's auto-skip forwards a ``setupAutoSkip`` payload
 * through router state; this banner reads it and surfaces the
 * auto-selected configuration non-blockingly on the landing screen.
 * Dismissing clears the router state in place (replace-navigation to
 * the same route), so the banner is one-shot and never persists.
 */

import { useLocation, useNavigate } from "react-router-dom";
import { Button } from "@kui/react";
import { X } from "lucide-react";

import { InfoBanner } from "@/components/common/InfoBanner";
import type { SetupAutoSkipState } from "@/types/setupChain";
import type { RecommendedMode } from "@/types/nim";

// One line per auto-configured stack. A local mode in this payload is already
// usable: setup only carries ``cameFromAutoSkip=true`` when no local deploy
// remains to dispatch.
function modeSummary(teacherMode: RecommendedMode, embeddingMode: RecommendedMode) {
  if (teacherMode === "hosted" && embeddingMode === "local") {
    return "Teacher: hosted NIM. Embeddings: deploying locally.";
  }
  if (teacherMode === "hosted" && embeddingMode === "hosted") {
    return "Using hosted NIM for Teacher and embeddings.";
  }
  if (teacherMode === "local" && embeddingMode === "local") {
    return "Using local NIMs for Teacher and embeddings.";
  }
  if (teacherMode === "local" && embeddingMode === "hosted") {
    return "Teacher: local NIM already running. Embeddings: hosted NIM.";
  }
  return `Teacher: ${teacherMode}. Embeddings: ${embeddingMode}.`;
}

export function SetupAutoSkipBanner(): JSX.Element | null {
  const location = useLocation();
  const navigate = useNavigate();
  const payload = (location.state as { setupAutoSkip?: SetupAutoSkipState } | null)
    ?.setupAutoSkip;

  if (!payload) return null;

  const defaultsLine =
    payload.teacherName !== null
      ? `Using defaults: Teacher = ${payload.teacherName}.`
      : undefined;

  return (
    <InfoBanner
      tone="info"
      border="edge"
      role="status"
      heading="Setup completed automatically"
      body={modeSummary(payload.teacherMode, payload.embeddingMode)}
      extra={defaultsLine}
      data-testid="setup-auto-skip-banner"
      actions={
        <>
          <Button
            kind="tertiary"
            size="tiny"
            onClick={() => navigate("../settings/nim")}
          >
            Review
          </Button>
          <Button
            kind="tertiary"
            size="tiny"
            onClick={() => navigate(".", { replace: true, state: {} })}
            aria-label="Dismiss"
            data-testid="setup-auto-skip-banner-dismiss"
          >
            <X size={14} style={{ color: "var(--text-muted)" }} />
          </Button>
        </>
      }
    />
  );
}

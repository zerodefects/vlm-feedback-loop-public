// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Display-label badge for TAOJob canonical statuses.
 *
 * The canonical string is NEVER shown as badge text — see
 * src/lib/training/statusDisplay.ts for the enforced mapping.  This
 * component is a thin, domain-specific wrapper around the shared
 * <StatusPill/> (src/components/common/StatusPill.tsx): it resolves the
 * TAOJob status + chain-halted reason into a tone/label via
 * ``statusDisplay()`` and delegates all chrome to the shared pill.
 */

import { StatusPill } from "@/components/common/StatusPill";
import { statusDisplay } from "@/lib/training/statusDisplay";
import type { TAOJobOutputsFetchStatus, TAOJobStatus } from "@/types/training";

interface TrainingStatusBadgeProps {
  status: TAOJobStatus;
  chainHaltedReason?: string | null;
  outputsFetchStatus?: TAOJobOutputsFetchStatus | null;
  "data-testid"?: string;
}

export function TrainingStatusBadge({
  status,
  chainHaltedReason = null,
  outputsFetchStatus = null,
  "data-testid": testid = "training-status-badge",
}: TrainingStatusBadgeProps) {
  const d = statusDisplay(status, chainHaltedReason, outputsFetchStatus);
  return (
    <StatusPill
      tone={d.tone}
      label={d.label}
      spinner={d.spinner}
      data-testid={testid}
      data-status={status}
      data-halted={chainHaltedReason ? "true" : "false"}
    />
  );
}

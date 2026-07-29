// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Chain progress summary for the Training Job Monitor. Rendered next to
 * a section heading that already names the base model, so no label
 * prefix here.
 *
 * Example rendered output:
 *   "done"
 *   "5 of 6"
 */

import { Text } from "@kui/react";

import { formatChainProgress } from "@/lib/training/formatters";

interface ChainProgressLineProps {
  /** Number of jobs that have reached ``succeeded``. */
  succeeded: number;
  /** Total job count in this chain. */
  total: number;
  "data-testid"?: string;
}

export function ChainProgressLine({
  succeeded,
  total,
  "data-testid": testid = "chain-progress-line",
}: ChainProgressLineProps) {
  const progress = formatChainProgress(succeeded, total);
  return (
    <Text
      kind="body/regular/sm"
      style={{ color: "var(--text-muted)" }}
      data-testid={testid}
    >
      {progress}
    </Text>
  );
}

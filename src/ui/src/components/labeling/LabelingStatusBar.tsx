// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Status bar showing Verified / Unlabeled / Omitted counts
 * on the labeling screen (persistent across labeling states).
 */

import { Text } from "@kui/react";

interface LabelingStatusBarProps {
  verified: number;
  unlabeled: number;
  omitted: number;
}

export function LabelingStatusBar({
  verified,
  unlabeled,
  omitted,
}: LabelingStatusBarProps) {
  return (
    <div
      className="glass-info flex items-center gap-4 px-4 py-2"
      data-testid="labeling-status-bar"
    >
      <Text
        kind="label/bold/xs"
        className="glass-pill green"
        data-testid="count-verified"
      >
        Verified: {verified}
      </Text>
      <Text kind="label/bold/xs" className="glass-pill" data-testid="count-unlabeled">
        Unlabeled: {unlabeled}
      </Text>
      <Text
        kind="label/bold/xs"
        className="glass-pill muted"
        data-testid="count-omitted"
        style={{ color: "var(--text-muted)" }}
      >
        Omitted: {omitted}
      </Text>
    </div>
  );
}

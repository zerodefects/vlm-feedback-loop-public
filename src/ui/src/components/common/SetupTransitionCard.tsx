// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Spinner, Text } from "@kui/react";

interface SetupTransitionCardProps {
  title: string;
  description: string;
  testId?: string;
}

/** Visible handoff state for setup-chain steps that navigate automatically. */
export function SetupTransitionCard({
  title,
  description,
  testId = "setup-transition-card",
}: SetupTransitionCardProps): JSX.Element {
  return (
    <div className="flex flex-1 flex-col items-center justify-center p-6">
      <div
        className="glass-card glass-card--elevated flex w-full max-w-[640px] flex-col items-center gap-4 p-8"
        data-testid={testId}
      >
        <Spinner size="large" aria-label={title} />
        <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
          {title}
        </Text>
        <Text
          kind="body/regular/sm"
          style={{ color: "var(--text-muted)", textAlign: "center" }}
        >
          {description}
        </Text>
      </div>
    </div>
  );
}

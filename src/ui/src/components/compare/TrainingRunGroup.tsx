// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Visual boundary and immutable provenance summary for one Training Suite. */

import type { ReactNode } from "react";
import { Text } from "@kui/react";
import { AlertTriangle } from "lucide-react";

export interface TrainingRunGroupProps {
  groupKey: string;
  latest: boolean;
  unassigned: boolean;
  startedAt: string | null;
  presetLabel: string | null;
  guidanceVersion: number | null;
  trainingExampleCount: number | null;
  evaluationExampleCount: number | null;
  modelSummary: string;
  warning: string | null;
  children: ReactNode;
}

function countLabel(count: number | null, singular: string): string {
  if (count == null) return `${singular} count unavailable`;
  return `${count.toLocaleString()} ${singular} image${count === 1 ? "" : "s"}`;
}

export function TrainingRunGroup({
  groupKey,
  latest,
  unassigned,
  startedAt,
  presetLabel,
  guidanceVersion,
  trainingExampleCount,
  evaluationExampleCount,
  modelSummary,
  warning,
  children,
}: TrainingRunGroupProps) {
  const title = unassigned
    ? "Unassigned historical models"
    : latest
      ? "Latest training run"
      : "Training run";
  const provenance = [
    startedAt,
    presetLabel ? `${presetLabel} preset` : null,
    guidanceVersion != null ? `Guidance v${guidanceVersion}` : "Guidance unavailable",
  ].filter((value): value is string => value != null);

  return (
    <section
      className="flex flex-col gap-3"
      data-testid={`training-run-group-${groupKey}`}
    >
      <div className="glass-card glass-card--static px-5 py-4 flex flex-col gap-3">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex min-w-0 flex-col gap-1">
            <Text kind="title/sm">{title}</Text>
            <Text kind="body/regular/xs" style={{ color: "var(--text-secondary)" }}>
              {provenance.join(" · ")}
            </Text>
            {!unassigned && (
              <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
                {countLabel(trainingExampleCount, "training")} ·{" "}
                {countLabel(evaluationExampleCount, "Test Pool")}
              </Text>
            )}
          </div>
          <Text
            kind="label/regular/xs"
            style={{ color: "var(--text-secondary)", maxWidth: 480 }}
          >
            {modelSummary}
          </Text>
        </div>

        {warning && (
          <div
            className="flex items-start gap-2 rounded-xl px-3 py-2"
            style={{
              color: "var(--warning-amber)",
              background: "color-mix(in srgb, var(--warning-amber) 8%, transparent)",
              border:
                "1px solid color-mix(in srgb, var(--warning-amber) 28%, transparent)",
            }}
            role="note"
            data-testid={`training-run-warning-${groupKey}`}
          >
            <AlertTriangle size={15} aria-hidden="true" className="mt-0.5 shrink-0" />
            <Text kind="body/regular/xs" style={{ color: "inherit" }}>
              {warning}
            </Text>
          </div>
        )}
      </div>

      {children}
    </section>
  );
}

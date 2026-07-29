// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared Accepted/Skipped/Errors count row for the ingestion screen.
 *
 * Both the in-progress state and the completion summary show
 * the same three count rows. Rendering them through one primitive keeps
 * the row heading + item detail rhythm identical across both states, and
 * (critically) lets both always reserve the three rows so the card does
 * not reshape when the first Skip or Error arrives.
 */

import { Text } from "@kui/react";
import type { FailureItem } from "@/components/ingest/IngestionProgress";

type Tone = "primary" | "secondary" | "error";

interface IngestionCountRowProps {
  label: string;
  count: number;
  items?: FailureItem[];
  /** Text color for the heading row. Detail items are always muted. */
  tone?: Tone;
  /** Optional text appended after the count on the heading row (e.g. "images (now Unlabeled)"). */
  suffix?: string;
}

const toneColor: Record<Tone, string> = {
  primary: "var(--text-primary)",
  secondary: "var(--text-secondary)",
  error: "var(--error-red-text)",
};

export function IngestionCountRow({
  label,
  count,
  items,
  tone = "secondary",
  suffix,
}: IngestionCountRowProps) {
  return (
    <div className="flex flex-col">
      <Text kind="body/regular/sm" style={{ color: toneColor[tone], display: "block" }}>
        {label}: {count}
        {suffix ? ` ${suffix}` : null}
      </Text>
      {items && items.length > 0 ? (
        // Cap the detail list at a fixed max-height with overflow-y
        // scroll — hundreds of failures would otherwise blow out the
        // page vertically, since every failure is a sibling <Text> line
        // stacked inline. The box auto-sizes down for small lists (a few
        // items render at their natural height with no scrollbar) and
        // starts scrolling once the list would exceed max-h-48.
        <div
          className="ml-4 mt-1 max-h-48 overflow-y-auto rounded-md"
          style={{
            border: items.length > 4 ? "1px solid var(--glass-border-subtle)" : "none",
            padding: items.length > 4 ? "6px 8px" : "0",
            backgroundColor:
              items.length > 4 ? "var(--block-bg-elevated)" : "transparent",
          }}
          data-testid={`count-row-items-${label.toLowerCase()}`}
        >
          {items.map((i) => (
            <Text
              key={i.name}
              kind="body/regular/sm"
              style={{ color: "var(--text-muted)", display: "block" }}
            >
              {i.name} ({i.reason})
            </Text>
          ))}
        </div>
      ) : null}
    </div>
  );
}

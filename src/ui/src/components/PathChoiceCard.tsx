// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * PathChoiceCard.
 *
 * Glass-card surface used by the setup-choice screen to render either
 * the primary
 * recommended path (the backend-recommended local Teacher on a GPU box;
 * the configured hosted default when its API key is ready) or the peer
 * alternate path (the path that's
 * "still available, one click away").
 *
 * Visual hierarchy: ``kind="primary"`` uses ``glass-card--elevated``
 * with full padding and a bold ``title/lg`` heading — the SME's eye
 * lands here first. ``kind="peer"`` uses the lighter ``glass-card``
 * with smaller chrome and a ``title/md`` heading so it reads as
 * deliberate alternate, not afterthought.
 *
 * Children render below the meta line — most callers pass either a
 * primary CTA button (Case A: [Deploy locally →]) or a credential
 * input + secondary button (Case B peer: paste API key + [Test &
 * continue]).
 */

import type { ReactNode } from "react";
import { Text } from "@kui/react";

interface PathChoiceCardProps {
  kind: "primary" | "peer";
  eyebrow?: string;
  title: string;
  meta?: string;
  children?: ReactNode;
  /**
   * Optional test ID for the card root — primarily used by the
   * NIMNvidiaKeyPage tests to distinguish Case A's local primary card
   * from Case B's hosted summary card without coupling to copy.
   */
  testId?: string;
}

export function PathChoiceCard({
  kind,
  eyebrow,
  title,
  meta,
  children,
  testId,
}: PathChoiceCardProps): JSX.Element {
  const cardClass =
    kind === "primary" ? "glass-card glass-card--elevated" : "glass-card";
  const padding = kind === "primary" ? "p-6" : "p-5";
  const titleKind = kind === "primary" ? "title/lg" : "title/md";

  return (
    <div className={`${cardClass} flex flex-col gap-3 ${padding}`} data-testid={testId}>
      {eyebrow !== undefined && (
        <Text
          kind="label/bold/xs"
          style={{
            color: "var(--text-muted)",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
          }}
        >
          {eyebrow}
        </Text>
      )}
      <div className="flex flex-col gap-1">
        <Text kind={titleKind} style={{ color: "var(--text-primary)" }}>
          {title}
        </Text>
        {meta !== undefined && (
          <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
            {meta}
          </Text>
        )}
      </div>
      {children !== undefined && (
        <div className="flex flex-col gap-2 pt-1">{children}</div>
      )}
    </div>
  );
}

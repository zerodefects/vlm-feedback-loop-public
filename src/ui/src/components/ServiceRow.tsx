// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Service configuration row for the NIM Connection screen.
 * Used for Teacher and Embeddings service rows.
 */

import type { ReactNode } from "react";
import { Button, Text } from "@kui/react";
import { ChevronDown, ChevronUp } from "lucide-react";

interface ServiceRowProps {
  serviceName: string;
  recommendedMode: string;
  description: string;
  onConfigureClick: () => void;
  isExpanded: boolean;
  children?: ReactNode;
}

export function ServiceRow({
  serviceName,
  recommendedMode,
  description,
  onConfigureClick,
  isExpanded,
  children,
}: ServiceRowProps) {
  const modeLabel =
    recommendedMode === "hosted"
      ? "NVIDIA hosted NIM"
      : recommendedMode === "local"
        ? "Deploy locally"
        : "Not available";

  return (
    <div className="glass-card glass-card--elevated overflow-hidden">
      {/* Summary row */}
      <div className="flex items-center justify-between px-5 py-4">
        <div className="flex flex-col gap-1">
          <Text kind="label/bold/sm" style={{ color: "var(--text-primary)" }}>
            {serviceName}
          </Text>
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            {modeLabel}
          </Text>
          <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
            {description}
          </Text>
        </div>
        <Button kind="secondary" onClick={onConfigureClick}>
          Configure {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </Button>
      </div>

      {/* Expanded override panel */}
      {isExpanded && children && (
        <div
          className="px-5 pb-5"
          style={{ borderTop: "1px solid var(--glass-border)" }}
        >
          <div className="pt-4">{children}</div>
        </div>
      )}
    </div>
  );
}

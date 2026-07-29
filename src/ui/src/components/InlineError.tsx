// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Inline X + error-text primitive — the single red "icon + message" row
 * used everywhere an error replaces or annotates a content region:
 * Image Ingestion path errors and the Guidance builder's
 * per-field validation rows (which re-export it as ``ErrorRow``).
 *
 * Uses the .text-error token so the chrome matches every other
 * destructive text surface in the app and aligns with KUI Foundations
 * --text-color-feedback-danger used by the Retail Blueprints.
 */

import { Text } from "@kui/react";
import { X } from "lucide-react";

interface InlineErrorProps {
  message: string;
  /** data-testid for the row, e.g. "path-error" or "error-NO_CORE_FIELDS". */
  testId?: string;
  /** Extra layout classes (spacing/indent) appended to the shared treatment. */
  className?: string;
  /** Optional fix-it affordance rendered after the message. */
  children?: React.ReactNode;
}

export function InlineError({
  message,
  testId,
  className,
  children,
}: InlineErrorProps) {
  return (
    <div
      className={`text-error flex items-center gap-2 text-sm${className ? ` ${className}` : ""}`}
      data-testid={testId}
    >
      <X size={14} className="flex-shrink-0" aria-hidden="true" />
      <Text kind="body/regular/sm" style={{ color: "inherit" }}>
        {message}
      </Text>
      {children}
    </div>
  );
}

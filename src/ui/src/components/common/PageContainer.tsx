// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Standard page-level shell used by the configuration-summary / dashboard
 * screens (Scale-Up Hub, Batch Pre-Run, Batch Run Status, Student
 * Training, Training Job Monitor).
 *
 * Centralises the max-width, padding, and vertical rhythm so those
 * screens can't drift out of alignment with each other.  Matches the
 * Retail Blueprint density: 1024 px max content width, 24 px card
 * padding, 24 px between sections.
 */

import type { ReactNode } from "react";

export interface PageContainerProps {
  children: ReactNode;
  "data-testid"?: string;
  /**
   * Escape hatch for screens that need a wider canvas (e.g. a three-up
   * comparison). Defaults to ``max-w-5xl`` (64rem / 1024px). Prefer the
   * default unless the content explicitly requires more width.
   */
  maxWidthClass?: string;
}

export function PageContainer({
  children,
  "data-testid": testid,
  maxWidthClass = "max-w-5xl",
}: PageContainerProps) {
  return (
    <div
      className={`mx-auto w-full ${maxWidthClass} p-6 flex flex-col gap-6`}
      data-testid={testid}
    >
      {children}
    </div>
  );
}

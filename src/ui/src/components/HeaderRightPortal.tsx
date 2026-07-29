// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from "react";
import { createPortal } from "react-dom";

/**
 * Portal children into the AppShell header-right slot.
 *
 * Falls back to inline rendering when the slot is absent (unit tests that
 * mount the page without AppShell, storybook) so the CTA never silently
 * disappears if a consumer skips the shell.
 */
export function HeaderRightPortal({ children }: { children: ReactNode }) {
  const target =
    typeof document !== "undefined"
      ? document.getElementById("header-right-slot")
      : null;
  return target ? createPortal(children, target) : <>{children}</>;
}

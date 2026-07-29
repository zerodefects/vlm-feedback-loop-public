// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Small icon-only button with disabled state. Uses KUI Button kind="tertiary". */

import { Button } from "@kui/react";

interface IconBtnProps {
  onClick: () => void;
  disabled?: boolean;
  label: string;
  children: React.ReactNode;
  testId?: string;
}

export function IconBtn({ onClick, disabled, label, children, testId }: IconBtnProps) {
  // Each call site renders its lucide icon with an inline `style={{ color:
  // "var(--text-muted)" }}` so the active-state colour is consistent. That
  // inline colour overrides KUI Button's disabled token, so without this
  // wrapper a disabled chevron looks identical to an active one. Dropping the
  // child opacity to ~0.4 when disabled gives the eye a clear cue without
  // touching any of the call sites.
  return (
    <Button
      kind="tertiary"
      size="tiny"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      data-testid={testId}
      className="rounded p-1"
    >
      <span style={{ display: "inline-flex", opacity: disabled ? 0.4 : 1 }}>
        {children}
      </span>
    </Button>
  );
}

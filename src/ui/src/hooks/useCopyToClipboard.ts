// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Copy-to-clipboard with the shared "Copied" feedback contract: ``copy``
 * writes the text, flips ``copied`` true, and resets it after 2s so the
 * button label can flash "Copied" and revert. Clipboard errors (API
 * unavailable — e.g. non-secure context) are swallowed; ``copy``
 * resolves ``false`` and ``copied`` stays false, so callers can gate
 * copy-dependent side effects on the result.
 */

import { useCallback, useEffect, useRef, useState } from "react";

const COPIED_RESET_MS = 2000;

export function useCopyToClipboard(): {
  copied: boolean;
  copy: (text: string) => Promise<boolean>;
} {
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clear a pending reset on unmount so it never fires into an
  // unmounted component.
  useEffect(
    () => () => {
      if (timeoutRef.current != null) clearTimeout(timeoutRef.current);
    },
    [],
  );

  const copy = useCallback(async (text: string): Promise<boolean> => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      return false;
    }
    setCopied(true);
    if (timeoutRef.current != null) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => setCopied(false), COPIED_RESET_MS);
    return true;
  }, []);

  return { copied, copy };
}

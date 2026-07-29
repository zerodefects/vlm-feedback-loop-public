// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Green "get a key" portal link — the ExternalLink-suffixed anchor
 * repeated across the FTU setup screens and the NIM Connection
 * credential rows. Opens the key portal in a new tab.
 */

import { Text } from "@kui/react";
import { ExternalLink } from "lucide-react";

interface KeyPortalLinkProps {
  href: string;
  label: string;
  /** Callers embedded in clickable rows pass a stopPropagation handler. */
  onClick?: (e: React.MouseEvent<HTMLAnchorElement>) => void;
  /** Layout tweaks appended to the base classes (e.g. "self-start"). */
  className?: string;
  /** Dense variant for credential rows: xs label text + 11px icon. */
  dense?: boolean;
}

export function KeyPortalLink({
  href,
  label,
  onClick,
  className,
  dense = false,
}: KeyPortalLinkProps): JSX.Element {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      onClick={onClick}
      className={`flex items-center gap-1 hover:underline${className ? ` ${className}` : ""}`}
      style={{ color: "var(--accent-green)" }}
    >
      <Text
        kind={dense ? "label/regular/xs" : "body/regular/sm"}
        style={{ color: "inherit" }}
      >
        {label}
      </Text>
      <ExternalLink size={dense ? 11 : 14} />
    </a>
  );
}

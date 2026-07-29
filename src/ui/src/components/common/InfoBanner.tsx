// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Icon + text banner shared across the configuration-summary / status
 * screens.  Centralizes the banner chrome that call sites otherwise
 * hand-roll inconsistently — ``.glass-info`` vs ``.toast-error``, inline
 * ``borderColor`` overrides, icon sizes, and the ``block`` class on each
 * text line.  The ``block`` class is load-bearing: two sibling ``<Text>``
 * elements without it end up on the same visual line inside
 * ``glass-info``, even though the surrounding ``space-y-*`` utility
 * suggests otherwise.
 *
 * This component picks sensible defaults per tone so call sites don't
 * need to remember any of that:
 *   - tone="info"    → ``glass-info`` base, ``<Info/>`` icon
 *   - tone="warning" → ``glass-info`` + amber border, ``<AlertTriangle/>``
 *   - tone="success" → ``glass-info`` + green border, ``<CheckCircle2/>``
 *   - tone="error"   → ``toast-error`` (its own red chrome), ``<X/>`` icon
 *
 * Use ``heading`` + ``body`` for the common 2-line banner.  ``extra`` is
 * the optional third line (e.g. retained-count summary).  For anything
 * more complex, pass ``children`` — the icon column is preserved.
 *
 * Banners with right-aligned affordances (dismiss [X], inline CTAs)
 * pass ``actions`` — the banner switches to the two-column
 * text-left / actions-right layout shared by the labeling-screen
 * notice, trigger-recommendation, and auto-skip dismissable banners.
 */

import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import { isValidElement } from "react";
import type { ComponentType, ReactElement, ReactNode } from "react";

import { Text } from "@kui/react";

type BannerTone = "info" | "warning" | "success" | "error";

interface BannerToneConfig {
  /** Base chrome class — either ``glass-info`` or ``toast-error``. */
  baseClass: string;
  /** Default lucide icon for the tone. */
  defaultIcon: ComponentType<{
    size?: number;
    className?: string;
    style?: React.CSSProperties;
  }>;
  /** Inline ``borderColor`` override on top of ``glass-info`` (null for error, which uses its own border). */
  borderColor: string | null;
  /** Left-edge color used by ``border="edge"`` (null for error, which uses its own border). */
  edgeColor: string | null;
  /** Inline color for the icon (null = inherit from class). */
  iconColor: string | null;
}

const TONE_CONFIG: Readonly<Record<BannerTone, BannerToneConfig>> = {
  info: {
    baseClass: "glass-info",
    defaultIcon: Info,
    borderColor: null,
    edgeColor: "var(--glass-border)",
    iconColor: "var(--text-muted)",
  },
  warning: {
    baseClass: "glass-info",
    defaultIcon: AlertTriangle,
    borderColor: "var(--warning-amber)",
    edgeColor: "var(--warning-amber)",
    iconColor: "var(--warning-amber)",
  },
  success: {
    baseClass: "glass-info",
    defaultIcon: CheckCircle2,
    borderColor: "var(--accent-green)",
    edgeColor: "var(--accent-green-border)",
    iconColor: "var(--accent-green)",
  },
  error: {
    baseClass: "toast-error",
    defaultIcon: X,
    borderColor: null, // toast-error sets its own border
    edgeColor: null, // toast-error sets its own border
    iconColor: null, // inherits toast-error text color
  },
};

export interface InfoBannerProps {
  tone: BannerTone;
  /**
   * Bold heading line. Rendered in the tone's accent color for warning
   * (amber) and inherits the surrounding class color otherwise.
   */
  heading?: ReactNode;
  /** Secondary body line (white-secondary for every tone; the tone color stays on icon + heading). */
  body?: ReactNode;
  /** Optional third line below the body (e.g. retained-count summary). */
  extra?: ReactNode;
  /**
   * Custom content in place of heading/body/extra. Use when the banner
   * carries conditional children (e.g. the Completed banner where the
   * count interpolates into the middle of the sentence). The icon column
   * is preserved so custom layouts still align with standard banners.
   */
  children?: ReactNode;
  /**
   * Lucide icon override (defaults to tone-appropriate icon). Accepts
   * either a component type (sized/colored per tone) or a pre-built
   * element (rendered verbatim — e.g. a KUI ``<Spinner/>`` standing in
   * for the icon on in-progress banners).
   */
  icon?:
    | ComponentType<{
        size?: number;
        className?: string;
        style?: React.CSSProperties;
      }>
    | ReactElement;
  /** Icon size — defaults to 16 for warning/error, 14 for info/success (higher-urgency tones read slightly larger). */
  iconSize?: number;
  /**
   * Layout inside the banner: ``"top"`` (default) aligns the icon to
   * the first line via ``items-start`` + ``mt-0.5``.  ``"center"`` uses
   * ``items-center`` with no icon offset — used for single-line banners.
   */
  align?: "top" | "center";
  /**
   * Border treatment: ``"full"`` (default) applies the tone's border
   * color to the whole banner border; ``"edge"`` keeps the base glass
   * border and renders a 2px tone-colored left edge instead (the
   * auto-skip dismissable banner treatment).
   */
  border?: "full" | "edge";
  /**
   * Right-aligned action group (dismiss buttons, CTAs). Providing this
   * prop — even conditionally ``null`` — switches the banner to the
   * two-column text-left / actions-right layout shared by the
   * dismissable, trigger-recommendation, and notice banners.
   */
  actions?: ReactNode;
  /** ARIA role (e.g. ``"status"`` for polite live-region announcements). */
  role?: React.AriaRole;
  className?: string;
  "data-testid"?: string;
}

export function InfoBanner({
  tone,
  heading,
  body,
  extra,
  children,
  icon,
  iconSize,
  align = "top",
  border = "full",
  actions,
  role,
  className,
  "data-testid": testid,
}: InfoBannerProps) {
  const cfg = TONE_CONFIG[tone];
  const iconIsElement = isValidElement(icon);
  const Icon = (iconIsElement ? undefined : icon) ?? cfg.defaultIcon;
  const effectiveSize = iconSize ?? (tone === "warning" || tone === "error" ? 16 : 14);

  const wrapperStyle: React.CSSProperties = {};
  if (border === "edge") {
    if (cfg.edgeColor) wrapperStyle.borderLeft = `2px solid ${cfg.edgeColor}`;
  } else if (cfg.borderColor) {
    wrapperStyle.borderColor = cfg.borderColor;
  }

  const iconStyle: React.CSSProperties = {};
  if (cfg.iconColor) iconStyle.color = cfg.iconColor;

  const flexAlign = align === "center" ? "items-center" : "items-start";
  const iconOffset = align === "center" ? "" : "mt-0.5";

  // Heading color: warning = amber, success = green, error/info = inherit.
  const headingColor =
    tone === "warning"
      ? "var(--warning-amber)"
      : tone === "success"
        ? "var(--accent-green)"
        : undefined;

  const iconEl = iconIsElement ? (
    icon
  ) : (
    <Icon
      size={effectiveSize}
      className={`flex-shrink-0 ${iconOffset}`.trim()}
      style={Object.keys(iconStyle).length > 0 ? iconStyle : undefined}
    />
  );

  const content =
    children !== undefined ? (
      <div className="min-w-0 flex-1">{children}</div>
    ) : (
      <div className="min-w-0 flex-1 space-y-1">
        {heading !== undefined && (
          <Text
            kind="label/bold/sm"
            className="block"
            style={headingColor ? { color: headingColor } : undefined}
          >
            {heading}
          </Text>
        )}
        {body !== undefined && (
          <Text
            kind="body/regular/sm"
            className="block"
            style={{ color: "var(--text-secondary)" }}
          >
            {body}
          </Text>
        )}
        {extra !== undefined && (
          <Text
            kind="body/regular/sm"
            className="block"
            style={{ color: "var(--text-secondary)" }}
          >
            {extra}
          </Text>
        )}
      </div>
    );

  if (actions !== undefined) {
    return (
      <div
        className={`${cfg.baseClass} flex ${flexAlign} justify-between gap-3 px-4 py-3 ${className ?? ""}`.trim()}
        style={Object.keys(wrapperStyle).length > 0 ? wrapperStyle : undefined}
        role={role}
        data-testid={testid}
        data-tone={tone}
      >
        <div className={`flex ${flexAlign} gap-2 min-w-0 flex-1`}>
          {iconEl}
          {content}
        </div>
        {actions ? (
          <div className="flex items-center gap-2 shrink-0">{actions}</div>
        ) : null}
      </div>
    );
  }

  return (
    <div
      className={`${cfg.baseClass} flex ${flexAlign} gap-2 px-4 py-3 ${className ?? ""}`.trim()}
      style={Object.keys(wrapperStyle).length > 0 ? wrapperStyle : undefined}
      role={role}
      data-testid={testid}
      data-tone={tone}
    >
      {iconEl}
      {content}
    </div>
  );
}

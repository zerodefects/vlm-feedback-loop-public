// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared page chrome for the Create and Edit Guidance screens: sticky
 * glass header (title/subtitle + StatusBadge), sr-only aria-live
 * validation announcer, scrollable body, sticky glass footer whose
 * action row is constrained to the same max-w-3xl column as the form,
 * and a trailing slot for page-specific modals.
 *
 * Page-specific behavior (data flow, navigation, footer buttons,
 * dialogs) stays in the pages and arrives through slots.
 */

import type { CSSProperties, ReactNode, RefObject } from "react";
import { Text } from "@kui/react";

import { scrollToFirstError } from "@/lib/scroll-to-first-error";
import { StatusBadge } from "./StatusBadge";
import type { GuidanceForm } from "./field-handlers";

/** Shared glass chrome for the sticky header/footer bars — the standard
 *  blur/saturate glass token over a heavier 0.85 fill so scrolled content
 *  stays legible behind the bars. */
const stickyBarStyle: CSSProperties = {
  borderColor: "var(--glass-border)",
  backgroundColor: "rgba(12, 12, 12, 0.85)",
  backdropFilter: "blur(20px) saturate(150%)",
  WebkitBackdropFilter: "blur(20px) saturate(150%)",
};

interface GuidanceEditorLayoutProps {
  /** Root-container testid ("create-guidance-page" / "edit-guidance-page"). */
  testId: string;
  /** Header title ("Create Guidance" / "Edit Guidance"). */
  title: string;
  /** Header subtitle line (project name, version). */
  subtitle: ReactNode;
  /** Shared form hook — drives the badge, badge click, and aria-live text. */
  form: GuidanceForm;
  /** Scroll-container ref; the owning page also uses it in its save flow. */
  scrollBodyRef: RefObject<HTMLDivElement | null>;
  /** Pass form.saveAttempted so the badge flips from grey "N required"
   *  to red "N errors" after a failed save. */
  badgeSaveAttempted?: boolean;
  /** Optional testid on the aria-live region (Create-only test contract). */
  ariaLiveTestId?: string;
  /** Success toast line ("Guidance saved" etc.); rendered at the top
   *  of the body with the shared ``save-toast`` testid. */
  saveToast?: string | null;
  /** Prebuilt save-failure message; rendered below the toast with the
   *  shared ``save-error`` testid. Each page passes its own string. */
  saveErrorMessage?: string | null;
  /** Footer flex-alignment classes, verbatim from each page — Create
   *  "justify-end gap-3", Edit "justify-between". */
  footerJustifyClass: string;
  /** Footer contents (action buttons). */
  footerSlot: ReactNode;
  /** Page-specific modals, rendered after the footer (same DOM position
   *  they occupied before the extraction). */
  dialogSlot?: ReactNode;
  /** Scrollable body contents. */
  children: ReactNode;
}

export function GuidanceEditorLayout({
  testId,
  title,
  subtitle,
  form,
  scrollBodyRef,
  badgeSaveAttempted,
  ariaLiveTestId,
  saveToast,
  saveErrorMessage,
  footerJustifyClass,
  footerSlot,
  dialogSlot,
  children,
}: GuidanceEditorLayoutProps) {
  // Badge click reveals inline errors and jumps to the first one.
  function handleBadgeClick() {
    form.markSaveAttempted();
    scrollToFirstError(scrollBodyRef);
  }

  return (
    <div
      className="flex flex-col"
      style={{ height: "calc(100vh - 56px)" }}
      data-testid={testId}
    >
      {/* Sticky header */}
      <div
        className="sticky z-40 flex items-center justify-between border-b px-6 py-3"
        style={{ top: 56, ...stickyBarStyle }}
      >
        <div>
          <Text
            kind="title/sm"
            style={{ color: "var(--text-primary)", display: "block" }}
          >
            {title}
          </Text>
          <Text
            kind="label/regular/xs"
            style={{ color: "var(--text-muted)", display: "block" }}
          >
            {subtitle}
          </Text>
        </div>
        {/* Validation is backend-driven; until the first validate_draft
            response lands there is no verdict to show, so the badge and
            the announcer stay empty rather than claiming "Valid". */}
        {form.backendValidation !== null && (
          <StatusBadge
            errorCount={form.totalErrorCount}
            saveAttempted={badgeSaveAttempted}
            onClick={handleBadgeClick}
          />
        )}
      </div>

      <div aria-live="polite" className="sr-only" data-testid={ariaLiveTestId}>
        {form.backendValidation === null
          ? ""
          : form.totalErrorCount > 0
            ? `${form.totalErrorCount} validation error${form.totalErrorCount === 1 ? "" : "s"}`
            : "Schema is valid"}
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto p-6" ref={scrollBodyRef}>
        <div className="mx-auto max-w-3xl space-y-6 fade-in">
          {saveToast && (
            <div
              className="px-4 py-3 toast-success"
              data-testid="save-toast"
              role="status"
            >
              <Text kind="body/regular/sm" style={{ color: "inherit" }}>
                ✓ {saveToast}
              </Text>
            </div>
          )}
          {saveErrorMessage && (
            <div className="rounded-lg px-4 py-3 toast-error" data-testid="save-error">
              <Text kind="body/regular/sm" style={{ color: "inherit" }}>
                {saveErrorMessage}
              </Text>
            </div>
          )}
          {children}
        </div>
      </div>

      {/* Sticky footer — the action row shares the body's max-w-3xl column
          so the CTAs align with the form edge instead of the viewport. */}
      <div className="sticky bottom-0 z-40 border-t px-6 py-3" style={stickyBarStyle}>
        <div
          className={`mx-auto flex w-full max-w-3xl items-center ${footerJustifyClass}`}
        >
          {footerSlot}
        </div>
      </div>

      {dialogSlot}
    </div>
  );
}

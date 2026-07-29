// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Inline dismissable notices for the labeling screen.
 *
 * Notice types (priority order — only one renders at a time, the
 * single-nudge-at-a-time slot allocation):
 *   1. Cold start: Verified=0, first proposal
 *   2. Prior-label progress: post-schema-change re-labeling
 *   3. Schema refinement reminders: which reminder (if any) is
 *      active comes from the backend (`GET .../guidance:reminder_status`)
 *      — thresholds, the higher-of-two rule, dismissal counting, and
 *      suppression once Guidance was edited all live server-side. This
 *      component only renders what `reminderStatus.active_reminder` says.
 *   4. Embeddings unavailable: persistent — when
 *      ``embedding_provider === "none"``. Lowest priority because it's
 *      not time-bounded and stays surfaceable until the SME configures
 *      a provider via the NIM settings page.
 */

import { Text, Button } from "@kui/react";
import { X, Info } from "lucide-react";

import { EmbeddingsUnavailableNotice } from "@/components/common/EmbeddingsUnavailableNotice";
import { InfoBanner } from "@/components/common/InfoBanner";
import type { ReminderStatusResponse } from "@/types/guidance";

interface InlineNoticesProps {
  verifiedCount: number;
  /**
   * Backend-owned schema refinement reminder state. Undefined while
   * loading — no reminder renders until the backend has answered.
   */
  reminderStatus: ReminderStatusResponse | undefined;
  coldStartDismissed: boolean;
  pendingRelabel: number;
  priorRelabeled: number;
  /**
   * Project's effective embedding provider. When ``"none"``, render a
   * persistent "Embeddings unavailable" notice with a [Configure] CTA —
   * last in the priority order so it never displaces a cold-start or
   * schema-refinement nudge.
   */
  embeddingProvider: string;
  onDismissColdStart: () => void;
  onDismissReminder: () => void;
  onReviewSchema: () => void;
}

export function InlineNotices({
  verifiedCount,
  reminderStatus,
  coldStartDismissed,
  pendingRelabel,
  priorRelabeled,
  embeddingProvider,
  onDismissColdStart,
  onDismissReminder,
  onReviewSchema,
}: InlineNoticesProps) {
  // ── Cold start notice ─────────────────────────────────────────────────
  if (verifiedCount === 0 && !coldStartDismissed) {
    return (
      <NoticeBanner testId="cold-start-notice" onDismiss={onDismissColdStart}>
        This is your first label. The model has no examples to learn from yet. Accuracy
        improves immediately with every Edit.
      </NoticeBanner>
    );
  }

  // ── Prior-label progress indicator ────────────────────────────────────
  // Shown as informational text, not a dismissable notice. Renders the
  // "N of M re-labeled (prior edits first)" fraction so
  // the SME sees progress against the schema-change cohort. M is
  // computed from the prior cohort: already-re-labeled + still-pending.
  if (pendingRelabel > 0) {
    const totalPrior = priorRelabeled + pendingRelabel;
    return (
      <div
        className="glass-info flex items-center gap-2 px-4 py-2"
        data-testid="prior-label-progress"
      >
        <Info size={14} style={{ color: "var(--accent-green)" }} />
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            Prior labels: {priorRelabeled} of {totalPrior} re-labeled (prior edits
            first).
          </Text>
          <div
            className="h-1 w-full rounded-full"
            style={{ background: "var(--bar-track)" }}
          >
            <div
              className="h-1 rounded-full"
              style={{
                width: `${(priorRelabeled / totalPrior) * 100}%`,
                background: "var(--accent-green)",
                transition: "width 300ms",
              }}
            />
          </div>
        </div>
      </div>
    );
  }

  // ── Schema refinement reminders ──────────────────────────────────────
  // Eligibility is the backend's call; render whichever reminder
  // it reports as active. Copy uses the backend's verified_count so the
  // banner text matches the count the decision was made on.

  if (reminderStatus?.active_reminder === 2) {
    return (
      <NoticeBanner
        testId="schema-reminder-2"
        onDismiss={onDismissReminder}
        action={
          <Button
            kind="secondary"
            onClick={onReviewSchema}
            data-testid="review-schema-btn"
          >
            Review Schema
          </Button>
        }
      >
        You have {reminderStatus.verified_count} labels. Schema changes mean more images
        to re-label.
      </NoticeBanner>
    );
  }

  if (reminderStatus?.active_reminder === 1) {
    return (
      <NoticeBanner
        testId="schema-reminder-1"
        onDismiss={onDismissReminder}
        action={
          <Button
            kind="secondary"
            onClick={onReviewSchema}
            data-testid="review-schema-btn"
          >
            Review Schema
          </Button>
        }
      >
        Need to adjust your schema? Fewer labels to re-do now.
      </NoticeBanner>
    );
  }

  // ── Embeddings unavailable ───────────────────────────────────────────
  // Persistent (non-dismissable) advisory when the project has no
  // embedding provider. Lives in the inline-notices region so it
  // respects the single-nudge-at-a-time slot allocation — only renders
  // when no higher-priority nudge is on screen. Copy + [Configure] CTA
  // are shared with the ingestion summary via
  // EmbeddingsUnavailableNotice.
  if (embeddingProvider === "none") {
    return <EmbeddingsUnavailableNotice variant="banner" />;
  }

  return null;
}

// ── Reusable notice banner ──────────────────────────────────────────────────
// Dismissable-nudge adapter over the shared <InfoBanner/>: contributes the
// per-notice testids and the tiny dismiss [X]; all banner chrome lives in
// InfoBanner.

function NoticeBanner({
  testId,
  onDismiss,
  action,
  children,
}: {
  testId: string;
  onDismiss: () => void;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <InfoBanner
      tone="info"
      align="center"
      role="status"
      body={children}
      data-testid={testId}
      actions={
        <>
          {action}
          <Button
            kind="tertiary"
            size="tiny"
            onClick={onDismiss}
            aria-label="Dismiss"
            data-testid={`${testId}-dismiss`}
          >
            <X size={14} style={{ color: "var(--text-muted)" }} />
          </Button>
        </>
      }
    />
  );
}

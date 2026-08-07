// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Rationale panel for the labeling screen edit flow.
 *
 * Appears below the proposal form when the SME modifies any field.
 * Enforces the invariant: every Verified Edit has an SME-reviewed
 * rationale note. Four visible states: needs_review, regenerating,
 * ai_review_required, and edited/approved.
 */

import { Button, Spinner, Text } from "@kui/react";
import { Sparkles, Check, AlertTriangle } from "lucide-react";
import { useId } from "react";

// ── Types ────────────────────────────────────────────────────────────────────

export type RationalePanelState =
  | "hidden"
  | "needs_review"
  | "regenerating"
  | "ai_review_required"
  | "edited"
  | "approved";

interface RationalePanelProps {
  state: RationalePanelState;
  rationaleText: string;
  onRationaleTextChange: (text: string) => void;
  onGenerateAI: () => void;
  onApproveAI: () => void;
  regenerationError: string | null;
  disabled?: boolean;
}

// ── Component ────────────────────────────────────────────────────────────────

export function RationalePanel({
  state,
  rationaleText,
  onRationaleTextChange,
  onGenerateAI,
  onApproveAI,
  regenerationError,
  disabled = false,
}: RationalePanelProps) {
  const textareaId = useId();
  const helperId = useId();
  const errorId = useId();

  if (state === "hidden") return null;

  const helperVisible = state === "needs_review" || state === "ai_review_required";
  const descriptionIds = [
    helperVisible ? helperId : null,
    regenerationError ? errorId : null,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className="mt-4 flex flex-col gap-3 border-t pt-4 fade-in"
      style={{ borderColor: "var(--glass-border)" }}
      data-testid="rationale-panel"
    >
      {/* Header — uppercase tracked-caps eyebrow ("RATIONALE"),
         matching the Retail Blueprint section-eyebrow pattern
         (CONFIGURATION / AGENT PERFORMANCE). Title-case rendering reads as a
         body label rather than a section divider. */}
      <div className="flex items-center gap-2">
        <label htmlFor={textareaId}>
          <Text
            kind="label/regular/xs"
            className="section-eyebrow"
            style={{
              color: "var(--text-secondary)",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            Rationale
          </Text>
        </label>
        {state === "needs_review" && (
          <span className="glass-pill yellow" data-testid="rationale-state-badge">
            <Text kind="body/regular/xs">Needs review</Text>
          </span>
        )}
        {state === "edited" && (
          <span className="glass-pill green" data-testid="rationale-state-badge">
            <Text kind="body/regular/xs">Edited</Text>
          </span>
        )}
        {state === "approved" && (
          <span className="glass-pill green" data-testid="rationale-state-badge">
            <Text kind="body/regular/xs">Approved</Text>
          </span>
        )}
        {state === "ai_review_required" && (
          <span className="glass-pill yellow" data-testid="rationale-state-badge">
            <Text kind="body/regular/xs">AI-generated, review required</Text>
          </span>
        )}
      </div>

      {/* Needs review: helper text */}
      {state === "needs_review" && (
        <Text
          id={helperId}
          kind="body/regular/sm"
          style={{ color: "var(--text-muted)" }}
          data-testid="rationale-helper"
        >
          Your label changed. Update the rationale to match.
        </Text>
      )}

      {/* AI review required: helper text */}
      {state === "ai_review_required" && (
        <Text
          id={helperId}
          kind="body/regular/sm"
          style={{ color: "var(--text-muted)" }}
          data-testid="rationale-helper"
        >
          Review the AI-generated rationale before saving.
        </Text>
      )}

      {/* Regenerating: spinner */}
      {state === "regenerating" ? (
        <div
          className="glass-info flex items-center gap-3 px-4 py-6"
          data-testid="rationale-regenerating"
        >
          <Spinner aria-label="Generating rationale" />
          <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
            Generating rationale for your corrected label...
          </Text>
        </div>
      ) : (
        /* Textarea — visible in all states except hidden and regenerating */
        <textarea
          id={textareaId}
          className="glass-input w-full px-3 py-2 text-sm leading-relaxed"
          style={{ minHeight: 80, resize: "vertical" }}
          value={rationaleText}
          onChange={(e) => onRationaleTextChange(e.target.value)}
          disabled={disabled}
          aria-describedby={descriptionIds || undefined}
          placeholder="Describe the visible evidence that supports this label..."
          data-testid="rationale-textarea"
        />
      )}

      {/* Regeneration error */}
      {regenerationError && (
        <div
          id={errorId}
          className="flex items-center gap-2"
          role="alert"
          data-testid="rationale-regen-error"
        >
          <AlertTriangle size={14} className="text-error" />
          <Text kind="body/regular/sm" className="text-error">
            {regenerationError}
          </Text>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center gap-3">
        {/* Generate AI Rationale — shown in needs_review state */}
        {state === "needs_review" && (
          <Button
            kind="secondary"
            onClick={onGenerateAI}
            disabled={disabled}
            data-testid="generate-ai-rationale-btn"
          >
            <Sparkles size={14} /> Generate AI Rationale
          </Button>
        )}

        {/* Approve AI Rationale — shown in ai_review_required state */}
        {state === "ai_review_required" && (
          <Button
            kind="secondary"
            onClick={onApproveAI}
            disabled={disabled}
            data-testid="approve-ai-rationale-btn"
          >
            <Check size={14} /> Approve AI Rationale
          </Button>
        )}
      </div>
    </div>
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Apply Controls.
 *
 * Renders after ``ConnectionTestPanel`` reports success on the NIM
 * Connection screen (both initial onboarding and post-onboarding edit
 * mode). Lets
 * the SME choose how their pasted key is applied:
 *
 *   - **Apply this session** (always-on, disabled checkbox): the key is
 *     installed in the process-level runtime-override layer
 *     and applies to the next NIM call. Lost on backend
 *     restart.
 *   - **Save to ~/.vlm_feedback_loop/.env** (opt-in, hidden when the
 *     deployment-level ``ALLOW_UI_SECRET_PERSIST`` flag is false): the
 *     key is additionally upserted into the .env file and Settings is
 *     reloaded so it survives restart.
 *
 * Production / container deployments typically run with
 * ``ALLOW_UI_SECRET_PERSIST=false`` (the .env file is managed
 * externally); in that posture the persist checkbox is hidden entirely
 * and the SME is left with the session-only path.
 */

import { useState } from "react";
import { Text } from "@kui/react";

export interface ApplyControlsValue {
  /** Whether the SME opted in to disk persistence. */
  persist: boolean;
}

interface ApplyControlsProps {
  /**
   * Whether the deployment allows persistent disk writes from the UI.
   * Sourced from ``GET /v1/environment.allow_secret_persist``. When
   * false, the persist checkbox is hidden entirely.
   */
  allowPersist: boolean;
  /** The current value (controlled component). */
  value: ApplyControlsValue;
  /** Called whenever the SME toggles the persist checkbox. */
  onChange: (value: ApplyControlsValue) => void;
}

export function ApplyControls({
  allowPersist,
  value,
  onChange,
}: ApplyControlsProps): JSX.Element {
  // Local state mirrors the persist checkbox. Kept in sync with the
  // controlled ``value`` prop on every render so a parent reset clears
  // the local state too.
  const [persistChecked, setPersistChecked] = useState(value.persist);
  if (value.persist !== persistChecked) {
    // Reconcile to the controlled value when the parent changes it
    // externally (e.g. resets the form). React's render-time setter is
    // safe here because the equality check above prevents loops.
    setPersistChecked(value.persist);
  }

  return (
    <div className="flex flex-col gap-2 mt-3" data-testid="apply-controls">
      <Text
        kind="label/bold/xs"
        style={{ color: "var(--text-primary)", display: "block" }}
      >
        Apply your key
      </Text>

      {/* "Apply this session" — always on. Rendered as a disabled
          checkbox plus muted helper text so the SME understands what
          will happen when they click [Continue] / [Save]. */}
      <label className="flex items-center gap-2 opacity-80 cursor-not-allowed">
        <input
          type="checkbox"
          checked
          disabled
          className="glass-input"
          aria-label="Apply this session"
          data-testid="apply-session-checkbox"
        />
        <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
          Apply this session
        </Text>
      </label>

      {/* Persist checkbox — opt-in, hidden when not allowed. */}
      {allowPersist && (
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={persistChecked}
            onChange={(e) => {
              const next = e.currentTarget.checked;
              setPersistChecked(next);
              onChange({ persist: next });
            }}
            className="glass-input"
            aria-label="Save to env file"
            data-testid="apply-persist-checkbox"
          />
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            Save to <code>~/.vlm_feedback_loop/.env</code> (persist across restarts)
          </Text>
        </label>
      )}

      <Text
        kind="body/regular/xs"
        style={{ color: "var(--text-muted)", display: "block", marginTop: 4 }}
      >
        Project-scoped settings save immediately. Session keys are kept in memory until
        restart{allowPersist ? " unless you save them to the .env file" : ""}.
      </Text>
    </div>
  );
}

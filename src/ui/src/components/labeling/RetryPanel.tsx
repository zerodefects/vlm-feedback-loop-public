// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Inline retry parameter selection panel.
 *
 * Replaces the action buttons when the SME clicks [Retry].
 * 5 controls pre-populated from current project settings.
 * Thinking and Visual Budget hide/show based on the selected
 * teacher model's capabilities.
 */

import { useMemo, useState } from "react";
import { Button, Text } from "@kui/react";

import { SegmentedControl } from "@/components/SegmentedControl";
import { thinkingToggleVisible, visualBudgetVisible } from "@/lib/model-display";
import type { ModelConfigResponse } from "@/types/nim";
import type { GuidanceResponse } from "@/types/guidance";

// ── Types ────────────────────────────────────────────────────────────────────

export interface RetryOverrides {
  teacher_model_config_id_override?: string;
  guidance_id_override?: string;
  generation_preset_key_override?: string;
  thinking_mode_override?: "on" | "off";
  visual_budget_preset_key_override?: string;
}

interface RetryPanelProps {
  teacherConfigs: ModelConfigResponse[];
  guidanceVersions: GuidanceResponse[];
  currentTeacherId: string;
  currentGuidanceId: string;
  currentPreset: string;
  currentThinking: boolean;
  currentVisualBudget: string;
  onRetry: (overrides: RetryOverrides) => void;
  onCancel: () => void;
}

// ── Component ────────────────────────────────────────────────────────────────

export function RetryPanel({
  teacherConfigs,
  guidanceVersions,
  currentTeacherId,
  currentGuidanceId,
  currentPreset,
  currentThinking,
  currentVisualBudget,
  onRetry,
  onCancel,
}: RetryPanelProps) {
  const [teacherId, setTeacherId] = useState(currentTeacherId);
  const [guidanceId, setGuidanceId] = useState(currentGuidanceId);
  const [preset, setPreset] = useState(currentPreset);
  const [thinking, setThinking] = useState(currentThinking);
  const [visualBudget, setVisualBudget] = useState(currentVisualBudget);

  // Look up selected teacher to determine control visibility
  const selectedTeacher = useMemo(
    () => teacherConfigs.find((mc) => mc.model_config_id === teacherId),
    [teacherConfigs, teacherId],
  );

  const thinkingVisible = thinkingToggleVisible(selectedTeacher);
  const visualBudgetShown = visualBudgetVisible(selectedTeacher);

  function handleRetry() {
    const overrides: RetryOverrides = {};
    if (teacherId !== currentTeacherId)
      overrides.teacher_model_config_id_override = teacherId;
    if (guidanceId !== currentGuidanceId) overrides.guidance_id_override = guidanceId;
    if (preset !== currentPreset) overrides.generation_preset_key_override = preset;
    if (thinking !== currentThinking)
      overrides.thinking_mode_override = thinking ? "on" : "off";
    if (visualBudget !== currentVisualBudget)
      overrides.visual_budget_preset_key_override = visualBudget;
    onRetry(overrides);
  }

  return (
    <div className="flex flex-1 flex-col gap-4 fade-in" data-testid="retry-panel">
      <Text kind="label/bold/sm" style={{ color: "var(--text-primary)" }}>
        Retry with different settings
      </Text>

      {/* Teacher */}
      <div data-testid="retry-teacher">
        <Text
          kind="label/regular/xs"
          style={{ color: "var(--text-muted)" }}
          className="mb-1 block"
        >
          Teacher
        </Text>
        <select
          className="glass-input w-full px-3 py-2 text-sm"
          value={teacherId}
          onChange={(e) => setTeacherId(e.target.value)}
          data-testid="retry-teacher-select"
        >
          {teacherConfigs
            .filter((mc) => mc.availability?.available ?? true)
            .map((mc) => (
              <option key={mc.model_config_id} value={mc.model_config_id}>
                {mc.model_name}
                {mc.model_config_id === currentTeacherId ? " (current)" : ""}
              </option>
            ))}
        </select>
      </div>

      {/* Guidance */}
      <div data-testid="retry-guidance">
        <Text
          kind="label/regular/xs"
          style={{ color: "var(--text-muted)" }}
          className="mb-1 block"
        >
          Guidance
        </Text>
        <select
          className="glass-input w-full px-3 py-2 text-sm"
          value={guidanceId}
          onChange={(e) => setGuidanceId(e.target.value)}
          data-testid="retry-guidance-select"
        >
          {guidanceVersions.map((g) => (
            <option key={g.guidance_id} value={g.guidance_id}>
              v{g.version_number}
              {g.guidance_id === currentGuidanceId ? " (current)" : ""}
            </option>
          ))}
        </select>
      </div>

      {/* Stability — shared segmented pill; label matches the top bar's
          short form so the two surfaces name the setting identically */}
      <div data-testid="retry-stability">
        <Text
          kind="label/regular/xs"
          style={{ color: "var(--text-muted)" }}
          className="mb-1 block"
        >
          Stability
        </Text>
        <SegmentedControl
          testId="retry-stability-group"
          options={[
            { key: "precise", label: "Precise" },
            { key: "explore", label: "Explore" },
          ]}
          value={preset}
          onChange={setPreset}
        />
      </div>

      {/* Thinking */}
      {thinkingVisible && (
        <div data-testid="retry-thinking">
          <Text
            kind="label/regular/xs"
            style={{ color: "var(--text-muted)" }}
            className="mb-1 block"
          >
            Thinking
          </Text>
          <SegmentedControl
            testId="retry-thinking-group"
            options={[
              { key: "on", label: "ON" },
              { key: "off", label: "OFF" },
            ]}
            value={thinking ? "on" : "off"}
            onChange={(k) => setThinking(k === "on")}
          />
        </div>
      )}

      {/* Detail — label and option names match TopBarControls */}
      {visualBudgetShown && (
        <div data-testid="retry-visual-budget">
          <Text
            kind="label/regular/xs"
            style={{ color: "var(--text-muted)" }}
            className="mb-1 block"
          >
            Detail
          </Text>
          <SegmentedControl
            testId="retry-vb-group"
            options={[
              { key: "fast", label: "Fast" },
              { key: "balanced", label: "Balanced" },
              { key: "high_detail", label: "High" },
            ]}
            value={visualBudget}
            onChange={setVisualBudget}
          />
        </div>
      )}

      {/* Actions — bottom-anchored (mt-auto) so Cancel/Retry sit where the
          proposal action row lives instead of floating mid-card */}
      <div
        className="mt-auto flex items-center justify-end gap-3 border-t pt-3"
        style={{ borderColor: "var(--glass-border)" }}
      >
        <Button kind="secondary" onClick={onCancel} data-testid="retry-cancel-btn">
          Cancel
        </Button>
        <Button
          kind="primary"
          className="nvidia-green-button"
          onClick={handleRetry}
          data-testid="retry-confirm-btn"
        >
          Retry
        </Button>
      </div>
    </div>
  );
}

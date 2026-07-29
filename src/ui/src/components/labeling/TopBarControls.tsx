// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Top bar inference controls for the labeling screen.
 *
 * Output Stability (Precise/Explore), Thinking toggle (ON/OFF),
 * Visual Budget (Fast/Balanced/High Detail). Controls hide based on
 * the active teacher model's capabilities.
 */

import { SegmentedControl } from "@/components/SegmentedControl";

// ── Props ────────────────────────────────────────────────────────────────────

interface TopBarControlsProps {
  generationPreset: string;
  thinkingOn: boolean;
  visualBudgetPreset: string;
  thinkingVisible: boolean;
  visualBudgetVisible: boolean;
  disabled?: boolean;
  onGenerationPresetChange: (key: string) => void;
  onThinkingChange: (on: boolean) => void;
  onVisualBudgetChange: (key: string) => void;
}

// ── Component ────────────────────────────────────────────────────────────────

export function TopBarControls({
  generationPreset,
  thinkingOn,
  visualBudgetPreset,
  thinkingVisible,
  visualBudgetVisible,
  disabled = false,
  onGenerationPresetChange,
  onThinkingChange,
  onVisualBudgetChange,
}: TopBarControlsProps) {
  return (
    <div
      className="flex items-center gap-4 transition-opacity"
      style={{ opacity: disabled ? 0.5 : 1 }}
      data-testid="top-bar-controls-inner"
    >
      {/* Output Stability */}
      <SegmentedControl
        testId="output-stability"
        label="Stability"
        options={[
          { key: "precise", label: "Precise" },
          { key: "explore", label: "Explore" },
        ]}
        value={generationPreset}
        disabled={disabled}
        onChange={onGenerationPresetChange}
      />

      {/* Thinking toggle */}
      {thinkingVisible && (
        <SegmentedControl
          testId="thinking-toggle"
          label="Thinking"
          options={[
            { key: "on", label: "ON" },
            { key: "off", label: "OFF" },
          ]}
          value={thinkingOn ? "on" : "off"}
          disabled={disabled}
          onChange={(k) => onThinkingChange(k === "on")}
        />
      )}

      {/* Visual Budget */}
      {visualBudgetVisible && (
        <SegmentedControl
          testId="visual-budget"
          label="Detail"
          options={[
            { key: "fast", label: "Fast" },
            { key: "balanced", label: "Balanced" },
            { key: "high_detail", label: "High" },
          ]}
          value={visualBudgetPreset}
          disabled={disabled}
          onChange={onVisualBudgetChange}
        />
      )}
    </div>
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Base-model checkbox selector for the Student Training screen.
 *
 * Every ``student_base`` catalog entry is selectable. Entries whose TAO base
 * experiment is missing are identified as first-run provisioning work.
 */

import { Text } from "@kui/react";

import { formatModelDisplayName } from "@/lib/model-display";

interface BaseModelOption {
  modelConfigId: string;
  modelName: string;
  provisioned: boolean;
}

interface BaseModelSelectorProps {
  options: BaseModelOption[];
  selected: string[];
  onChange: (selected: string[]) => void;
  disabled?: boolean;
}

export function BaseModelSelector({
  options,
  selected,
  onChange,
  disabled = false,
}: BaseModelSelectorProps) {
  function toggle(modelConfigId: string) {
    if (selected.includes(modelConfigId)) {
      onChange(selected.filter((id) => id !== modelConfigId));
    } else {
      onChange([...selected, modelConfigId]);
    }
  }

  return (
    <div className="flex flex-col gap-2" data-testid="training-base-model-selector">
      <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
        Each selected base trains its own Student; selecting more than one gives you a
        size/accuracy comparison.
      </Text>
      {options.map((opt) => {
        const checked = selected.includes(opt.modelConfigId);
        return (
          <label
            key={opt.modelConfigId}
            className={`flex items-center gap-3 ${
              disabled ? "cursor-not-allowed opacity-70" : "cursor-pointer"
            }`}
          >
            <input
              type="checkbox"
              className="glass-input"
              checked={checked}
              disabled={disabled}
              onChange={() => toggle(opt.modelConfigId)}
              data-testid={`base-model-checkbox-${opt.modelConfigId}`}
            />
            <Text kind="body/regular/sm">{formatModelDisplayName(opt.modelName)}</Text>
            {!opt.provisioned && (
              <Text
                kind="body/regular/xs"
                style={{ color: "var(--text-muted)" }}
                data-testid={`base-model-first-run-${opt.modelConfigId}`}
              >
                Will be provisioned in Training Jobs
              </Text>
            )}
          </label>
        );
      })}
    </div>
  );
}

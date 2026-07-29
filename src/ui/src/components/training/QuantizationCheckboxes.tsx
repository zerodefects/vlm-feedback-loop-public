// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Quantization scheme checkbox group for the Student Training screen.
 *
 * The parent owns the selection policy. The first-run validation workflow
 * selects FP8_DYNAMIC only; deselecting everything means "train + evaluate
 * baseline only — skip quantization".
 */

import { Text } from "@kui/react";

import type { QuantizationScheme } from "@/types/training";

interface QuantizationCheckboxesProps {
  schemes: QuantizationScheme[];
  onChange: (schemes: QuantizationScheme[]) => void;
  disabled?: boolean;
}

interface SchemeRow {
  key: QuantizationScheme;
  label: string;
  description: string;
}

const QUANT_SCHEMES: readonly SchemeRow[] = [
  {
    key: "FP8_DYNAMIC",
    label: "FP8 Dynamic",
    description: "Broadly compatible, modest quality trade-off.",
  },
  { key: "W8A8", label: "W8A8", description: "8-bit weights + activations." },
  {
    key: "W8A16",
    label: "W8A16",
    description: "8-bit weights, 16-bit activations.",
  },
  {
    key: "W4A16",
    label: "W4A16",
    description: "4-bit weights, most aggressive compression.",
  },
];

export function QuantizationCheckboxes({
  schemes,
  onChange,
  disabled = false,
}: QuantizationCheckboxesProps) {
  function toggle(scheme: QuantizationScheme) {
    if (schemes.includes(scheme)) {
      onChange(schemes.filter((s) => s !== scheme));
    } else {
      onChange([...schemes, scheme]);
    }
  }

  return (
    <div className="flex flex-col gap-2" data-testid="training-quantization-checkboxes">
      {QUANT_SCHEMES.map((row) => {
        const checked = schemes.includes(row.key);
        return (
          <label
            key={row.key}
            className={`flex items-start gap-3 ${
              disabled ? "cursor-not-allowed opacity-70" : "cursor-pointer"
            }`}
          >
            <input
              type="checkbox"
              className="glass-input mt-1"
              checked={checked}
              disabled={disabled}
              onChange={() => toggle(row.key)}
              data-testid={`quant-checkbox-${row.key}`}
            />
            <div className="flex flex-col">
              <Text kind="body/regular/sm">{row.label}</Text>
              <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
                {row.description}
              </Text>
            </div>
          </label>
        );
      })}
      {schemes.length === 0 && (
        <Text
          kind="body/regular/xs"
          style={{ color: "var(--warning-amber, #f59e0b)" }}
          data-testid="quantization-skip-notice"
        >
          No quantization scheme selected. The suite will train and evaluate the
          full-precision baseline only.
        </Text>
      )}
    </div>
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Advanced (read-only) hyperparameter JSON viewer for the Student
 * Training screen.
 *
 * Renders the server-resolved patches from the training-preset resolver,
 * so the SME sees exactly what TAO
 * will receive. A frontend mirror of the backend resolver used to live
 * here and drifted (wrong ``max_keep``, wrong cosmos3-super epochs) —
 * the backend is the single source of truth.
 */

import { Text } from "@kui/react";
import { ChevronRight } from "lucide-react";

import { formatModelDisplayName } from "@/lib/model-display";
import type { ResolvedTrainingPatch, TrainingPreset } from "@/types/training";

interface TrainingAdvancedExpanderProps {
  preset: TrainingPreset;
  /** Display-name-labelled list of base models to preview. */
  baseModels: Array<{ modelConfigId: string; modelName: string }>;
  /** Server-resolved patches for the selected bases. */
  resolvedPresets:
    | Record<string, Record<TrainingPreset, ResolvedTrainingPatch>>
    | undefined;
}

export function TrainingAdvancedExpander({
  preset,
  baseModels,
  resolvedPresets,
}: TrainingAdvancedExpanderProps) {
  return (
    <details
      className="group border"
      style={{
        borderColor: "var(--glass-border, rgba(255,255,255,0.08))",
        borderRadius: "var(--glass-radius-sm, 14px)",
      }}
      data-testid="training-advanced-expander"
    >
      <summary
        className="flex cursor-pointer select-none items-center gap-2 px-4 py-2 text-sm list-none [&::-webkit-details-marker]:hidden"
        style={{ color: "var(--text-secondary)" }}
      >
        <ChevronRight
          size={16}
          className="transition-transform duration-[140ms] group-open:rotate-90"
          aria-hidden
        />
        Advanced (resolved hyperparameters)
      </summary>
      <div className="flex flex-col gap-3 px-4 pb-4">
        {baseModels.length === 0 && (
          <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
            Select at least one base model to see the resolved hyperparameters.
          </Text>
        )}
        {baseModels.map(({ modelConfigId, modelName }) => {
          const patch = resolvedPresets?.[modelConfigId]?.[preset];
          if (!patch) {
            return (
              <div key={modelConfigId} className="flex flex-col gap-1">
                <Text kind="label/bold/xs" style={{ color: "var(--text-muted)" }}>
                  {formatModelDisplayName(modelName)}
                </Text>
                <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
                  Resolving hyperparameters…
                </Text>
              </div>
            );
          }
          return (
            <div key={modelConfigId} className="flex flex-col gap-1">
              <Text kind="label/bold/xs" style={{ color: "var(--text-muted)" }}>
                {formatModelDisplayName(modelName)}
              </Text>
              <pre
                className="overflow-auto border p-4 text-xs leading-relaxed"
                style={{
                  background: "rgba(0, 0, 0, 0.3)",
                  borderColor: "rgba(255, 255, 255, 0.08)",
                  borderRadius: "var(--glass-radius-sm, 14px)",
                  color: "var(--text-secondary)",
                  // Tall enough for a full resolved patch (~18 lines);
                  // overflow-auto covers anything larger.
                  maxHeight: "400px",
                }}
                data-testid={`training-advanced-json-${modelConfigId}`}
              >
                {JSON.stringify(patch, null, 2)}
              </pre>
            </div>
          );
        })}
        <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
          Read-only. These values are applied deterministically by the backend.
        </Text>
      </div>
    </details>
  );
}

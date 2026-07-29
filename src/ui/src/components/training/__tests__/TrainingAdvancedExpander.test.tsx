// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { TrainingAdvancedExpander } from "../TrainingAdvancedExpander";
import type { ResolvedTrainingPatch, TrainingPreset } from "@/types/training";

function patch(epoch: number): ResolvedTrainingPatch {
  return {
    train: {
      epoch,
      resume: false,
      ckpt: {
        enable_checkpoint: true,
        save_freq_in_epoch: 1,
        max_keep: 1,
        export_safetensors: true,
      },
    },
  };
}

function patches(
  epochs: Partial<Record<TrainingPreset, number>>,
): Record<TrainingPreset, ResolvedTrainingPatch> {
  return {
    quick: patch(epochs.quick ?? 1),
    standard: patch(epochs.standard ?? 3),
    high_quality: patch(epochs.high_quality ?? 9),
    max_quality: patch(epochs.max_quality ?? 18),
  };
}

describe("TrainingAdvancedExpander", () => {
  const BASES = [
    { modelConfigId: "mc-8b", modelName: "nvidia/cosmos-reason2-8b" },
    { modelConfigId: "mc-2b", modelName: "nvidia/cosmos-reason2-2b" },
  ];
  // Server-resolved patches (the component renders these VERBATIM — it
  // no longer recomputes them; the old frontend mirror drifted).
  const RESOLVED = {
    "mc-8b": patches({}),
    "mc-2b": patches({ high_quality: 12, max_quality: 24 }),
  };

  it("renders the server-resolved JSON per base model", () => {
    render(
      <TrainingAdvancedExpander
        preset="standard"
        baseModels={BASES}
        resolvedPresets={RESOLVED}
      />,
    );
    const json8b = screen.getByTestId("training-advanced-json-mc-8b").textContent ?? "";
    const json2b = screen.getByTestId("training-advanced-json-mc-2b").textContent ?? "";
    expect(json8b).toContain('"epoch": 3');
    expect(json2b).toContain('"epoch": 3');
    expect(json8b).toContain('"export_safetensors": true');
    // Drift-sensitive F27 value the old mirror got wrong (it said 8).
    expect(json8b).toContain('"max_keep": 1');
  });

  it("renders per-model preset differences from the server data", () => {
    render(
      <TrainingAdvancedExpander
        preset="max_quality"
        baseModels={BASES}
        resolvedPresets={RESOLVED}
      />,
    );
    expect(
      screen.getByTestId("training-advanced-json-mc-8b").textContent ?? "",
    ).toContain('"epoch": 18');
    expect(
      screen.getByTestId("training-advanced-json-mc-2b").textContent ?? "",
    ).toContain('"epoch": 24');
  });

  it("shows a waiting line until preset resolution completes", () => {
    render(
      <TrainingAdvancedExpander
        preset="standard"
        baseModels={BASES}
        resolvedPresets={undefined}
      />,
    );
    expect(screen.getAllByText(/resolving hyperparameters/i).length).toBeGreaterThan(0);
  });

  it("renders guidance text when no base model is selected", () => {
    render(
      <TrainingAdvancedExpander
        preset="standard"
        baseModels={[]}
        resolvedPresets={RESOLVED}
      />,
    );
    expect(screen.getByText(/select at least one base model/i)).toBeTruthy();
  });
});

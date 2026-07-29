// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * DetectedHardwareEyebrow.
 *
 * Renders a confident "what we saw" header above the setup-choice
 * screen so the SME
 * trusts the recommendation that follows. Green ✓ chip per detected
 * capability (GPU / Docker / Toolkit); red ✗ chip with an install hint
 * link when the prerequisite for local NIM is missing on a GPU machine.
 *
 * Used as the topmost element on the setup-choice screen whenever
 * ``env.gpus.length > 0`` OR a Docker/Toolkit prerequisite is missing
 * (so the SME on a partially-equipped GPU host sees actionable hints).
 * No-GPU machines with Docker and the Toolkit both present skip the
 * eyebrow entirely; the existing "GPU not detected" note in the hosted
 * card carries that signal.
 */

import { Text } from "@kui/react";
import { Check, X } from "lucide-react";

import type { EnvironmentResponse } from "@/types/nim";

interface DetectedHardwareEyebrowProps {
  env: EnvironmentResponse;
}

interface ChipProps {
  ok: boolean;
  label: string;
}

function Chip({ ok, label }: ChipProps): JSX.Element {
  const color = ok ? "var(--accent-green)" : "var(--text-error, #ef4444)";
  return (
    <span
      className="flex items-center gap-1"
      style={{ color }}
      data-testid={`hw-chip-${ok ? "ok" : "missing"}`}
    >
      {ok ? <Check size={14} strokeWidth={2.5} /> : <X size={14} strokeWidth={2.5} />}
      <Text kind="label/bold/xs" style={{ color: "inherit", letterSpacing: "0.04em" }}>
        {label}
      </Text>
    </span>
  );
}

export function DetectedHardwareEyebrow({
  env,
}: DetectedHardwareEyebrowProps): JSX.Element | null {
  const gpu = env.gpus[0];
  const hasGpu = env.gpus.length > 0;

  // No-signal case: no GPU AND no prereq gap to flag → don't render.
  // The setup-choice screen's hosted card already carries the "GPU not detected" line.
  if (!hasGpu && env.docker_available && env.nvidia_toolkit_available) {
    return null;
  }

  return (
    <div
      className="flex flex-wrap items-center gap-x-4 gap-y-1"
      data-testid="detected-hardware-eyebrow"
    >
      {hasGpu && gpu !== undefined && (
        <Chip
          ok={true}
          label={`GPU: ${gpu.name} · ${gpu.memory_total_gb.toFixed(0)} GB`}
        />
      )}
      {!hasGpu && <Chip ok={false} label="GPU not detected" />}
      <Chip
        ok={env.docker_available}
        label={env.docker_available ? "Docker" : "Docker missing"}
      />
      <Chip
        ok={env.nvidia_toolkit_available}
        label={
          env.nvidia_toolkit_available ? "NVIDIA Toolkit" : "NVIDIA Toolkit missing"
        }
      />
    </div>
  );
}

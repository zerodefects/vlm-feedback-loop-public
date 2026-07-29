// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for DetectedHardwareEyebrow (F-amendment NIM-FTU-Local-Peer).
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { DetectedHardwareEyebrow } from "@/components/DetectedHardwareEyebrow";
import { makeEnvironmentResponse } from "@/test/fixtures";
import type { EnvironmentResponse } from "@/types/nim";

function env(overrides: Partial<EnvironmentResponse> = {}): EnvironmentResponse {
  return makeEnvironmentResponse({
    embedding_deployment: {
      model_name: "x",
      nim_container_image: "x",
      gpu_memory_minimum_gb: 0,
      fits: false,
      provider: "none",
    },
    recommended_teacher_mode: "hosted",
    recommended_embedding_mode: "hosted",
    ...overrides,
  });
}

describe("DetectedHardwareEyebrow", () => {
  it("renders nothing when no GPU AND Docker + Toolkit are present (no signal to show)", () => {
    const { container } = render(
      <DetectedHardwareEyebrow
        env={env({ docker_available: true, nvidia_toolkit_available: true })}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders GPU chip with name and memory when a GPU is present", () => {
    render(
      <DetectedHardwareEyebrow
        env={env({
          gpus: [{ name: "NVIDIA A100", memory_total_gb: 80 }],
          docker_available: true,
          nvidia_toolkit_available: true,
        })}
      />,
    );
    expect(screen.getByText(/NVIDIA A100/)).toBeInTheDocument();
    expect(screen.getByText(/80 GB/)).toBeInTheDocument();
  });

  it("renders red Docker missing chip when Docker isn't installed but a GPU is present", () => {
    render(
      <DetectedHardwareEyebrow
        env={env({
          gpus: [{ name: "NVIDIA A100", memory_total_gb: 80 }],
          docker_available: false,
          nvidia_toolkit_available: true,
        })}
      />,
    );
    expect(screen.getByText(/Docker missing/i)).toBeInTheDocument();
    // The Toolkit chip is the green/ok variant.
    expect(screen.getByText(/^NVIDIA Toolkit$/)).toBeInTheDocument();
  });
});

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { RetryPanel } from "@/components/labeling/RetryPanel";
import type { GuidanceResponse } from "@/types/guidance";
import type { ModelConfigResponse } from "@/types/nim";

const teacher: ModelConfigResponse = {
  model_config_id: "teacher-1",
  project_id: "project-1",
  endpoint_id: "endpoint-1",
  model_name: "nvidia/cosmos3-nano-reasoner",
  context_window_tokens: 131072,
  eligible_roles: ["teacher"],
  supports_image_input: true,
  structured_generation_support: "supported",
  thinking_toggle_mode: "request_boolean",
  thinking_toggle_support: "supported",
  visual_budget_mode: "preset",
  visual_budget_support: "supported",
  model_quantization: null,
  nim_model_profile: null,
  nim_profile_metadata: null,
  local_deploy_metadata: null,
  hosted_compatible: false,
  availability: { available: true, reason: null },
  created_at: "2026-08-04T00:00:00Z",
};

const guidance: GuidanceResponse = {
  guidance_id: "guidance-1",
  project_id: "project-1",
  version_number: 1,
  description: "Classify the image.",
  schema_fields: [],
  rules: "Use visible evidence.",
  derived_json_schema: {},
  generation_order: [],
  schema_hash: "schema-hash",
  created_at: "2026-08-04T00:00:00Z",
};

describe("RetryPanel accessibility", () => {
  it("identifies every retry setting and segmented choice group", () => {
    render(
      <RetryPanel
        teacherConfigs={[teacher]}
        guidanceVersions={[guidance]}
        currentTeacherId={teacher.model_config_id}
        currentGuidanceId={guidance.guidance_id}
        currentPreset="precise"
        currentThinking={true}
        currentVisualBudget="balanced"
        onRetry={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByRole("combobox", { name: "Teacher" })).toHaveValue("teacher-1");
    expect(screen.getByRole("combobox", { name: "Guidance" })).toHaveValue(
      "guidance-1",
    );

    const stability = screen.getByRole("group", { name: "Stability" });
    expect(within(stability).getByRole("button", { name: "Precise" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    const thinking = screen.getByRole("group", { name: "Thinking" });
    expect(within(thinking).getByRole("button", { name: "ON" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    const detail = screen.getByRole("group", { name: "Detail" });
    expect(within(detail).getByRole("button", { name: "Balanced" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });
});

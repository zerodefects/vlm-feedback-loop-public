// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";

import { TeacherModelPicker } from "@/components/labeling/TeacherModelPicker";
import type { ModelConfigResponse } from "@/types/nim";

const BASE_CONFIG: ModelConfigResponse = {
  model_config_id: "mc-cosmos",
  project_id: "p",
  endpoint_id: "ep",
  model_name: "nvidia/cosmos-reason2-8b",
  context_window_tokens: 256000,
  eligible_roles: ["teacher"],
  supports_image_input: true,
  structured_generation_support: "unknown",
  thinking_toggle_mode: "qwen_enable_thinking",
  thinking_toggle_support: "supported",
  visual_budget_mode: "mm_processor_size",
  visual_budget_support: "supported",
  model_quantization: null,
  nim_model_profile: null,
  nim_profile_metadata: null,
  local_deploy_metadata: null,
  hosted_compatible: false,
  availability: { available: true, reason: null },
  created_at: "2026-04-14T00:00:00Z",
};

const MISTRAL_CONFIG: ModelConfigResponse = {
  ...BASE_CONFIG,
  model_config_id: "mc-mistral",
  model_name: "mistralai/mistral-large-3-675b-instruct-2512",
  thinking_toggle_mode: "none",
  thinking_toggle_support: "unsupported",
  visual_budget_mode: "none",
  visual_budget_support: "unsupported",
  hosted_compatible: true,
  availability: { available: true, reason: null },
};

function renderPicker(props: Parameters<typeof TeacherModelPicker>[0]) {
  return render(
    <MemoryRouter>
      <TeacherModelPicker {...props} />
    </MemoryRouter>,
  );
}

describe("TeacherModelPicker", () => {
  it("renders an empty-state label when no teacher-eligible entries exist", () => {
    renderPicker({
      teacherConfigs: [],
      currentTeacherId: null,
      onChange: () => {},
      projectId: "p",
    });
    expect(screen.getByTestId("teacher-model-picker-empty")).toHaveTextContent(
      "Teacher",
    );
    expect(screen.queryByTestId("teacher-model-picker-select")).not.toBeInTheDocument();
  });

  it("renders one option per available teacher entry and highlights the current one", () => {
    renderPicker({
      teacherConfigs: [BASE_CONFIG, MISTRAL_CONFIG],
      currentTeacherId: "mc-mistral",
      onChange: () => {},
      projectId: "p",
    });
    const select = screen.getByTestId(
      "teacher-model-picker-select",
    ) as HTMLSelectElement;
    const values = Array.from(select.options).map((o) => o.value);
    expect(values).toEqual(["mc-cosmos", "mc-mistral"]);
    expect(select.value).toBe("mc-mistral");
  });

  it("hides entries whose availability.available is false", () => {
    // Real-world scenario: Cosmos bound to a hosted endpoint is
    // unavailable on an account without an NVCF subscription. The picker
    // drops it without affecting the available options.
    const cosmosUnavailable: ModelConfigResponse = {
      ...BASE_CONFIG,
      availability: { available: false, reason: "hosted_not_compatible" },
    };
    renderPicker({
      teacherConfigs: [cosmosUnavailable, MISTRAL_CONFIG],
      currentTeacherId: "mc-mistral",
      onChange: () => {},
      projectId: "p",
    });
    const select = screen.getByTestId(
      "teacher-model-picker-select",
    ) as HTMLSelectElement;
    const values = Array.from(select.options).map((o) => o.value);
    expect(values).toEqual(["mc-mistral"]);
  });

  it("keeps the selected unavailable Teacher visible without substituting another model", () => {
    const cosmosUnavailable: ModelConfigResponse = {
      ...BASE_CONFIG,
      availability: { available: false, reason: "endpoint_unhealthy" },
    };
    renderPicker({
      teacherConfigs: [cosmosUnavailable, MISTRAL_CONFIG],
      currentTeacherId: "mc-cosmos",
      onChange: () => {},
      projectId: "p123",
    });

    const select = screen.getByTestId(
      "teacher-model-picker-select",
    ) as HTMLSelectElement;
    expect(select.value).toBe("mc-cosmos");
    expect(select.selectedOptions[0]).toHaveTextContent(
      "nvidia/cosmos-reason2-8b (unavailable)",
    );
    expect(select.selectedOptions[0]).toBeDisabled();
    expect(Array.from(select.options).map((option) => option.value)).toEqual([
      "mc-cosmos",
      "mc-mistral",
    ]);
    expect(screen.getByTestId("teacher-model-picker-configure-link")).toHaveAttribute(
      "href",
      "/projects/p123/settings/nim",
    );
  });

  it("keeps an orphaned selected Teacher visible instead of naming the first available model", () => {
    renderPicker({
      teacherConfigs: [MISTRAL_CONFIG],
      currentTeacherId: "mc-retired",
      onChange: () => {},
      projectId: "p",
    });

    const select = screen.getByTestId(
      "teacher-model-picker-select",
    ) as HTMLSelectElement;
    expect(select.value).toBe("mc-retired");
    expect(select.selectedOptions[0]).toHaveTextContent(
      "Selected Teacher (unavailable)",
    );
  });

  it("shows the configure-endpoint empty-state when every entry is unavailable", () => {
    const cosmosUnavailable: ModelConfigResponse = {
      ...BASE_CONFIG,
      availability: { available: false, reason: "hosted_not_compatible" },
    };
    const mistralUnavailable: ModelConfigResponse = {
      ...MISTRAL_CONFIG,
      availability: { available: false, reason: "no_nvidia_api_key" },
    };
    renderPicker({
      teacherConfigs: [cosmosUnavailable, mistralUnavailable],
      currentTeacherId: null,
      onChange: () => {},
      projectId: "p123",
    });
    expect(screen.getByTestId("teacher-model-picker-unavailable")).toBeInTheDocument();
    // The NIM Connection screen is routed at settings/nim (App.tsx); any
    // other target renders a blank page under the header.
    const link = screen.getByTestId("teacher-model-picker-configure-link");
    expect(link).toHaveAttribute("href", "/projects/p123/settings/nim");
    expect(screen.queryByTestId("teacher-model-picker-select")).not.toBeInTheDocument();
  });

  it("shows a placeholder option when no teacher is currently assigned", () => {
    renderPicker({
      teacherConfigs: [BASE_CONFIG],
      currentTeacherId: null,
      onChange: () => {},
      projectId: "p",
    });
    const select = screen.getByTestId(
      "teacher-model-picker-select",
    ) as HTMLSelectElement;
    expect(select.value).toBe("");
    const values = Array.from(select.options).map((o) => o.value);
    expect(values).toEqual(["", "mc-cosmos"]);
  });

  it("invokes onChange with the new model_config_id on selection", async () => {
    const onChange = vi.fn();
    renderPicker({
      teacherConfigs: [BASE_CONFIG, MISTRAL_CONFIG],
      currentTeacherId: "mc-cosmos",
      onChange,
      projectId: "p",
    });
    await userEvent
      .setup()
      .selectOptions(screen.getByTestId("teacher-model-picker-select"), "mc-mistral");
    expect(onChange).toHaveBeenCalledWith("mc-mistral");
  });

  it("exposes an accessible Teacher-model aria label", () => {
    renderPicker({
      teacherConfigs: [BASE_CONFIG],
      currentTeacherId: "mc-cosmos",
      onChange: () => {},
      projectId: "p",
    });
    expect(screen.getByLabelText("Teacher model")).toBe(
      screen.getByTestId("teacher-model-picker-select"),
    );
  });
});

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

import { BaseModelSelector } from "../BaseModelSelector";

describe("BaseModelSelector", () => {
  it("describes an unavailable TAO base experiment as future provisioning work", () => {
    render(
      <BaseModelSelector
        options={[
          {
            modelConfigId: "mc-8b",
            modelName: "nvidia/cosmos-reason2-8b",
            provisioned: false,
          },
        ]}
        selected={[]}
        onChange={vi.fn()}
      />,
    );

    const helper = screen.getByTestId("base-model-first-run-mc-8b");
    expect(helper.textContent).toBe("Will be provisioned in Training Jobs");
    expect(screen.queryByText(/^Provisioned in Training Jobs$/)).toBeNull();
  });

  it("does not show provisioning guidance for an already available base", () => {
    render(
      <BaseModelSelector
        options={[
          {
            modelConfigId: "mc-2b",
            modelName: "nvidia/cosmos-reason2-2b",
            provisioned: true,
          },
        ]}
        selected={["mc-2b"]}
        onChange={vi.fn()}
      />,
    );

    expect(screen.queryByTestId("base-model-first-run-mc-2b")).toBeNull();
  });
});

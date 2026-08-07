// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  RationalePanel,
  type RationalePanelState,
} from "@/components/labeling/RationalePanel";

function renderPanel(
  state: RationalePanelState,
  regenerationError: string | null = null,
) {
  render(
    <RationalePanel
      state={state}
      rationaleText="Visible dent on the front edge."
      onRationaleTextChange={vi.fn()}
      onGenerateAI={vi.fn()}
      onApproveAI={vi.fn()}
      regenerationError={regenerationError}
    />,
  );
}

describe("RationalePanel accessibility", () => {
  it("names the SME rationale editor and associates the edit-review guidance", () => {
    renderPanel("needs_review");

    expect(
      screen.getByRole("textbox", { name: "Rationale" }),
    ).toHaveAccessibleDescription("Your label changed. Update the rationale to match.");
  });

  it("associates both AI-review guidance and a regeneration failure", () => {
    renderPanel("ai_review_required", "The Teacher could not regenerate a rationale.");

    expect(
      screen.getByRole("textbox", { name: "Rationale" }),
    ).toHaveAccessibleDescription(
      "Review the AI-generated rationale before saving. The Teacher could not regenerate a rationale.",
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "The Teacher could not regenerate a rationale.",
    );
  });

  it("replaces the editor with an identified progress state while regenerating", () => {
    renderPanel("regenerating");

    expect(
      screen.queryByRole("textbox", { name: "Rationale" }),
    ).not.toBeInTheDocument();
    expect(screen.getByLabelText("Generating rationale")).toBeInTheDocument();
  });
});

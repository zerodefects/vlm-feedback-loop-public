// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for SetupAutoSkipBanner — the acknowledgment banner shown on
 * the landing screen after a fully-auto-skipped FTUE setup chain.
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import { SetupAutoSkipBanner } from "@/components/common/SetupAutoSkipBanner";
import type { SetupAutoSkipState } from "@/types/setupChain";

function renderAtReady(state?: { setupAutoSkip: SetupAutoSkipState }) {
  render(
    <MemoryRouter initialEntries={[{ pathname: "/projects/test-pid/ready", state }]}>
      <Routes>
        <Route path="/projects/:projectId">
          <Route
            path="ready"
            element={
              <>
                <SetupAutoSkipBanner />
                <div data-testid="ready-page" />
              </>
            }
          />
          <Route
            path="settings/nim"
            element={<div data-testid="nim-settings-page" />}
          />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

function makePayload(overrides: Partial<SetupAutoSkipState> = {}): SetupAutoSkipState {
  return {
    teacherMode: "hosted",
    embeddingMode: "hosted",
    teacherName: "mistralai/mistral-large-3-675b-instruct-2512",
    ...overrides,
  };
}

describe("SetupAutoSkipBanner", () => {
  it("renders nothing when router state carries no setupAutoSkip payload", () => {
    // A plain visit (reload, deep-link, revisit after dismiss) must not
    // resurrect the banner — the payload lives only in router state.
    renderAtReady(undefined);

    expect(screen.getByTestId("ready-page")).toBeInTheDocument();
    expect(screen.queryByTestId("setup-auto-skip-banner")).toBeNull();
  });

  it("hosted-only chain: states that both Teacher and embeddings use hosted NIM", () => {
    renderAtReady({ setupAutoSkip: makePayload() });

    const banner = screen.getByTestId("setup-auto-skip-banner");
    expect(banner).toHaveTextContent("Setup completed automatically");
    expect(banner).toHaveTextContent("Using hosted NIM for Teacher and embeddings.");
  });

  it("hosted Teacher + local embeddings chain: names each service's destination", () => {
    renderAtReady({ setupAutoSkip: makePayload({ embeddingMode: "local" }) });

    expect(screen.getByTestId("setup-auto-skip-banner")).toHaveTextContent(
      "Teacher: hosted NIM. Embeddings: deploying locally.",
    );
  });

  it("all-local auto-skip reports that both local NIMs are ready", () => {
    renderAtReady({
      setupAutoSkip: makePayload({ teacherMode: "local", embeddingMode: "local" }),
    });

    expect(screen.getByTestId("setup-auto-skip-banner")).toHaveTextContent(
      "Using local NIMs for Teacher and embeddings.",
    );
  });

  it("reused local Teacher + hosted embeddings reports the resident as running", () => {
    renderAtReady({
      setupAutoSkip: makePayload({
        teacherMode: "local",
        embeddingMode: "hosted",
      }),
    });

    expect(screen.getByTestId("setup-auto-skip-banner")).toHaveTextContent(
      "Teacher: local NIM already running. Embeddings: hosted NIM.",
    );
  });

  it("shows the auto-confirmed model defaults line", () => {
    renderAtReady({
      setupAutoSkip: makePayload({
        teacherName: "nvidia/cosmos-reason2-8b",
      }),
    });

    expect(screen.getByTestId("setup-auto-skip-banner")).toHaveTextContent(
      "Using defaults: Teacher = nvidia/cosmos-reason2-8b.",
    );
  });

  it("omits the defaults line when the model name could not be resolved", () => {
    renderAtReady({ setupAutoSkip: makePayload({ teacherName: null }) });

    const banner = screen.getByTestId("setup-auto-skip-banner");
    expect(banner).toHaveTextContent("Setup completed automatically");
    expect(banner).not.toHaveTextContent("Using defaults:");
  });

  it("dismiss clears the payload from router state and hides the banner in place", async () => {
    const user = userEvent.setup();
    renderAtReady({ setupAutoSkip: makePayload() });

    await user.click(screen.getByTestId("setup-auto-skip-banner-dismiss"));

    // Replace-navigation to the same route: the screen stays, the
    // banner is gone.
    expect(screen.getByTestId("ready-page")).toBeInTheDocument();
    expect(screen.queryByTestId("setup-auto-skip-banner")).toBeNull();
  });

  it("[Review] navigates to the NIM Configuration screen", async () => {
    const user = userEvent.setup();
    renderAtReady({ setupAutoSkip: makePayload() });

    await user.click(screen.getByRole("button", { name: "Review" }));

    expect(await screen.findByTestId("nim-settings-page")).toBeInTheDocument();
  });
});

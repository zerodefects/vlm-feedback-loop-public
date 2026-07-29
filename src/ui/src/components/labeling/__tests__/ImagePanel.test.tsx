// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ImagePanel } from "@/components/labeling/ImagePanel";

describe("ImagePanel zoom controls", () => {
  it("changes wheel zoom by one incremental step", async () => {
    render(
      <ImagePanel
        projectId="project-1"
        exampleKey="image-1.png"
        isLoading={false}
        onImageMissing={vi.fn()}
      />,
    );

    const imageViewer = screen.getByLabelText("Zoomable labeling image");

    fireEvent.wheel(imageViewer, { deltaY: -100 });

    await waitFor(() => {
      expect(screen.getByTestId("zoom-level")).toHaveTextContent("115%");
    });

    fireEvent.wheel(imageViewer, { deltaY: 100 });

    await waitFor(() => {
      expect(screen.getByTestId("zoom-level")).toHaveTextContent("100%");
    });
  });

  it("zooms in and resets to the fitted view", async () => {
    const user = userEvent.setup();

    render(
      <ImagePanel
        projectId="project-1"
        exampleKey="image-1.png"
        isLoading={false}
        onImageMissing={vi.fn()}
      />,
    );

    expect(screen.getByRole("group", { name: "Image zoom controls" })).toBeVisible();
    expect(screen.getByTestId("zoom-level")).toHaveTextContent("100%");
    expect(screen.getByRole("button", { name: "Zoom out" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Reset zoom" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Zoom in" }));

    await waitFor(() => {
      expect(screen.getByTestId("zoom-level")).not.toHaveTextContent("100%");
    });
    expect(screen.getByRole("button", { name: "Zoom out" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Reset zoom" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Reset zoom" }));

    await waitFor(() => {
      expect(screen.getByTestId("zoom-level")).toHaveTextContent("100%");
      expect(screen.getByRole("button", { name: "Zoom out" })).toBeDisabled();
      expect(screen.getByRole("button", { name: "Reset zoom" })).toBeDisabled();
    });
  });

  it("starts each new image at the fitted view", async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <ImagePanel
        projectId="project-1"
        exampleKey="image-1.png"
        isLoading={false}
        onImageMissing={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Zoom in" }));
    await waitFor(() => {
      expect(screen.getByTestId("zoom-level")).not.toHaveTextContent("100%");
    });

    rerender(
      <ImagePanel
        projectId="project-1"
        exampleKey="image-2.png"
        isLoading={false}
        onImageMissing={vi.fn()}
      />,
    );

    expect(screen.getByTestId("zoom-level")).toHaveTextContent("100%");
    expect(screen.getByRole("button", { name: "Zoom out" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Reset zoom" })).toBeDisabled();
  });
});

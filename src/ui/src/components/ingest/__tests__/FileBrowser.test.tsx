// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for FileBrowser — pins the breadcrumb to the backend-selected
 * browse root so configured deployments never render escape links above it.
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { FileBrowser } from "@/components/ingest/FileBrowser";
import type { BrowseEntry } from "@/types/filesystem";

const entries: BrowseEntry[] = [
  { name: "folder_a", type: "directory", path: "/root/folder_a", size_bytes: null },
  {
    name: "img_001.jpg",
    type: "file",
    path: "/root/img_001.jpg",
    size_bytes: 2_097_152,
  },
];

function renderAt(
  currentPath: string,
  rootPath = "/",
  onNavigate = vi.fn(),
  listedEntries = entries,
) {
  render(
    <FileBrowser
      entries={listedEntries}
      rootPath={rootPath}
      currentPath={currentPath}
      selectedPaths={new Set()}
      onNavigate={onNavigate}
      onToggleSelect={vi.fn()}
    />,
  );
  return onNavigate;
}

describe("FileBrowser breadcrumb", () => {
  it("hides the breadcrumb at filesystem root", () => {
    renderAt("/");
    expect(screen.queryByTestId("breadcrumb")).toBeNull();
  });

  it("renders a clickable segment trail at depth >= 1", async () => {
    const onNavigate = renderAt("/data/images");
    const user = userEvent.setup();

    const breadcrumb = screen.getByTestId("breadcrumb");
    expect(breadcrumb).toHaveTextContent("data");
    expect(breadcrumb).toHaveTextContent("images");

    await user.click(screen.getByRole("button", { name: "data" }));
    expect(onNavigate).toHaveBeenCalledWith("/data");

    await user.click(screen.getByRole("button", { name: "/" }));
    expect(onNavigate).toHaveBeenCalledWith("/");
  });

  it("hides the breadcrumb at a configured image root", () => {
    renderAt("/data/images", "/data/images");
    expect(screen.queryByTestId("breadcrumb")).toBeNull();
  });

  it("never renders ancestors above a configured image root", async () => {
    const onNavigate = renderAt("/data/images/batch-01", "/data/images");
    const user = userEvent.setup();

    const breadcrumb = screen.getByTestId("breadcrumb");
    expect(breadcrumb).toHaveTextContent("images");
    expect(breadcrumb).toHaveTextContent("batch-01");
    expect(screen.queryByRole("button", { name: "/" })).toBeNull();
    expect(screen.queryByRole("button", { name: "data" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "images" }));
    expect(onNavigate).toHaveBeenCalledWith("/data/images");
  });
});

describe("FileBrowser selection controls", () => {
  it("names directory and file checkboxes for assistive technology", () => {
    renderAt("/data/images", "/data/images", vi.fn(), [
      {
        name: "class-a",
        path: "/data/images/class-a",
        type: "directory",
        size_bytes: 0,
      },
      {
        name: "sample.png",
        path: "/data/images/sample.png",
        type: "file",
        size_bytes: 1024,
      },
    ]);

    expect(
      screen.getByRole("checkbox", { name: "Select directory class-a" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("checkbox", { name: "Select file sample.png" }),
    ).toBeInTheDocument();
  });
});

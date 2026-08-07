// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { IclChip } from "../IclChip";

describe("IclChip", () => {
  it("does not imply context selection when no proposal is active", () => {
    render(<IclChip count={null} idle />);
    expect(screen.getByTestId("icl-chip-idle")).toHaveTextContent(
      "ICL: no active proposal",
    );
    expect(screen.queryByTestId("icl-chip-pending")).not.toBeInTheDocument();
  });

  it("does not claim cold start while the next proposal is selecting context", () => {
    render(<IclChip count={null} />);
    expect(screen.getByTestId("icl-chip-pending")).toBeInTheDocument();
    expect(screen.getByText("ICL: selecting context…")).toBeInTheDocument();
    expect(screen.queryByText("ICL: no edits yet")).not.toBeInTheDocument();
  });

  it("renders cold-start state when count is 0", () => {
    render(<IclChip count={0} />);
    expect(screen.getByTestId("icl-chip-coldstart")).toBeInTheDocument();
    expect(screen.getByText("ICL: no edits yet")).toBeInTheDocument();
  });

  it("renders cold-start state when count is negative", () => {
    // Defensive: count <= 0 collapses to coldstart even if a stale
    // proposal returns -1.
    render(<IclChip count={-1} />);
    expect(screen.getByTestId("icl-chip-coldstart")).toBeInTheDocument();
  });

  it("renders active state with singular noun for count=1", () => {
    render(<IclChip count={1} />);
    expect(screen.getByTestId("icl-chip-active")).toBeInTheDocument();
    expect(screen.getByText("ICL: 1 edit in context")).toBeInTheDocument();
  });

  it("renders active state with plural noun for count>1", () => {
    render(<IclChip count={5} />);
    expect(screen.getByTestId("icl-chip-active")).toBeInTheDocument();
    expect(screen.getByText("ICL: 5 edits in context")).toBeInTheDocument();
  });
});

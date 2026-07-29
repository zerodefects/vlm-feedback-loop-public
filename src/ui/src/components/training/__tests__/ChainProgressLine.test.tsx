// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { ChainProgressLine } from "../ChainProgressLine";

describe("ChainProgressLine", () => {
  // No model-label prefix: the section heading beside this line already
  // names the base model, so the line carries only the progress text.
  it("renders 'done' when all jobs succeeded", () => {
    render(<ChainProgressLine succeeded={6} total={6} />);
    expect(screen.getByTestId("chain-progress-line").textContent).toBe("done");
  });

  it("renders 'N of M' while jobs remain", () => {
    render(<ChainProgressLine succeeded={5} total={6} />);
    expect(screen.getByTestId("chain-progress-line").textContent).toBe("5 of 6");
  });

  it("renders em-dash for zero-total", () => {
    render(<ChainProgressLine succeeded={0} total={0} />);
    expect(screen.getByTestId("chain-progress-line").textContent).toBe("—");
  });
});

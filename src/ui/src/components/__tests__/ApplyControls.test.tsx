// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for ApplyControls.
 *
 * Verifies the two-checkbox cluster behavior:
 *   - "Apply this session" is always rendered and disabled (always on).
 *   - "Save to .env" is shown only when ``allowPersist`` is true.
 *   - Toggling persist propagates through ``onChange``.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { ApplyControls } from "@/components/ApplyControls";

describe("ApplyControls", () => {
  it("renders the persist checkbox when allowPersist=true", () => {
    render(
      <ApplyControls
        allowPersist={true}
        value={{ persist: false }}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByTestId("apply-session-checkbox")).toBeInTheDocument();
    expect(screen.getByTestId("apply-persist-checkbox")).toBeInTheDocument();
  });

  it("hides the persist checkbox when allowPersist=false", () => {
    render(
      <ApplyControls
        allowPersist={false}
        value={{ persist: false }}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByTestId("apply-session-checkbox")).toBeInTheDocument();
    expect(screen.queryByTestId("apply-persist-checkbox")).not.toBeInTheDocument();
  });

  it("'Apply this session' is always checked and disabled", () => {
    render(
      <ApplyControls
        allowPersist={true}
        value={{ persist: false }}
        onChange={vi.fn()}
      />,
    );
    const sessionCheckbox = screen.getByTestId(
      "apply-session-checkbox",
    ) as HTMLInputElement;
    expect(sessionCheckbox.checked).toBe(true);
    expect(sessionCheckbox.disabled).toBe(true);
  });

  it("toggling persist fires onChange with persist=true", () => {
    const onChange = vi.fn();
    render(
      <ApplyControls
        allowPersist={true}
        value={{ persist: false }}
        onChange={onChange}
      />,
    );
    const persistCheckbox = screen.getByTestId(
      "apply-persist-checkbox",
    ) as HTMLInputElement;
    fireEvent.click(persistCheckbox);
    expect(onChange).toHaveBeenCalledWith({ persist: true });
  });
});

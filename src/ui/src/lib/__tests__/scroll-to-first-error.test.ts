// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { scrollToFirstError } from "../scroll-to-first-error";

describe("scrollToFirstError", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    document.body.innerHTML = "";
  });

  it("scrolls the first error-testid node into view after a tick", () => {
    const container = document.createElement("div");
    container.innerHTML = `
      <div data-testid="error-NO_CORE_FIELDS"></div>
      <div data-testid="error-MISSING_FIELD_NAME"></div>
    `;
    document.body.appendChild(container);
    const first = container.querySelector('[data-testid="error-NO_CORE_FIELDS"]')!;
    const scrollIntoView = vi.fn();
    (first as HTMLElement).scrollIntoView = scrollIntoView;

    scrollToFirstError({ current: container });
    // Deferred: nothing happens synchronously (errors render on this tick).
    expect(scrollIntoView).not.toHaveBeenCalled();

    vi.runAllTimers();
    expect(scrollIntoView).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "center",
    });
  });

  it("is a no-op when the container is gone or has no errors", () => {
    // Null container: must not throw.
    scrollToFirstError({ current: null });
    // Container without error nodes: must not throw.
    scrollToFirstError({ current: document.createElement("div") });
    expect(() => vi.runAllTimers()).not.toThrow();
  });
});

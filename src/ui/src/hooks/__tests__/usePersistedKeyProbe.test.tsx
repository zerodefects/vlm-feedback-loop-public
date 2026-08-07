// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StrictMode, type ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  PERSISTED_KEY_PROBE_TIMEOUT_MS,
  usePersistedKeyProbe,
} from "@/hooks/usePersistedKeyProbe";

function StrictWrapper({ children }: { children: ReactNode }) {
  return <StrictMode>{children}</StrictMode>;
}

describe("usePersistedKeyProbe", () => {
  it("sends one credential check when Strict Mode replays mount effects", async () => {
    const probe = vi.fn().mockResolvedValue({ success: true });
    const onRejected = vi.fn();
    const onSettled = vi.fn();

    renderHook(
      () =>
        usePersistedKeyProbe({
          configured: true,
          probe,
          fallbackError: "Credential validation failed.",
          onRejected,
          onSettled,
        }),
      { wrapper: StrictWrapper },
    );

    await waitFor(() => expect(onSettled).toHaveBeenCalledTimes(1));
    expect(probe).toHaveBeenCalledTimes(1);
    expect(onRejected).not.toHaveBeenCalled();
  });

  it("stops a hung provider check from blocking an automatic setup transition", async () => {
    vi.useFakeTimers();
    const probe = vi.fn(
      (_signal?: AbortSignal) =>
        new Promise<{ success: boolean }>(() => {
          // Deliberately never settles: this reproduces a provider request
          // waiting for the backend's much longer inference deadline.
        }),
    );
    const onRejected = vi.fn();
    const onSettled = vi.fn();

    renderHook(() =>
      usePersistedKeyProbe({
        configured: true,
        probe,
        fallbackError: "Credential validation failed.",
        onRejected,
        onSettled,
      }),
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(PERSISTED_KEY_PROBE_TIMEOUT_MS + 1);
    });

    expect(probe).toHaveBeenCalledTimes(1);
    expect(onSettled).toHaveBeenCalledTimes(1);
    expect(onRejected).not.toHaveBeenCalled();
    vi.useRealTimers();
  });
});

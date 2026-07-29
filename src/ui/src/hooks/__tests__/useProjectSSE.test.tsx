// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the useProjectSSE React hook.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useProjectSSE } from "@/hooks/useProjectSSE";
import { useSSEStore } from "@/stores/sse-store";
import { MockEventSource, installEventSourceMock } from "@/test/event-source-mock";

installEventSourceMock();

// ---------------------------------------------------------------------------
// Wrapper with QueryClientProvider
// ---------------------------------------------------------------------------

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return { Wrapper, queryClient };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useProjectSSE", () => {
  beforeEach(() => {
    MockEventSource.reset();
    useSSEStore.setState({ connections: {} });
  });

  afterEach(() => {
    // Clean up all connections
    const state = useSSEStore.getState();
    for (const pid of Object.keys(state.connections)) {
      state.disconnect(pid);
    }
    vi.restoreAllMocks();
  });

  it("calls connect on mount with the projectId", () => {
    const { Wrapper } = createWrapper();

    renderHook(() => useProjectSSE("proj-1"), { wrapper: Wrapper });

    expect(MockEventSource.instances).toHaveLength(1);
    expect(MockEventSource.instances[0].url).toBe("/v1/projects/proj-1/events");
  });

  it("calls disconnect on unmount", () => {
    const { Wrapper } = createWrapper();

    const { unmount } = renderHook(() => useProjectSSE("proj-1"), {
      wrapper: Wrapper,
    });
    const es = MockEventSource.instances[0];

    unmount();

    expect(es.close).toHaveBeenCalled();
    expect(useSSEStore.getState().connections["proj-1"]).toBeUndefined();
  });

  it("returns null lastEvent initially", () => {
    const { Wrapper } = createWrapper();

    const { result } = renderHook(() => useProjectSSE("proj-1"), {
      wrapper: Wrapper,
    });

    expect(result.current.lastEvent).toBeNull();
  });

  it("does not connect when projectId is undefined", () => {
    const { Wrapper } = createWrapper();

    const { result } = renderHook(() => useProjectSSE(undefined), {
      wrapper: Wrapper,
    });

    expect(MockEventSource.instances).toHaveLength(0);
    expect(result.current.lastEvent).toBeNull();
  });
});

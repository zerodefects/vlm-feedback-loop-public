// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the SSE Zustand store — verifies all recovery contract
 * behaviors with a mock EventSource.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { QueryClient } from "@tanstack/react-query";
import { useSSEStore } from "@/stores/sse-store";
import { MockEventSource, installEventSourceMock } from "@/test/event-source-mock";

// ---------------------------------------------------------------------------
// Test setup
// ---------------------------------------------------------------------------

installEventSourceMock();

function latestMock(): MockEventSource {
  return MockEventSource.instances[MockEventSource.instances.length - 1];
}

describe("SSE Store — recovery contract", () => {
  let queryClient: QueryClient;
  let invalidateSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    MockEventSource.reset();
    useSSEStore.setState({ connections: {} });
    vi.useFakeTimers();

    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
  });

  afterEach(() => {
    // Disconnect any remaining connections.
    const state = useSSEStore.getState();
    for (const pid of Object.keys(state.connections)) {
      state.disconnect(pid);
    }
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  // -----------------------------------------------------------------------
  // Connection lifecycle
  // -----------------------------------------------------------------------

  it("connect() creates EventSource with correct URL", () => {
    useSSEStore.getState().connect("proj-1", queryClient);

    expect(MockEventSource.instances).toHaveLength(1);
    expect(latestMock().url).toBe("/v1/projects/proj-1/events");
  });

  it("connect() is a no-op when already connected", () => {
    useSSEStore.getState().connect("proj-1", queryClient);
    useSSEStore.getState().connect("proj-1", queryClient);

    expect(MockEventSource.instances).toHaveLength(1);
  });

  it("disconnect() closes EventSource and removes state", () => {
    useSSEStore.getState().connect("proj-1", queryClient);
    const es = latestMock();

    useSSEStore.getState().disconnect("proj-1");

    expect(es.close).toHaveBeenCalled();
    expect(useSSEStore.getState().connections["proj-1"]).toBeUndefined();
  });

  // -----------------------------------------------------------------------
  // AC-4: onopen → invalidate queries (reconnect refresh)
  // -----------------------------------------------------------------------

  it("onopen invalidates queries (connect refresh)", () => {
    useSSEStore.getState().connect("proj-1", queryClient);
    latestMock().simulateOpen();

    expect(invalidateSpy).toHaveBeenCalled();
  });

  it("onopen after error stops polling and invalidates queries (AC-4)", () => {
    useSSEStore.getState().connect("proj-1", queryClient);

    // Simulate disconnect → polling starts
    latestMock().simulateError();
    expect(
      useSSEStore.getState().connections["proj-1"]?.pollingIntervalId,
    ).not.toBeNull();

    // Clear spy to track only reconnect invalidation
    invalidateSpy.mockClear();

    // Simulate reconnect
    latestMock().simulateOpen();

    expect(useSSEStore.getState().connections["proj-1"]?.pollingIntervalId).toBeNull();
    expect(invalidateSpy).toHaveBeenCalled();
  });

  // -----------------------------------------------------------------------
  // AC-3: onerror → disconnected + REST polling until reconnect
  // -----------------------------------------------------------------------

  it("onerror starts 5s REST polling (AC-3)", () => {
    useSSEStore.getState().connect("proj-1", queryClient);
    invalidateSpy.mockClear();

    latestMock().simulateError();

    // Polling interval should be set
    const conn = useSSEStore.getState().connections["proj-1"];
    expect(conn?.pollingIntervalId).not.toBeNull();

    // Advance timer — invalidateQueries called on each tick
    vi.advanceTimersByTime(5_000);
    expect(invalidateSpy).toHaveBeenCalled();
  });

  it("repeated onerror keeps a single polling interval (AC-3)", () => {
    useSSEStore.getState().connect("proj-1", queryClient);

    latestMock().simulateError();
    const first = useSSEStore.getState().connections["proj-1"]?.pollingIntervalId;
    latestMock().simulateError();
    const second = useSSEStore.getState().connections["proj-1"]?.pollingIntervalId;

    expect(first).not.toBeNull();
    expect(second).toBe(first);
  });

  // -----------------------------------------------------------------------
  // AC-2 + AC-5: named events + terminal events
  // -----------------------------------------------------------------------

  it("named event updates lastEvent (AC-2)", () => {
    useSSEStore.getState().connect("proj-1", queryClient);
    latestMock().simulateOpen();
    invalidateSpy.mockClear();

    latestMock().simulateEvent("evaluation_progress", {
      run_id: "r1",
      processed: 3,
      total: 10,
    });

    const conn = useSSEStore.getState().connections["proj-1"];
    expect(conn?.lastEvent?.type).toBe("evaluation_progress");
    expect(conn?.lastEvent?.data.processed).toBe(3);
    // Non-terminal event should NOT trigger extra invalidation
    expect(invalidateSpy).not.toHaveBeenCalled();
  });

  it("terminal event (*_completed) triggers query invalidation (AC-5)", () => {
    useSSEStore.getState().connect("proj-1", queryClient);
    latestMock().simulateOpen();
    invalidateSpy.mockClear();

    latestMock().simulateEvent("evaluation_completed", {
      run_id: "r1",
      status: "completed",
      summary: { accuracy: 0.9 },
    });

    expect(invalidateSpy).toHaveBeenCalled();
  });

  it("terminal event (run_failed) triggers query invalidation (AC-5)", () => {
    useSSEStore.getState().connect("proj-1", queryClient);
    latestMock().simulateOpen();
    invalidateSpy.mockClear();

    latestMock().simulateEvent("run_failed", {
      run_id: "r1",
      run_type: "evaluation_run",
      error_summary: "timeout",
    });

    expect(invalidateSpy).toHaveBeenCalled();
  });

  it("non-terminal event does NOT trigger query invalidation", () => {
    useSSEStore.getState().connect("proj-1", queryClient);
    latestMock().simulateOpen();
    invalidateSpy.mockClear();

    latestMock().simulateEvent("evaluation_started", { run_id: "r1" });

    expect(invalidateSpy).not.toHaveBeenCalled();
  });

  it("terminal event invalidates run-scoped caches, not just project keys", () => {
    // Regression guard: page-level caches (studentModels, evaluations,
    // …) key on projectId but NOT under the ["project"]/["projects"] prefix.
    // A prefix-only invalidation left the Compare page (SSE-only, no
    // refetchInterval) stuck on a benchmark that finished during an SSE
    // outage. The predicate must match any query whose key contains the id.
    const studentQueryKey = ["studentModels", "proj-1", "list"];
    void queryClient.fetchQuery({
      queryKey: studentQueryKey,
      queryFn: () => Promise.resolve([]),
    });

    useSSEStore.getState().connect("proj-1", queryClient);
    latestMock().simulateOpen();
    invalidateSpy.mockClear();

    latestMock().simulateEvent("nim_benchmark_completed", {
      run_id: "r1",
      student_model_id: "s1",
    });

    // The predicate invalidation must have been asked to run for a key
    // containing the projectId.
    const predicateCall = invalidateSpy.mock.calls.find(
      ([arg]) => typeof arg === "object" && arg !== null && "predicate" in arg,
    );
    expect(predicateCall).toBeDefined();
    const predicate = (
      predicateCall![0] as {
        predicate: (q: { queryKey: readonly unknown[] }) => boolean;
      }
    ).predicate;
    expect(predicate({ queryKey: studentQueryKey })).toBe(true);
    expect(predicate({ queryKey: ["environment", "assessment"] })).toBe(false);
  });

  // -----------------------------------------------------------------------
  // disconnect cleanup
  // -----------------------------------------------------------------------

  it("disconnect clears polling interval", () => {
    useSSEStore.getState().connect("proj-1", queryClient);
    latestMock().simulateError();

    // Polling is running
    expect(
      useSSEStore.getState().connections["proj-1"]?.pollingIntervalId,
    ).not.toBeNull();

    useSSEStore.getState().disconnect("proj-1");

    // No leaked intervals
    expect(useSSEStore.getState().connections["proj-1"]).toBeUndefined();
  });
});

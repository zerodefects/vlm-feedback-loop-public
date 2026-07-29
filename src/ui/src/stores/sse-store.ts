// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * SSE connection store implementing the frontend recovery contract.
 *
 * Core principle: REST is the source of truth, SSE is a performance
 * optimization.  The store manages per-project EventSource connections
 * and triggers React Query cache invalidation on:
 *   - reconnect
 *   - terminal events  (*_completed, run_failed)
 *   - disconnect polling  (5 s interval while active work exists)
 */

import { create } from "zustand";
import type { QueryClient } from "@tanstack/react-query";

import { projectKeys } from "@/api/query-keys";
import type { SSEEvent } from "@/types/sse";

// ---------------------------------------------------------------------------
// Event contract
// ---------------------------------------------------------------------------

/**
 * Known SSE event types from the backend.
 *
 * Native EventSource dispatches **named events** based on the `event:` field.
 * The generic `onmessage` handler only fires for events without an `event:`
 * field.  Since the backend always sends `event: {type}`, we must register
 * `addEventListener(type, handler)` for each known type.
 *
 * Expand this array as new event types are added (one line each).
 */
const KNOWN_SSE_EVENT_TYPES = [
  // Evaluation
  "evaluation_started",
  "evaluation_progress",
  "evaluation_completed",
  // CLIP Embedding
  "embedding_progress",
  "embedding_completed",
  // Ingest pHash sweeper
  "ingest_progress",
  "ingest_completed",
  // Batch Labeling
  "batch_label_progress",
  "batch_label_completed",
  // TAO jobs
  "tao_job_progress",
  "tao_job_completed",
  // NIM benchmarking
  "nim_benchmark_progress",
  "nim_benchmark_completed",
  // Failure (any producer)
  "run_failed",
] as const;

/**
 * Check whether an event type is terminal — triggers REST refresh.
 *
 * Terminal events: anything ending in `_completed` or the literal `run_failed`.
 * This pattern-matches the backend's `*_completed` and `run_failed` naming
 * and is forward-compatible with newly added event types.
 */
function isTerminalEvent(eventType: string): boolean {
  return eventType.endsWith("_completed") || eventType === "run_failed";
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ProjectConnection {
  lastEvent: SSEEvent | null;
  eventSource: EventSource | null;
  pollingIntervalId: ReturnType<typeof setInterval> | null;
}

interface SSEState {
  /** Per-project connection tracking. */
  connections: Record<string, ProjectConnection>;

  /** Open an SSE connection for a project. */
  connect: (projectId: string, queryClient: QueryClient) => void;

  /** Close the SSE connection for a project. */
  disconnect: (projectId: string) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const POLLING_INTERVAL_MS = 5_000;

function emptyConnection(): ProjectConnection {
  return {
    lastEvent: null,
    eventSource: null,
    pollingIntervalId: null,
  };
}

function stopPolling(conn: ProjectConnection): void {
  if (conn.pollingIntervalId !== null) {
    clearInterval(conn.pollingIntervalId);
    conn.pollingIntervalId = null;
  }
}

function invalidateProject(queryClient: QueryClient, projectId: string): void {
  // Refresh the project list (home-screen counts) via its ["projects"] prefix…
  void queryClient.invalidateQueries({ queryKey: projectKeys.all });
  // …plus every cache scoped to this project. Page-level factories
  // (studentModels, evaluations, training, batch, …) key on
  // projectId but do NOT share the ["project"]/["projects"] prefix, so a
  // prefix-only invalidation left the Compare page — SSE-only, no
  // refetchInterval — stuck on a benchmark that finished during an SSE
  // outage. Match any query whose key contains the projectId so new
  // project-scoped factories are covered automatically.
  void queryClient.invalidateQueries({
    predicate: (query) => query.queryKey.includes(projectId),
  });
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useSSEStore = create<SSEState>()((set, get) => ({
  connections: {},

  connect: (projectId: string, queryClient: QueryClient) => {
    const existing = get().connections[projectId];
    // Already connected or connecting — no-op.
    if (existing?.eventSource) return;

    const url = `/v1/projects/${projectId}/events`;
    const es = new EventSource(url);

    // ------ onopen (reconnect → refresh) ------
    es.onopen = () => {
      const conn = get().connections[projectId];
      if (conn) stopPolling(conn);

      set((state) => ({
        connections: {
          ...state.connections,
          [projectId]: {
            ...(state.connections[projectId] ?? emptyConnection()),
            eventSource: es,
            pollingIntervalId: null,
          },
        },
      }));

      // Immediately refresh REST state on (re)connect
      invalidateProject(queryClient, projectId);
    };

    // ------ onerror (disconnect → polling) ------
    es.onerror = () => {
      set((state) => {
        const prev = state.connections[projectId] ?? emptyConnection();

        // While the stream is down, fall back to 5 s REST polling
        // so run progress keeps flowing when SSE is broken but REST is
        // fine (e.g. a proxy severing idle streams). Stops on reconnect
        // (onopen) or disconnect(). Unconditional — the store cannot
        // know whether background work exists, and REST is authoritative
        // either way.
        let pollingId = prev.pollingIntervalId;
        if (pollingId === null) {
          pollingId = setInterval(() => {
            invalidateProject(queryClient, projectId);
          }, POLLING_INTERVAL_MS);
        }

        return {
          connections: {
            ...state.connections,
            [projectId]: {
              ...prev,
              pollingIntervalId: pollingId,
            },
          },
        };
      });
    };

    // ------ Named event listeners ------
    const handleEvent = (msgEvent: Event) => {
      const me = msgEvent as MessageEvent;
      const eventType = me.type;

      let data: Record<string, unknown>;
      try {
        data = JSON.parse(me.data as string) as Record<string, unknown>;
      } catch {
        return; // malformed data — skip
      }

      const sseEvent: SSEEvent = { type: eventType, data };

      set((state) => ({
        connections: {
          ...state.connections,
          [projectId]: {
            ...(state.connections[projectId] ?? emptyConnection()),
            lastEvent: sseEvent,
          },
        },
      }));

      // Terminal events trigger immediate REST refresh.
      if (isTerminalEvent(eventType)) {
        invalidateProject(queryClient, projectId);
      }
    };

    for (const type of KNOWN_SSE_EVENT_TYPES) {
      es.addEventListener(type, handleEvent);
    }

    // Record the EventSource so connect() dedupes and disconnect() can close it.
    set((state) => ({
      connections: {
        ...state.connections,
        [projectId]: {
          ...(state.connections[projectId] ?? emptyConnection()),
          eventSource: es,
        },
      },
    }));
  },

  disconnect: (projectId: string) => {
    const conn = get().connections[projectId];
    if (!conn) return;

    conn.eventSource?.close();
    stopPolling(conn);

    set((state) => {
      const { [projectId]: _, ...rest } = state.connections;
      return { connections: rest };
    });
  },
}));

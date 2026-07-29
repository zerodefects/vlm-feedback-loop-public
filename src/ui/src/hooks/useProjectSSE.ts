// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * React hook for subscribing to project-scoped SSE events.
 *
 * Connects on mount, disconnects on unmount.  Returns the last
 * received event.  Safe to call from any page that needs SSE — the
 * store deduplicates connections per project.
 */

import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { useSSEStore } from "@/stores/sse-store";
import type { SSEEvent } from "@/types/sse";

interface UseProjectSSEResult {
  lastEvent: SSEEvent | null;
}

export function useProjectSSE(projectId: string | undefined): UseProjectSSEResult {
  const queryClient = useQueryClient();
  const connect = useSSEStore((s) => s.connect);
  const disconnect = useSSEStore((s) => s.disconnect);

  const lastEvent = useSSEStore((s) =>
    projectId ? (s.connections[projectId]?.lastEvent ?? null) : null,
  );

  useEffect(() => {
    if (!projectId) return;
    connect(projectId, queryClient);
    return () => disconnect(projectId);
  }, [projectId, connect, disconnect, queryClient]);

  return { lastEvent };
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * SSE client types for the frontend recovery contract.
 */

/** A parsed SSE event from the backend. */
export interface SSEEvent {
  /** The named event type (e.g. "evaluation_progress"). */
  type: string;
  /** Parsed JSON payload. */
  data: Record<string, unknown>;
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Thin fetch wrapper for REST API calls.
 *
 * Uses relative URLs so the Vite dev proxy handles routing in development
 * and same-origin works in production.
 */

const BASE = "/v1";

export class ApiError extends Error {
  readonly status: number;
  readonly body: string;

  constructor(status: number, body: string) {
    super(`API ${status}: ${body}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

/**
 * Extract the backend's structured ``detail`` string from a failed
 * request. ``ApiError.body`` is raw response text; the backend's error
 * handlers emit JSON like ``{"detail": "..."}``, but proxy errors and
 * plain-text bodies also flow through here, so parsing must never
 * throw. Returns null when ``err`` is not an ApiError or the body
 * carries no non-empty string detail — callers supply their own
 * fallback message.
 */
export function parseApiErrorDetail(err: unknown): string | null {
  if (!(err instanceof ApiError)) return null;
  try {
    const parsed = JSON.parse(err.body) as { detail?: unknown };
    if (typeof parsed.detail === "string" && parsed.detail.length > 0) {
      return parsed.detail;
    }
  } catch {
    // Non-JSON body (proxy error page, plain text) — no structured detail.
  }
  return null;
}

/**
 * Companion to ``parseApiErrorDetail`` for the busy-gate 409 shape,
 * whose body carries a ``reasons: string[]`` list alongside ``detail``.
 * Returns an empty list when absent or unparseable.
 */
export function parseApiErrorReasons(err: unknown): string[] {
  if (!(err instanceof ApiError)) return [];
  try {
    const parsed = JSON.parse(err.body) as { reasons?: unknown };
    if (
      Array.isArray(parsed.reasons) &&
      parsed.reasons.every((r): r is string => typeof r === "string")
    ) {
      return parsed.reasons;
    }
  } catch {
    // Non-JSON body — no structured reasons.
  }
  return [];
}

/**
 * Typed fetch wrapper.  All REST calls go through this.
 *
 * @param path - API path **without** the `/v1` prefix (e.g. `/projects`).
 * @param init - Standard `RequestInit` overrides.
 */
export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!res.ok) {
    const text = await res.text();
    throw new ApiError(res.status, text);
  }

  return res.json() as Promise<T>;
}

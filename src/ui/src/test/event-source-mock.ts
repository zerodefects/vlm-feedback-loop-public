// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared EventSource test double.
 *
 * Two usage tiers, one class:
 *
 * 1. Page tests that merely need the SSE store not to crash on mount call
 *    `installEventSourceMock()` at module scope and ignore the return value.
 * 2. Event-driven tests (sse-store, useProjectSSE) use the captured
 *    `MockEventSource.instances` plus the `simulate*` helpers to drive
 *    open/error/named-event flows.
 *
 * Reset `MockEventSource.instances` (or call `MockEventSource.reset()`) in
 * `beforeEach` when a test reads the instances list.
 */

import { vi } from "vitest";

type EventHandler = (event: Event) => void;

export class MockEventSource {
  static instances: MockEventSource[] = [];

  static reset(): void {
    MockEventSource.instances = [];
  }

  url: string;
  readyState = 0; // CONNECTING

  onopen: ((ev: Event) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;

  private listeners: Record<string, EventHandler[]> = {};

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(type: string, handler: EventHandler): void {
    if (!this.listeners[type]) this.listeners[type] = [];
    this.listeners[type].push(handler);
  }

  removeEventListener(type: string, handler: EventHandler): void {
    const list = this.listeners[type];
    if (list) {
      this.listeners[type] = list.filter((h) => h !== handler);
    }
  }

  close = vi.fn();

  // -- Simulation helpers --

  simulateOpen(): void {
    this.readyState = 1; // OPEN
    this.onopen?.(new Event("open"));
  }

  simulateError(): void {
    this.readyState = 2; // CLOSED
    this.onerror?.(new Event("error"));
  }

  simulateEvent(type: string, data: Record<string, unknown>): void {
    const me = new MessageEvent(type, { data: JSON.stringify(data) });
    for (const handler of this.listeners[type] ?? []) {
      handler(me);
    }
  }
}

/**
 * Stub the global `EventSource` with {@link MockEventSource} for the current
 * test file. Call at module scope, before the first render that mounts the
 * SSE store. Returns the class so event-driven tests can reach the captured
 * instances.
 */
export function installEventSourceMock(): typeof MockEventSource {
  vi.stubGlobal("EventSource", MockEventSource);
  return MockEventSource;
}

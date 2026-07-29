// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterAll, beforeAll } from "vitest";

// React calls console.error with format-string + arg form, e.g.
// `console.error("An update to %s inside a test was not wrapped in
// act(...)", "TestComponent")`, so the filter inspects args[0] (the
// format string) for this static fragment.
const ACT_WARNING_FRAGMENT = "An update to %s inside a test was not wrapped in act";

/**
 * Per-file opt-in filter for React's "not wrapped in act(...)" warning.
 * Call once at module scope of a test file; installs via beforeAll and
 * restores the real console.error in afterAll.
 *
 * Only opt in for KNOWN false positives of testing-library's act-wrapper
 * detection, where the component code is correct:
 *
 * - External-store subscriptions: Zustand's `set()` notifies subscribers
 *   synchronously, so re-renders it triggers inside an effect-scoped call
 *   are flagged even though the hook code is fine.
 * - Mid-flight async loops observed via `waitFor`: a handler that awaits
 *   sequential requests and sets progress state after each one (e.g.
 *   ImageIngestPage's batch-ingest loop driven by deferred-resolution
 *   mocks) resumes in microtasks between `waitFor` polls, outside any
 *   act scope.
 *
 * New test files should NOT opt in by default — an unexpected act warning
 * often points at a real bug (setState after unmount, un-awaited async in
 * an effect). Diagnose first; suppress only once the trigger is understood
 * and benign.
 */
export function suppressActWarnings(): void {
  let realConsoleError: typeof console.error;

  beforeAll(() => {
    realConsoleError = console.error.bind(console);
    console.error = (...args: unknown[]): void => {
      const first = args[0];
      if (typeof first === "string" && first.includes(ACT_WARNING_FRAGMENT)) {
        return;
      }
      realConsoleError(...args);
    };
  });

  afterAll(() => {
    console.error = realConsoleError;
  });
}

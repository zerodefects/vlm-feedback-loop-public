// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import "@testing-library/jest-dom";

// JSDOM does not implement scrollIntoView; stub so components that call it in
// handlers (e.g. scroll-to-first-error on Save click) don't throw in tests.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function () {};
}

// JSDOM does not implement ResizeObserver. The image zoom viewer uses it to
// keep pan bounds aligned with its card when the desktop window changes size.
if (!globalThis.ResizeObserver) {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }

  Object.defineProperty(globalThis, "ResizeObserver", {
    configurable: true,
    writable: true,
    value: ResizeObserverMock,
  });
}

// React act(...) warnings are NOT suppressed globally — a genuine
// async-setState bug in any component must stay visible. Test files whose
// warnings are a known false positive opt in per-file via
// `suppressActWarnings()` from src/test/suppress-act-warnings.ts.

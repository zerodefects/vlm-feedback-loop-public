// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Scroll a form's first visible validation error into view.
 *
 * Shared by the Create/Edit Guidance screens: errors become visible on
 * the render that flips ``saveAttempted``, so the lookup is deferred a
 * tick (setTimeout 0) to let the DOM catch up before scrolling. Error
 * nodes are located by the ``data-testid="error-*"`` convention the
 * guidance builders already use.
 *
 * Accepts the container *ref* (not the element) so the deref also
 * happens after the deferred tick.
 */
export function scrollToFirstError(containerRef: {
  current: HTMLElement | null;
}): void {
  setTimeout(() => {
    containerRef.current
      ?.querySelector('[data-testid^="error-"]')
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, 0);
}

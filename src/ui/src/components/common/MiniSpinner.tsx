// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * 14 px inline spinner for pill and body-text contexts.
 *
 * KUI Spinner ``size="small"`` renders ~32 px tall regardless of container —
 * it has its own intrinsic size that ignores the surrounding text-size
 * context, so dropped raw next to ``text-xs`` pill text or ``body/regular/sm``
 * copy it inflates the line and shifts sibling layout. ``transform: scale``
 * is the only mechanism that resizes the rendered spinner without clipping
 * the spinning hex animation; the fixed-dimension wrapper makes adjacent
 * text align as if the spinner were natively 14 px.
 *
 * The ``.mini-spinner`` class (index.css) redirects KUI's arrow-fill
 * token to ``currentColor`` across the subtree, so the spinner takes
 * the host's text color — a blue info pill gets a blue spinner instead
 * of a fixed-green one clashing with its tone.
 *
 * Decorative by contract: the wrapper is ``aria-hidden`` and the host
 * pill/text conveys the in-progress state, so no accessible label is
 * exposed (the empty ``aria-label`` only satisfies KUI's prop union).
 */

import { Spinner } from "@kui/react";

export function MiniSpinner() {
  return (
    <span
      className="mini-spinner inline-flex items-center justify-center shrink-0"
      style={{ width: 14, height: 14 }}
      aria-hidden="true"
    >
      <span
        className="inline-flex"
        style={{ transform: "scale(0.45)", transformOrigin: "center" }}
      >
        <Spinner aria-label="" size="small" />
      </span>
    </span>
  );
}

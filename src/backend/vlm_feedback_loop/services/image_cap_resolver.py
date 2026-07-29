# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolution helper for the per-endpoint ``max_images_per_request`` override.

Background
==========

The image-content-parts cap is a property of the **endpoint** (gateway
clamp / NIM container / model layer), not the model alone:

* Hosted Mistral and hosted cosmos cap at 8 images per request because
  of a ``build.nvidia.com`` gateway clamp (the rejection text is
  literally ``"At most 8 image(s) may be provided in one prompt"`` —
  that's a gateway pattern uniform across hosted Teachers).
* The SAME cosmos-reason2-2b/8b model running locally caps at whatever
  the NIM container's ``NIM_MAX_IMAGES_PER_PROMPT`` resolves to, which
  feeds vLLM's ``limit_mm_per_prompt['image']`` validator (rejection
  text: ``"At most N image(s) may be provided in one prompt. None"`` —
  the trailing ``None`` is the vLLM-layer tell vs. the gateway clamp).
  That default is **model-profile-specific and changes across NIM
  versions**: the ``:1.6.0`` cosmos profile ships a very high default
  (≈999 — effectively uncapped), but the ``:1.7.0`` family (cosmos-
  reason2-2b/8b AND cosmos3-reasoner nano/super) silently drops it to
  **5** (empirically: N=6 → HTTP 400). Lesson: never trust a
  hardcoded per-version table for the local cap.

So the same model can run on two endpoints with different effective
caps. ``ModelConfig.max_images_per_request`` is per-model; storing the
cap there forces hosted and local to share a value, which is wrong.
``local_nim_service._build_docker_run_command`` sets
``NIM_MAX_IMAGES_PER_PROMPT`` on app-driven local deploys to the
resolved cap, so the NIM accepts exactly what this resolver lets the
backend send (no silent 400 on a 6th ICL image against ``:1.7.0``).

This module's resolver returns the **effective** cap by preferring the
per-endpoint override (``NimEndpoint.max_images_per_request``, set to
non-NULL by the operator or by
``local_nim_service._auto_set_endpoint_image_cap`` on healthy local
deploys)
and falling back to the per-model value (``ModelConfig.max_images_per_request``)
when the override is unset. The fallback keeps hosted-NIM workflows
that never set the override on the gateway-correct per-model cap.

Same pattern for the capability-probe field (``image_cap_support``).

The override columns default to NULL, so rows without
an operator-set override always take the per-model fallback.
"""

from __future__ import annotations

from typing import Any


def resolve_max_images_per_request(
    *,
    model_config: Any,
    nim_endpoint: Any | None = None,
) -> int:
    """Return the effective ``max_images_per_request`` for an
    (model_config, nim_endpoint) pair.

    Prefers ``NimEndpoint.max_images_per_request`` when non-NULL; falls
    back to ``ModelConfig.max_images_per_request``.

    The function accepts ``None`` for ``nim_endpoint`` (some test paths
    construct ``ModelConfigInput`` without an endpoint reference); in
    that case the per-model value is returned unchanged.

    Both arguments are passed by keyword to make call sites self-
    documenting at the reader sites in ``proposal_service`` /
    ``evaluation_service`` / ``batch_label_service``.
    """
    if nim_endpoint is not None:
        override = getattr(nim_endpoint, "max_images_per_request", None)
        if override is not None:
            return int(override)
    return int(model_config.max_images_per_request)

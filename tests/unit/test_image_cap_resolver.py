# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``services/image_cap_resolver.py``.

``max_images_per_request`` and ``image_cap_support`` are nullable overrides on
``NimEndpoint``. The resolver picks the
endpoint override when set, falls back to the per-model
``ModelConfig`` value otherwise. These tests pin the four cases the
proposal/eval/batch services rely on.

Why this matters: the same model can run on two endpoints with
different effective caps (hosted Mistral capped at 8 by build.nvidia.com,
local cosmos-reason2-2b NIM accepts 999). Storing the cap on
``ModelConfig`` alone forced both endpoints to share a value, which is
wrong. The resolver lets the local-NIM auto-registered endpoint carry
its own override without changing the seeded per-model value.
"""

from __future__ import annotations

from types import SimpleNamespace

from vlm_feedback_loop.services.image_cap_resolver import (
    resolve_max_images_per_request,
)


def _mc(value: int = 8, support: str = "supported") -> SimpleNamespace:
    return SimpleNamespace(
        max_images_per_request=value,
        image_cap_support=support,
    )


def _ep(value: int | None = None, support: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        max_images_per_request=value,
        image_cap_support=support,
    )


class TestResolveMaxImagesPerRequest:
    def test_falls_back_to_model_config_when_endpoint_override_unset(self):
        """The default state for hosted endpoints — the override column
        is NULL on every existing nim_endpoints row post-migration. The
        resolver MUST return the per-model value to preserve backwards
        compatibility (hosted-NIM workflows see no behavior change)."""
        assert (
            resolve_max_images_per_request(
                model_config=_mc(value=8), nim_endpoint=_ep(value=None)
            )
            == 8
        )

    def test_endpoint_override_wins_when_set(self):
        """Local-NIM auto-registered endpoints carry an override. The
        resolver MUST prefer the per-endpoint value over the per-model
        value — that's the point of the override."""
        assert (
            resolve_max_images_per_request(
                model_config=_mc(value=8), nim_endpoint=_ep(value=32)
            )
            == 32
        )

    def test_no_endpoint_returns_model_config(self):
        """Some test paths construct ``ModelConfigInput`` from a
        ``ModelConfig`` row without loading the paired ``NimEndpoint``.
        The resolver MUST handle ``nim_endpoint=None`` cleanly."""
        assert (
            resolve_max_images_per_request(model_config=_mc(value=8), nim_endpoint=None)
            == 8
        )

    def test_endpoint_zero_does_not_fall_back(self):
        """Edge: a deliberate ``0`` on the endpoint MUST NOT be treated
        as ``None``. Only Python ``None`` falls back. (``0`` is a nonsense
        cap in practice — the system rejects N=0 image-content-parts in
        prompt rendering anyway — but the resolver must still respect
        an explicit zero.)"""
        # Note: ``getattr(..., None)`` returns 0, which is falsy but not
        # None. The resolver's check is ``is None`` so 0 is honored.
        assert (
            resolve_max_images_per_request(
                model_config=_mc(value=8), nim_endpoint=_ep(value=0)
            )
            == 0
        )

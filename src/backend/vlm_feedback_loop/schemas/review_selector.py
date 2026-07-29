# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for the review selector endpoint."""

from __future__ import annotations

from pydantic import BaseModel


class ReviewSelectorNextResponse(BaseModel):
    """Response from ``GET .../review_selector/next``."""

    example_key: str | None
    example_state: str | None
    has_existing_label: bool
    selection_mode: str
    queue_empty: bool
    # Example fields the labeling screen needs alongside the selection, so
    # the frontend does not have to re-query the examples list to find the
    # record it was just handed (these mirror Example columns).
    # ``storage_ref`` feeds the missing-image diagnostic ("Expected:" path);
    # ``prior_verified_label_ref`` is the JSON prior-label snapshot that
    # powers the labeling screen's per-field re-labeling hints after a
    # semantic schema change. Both are null when ``queue_empty=true``.
    storage_ref: str | None
    prior_verified_label_ref: str | None

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal boto3 / botocore typing shim.

Boto3 ships no public type stubs (the third-party ``boto3-stubs`` package
is the closest, but this project's typing policy is "no stub packages").
Without this shim, the two TAO-side modules that need an S3 client
(``tao_polling_service`` for artifact downloads and
``tao_dataset_upload_service`` for dataset uploads) would each carry
their own ``import boto3`` / ``from botocore.config import Config`` pair
plus per-line ``# pyright: ignore`` comments. This module consolidates
the typing waiver into one place so every caller can import the symbols
without paying a per-call ignore.

The pattern matches Retail-Agentic-Commerce's ``a2a-sdk`` wrapper — a
single shim that takes the ``reportMissingTypeStubs`` hit once. The
``Any``-typed re-exports propagate downstream; the actual call-site
``boto3.client(...)`` returns ``Any`` and stays usable without further
ignores.

Usage::

    from vlm_feedback_loop.services._boto3_shim import boto3, BotoConfig
    client = boto3.client("s3", endpoint_url=..., ...)  # Any
"""

from __future__ import annotations

from typing import Any

import boto3 as _boto3  # pyright: ignore[reportMissingTypeStubs] — no public stubs ship for boto3
from botocore.config import (  # pyright: ignore[reportMissingTypeStubs] — botocore has no public stubs either
    Config as _BotoConfig,
)

# Re-export with an explicit ``Any`` annotation so downstream call sites
# don't pay the partial-unknown cascade for ``boto3.client(...)``.
boto3: Any = _boto3
BotoConfig: Any = _BotoConfig

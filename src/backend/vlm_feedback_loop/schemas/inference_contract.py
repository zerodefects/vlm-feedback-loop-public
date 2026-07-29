# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference Contract model.

The Inference Contract specifies the effective output format, ICL
demonstration format, and ICL sizing controls that apply at inference time.
The Teacher contract is fixed; Student contracts are derived from training.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class InferenceContract(BaseModel):
    """Per-model inference contract.

    Fields:
        output_field_mode: Which field groups the model produces in output.
        icl_field_mode: Which field groups appear in ICL demonstrations.
        icl_max_examples: Maximum ICL examples (None = no limit).
    """

    model_config = ConfigDict(extra="forbid")

    output_field_mode: Literal["all", "aux_and_core", "core_only"]
    icl_field_mode: Literal["all", "aux_and_core", "core_only"]
    icl_max_examples: int | None = None


# Fixed Teacher contract — Core-only demonstrations keep the correction signal
# compact; the Teacher still produces the full output schema.
TEACHER_CONTRACT = InferenceContract(
    output_field_mode="all",
    icl_field_mode="core_only",
)
